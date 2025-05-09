import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

from ._base import Distiller


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


def randomize(x):
    x_noise = x.clone()
    noise = torch.randn_like(x_noise)
    x_noise = 0.9 * x_noise + 0.1 * noise
    return x_noise


def hcl_loss_tekap(fstudent, fteacher, augnum):
    loss_all = 0.0
    for index in range(augnum+1):
        for fs, ft in zip(fstudent, fteacher):
            n, c, h, w = fs.shape
            if index != 0:
                fs_noise = randomize(fs)
                ft_noise = randomize(ft)
            else:
                fs_noise = fs
                ft_noise = ft
            
            loss = F.mse_loss(fs_noise, ft_noise, reduction="mean")
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs_noise, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft_noise, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
    return loss_all


class ReviewKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ReviewKD, self).__init__(student, teacher)
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL
        self.cfg = cfg

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.abfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.abfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        
        with torch.no_grad():
            if self.cfg.DIV.USAGE:
                logits_teacher, features_teacher, loss_dict = self.teacher(image, loss=True, target=target)
            else:
                logits_teacher, features_teacher = self.teacher(image)
        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        log_logits_student = F.log_softmax(logits_student/self.cfg.KD.TEMPERATURE, dim=1)
        if self.cfg.DIV.USAGE:
            kd_loss = F.kl_div(log_logits_student, logits_teacher, size_average=False) * (self.cfg.KD.TEMPERATURE**2) / logits_student.shape[0]
        else:
            kd_loss = F.kl_div(log_logits_student, F.softmax(logits_teacher/self.cfg.KD.TEMPERATURE,dim=1), size_average=False) * (self.cfg.KD.TEMPERATURE**2) / logits_student.shape[0]
        
        if self.cfg.TEKAP.USAGE:
            for i in range(self.cfg.TEKAP.AUGNUM+1):
                logits_teacher_clone = randomize(logits_teacher)
                logits_teacher_clone = F.softmax(logits_teacher_clone/self.cfg.KD.TEMPERATURE, dim=1)
                kd_loss += 0.8 * F.kl_div(log_logits_student, logits_teacher_clone, size_average=False) * (self.cfg.KD.TEMPERATURE**2) / logits_student.shape[0]
            
            loss_reviewkd = (
                self.reviewkd_loss_weight
                * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
                * hcl_loss_tekap(results, features_teacher, self.cfg.TEKAP.AUGNUM)
            )
            
            for i in range(self.cfg.TEKAP.AUGNUM+1):
                features_teacher_clone = [ randomize(x) for x in features_teacher ]                  
                loss_reviewkd += 0.8 * (
                    self.reviewkd_loss_weight
                    * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
                    * hcl_loss_tekap(results, features_teacher_clone, self.cfg.TEKAP.AUGNUM)
                )
            
        else:
            loss_reviewkd = (
                self.reviewkd_loss_weight
                * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
                * hcl_loss(results, features_teacher)
            )
            
        if self.cfg.DIV.USAGE:
            for k, v in loss_dict.items():
                loss_reviewkd += v
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd + kd_loss,
        }
        return logits_student, losses_dict


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x
