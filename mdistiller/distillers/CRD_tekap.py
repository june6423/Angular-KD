import torch
from torch import nn
import torch.nn.functional as F
import math

from ._base import Distiller
from .MLKD_tekap import MLKD_tekap



def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=4.0):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def randomize(x):
    x_noise = x.clone()
    noise = torch.randn_like(x_noise) * 0.1 + 0.0
    x_noise = 0.9 * x_noise + 0.1 * noise
    return x_noise

class CRD_tekap(Distiller):
    """Contrastive Representation Distillation"""

    def __init__(self, student, teacher, cfg, num_data):
        super(CRD_tekap, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CRD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.CRD.LOSS.FEAT_WEIGHT
        self.cfg = cfg
        self.teakp_augnum = cfg.TEKAP.AUGNUM
        self.init_crd_modules(
            cfg.CRD.FEAT.STUDENT_DIM,
            cfg.CRD.FEAT.TEACHER_DIM,
            cfg.CRD.FEAT.DIM,
            num_data,
            cfg.CRD.NCE.K,
            cfg.CRD.NCE.MOMENTUM,
            cfg.CRD.NCE.TEMPERATURE,
        )

    def init_crd_modules(
        self,
        feat_s_channel,
        feat_t_channel,
        feat_dim,
        num_data,
        k=16384,
        momentum=0.5,
        temperature=0.07,
    ):
        self.embed_s = Embed(feat_s_channel, feat_dim)
        self.embed_t = Embed(feat_t_channel, feat_dim)
        self.contrast = ContrastMemory(feat_dim, num_data, k, temperature, momentum)
        self.criterion_s = ContrastLoss(num_data)
        self.criterion_t = ContrastLoss(num_data)

    def get_learnable_parameters(self):
        return (
            super().get_learnable_parameters()
            + list(self.embed_s.parameters())
            + list(self.embed_t.parameters())
        )

    def get_extra_parameters(self):
        params = (
            list(self.embed_s.parameters())
            + list(self.embed_t.parameters())
            + list(self.contrast.buffers())
        )
        num_p = 0
        for p in params:
            num_p += p.numel()
        return num_p

    def crd_loss(self, f_s, f_t, idx, contrast_idx):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        return s_loss + t_loss

    def forward_train(self, image_weak, image_strong, target, index, contrastive_index, **kwargs):
        logits_student, feature_student = self.student(image_weak)
        
        assert self.cfg.TEKAP.USAGE is True
        
        logits_student_weak, feature_student = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        
        with torch.no_grad():
            logits_teacher_weak, feature_teacher = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)
        
        mlkd = MLKD_tekap(self.cfg)
        _ , loss_mlkd_dict = mlkd.mlkd_loss(logits_student_weak, logits_student_strong, logits_teacher_weak, logits_teacher_strong, target= target)                
        
        loss_crd = self.feat_loss_weight * self.crd_loss(
            feature_student["pooled_feat"],
            feature_teacher["pooled_feat"],
            index,
            contrastive_index,
        )
        loss_dict = {**loss_mlkd_dict}
        loss_dict['loss_crd'] = loss_crd
                    
        for i in range(self.cfg.TEKAP.AUGNUM):
            feat_student_noise = randomize(feature_student["pooled_feat"])
            feat_teacher_noise = randomize(feature_teacher["pooled_feat"])
            
            logits_student_weak_noise = randomize(logits_student_weak)
            logits_student_strong_noise = randomize(logits_student_strong)
            logits_teacher_weak_noise = randomize(logits_teacher_weak)
            logits_teacher_strong_noise = randomize(logits_teacher_strong)
            
            _, loss_mlkd_dict = mlkd.mlkd_loss(logits_student_weak_noise, logits_student_strong_noise, logits_teacher_weak_noise, logits_teacher_strong_noise,target= target)
            
            for key in loss_mlkd_dict:
                if key in loss_dict:
                    loss_dict[key] += loss_mlkd_dict[key]
                else:
                    loss_dict[key] = loss_mlkd_dict[key]
                    
            loss_dict['loss_crd'] +=  self.feat_loss_weight * self.crd_loss(
                feat_student_noise,
                feat_teacher_noise,
                index,
                contrastive_index,
            )

        return logits_student_weak, loss_dict
        


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class ContrastLoss(nn.Module):
    """contrastive loss"""

    def __init__(self, num_data):
        super(ContrastLoss, self).__init__()
        self.num_data = num_data

    def forward(self, x):
        eps = 1e-7
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.num_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class ContrastMemory(nn.Module):
    """memory buffer that supplies large amount of negative samples."""

    def __init__(self, inputSize, output_size, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.n_lem = output_size
        self.unigrams = torch.ones(self.n_lem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory_v1", torch.rand(output_size, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        self.register_buffer(
            "memory_v2", torch.rand(output_size, inputSize).mul_(2 * stdv).add_(-stdv)
        )

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            # print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            # print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """Draw N samples from multinomial"""
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj
