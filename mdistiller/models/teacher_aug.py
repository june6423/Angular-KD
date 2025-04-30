import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg
from .diverse_loss import Feature_inter_Loss, Feature_intra_Loss, Logit_inter_Loss, Logit_intra_Loss


class View_Generator(torch.nn.Module):
    def __init__(self, number_of_view, teacher_channel, num_classes=100, dropout_prob=0.20):
        super(View_Generator, self).__init__()
        self.number_of_view = number_of_view
        self.teacher_channel = teacher_channel
        self.num_classes = num_classes
        
        prob_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        batch_norm_mean = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        batch_norm_std = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        self.gt_noise_mean = [torch.normal(mean=0.0, std=1, size=(1,)).item() for i in range(number_of_view)]
        self.gt_noise_std = [torch.normal(mean=0.0, std=1, size=(1,)).item() + 1 for i in range(number_of_view)]

        self.generators = nn.ModuleList([nn.Linear(teacher_channel, teacher_channel) for i in range(number_of_view)])
        self.classifiers = nn.ModuleList([nn.Linear(teacher_channel, self.num_classes) for i in range(number_of_view)])
        self.dropout = nn.ModuleList([nn.Dropout(p=prob_list[i]) for i in range(number_of_view)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(teacher_channel) for i in range(number_of_view)])
        
        for i in range(number_of_view):
            nn.init.xavier_uniform_(self.classifiers[i].weight)
            nn.init.constant_(self.classifiers[i].bias, 0)
        
        for bn in range(number_of_view):
            nn.init.constant_(self.batch_norm[bn].weight, torch.normal(mean=batch_norm_mean[i], std=batch_norm_std[i], size=(1,)).item())
            nn.init.constant_(self.batch_norm[bn].bias, torch.normal(mean=batch_norm_mean[i], std=batch_norm_std[i], size=(1,)).item())

        ## Initialize the weights
        A = torch.randn(teacher_channel, teacher_channel, number_of_view) # [128, 128, 3]
        #Q,R = torch.qr(A)
        Q,R = torch.linalg.qr(A, mode='reduced')
        for i, generator in enumerate(self.generators):
            generator.weight = nn.Parameter(Q[:,:,i], requires_grad=True) # [256, 256]
            
        
    def forward(self, x):
        view_logit_list = []
        view_feature_list = []
        for generator, classifiers, drop_out, bn in zip(self.generators, self.classifiers, self.dropout, self.batch_norm):
            x_ = drop_out(x)
            x_ = generator(x_)
            x_ = bn(x_)
            view_feature_list.append(x_)
            x_ = classifiers(x_)
            view_logit_list.append(x_)
        return view_logit_list, view_feature_list


def weight_sum_logit(logit_t, view_list, temp=4):
    
    weight = [0.8] * len(view_list)
    weight.append(1.0)
    weight = torch.tensor(weight).cuda()
    weight = weight.view(1, -1, 1)
    weight = F.normalize(weight, p=2, dim=1)        
    
    view_list.append(logit_t)
    logit_tensor = torch.stack(view_list, dim=1) # [B, N, D]
    logit_tensor = F.softmax(logit_tensor/temp, dim=2) # [B, N, D]
    logit_tensor = torch.sum(logit_tensor * weight, dim=1) # [B, D]
    return logit_tensor


def weight_sum_feature(feat_t, view_list):
    
    weight = [0.8] * len(view_list)
    weight.append(1.0)
    weight = torch.tensor(weight).cuda()
    weight = weight.view(1, -1, 1)
    weight = nn.functional.normalize(weight, p=2, dim=1)

    view_list.append(feat_t)
    feat_tensor = torch.stack(view_list, dim=1) # [B, N, D]
    feat_tensor = torch.sum(feat_tensor * weight, dim=1) # [B, D]
    return feat_tensor
   

class TeacherEnsemble(nn.Module):
    def __init__(self, cfg, original_teacher, num_classes=100, dropout_prob=0.20):
        super(TeacherEnsemble, self).__init__()
        self.cfg = cfg
        self.teacher_channel = cfg.CRD.FEAT.TEACHER_DIM
        self.original_teacher = original_teacher
        self.number_of_view = cfg.DIV.AUGNUM
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        self.view_generator = View_Generator(self.number_of_view, self.teacher_channel, self.num_classes, self.dropout_prob)
        
        self.feature_inter_loss = Feature_inter_Loss()
        self.feature_intra_loss = Feature_intra_Loss()
        self.logit_inter_loss = Logit_inter_Loss()
        self.logit_intra_loss = Logit_intra_Loss()
        
    def forward(self, x, loss = False, target = None, temp=4, various_temp = False):
        
        assert loss in [True, False]
        
        loss_dict = {}
        
        with torch.no_grad():
            logit_t, feat_t = self.original_teacher(x)
        
        pooled_feat_grad_on = feat_t["pooled_feat"].detach().requires_grad_()
        
        assert pooled_feat_grad_on.dim() == 2

        view_logit_list, view_feature_list = self.view_generator(pooled_feat_grad_on)
        
        
        loss_dict["feature_inter_loss"] = self.cfg.DIV.FEAT_INTERWEIGHT * self.feature_inter_loss(feat_t["pooled_feat"], view_feature_list)
        loss_dict["feature_intra_loss"] = self.cfg.DIV.FEAT_INTRAWEIGHT * self.feature_intra_loss(feat_t["pooled_feat"], view_feature_list)
        loss_dict["logit_inter_loss"] = self.cfg.DIV.LOGIT_INTERWEIGHT * self.logit_inter_loss(logit_t, view_logit_list)
        loss_dict["logit_intra_loss"] = self.cfg.DIV.LOGIT_INTRAWEIGHT * self.logit_intra_loss(logit_t, view_logit_list)
        
        if target is not None:
            ce_loss = 0
            for i in range(self.number_of_view):
                ce_loss += F.cross_entropy(view_logit_list[i], target)
            loss_dict["ce_loss"] = 0.8 * ce_loss          
        
        logit_tensor = weight_sum_logit(logit_t, view_logit_list, temp=temp)
        feat_tensor = weight_sum_feature(feat_t["pooled_feat"], view_feature_list)
        feat_t['pooled_feat'] = feat_tensor
        
        if various_temp:
            logit_dict = {}
            for temp in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                logit_dict[temp] = weight_sum_logit(logit_t, view_logit_list, temp=temp)
            return logit_tensor, feat_t, loss_dict, logit_dict
        
        if loss:
            return logit_tensor, feat_t, loss_dict
        return logit_tensor, feat_t
