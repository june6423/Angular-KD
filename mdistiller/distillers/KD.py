import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def randomize(x):
    x_noise = x.clone()
    noise = torch.randn_like(x_noise)
    x_noise = 0.9 * x_noise + 0.1 * noise
    return x_noise


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.cfg = cfg

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        p_s = F.log_softmax(logits_student / self.temperature, dim=1)
                
        if self.cfg.DIV.USAGE:
            logits_teacher, feature_teacher, losses_dict = self.teacher(image, loss=True, target=target)
            loss_kd = F.kl_div(p_s, logits_teacher, size_average=False) * (self.cfg.KD.TEMPERATURE**2) / logits_student.shape[0]
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            
            losses_dict['loss_kd'] = loss_kd
            losses_dict['loss_ce'] = loss_ce
        
        else:
            with torch.no_grad():
                logits_teacher, _ = self.teacher(image) 
                
            # losses
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
            )
            
            if self.cfg.TEKAP.USAGE:
                for i in range(self.cfg.TEKAP.AUGNUM):
                    logit_noise = randomize(logits_teacher)
                    logit_noise = F.softmax(logit_noise/self.temperature, dim=1)
                    loss_kd += self.ce_loss_weight * 0.8* (F.kl_div(p_s, logit_noise, size_average=False) * (self.temperature**2) / logits_student.shape[0])
            
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }
        return logits_student, losses_dict
