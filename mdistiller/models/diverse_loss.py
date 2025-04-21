import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature_inter_Loss(nn.Module):
    def __init__(self):
        super(Feature_inter_Loss, self).__init__()
        self.margin = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.temperature = 0.07
        
    def forward(self, f_t, view_feature_list, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The inter angle loss
        """
        
        N = len(view_feature_list)
        B, D = view_feature_list[0].size()
        
        assert N > 0
        assert f_t.shape == view_feature_list[0].shape
        
        view_feature_list_temp = view_feature_list.copy()
        
        view_feature_tensor = torch.stack(view_feature_list_temp, dim=1) # [B, N, D]
        view_feature_tensor = F.normalize(view_feature_tensor, p=2, dim=2)
        view_feature_tensor = view_feature_tensor.view(B*N, D) # [B*N, D]
        
        f_t = f_t # [B, D]
        f_t = F.normalize(f_t, p=2, dim=1) # [B, D]
        
        similarity_matrix = torch.matmul(view_feature_tensor, f_t.t()) # [B*N, B]
        labels = torch.arange(B * N, device=similarity_matrix.device) // N # [B * N], 0,0,0, 1,1,1,1, 2,2,2,2


        col_classes = torch.arange(B, device=labels.device)  # [B]
        positive_mask = (labels.unsqueeze(1) == col_classes.unsqueeze(0)) # [B*N, B]


        margin = self.margin.item()
        mask = positive_mask & (similarity_matrix >= (1-margin))
        similarity_matrix[mask] = 1.

        similarity_matrix /= self.temperature # [B*N, B]
        feature_cont_loss = F.cross_entropy(similarity_matrix, labels, reduction='mean') / (N) # [1]
        
        '''
        Diversity loss
        '''
        view_feature_list_temp = view_feature_list.copy()
        view_feature_list_temp_ = torch.stack(view_feature_list_temp, dim=1) # [B, N, D]
        view_feature_list_temp = F.normalize(view_feature_list_temp_, p=2, dim=1) # [batch_size, N, D]
        f_t = F.normalize(f_t, p=2, dim=1) # [batch_size, D]

        sim_with_teacher = torch.einsum('bd,bnd->bn', f_t, view_feature_list_temp) # [B, N]
        min_sim_with_teacher = torch.min(sim_with_teacher, dim=1)[0]# [B]
        within_margin_true = min_sim_with_teacher > (1-margin) # [B]

        sim_with_teacher = 1 - sim_with_teacher # [B, N]
    
        within_margin_true_views = view_feature_list_temp[within_margin_true] # [M, N, D]
        if within_margin_true_views.shape[0] > 0:
            cos_sim_matrix = torch.matmul(within_margin_true_views, within_margin_true_views.transpose(1,2)) # [N, N]
            cos_sim_matrix = F.softplus(cos_sim_matrix) # [M, N, N]

            mask = torch.ones_like(cos_sim_matrix) - torch.eye(cos_sim_matrix.shape[1], device=cos_sim_matrix.device)
            cos_sim_matrix = cos_sim_matrix * mask # [M, N, N]
            cos_sim_matrix = cos_sim_matrix.sum() # [M, N]
            
            diversify_loss = cos_sim_matrix / (N * (N-1) * B) # [1]
            
        else:
            diversify_loss = torch.zeros(1, device=sim_with_teacher.device)
        
        
        inter_loss = feature_cont_loss + diversify_loss.item()
        return inter_loss


class Feature_intra_Loss(nn.Module):
    def __init__(self):
        super(Feature_intra_Loss, self).__init__()
        
    def forward(self, f_t, view_feature_list, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The intra angle loss
        """
        
        N = len(view_feature_list)
        B, D = view_feature_list[0].size()
        feature_angle_loss = 0
        
        assert N > 0
        assert f_t.shape == view_feature_list[0].shape
        
        view_feature_tensor = torch.stack(view_feature_list, dim=0)  # [N, B, D]
        
        # Compute the difference between f_t and each view feature
        view_feature_diff = f_t.unsqueeze(0) - view_feature_tensor  # [N, B, D]
        
        view_feature_diff = view_feature_diff.view(N, B, D) # [N, B, D]
        view_feature_diff = view_feature_diff.permute(1, 0, 2) # [B, N, D]
        view_feature_diff = F.normalize(view_feature_diff, p=2, dim=-1) # [B, N, D]
        cos_sim_matrix = torch.matmul(view_feature_diff, view_feature_diff.transpose(1,2)).mean(dim=0) # [N, N]
        
        # Compute softplus of cosine similarity
        softplus_cos_sim = F.softplus(cos_sim_matrix)  # [N, N]
        
        # Exclude diagonal elements (self-similarity)
        mask = torch.ones_like(softplus_cos_sim) - torch.eye(N, device=softplus_cos_sim.device)
        feature_angle_loss = (softplus_cos_sim * mask).sum()
        
        # Normalize the loss
        return feature_angle_loss / (N * (N - 1))
        

class Logit_inter_Loss(nn.Module):
    def __init__(self, T=1):
        super(Logit_inter_Loss, self).__init__()
        self.margin = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.T = T
        self.temperature = 0.07
        
    def forward(self, y_t, view_logit_list):
    
        N = len(view_logit_list)
        B, D = view_logit_list[0].size()
        
        assert N > 0
        assert y_t.shape == view_logit_list[0].shape

        view_feature_list_temp = view_logit_list.copy()
    
        view_feature_tensor = torch.stack(view_feature_list_temp, dim=1) # [B, N, D]
        view_feature_tensor = view_feature_tensor.view(B*N, D) # [B*N, D]
        view_feature_tensor = F.softmax(view_feature_tensor/self.T, dim=1)

        f_t = F.softmax(y_t/self.T, dim=1) # [B, D]
        
        similarity_matrix = torch.matmul(view_feature_tensor, f_t.t()) # [B*N, B]
        labels = torch.arange(B * N, device=similarity_matrix.device) // N # [B * N], 0,0,0, 1,1,1,1, 2,2,2,2

        col_classes = torch.arange(B, device=labels.device)  # [B]
        positive_mask = (labels.unsqueeze(1) == col_classes.unsqueeze(0)) # [B*N, B*N]

        margin = self.margin.item()
        mask = positive_mask & (similarity_matrix >= (1-margin))
        similarity_matrix[mask] = 1.
        similarity_matrix /= self.temperature # [B*N, B]
        
        feature_cont_loss = F.cross_entropy(similarity_matrix, labels, reduction='mean') / (N) # [1]

        '''
        Diversiy loss
        '''
        view_feature_list_temp = view_logit_list.copy()
        view_feature_list_temp_ = torch.stack(view_feature_list_temp, dim=1) # [B, N, D]
        view_feature_list_temp = F.softmax(view_feature_list_temp_/self.T, dim=2)

        sim_with_teacher = torch.einsum('bd,bnd->bn', f_t, view_feature_list_temp) # [B, N]
        min_sim_with_teacher = torch.min(sim_with_teacher, dim=1)[0]# [B]
        within_margin_true = min_sim_with_teacher > (1-margin) # [B]
    
        within_margin_true_views = view_feature_list_temp[within_margin_true] # [M, N, D]
        if within_margin_true_views.shape[0] > 0:

            cos_sim_matrix = torch.matmul(within_margin_true_views, within_margin_true_views.transpose(1,2)) # [N, N]
            cos_sim_matrix = F.softplus(cos_sim_matrix) # [M, N, N]

            mask = torch.ones_like(cos_sim_matrix) - torch.eye(cos_sim_matrix.shape[1], device=cos_sim_matrix.device)
            cos_sim_matrix = cos_sim_matrix * mask # [M, N, N]
            cos_sim_matrix = cos_sim_matrix.sum() # [M, N]
            
            diversify_loss = cos_sim_matrix / (N * (N-1) * B) # [1]
        
        else:
            diversify_loss = torch.zeros(1, device=sim_with_teacher.device)
        
        inter_loss = feature_cont_loss + diversify_loss.item() 
        return inter_loss
    

class Logit_intra_Loss(nn.Module):
    def __init__(self, T=4):
        super(Logit_intra_Loss, self).__init__()
        self.T = T
        
        
    def forward(self, l_t, view_logit_list, contrast_idx=None):
        
        N = len(view_logit_list)
        B, D = view_logit_list[0].size()
        logit_angle_loss = 0
        
        assert N > 0
        assert l_t.shape == view_logit_list[0].shape
        
        view_logit_tensor = torch.stack(view_logit_list, dim=0)  # [N, B, D]
        view_logit_tensor = torch.softmax(view_logit_tensor / self.T, dim=2)
        l_t_softmax = torch.softmax(l_t / self.T, dim=1)
        
        # Compute the difference between f_t and each view logit
        view_logit_diff = l_t_softmax.unsqueeze(0) - view_logit_tensor  # [N, B, D]
        
        view_logit_diff = view_logit_diff.view(N, B, D) # [N, B, D]
        view_logit_diff = view_logit_diff.permute(1, 0, 2) # [B, N, D]
        
        view_logit_diff = F.normalize(view_logit_diff, p=2, dim=-1) # [B, N, D]
        cos_sim_matrix = torch.matmul(view_logit_diff, view_logit_diff.transpose(1,2)).mean(dim=0) # [N, N]
        
        # Compute softplus of cosine similarity
        softplus_cos_sim = F.softplus(cos_sim_matrix)  # [N, N]
        
        # Exclude diagonal elements (self-similarity)
        mask = torch.ones_like(softplus_cos_sim) - torch.eye(N, device=softplus_cos_sim.device)
        logit_angle_loss = (softplus_cos_sim * mask).sum()
        
        # Normalize the loss
        return logit_angle_loss / (N * (N - 1) )