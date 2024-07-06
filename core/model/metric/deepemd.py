# -*- coding: utf-8 -*-
"""
@InProceedings{Zhang_2020_CVPR,
author = {Zhang, Chi and Cai, Yujun and Lin, Guosheng and Shen, Chunhua},
title = {DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover's Distance and Structured Classifiers},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
@misc{zhang2020deepemdv2,
    title={DeepEMD: Differentiable Earth Mover's Distance for Few-Shot Learning},
    author={Chi Zhang and Yujun Cai and Guosheng Lin and Chunhua Shen},
    year={2020},
    eprint={2003.06777v3},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

Adapted from https://github.com/icoz69/DeepEMD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric_model import MetricModel
import cv2
from qpth.qp import QPFunction
import torch
import torch.nn.functional as F
from core.utils import accuracy

class EMDLayer(nn.Module):
    def __init__(self, temperature=12.5,norm='center',metric='cosine',solver='opencv',form='QP',l2_strength=0.0001):
        super(EMDLayer, self).__init__()
        self.temperature = temperature
        self.norm = norm
        self.metric = metric
        self.solver = solver
        self.form = form
        self.l2_strength = l2_strength

    def forward(self, support, query):
        weight_1 = self.get_weight_vector(query, support)
        weight_2 = self.get_weight_vector(support, query)

        support = self.normalize_feature(support)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(support, query)
        logits = self.get_emd_distance(similarity_map, weight_1, weight_2)
        return logits
    
    def normalize_feature(self, x):
        if self.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x
    
    def get_weight_vector(self, A, B):
        # get the weight vector for each node in A
        M = A.shape[0]
        N = B.shape[0]
        # average pooling of B's feature map and copy it to the size of A
        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])
        # raise the dimension of A and B
        A = A.unsqueeze(1)
        B = B.unsqueeze(0)
        # repeat A and B to the same size
        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)
        # calculate the weight vector
        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination
    
    def get_similiarity_map(self, support, query):
        way = support.shape[0]
        num_query = query.shape[0]
        # flatten the feature map
        query = query.view(query.shape[0], query.shape[1], -1)
        support = support.view(support.shape[0], support.shape[1], -1)
        # repeat the feature map
        support = support.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        # permute the feature map
        support = support.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = support.shape[-2]
        # calculate the similarity map
        if self.metric == 'cosine':
            support = support.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(support, query, dim=-1)
        if self.metric == 'l2':
            support = support.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (support - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map
    
    def get_emd_distance(self, similarity_map, weight_1, weight_2):
        num_query = similarity_map.shape[0]
        num_support = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if self.solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                for j in range(num_support):
                    _, flow = self.emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                    # multiply the similarity map with the flow
                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()
            # calculate the logitis and multiply it with the averaged temperature to make it smooth
            temperature=(self.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis

        elif self.solver == 'qpth': # use QPTH solver
            # permute the weight vector
            weight_2 = weight_2.permute(1, 0, 2)
            # reshape the similarity map and weight vector
            similarity_map = similarity_map.view(num_query * num_support, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_support, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_support, weight_2.shape[-1])
            # calculate the EMD distance
            _, flows = self.emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form=self.form, l2_strength=self.l2_strength)
            # reshape the flow
            logitis=(flows*similarity_map).view(num_query, num_support,flows.shape[-2],flows.shape[-1])
            # calculate the logitis and multiply it with the averaged temperature to make it smooth
            temperature = (self.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')
        
        return logitis
    
    def emd_inference_opencv(self, cost_matrix, weight1, weight2):
        # cost matrix is a tensor of shape [N,N]
        cost_matrix = cost_matrix.detach().cpu().numpy()
        # make sure the cost matrix is not negative
        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5
        # normalize the weight vector
        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()
        # calculate the EMD distance
        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow
    
    def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
        """
        to use the QP solver QPTH to derive EMD (LP problem),
        one can transform the LP problem to QP,
        or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
        :param distance_matrix: nbatch * element_number * element_number
        :param weight1: nbatch  * weight_number
        :param weight2: nbatch  * weight_number
        :return:
        emd distance: nbatch*1
        flow : nbatch * weight_number *weight_number
        """
        # normalize the weight vector
        weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
        weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)
        # get the shape of the input
        nbatch = distance_matrix.shape[0]
        nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
        nelement_weight1 = weight1.shape[1]
        nelement_weight2 = weight2.shape[1]
        # reshape the input
        Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()
        # set the QP problem
        if form == 'QP':
            # version: QTQ
            Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
                nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
            p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
        elif form == 'L2':
            # version: regularizer
            Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
            p = distance_matrix.view(nbatch, nelement_distmatrix).double()
        else:
            raise ValueError('Unkown form')
        # set the inequality constraints
        h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
        h_2 = torch.cat([weight1, weight2], 1).double()
        h = torch.cat((h_1, h_2), 1)
        # set the equality constraints
        G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
        # set G_2 according to the weight vector
        for i in range(nelement_weight1):
            G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
        for j in range(nelement_weight2):
            G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
        # cat the inequality constraints
        G = torch.cat((G_1, G_2), 1)
        A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
        b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
        flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)
        # calculate the EMD distance
        emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
        return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)
    
    

class DeepEMD(MetricModel):
    def __init__(self, **kwargs):
        super(DeepEMD, self).__init__(**kwargs)
        if self.feature_pyramid is not None:
            self.feature_pyramid = [int(x) for x in self.feature_pyramid.split(',')]
        if self.patch_list is not None:
            self.patch_list = [int(x) for x in self.patch_list.split(',')]
        self.EMDLayer = EMDLayer(temperature=self.temperature, 
                                 norm=self.norm, 
                                 metric=self.metric,
                                 solver=self.solver,
                                 form=self.form,
                                 l2_strength=self.l2_strength)
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        image, global_target = batch
        image = image.to(self.device)

        if self.mode == 'FCN':
            dense = True
        else:
            dense = False
        feat = self.encode(image, dense)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )

        if self.shot_num > 1:
            support_feat = self.get_sfc(support_feat)

        output = self.EMDLayer(support_feat, query_feat).reshape(-1, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc
    
    def set_forward_loss(self, batch):
        images, global_target = batch
        images = images.to(self.device)

        if self.mode == 'FCN':
            dense = True
        else:
            dense = False
        feat = self.encode(images, dense)

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=3
        )

        if self.shot_num > 1:
            support_feat = self.get_sfc(support_feat)

        output = self.EMDLayer(support_feat, query_feat).reshape(-1, self.way_num)
        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc, loss
    
    def encode(self, x, dense=True):
        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            # reshape the input
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.emb_func(x)
            # average pooling
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch,
                          x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            # 5D tensor to 4D tensor
            x = x.squeeze(-1)
            return x
        else:
            x = self.emb_func(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.feature_pyramid:
            # append the feature map according to the size
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        # concatenate the feature map
        out = torch.cat(feature_list, dim=-1)
        return out

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the support
        SFC = support.view(
            self.shot_num, -1, 640, support.shape[-2], support.shape[-1]
            ).mean(dim=0).clone().detach()
        # set the SFC as a learnable parameter
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)
        # set the optimizer
        optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)
        # crate label for finetune
        label_shot = torch.arange(self.way_num).repeat(self.shot_num)
        label_shot = label_shot.type(torch.cuda.LongTensor)
        # finetune the SFC
        with torch.enable_grad():
            for k in range(0, self.sfc_update_step):
                rand_id = torch.randperm(self.way_num * self.shot_num).cuda()
                for j in range(0, self.way_num * self.shot_num, self.sfc_bs):
                    # get the batch shot
                    selected_id = rand_id[j: min(j + self.sfc_bs, self.way_num * self.shot_num)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    # calculate the logits
                    out = self.EMDLayer(SFC, batch_shot.detach())
                    # calculate the loss
                    loss = F.cross_entropy(out, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC