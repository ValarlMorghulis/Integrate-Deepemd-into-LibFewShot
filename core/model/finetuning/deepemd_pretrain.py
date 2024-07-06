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
from torch import nn
from core.utils import accuracy
import torch.nn.functional as F
from .finetuning_model import FinetuningModel
from ..metric.deepemd import EMDLayer


class DeepEMD_Pretrain(FinetuningModel):
    def __init__(self,**kwargs):
        super(DeepEMD_Pretrain, self).__init__(**kwargs)

        if self.feature_pyramid is not None:
            self.feature_pyramid = [int(x) for x in self.feature_pyramid.split(',')]
        if self.patch_list is not None:
            self.patch_list = [int(x) for x in self.patch_list.split(',')]

        self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.loss_func = nn.CrossEntropyLoss()
        # DeepEMD's EMDLayer
        self.val_classifier = self.EMDLayer = EMDLayer(temperature=self.temperature, 
                                                       norm=self.norm, 
                                                       metric=self.metric,
                                                       solver=self.solver,
                                                       form=self.form,
                                                       l2_strength=self.l2_strength)

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
        images, target = batch
        images = images.to(self.device)
        target = target.to(self.device)
        output = self.classifier(self.encode(images,dense=False).squeeze(-1).squeeze(-1))
        acc = accuracy(output, target)
        loss = self.loss_func(output, target)
        return output, acc, loss
    
    def encode(self, x, dense=True):
        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.emb_func(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch,
                          x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
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
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(
                feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(
            feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out
        
    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the support
        SFC = support.view(
            self.shot_num, -1, 640, support.shape[-2], support.shape[-1]
            ).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.way_num).repeat(self.shot_num)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.sfc_update_step):
                rand_id = torch.randperm(self.way_num * self.shot_num).cuda()
                for j in range(0, self.way_num * self.shot_num, self.sfc_bs):
                    selected_id = rand_id[j: min(j + self.sfc_bs, self.way_num * self.shot_num)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.EMDLayer(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC