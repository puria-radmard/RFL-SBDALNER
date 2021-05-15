import logging

import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import *
from torch.utils.data.sampler import *


class EarlyStopper:

    def __init__(self, model, patience: int):  # , maximise: bool):
        """
        An early stopping & callback class.
        patience is an integer, the number of epochs that a non-optimal statistic is allowed (adding number of steps soon)
        maximise is set to True for scores, False for losses
        """
        self.patience = patience
        # self.maximise = maximise
        self.model_state_dict = None
        self.model = model
        self.scores = []
        self.min_score = float('inf')

    def is_overfitting(self, score):
        scores = self.scores
        if len(scores) < self.patience:
            self.scores.append(score)
            return False

        if score < self.min_score:
            self.model_state_dict = self.model.state_dict()
            self.min_score = score

        scores.append(score)
        all_increasing = True
        s0 = scores[0]
        for s1 in scores[1:]:
            if s0 >= s1:
                all_increasing = False
                break
            s0 = s1
        self.scores = scores[1:]

        if all_increasing:
            logging.info(f'reloading model')
            self.model.load_state_dict(self.model_state_dict)

        return all_increasing

    def check_stop(self, stats_list):

        if self.patience < 0 or len(stats_list) < self.patience:
            return False
        if len(stats_list) - np.argmin(stats_list) > self.patience:
            return True
        else:
            return False


class ModifiedKL(nn.Module):

    def __init__(self, weight):
        """
        KL mask not used at the moment as we bypassed the log in KL, so o.h.e vectors can be used for it
        Reverse KL is not supported here
        TODO: Looking into weighting labelled and non labelled loss?
        """

        super().__init__()

        # weights of size [1, 1, classes]
        self.weight = weight.reshape(1, 1, -1)

    def forward(self, pred_log_probs, target_probs, self_supervision_mask):
        # loss of size [batch, length, classes]
        loss = - pred_log_probs * target_probs
        loss *= self.weight
        loss *= self_supervision_mask
        # loss of size [batch, length]
        return loss.sum()


def group(data, breakpoints):
    groups = [[] for _ in range(len(breakpoints) + 1)]
    for idx, item in enumerate(data):
        # i.e. group into similar sentence sizes
        i = bisect.bisect_left(breakpoints, len(item[0]))
        groups[i].append(idx)
    data_groups = [Subset(data, g) for g in groups]
    return data_groups


class GroupBatchRandomSampler(object):
    def __init__(self, data_groups, batch_size, drop_last):
        self.batch_indices = []
        for data_group in data_groups:
            self.batch_indices.extend(
                list(
                    BatchSampler(
                        SubsetRandomSampler(data_group.indices),
                        batch_size,
                        drop_last=drop_last,
                    )
                )
            )

    def __iter__(self):
        return (self.batch_indices[i] for i in torch.randperm(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)
