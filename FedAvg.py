import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            #print('done')
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# def FedSGD(w, lr):
#     w_avg = copy.deepcopy(w[0])
#
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#
#         # 在这里应用FedSGD的更新规则
#         w_avg[k] = w_avg[k] / len(w)
#
#         # 在这里应用FedSGD的学习率调整
#         w[k] = w[k] - lr * w_avg[k]
#
#     return w
