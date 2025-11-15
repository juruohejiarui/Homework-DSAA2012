import torch
import torch.nn as nn
import torch.nn.functional as F
from .device import device

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.22, 0.28, 0.3], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction: 'reduction'参数，指定返回的loss形式，'mean'表示返回平均loss，'sum'表示返回总loss，其他表示返回每个样本的loss
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = alpha.detach().clone().to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target.cpu()]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_num=10, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
    def labels(self, logits):
        pred_y = torch.max(logits, 1)[1] #输出最大值的索引位置
        return pred_y
    def forward(self, preds, targets):
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        #out[i][j] = input[i][index[i][j]] # if dim == 1
        preds_softmax = preds_softmax.gather(1, targets.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, targets.view(-1, 1))
        # 矩阵惩罚 a*b,
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma)
        loss = self.alpha * loss
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss