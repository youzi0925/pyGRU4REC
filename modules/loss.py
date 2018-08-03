import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda=True, bpreg=1.0):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super().__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        self.bpreg = bpreg
        if loss_type == 'CrossEntropy':
            self._loss_fn = SampledCrossEntropyLoss(use_cuda)
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        elif loss_type == 'TOP1_Max':
            self._loss_fn = TOP1_MaxLoss()
        elif loss_type == 'BPR_Max':
            self._loss_fn = BPR_MaxLoss(bpreg)
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)


class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self, use_cuda):
        """
        See Balazs Hihasi(ICLR 2016), pg.5

        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super().__init__()
        self.xe_loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(0)
        target = Variable(torch.arange(batch_size).long())
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        target = target.to(self.device)
        #Fixing the instability 避免log0
        logit = logit + 1e-24
        return self.xe_loss(logit, target)


class BPRLoss(nn.Module):
    def __init__(self):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """

        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))

        return loss

class BPR_MaxLoss(nn.Module):
    def __init__(self, bpreg):

        super().__init__()
        self.bpreg = bpreg

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """

        softmax_scores = F.softmax(logit, 1)
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        # loss = -F.logsigmoid(diff)*softmax_scores + self.bpreg*(logit ** 2)*softmax_scores
        # loss = torch.sum(loss, 1).mean()

        loss = F.sigmoid(diff) * softmax_scores
        loss = (-torch.log(torch.sum(loss, 1) + 1e-24) + torch.sum(self.bpreg*(logit ** 2)*softmax_scores, 1)).mean()

        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        #diag()返回一个矩阵的对角线元素
        #logit.diag().view(-1, 1)
        # tensor(1.00000e-02 *
       # [[-4.3056],
       #  [ 6.1118],
       #  [-5.2414],
       #  [ 4.6450],
       #  [ 7.4447]])
        #logit.diag().view(-1, 1).expand_as(logit)
        # tensor(1.00000e-02 *
        #        [[-4.3056, -4.3056, -4.3056, -4.3056, -4.3056],
        #         [6.1118, 6.1118, 6.1118, 6.1118, 6.1118],
        #         [-5.2414, -5.2414, -5.2414, -5.2414, -5.2414],
        #         [4.6450, 4.6450, 4.6450, 4.6450, 4.6450],
        #         [7.4447, 7.4447, 7.4447, 7.4447, 7.4447]])
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        # final loss
        loss = F.sigmoid(diff).mean() + F.sigmoid(logit ** 2).mean()

        return loss

class TOP1_MaxLoss(nn.Module):
    def __init__(self):
        """
        See Balazs Hihasi(ICLR 2016), pg.5
        """
        super().__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        #diag()返回一个矩阵的对角线元素
        #logit.diag().view(-1, 1)
        # tensor(1.00000e-02 *
       # [[-4.3056],
       #  [ 6.1118],
       #  [-5.2414],
       #  [ 4.6450],
       #  [ 7.4447]])
        #logit.diag().view(-1, 1).expand_as(logit)
        # tensor(1.00000e-02 *
        #        [[-4.3056, -4.3056, -4.3056, -4.3056, -4.3056],
        #         [6.1118, 6.1118, 6.1118, 6.1118, 6.1118],
        #         [-5.2414, -5.2414, -5.2414, -5.2414, -5.2414],
        #         [4.6450, 4.6450, 4.6450, 4.6450, 4.6450],
        #         [7.4447, 7.4447, 7.4447, 7.4447, 7.4447]])
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        softmax_scores = F.softmax(logit, 1)
        # final loss
        loss = torch.sum(((F.sigmoid(diff) + F.sigmoid(logit ** 2))*softmax_scores), 1).mean()

        return loss
