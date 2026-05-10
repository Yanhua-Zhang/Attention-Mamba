import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np

from torch import Tensor


# __all__ = ['CE_and_Dice_loss', 'FL_and_CE_loss']


# -----------------------------------------------------------------------------
# from: https://github.com/JCruan519/MALUNet

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss_V2(nn.Module):
    def __init__(self):
        super(DiceLoss_V2, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class BCE_and_Dice_loss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BCE_and_Dice_loss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss_V2()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        
        pred = torch.sigmoid(pred)

        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss



# -----------------------------------------------------------------------------
# from: https://github.com/DengPingFan/PraNet

class BCE_and_wIoU_loss(nn.Module):
    def __init__(self, kernel_size = 31, padding = 15):
        super(BCE_and_wIoU_loss, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, pred, mask):

        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding) - mask)

        # target (Tensor) – Tensor of the same shape as input with values between 0 and 1
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)

        return (wbce + wiou).mean()


# -----------------------------------------------------------------------------
# dice loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()  # need trans to float

    def _dice_loss(self, score, target):
        target = target.float()     # need trans to float
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # 
        target = self._one_hot_encoder(target)  # one_hot_encode for ground true
        
        if weight is None:
            weight = [1] * self.n_classes  
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        # for each class dice loss 
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes  


# -----------------------------------------------------------------------------
# Cross Entropy Loss + Dice loss

class CE_and_Dice_loss(nn.Module):
    def __init__(self, num_classes = 2):
        super(CE_and_Dice_loss, self).__init__()

        self.num_classes = num_classes

        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, input, label):

        loss_ce = self.ce_loss(input, label.long())
        loss_dice = self.dice_loss(input, label.long(), softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        return loss
    

# -----------------------------------------------------------------------------
# CE + Focal loss from PICAI and nnUNet

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        # print(one_hot_key.size)
        
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1

            # b, 1, x, y or b, 1, x, y, z ---> b, x, y or b, x, y, z
            target = target[:, 0] 
            
        return super().forward(input, target.long())


class FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5, aggregate="sum"):
        super(FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        # self.fl = FocalLoss(apply_nonlin=nn.Softmax(), **fl_kwargs)
        self.fl = FocalLoss(apply_nonlin=nn.Softmax(dim=1), **fl_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

    def forward(self, net_output, target):
        fl_loss = self.fl(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        else:
            raise NotImplementedError("nah son")
        return result
    

# -----------------------------------------------------------------------------
# inconsistency loss

class Inconsistency_loss(nn.Module):
    def __init__(self, nonlin_name='Sigmoid', loss_type='L2'):
        super(Inconsistency_loss, self).__init__()

        self.loss_type = loss_type

        if nonlin_name == 'Softmax':
            self.apply_nonlin=nn.Softmax(dim=1)
        elif nonlin_name == 'Sigmoid':
            self.apply_nonlin=nn.Sigmoid()
        elif nonlin_name == 'Disabled':
            self.apply_nonlin=None
        else:
            raise ValueError('this nonlin is not supported: ' + self.apply_nonlin)

    def forward(self, input1, input2):

        if self.apply_nonlin is not None:
            input1 = self.apply_nonlin(input1)
            input2 = self.apply_nonlin(input2)

        if self.loss_type == 'L2':
            loss = F.mse_loss(input1, input2)
        elif self.loss_type == 'L1':
            loss = F.l1_loss(input1, input2)
        else:
            raise ValueError('this loss_type is not supported: ' + self.loss_type)

        return loss
    

# -----------------------------------------------------------------------------