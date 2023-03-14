import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # print(label)
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)

def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()

    # t=predict[0,0,:,:]
    # # print(t.shape)
    # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
    # # print(tt.shape)   #(256.256)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # m=target[1,:,:,:]
    # # mm=np.zeros((256,256))
    # mm=target[0,:,:]
    # ##print(np.maximum(mm, -1))
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # plt.subplot(121)
    # plt.imshow(tt)
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()

    target_mask = (target >= 0) * (target != 255)   #??
    target = target[target_mask]                    #??
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  #Predict(n,h,w,c)
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  #将target视为n,h,w,c，重复填充最后的c通道


    loss = F.cross_entropy(predict, target, size_average=True)
    # loss = F.cross_entropy(tt, target, size_average=True)

    return loss

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())   #造一个Predict大小的tensor
    y_truth_tensor.fill_(y_label)                       #用label填充它
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device()) #将其送入Predict对应的device里
    # print('y_truth_tensor,',y_truth_tensor.shape)
    # print('y_pred,',y_pred.shape)
    # t=y_pred[0,0,:,:]
    # # print(t.shape)
    # tt = np.argmax(F.softmax(predict).cpu().data[0].numpy().transpose(1, 2, 0),axis=2)
    # # print(tt.shape)   #(256.256)
    # # tt=np.zeros((256,256,3))
    # # tt[:,:,0]=t[0,:,:]
    # # # tt[:,:,1]=t[0,:,:]
    # # # tt[:,:,2]=t[1,:,:]
    # # print(tt[0,100,100])
    # # m=target[1,:,:,:]
    # # mm=np.zeros((256,256))
    # mm=y_label[0,:,:]
    # ##print(np.maximum(mm, -1))
    # # # mm[:,:,1]=target[1,:,:]
    # # # mm[:,:,2]=target[2,:,:]
    # # # t = t.transpose((1,2,0))
    # plt.subplot(121)
    # plt.imshow(tt)
    # # plt.show()
    # plt.subplot(122)
    # plt.imshow(mm.cpu().data)#, cmap='Greys_r') # 显示图片
    # # plt.axis('off') # 不显示坐标轴
    # plt.show()
    # tmp1 = nn.Sigmoid()(y_pred)
    # loss1= nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)
    # loss2 = nn.BCELoss()(tmp1, y_truth_tensor)
    # print('tmp1 and y_truth_tensor :',tmp1,y_truth_tensor)
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)       #[1, 1, 8, 8],[1, 1, 8, 8]

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


class SizeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SizeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        output_counts = torch.sum(torch.softmax(output, dim=1), dim=(2, 3))
        target_counts = torch.zeros_like(output_counts)
        for b in range(0, target.shape[0]):
            elements, counts = torch.unique(
                target[b, :, :, :, :], sorted=True, return_counts=True)
            assert torch.numel(target[b, :, :, :, :]) == torch.sum(counts)
            target_counts[b, :] = counts

        lower_bound = target_counts * (1 - self.margin)
        upper_bound = target_counts * (1 + self.margin)
        too_small = output_counts < lower_bound
        too_big = output_counts > upper_bound
        penalty_small = (output_counts - lower_bound) ** 2
        penalty_big = (output_counts - upper_bound) ** 2
        # do not consider background(i.e. channel 0)
        res = too_small.float()[:, 1:] * penalty_small[:, 1:] + \
            too_big.float()[:, 1:] * penalty_big[:, 1:]
        loss = res / (output.shape[2] * output.shape[3] * output.shape[4])
        return loss.mean()


class MumfordShah_Loss(nn.Module):
    def levelsetLoss(self, output, target, penalty='l1'):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        self.penalty = penalty
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(
                tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2, 3)
                                  ) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(
                    tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss

    def gradientLoss2d(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, image, prediction):
        loss_level = self.levelsetLoss(image, prediction)
        loss_tv = self.gradientLoss2d(image)
        return loss_level + loss_tv
