import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import copy
import cv2
import networks
import PIL
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
resize=Compose([
        # ToTensor(),
        Resize((256,256),interpolation=PIL.Image.NEAREST),
    ])


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def BlockMaskingGenerator(x, prob=0.5):
    # if random.random() <= 0.2:
    #     return x,None
    x = resize(x)
    masked_image1 = copy.deepcopy(x)
    masked_image2 = copy.deepcopy(x)
    img_deps,_,img_rows, img_cols = masked_image1.shape  #(3, 512, 512)
    num_patches = img_rows* img_cols 
    num_mask = int(prob * num_patches)
    mask_rows = 32
    mask_cols = 32
    num_patches = mask_rows* mask_cols 
    num_mask = int(prob * num_patches)
    # print("mask generating: total patches {}, mask patches {}".format(num_patches,num_mask))
    mask = torch.zeros([mask_rows, mask_cols])#np.random.rand(mask_rows, mask_cols)
    # print(mask.shape)
    # np.hstack(
    #     [np.ones(num_patches-num_mask),
    #      np.zeros(num_mask),]
    # ).reshape(mask_rows,mask_cols)#.astype(bool)
    # print('mask',mask)
    # np.random.shuffle(mask)
    mask=torch.reshape(mask,(num_patches,1))
    # print(mask.shape)
    mask[:num_mask] =1
    mask[num_mask:] = 0
    shuffle_index=torch.randperm(num_patches)
    # torch.random.shuffle(mask)
    # print(mask)
    mask=torch.reshape(mask[shuffle_index],(1,mask_rows, mask_cols))
    # print('mask',mask)
    
    # print(mask.shape)
    mask1 = resize(mask)
    # mask = cv2.resize(mask, dsize=None,fx=16,fy=16,interpolation=cv2.INTER_LINEAR)
    # mask = np.resize(mask, (img_rows, img_cols))
    # print(mask1.shape)
    mask1[mask1>0.5] =1
    mask1[mask1<0.5]=0
    ones = torch.ones(mask1.shape)#np.random.rand(mask_rows, mask_cols)
    mask2 = ones-mask1
    # mask[mask==2]=1
    # mask[mask>0] = 1
    for c in range(img_deps):
        # print(mask.reshape(img_rows,img_cols))
        new_img = masked_image1[c]
        # print(new_img.shape)  #(262144,) (3, 512, 512)
        masked_image1[c] = masked_image1[c]*mask1
        masked_image2[c] = masked_image2[c]*mask2
    # print('new_img',new_img)
    # print('x[2]',x[2])
    # out = orig_image*mask
    # x = np.dot(mask, orig_image)
    return masked_image1,masked_image2,mask1,mask2