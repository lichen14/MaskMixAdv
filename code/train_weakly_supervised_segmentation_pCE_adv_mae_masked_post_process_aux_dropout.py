import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.losses import *
from dataloaders import utils
from dataloaders.dataset1119 import BaseDataSets, RandomGenerator
from networks.net_factory import Discriminator, net_factory,get_fc_discriminator
from utils import losses, metrics, ramps
from utils.util import BlockMaskingGenerator
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from val_2D import *
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR,StepLR
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_SPS', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_mae_cct_0109', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--device', help='cuda|cpu', default="cuda")
parser.add_argument('--mask_ratio', type=float, default=0.5)

args = parser.parse_args()


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


def train(args, snapshot_path):
    device = torch.device(args.device)
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations    
    
    Tensor = torch.cuda.FloatTensor if args.device else torch.Tensor
    REAL_LABEL = Variable(Tensor(args.batch_size,1).fill_(1.0), requires_grad=False)
    FAKE_LABEL = Variable(Tensor(args.batch_size,1).fill_(0.0), requires_grad=False)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    model.train()
    
    dis_main = Discriminator(input_nc=1)
    dis_main.cuda()
    dis_main.train()

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
                                RandomGenerator(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    # db_adv = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
    #                             RandomGenerator(args.patch_size)]), fold="fold0", sup_type="label")
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=1, pin_memory=True,drop_last=True)
    # advloader = DataLoader(db_adv, batch_size=batch_size, shuffle=True,
    #                          num_workers=1, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)


    # optimizer = optim.SGD(model.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    optimizer_dis = optim.Adam(dis_main.parameters(), lr=base_lr)#, betas=(0.9, 0.99))
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    mae_loss = nn.MSELoss()
    adv_loss = torch.nn.MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_mean_hd95 = 0
    best_iter_num = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0

    # lr_scheduler = OneCycleLR(optimizer, max_lr=base_lr, total_steps =max_epoch,verbose=False)# args.num_epoch

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, real_mask = sampled_batch['image'], sampled_batch['label'], sampled_batch['mask']

            # print('volume_batch.shape,label_batch.shape',volume_batch.shape,label_batch.shape)
            masked_volume1,masked_volume2,mask1,mask2  =  BlockMaskingGenerator(volume_batch, args.mask_ratio)
            volume_batch, label_batch, real_mask = volume_batch.cuda(), label_batch.cuda(), real_mask.type(torch.FloatTensor).cuda().unsqueeze(1)
            # print(masked_volume1.shape,masked_volume2.shape,mask1.shape,mask2.shape)
            outputs1,outputs2,restore_img = model(masked_volume1.cuda())
            # outputs1,outputs2,restore_img = model(volume_batch)
            # outputs2,_ = model(masked_volume2.cuda())
            # outputs1, outputs2 = model(
            #     volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            outputs1_hard = torch.argmax(outputs1, dim=1, keepdim=False)
            outputs2_hard = torch.argmax(outputs2, dim=1, keepdim=False)
            t=volume_batch[0]
            mm=label_batch[0]

            
            
            loss_ce1 = ce_loss(outputs1, label_batch[:].long())
            loss_ce2 = ce_loss(outputs2, label_batch[:].long())
            # loss_ce3 = ce_loss((mask1.cuda() * outputs1 + mask2.cuda() * outputs2),label_batch[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)# + loss_ce3   #Eq.(1) L_pce: scribble supervision

            loss_mae = mae_loss(input=restore_img,target=volume_batch)
            # loss_mae = mae_loss(input=restore_img[:,mask1==0],target=volume_batch[:,mask1==0])

            beta = random.random() + 1e-10  # soft :avoid beta=zero
            pseudo_output = beta * outputs_soft1 + (1.0-beta) * outputs_soft2
            pseudo_supervision = torch.argmax(pseudo_output, dim=1, keepdim=True).type(torch.FloatTensor).cuda()
            zeros = torch.zeros_like(pseudo_supervision)
            if epoch_num >1:
                for i in range(pseudo_supervision.shape[0]):
                    tmp2 = pseudo_supervision.cpu().data[i].squeeze(0)
                    # plt.subplot(231)
                    # plt.imshow(tmp2)
                    for j in range(1,args.num_classes):
                        if j in tmp2.cpu().data: 
                            tmp = np.zeros_like(tmp2,np.uint8)
                            tmp[tmp2==j]=1
                        # print(tmp.shape,tmp)
                            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity= 8)
                            areas = stats[1:,cv2.CC_STAT_AREA]
                            max_area_index = np.argmax(areas)+1
                            # plt.subplot(231)
                            # plt.imshow(tmp2)
                            # # # plt.show()
                            # plt.subplot(232)
                            # plt.imshow(tmp)#, cmap='Greys_r') # 显示图片
                            # plt.subplot(233)
                            # plt.imshow(labels)#, cmap='Greys_r') # 显示图片
                            # max_area = areas[max_area_index-1]
                            zeros[i,0,labels==max_area_index]=j
                            # print(j,nlabels,max_area,max_area_index)
                            # plt.subplot(234)
                            # plt.imshow(tmp)
                            # plt.subplot(235)
                            # plt.imshow((labels!=max_area_index))
                            # plt.show()
                    # plt.subplot(232)
                    # plt.imshow(zeros.cpu().data[i].squeeze(0))
                    # plt.show()
                pseudo_supervision = zeros
            

            dis_out_main = dis_main(pseudo_supervision)   #softmax得到0-1的概率prob，prob_2_entropy转化为self-information I_x
            # print('dis_out_main',dis_out_main)    #[1, 1, 8, 8]
            loss_adv_trg_main = adv_loss(dis_out_main, REAL_LABEL)
            # loss = loss_adv_trg_main
            # loss.requires_grad_()
            # loss.backward(retain_graph=True) #以上是公式(8)的损失

            # print(outputs1_hard.shape,outputs2_hard.shape,mask1.shape,mask2.shape)
            # pseudo_output = torch.empty(outputs_soft1.shape).cuda()
            # pseudo_output[:,:,mask1.cuda().squeeze(0)>0] = outputs_soft1[:,:,mask1.cuda().squeeze(0)>0].detach()
            # pseudo_output[:,:,mask2.cuda().squeeze(0)>0] = outputs_soft2[:,:,mask2.cuda().squeeze(0)>0].detach()
            # print(pseudo_supervision.shape)
            # pseudo_output = beta * outputs_soft1 + (1.0-beta) * outputs_soft2
            # pseudo_supervision = torch.argmax(
            #     (mask1.cuda() * outputs_soft1.detach() + mask2.cuda() * outputs_soft2.detach()), dim=1, keepdim=False)
            

            # pseudo_supervision = torch.argmax(
            #     pseudo_output, dim=1, keepdim=False)        #Eq.(2) generate pseudo label
            # print(pseudo_supervision.shape)
            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision) + 
                                dice_loss(outputs_soft2, pseudo_supervision))        #Eq.(3) L_PLS : pseudo labels supervision
            # if i_batch >100:
            #     plt.subplot(231)
            #     plt.imshow((torch.argmax(
            #         (mask1.cuda() * outputs_soft1), dim=1, keepdim=False)+torch.argmax(
            #         (mask2.cuda() * outputs_soft2), dim=1, keepdim=False)).cpu().data[0])
            #     # # plt.show()
            #     plt.subplot(232)
            #     plt.imshow(outputs1_hard.cpu().data[0])#, cmap='Greys_r') # 显示图片
            #     plt.subplot(233)
            #     plt.imshow(real_mask.cpu().data[0])
            #     # plt.subplot(224)
            #     # plt.imshow(np.transpose(mask2,(1,2,0)))
            #     plt.subplot(234)
            #     plt.imshow(pseudo_supervision.cpu().data[0])
            #     plt.subplot(235)
            #     plt.imshow(torch.argmax(
            #         (mask1.cuda() * outputs_soft1), dim=1, keepdim=False).cpu().data[0])
            #     plt.subplot(236)
            #     plt.imshow(torch.argmax(
            #         (mask2.cuda() * outputs_soft2), dim=1, keepdim=False).cpu().data[0])
            #     plt.show()

            lamda = 10 if epoch_num >2 else 0
            loss = loss_ce * 10 + lamda*loss_pse_sup + 10 * loss_mae         #Eq.(4) L_total lamda=0.5
            optimizer.zero_grad()
            loss.backward()

            # loss_adv_trg_main.backward()
            optimizer.step()

            
            
            # for param in model.parameters():      #%%%%%%%%%%%%%%%!!!!!!!!!!!!!!
            #     param.requires_grad = False
            # for param in dis_main.parameters():
            #     param.requires_grad = True

            optimizer_dis.zero_grad()
            # _, batch = trainloader_iter.__next__()
            # images_source, labels, _, _ = batch


            optimizer_dis.zero_grad()
            dis_real = dis_main(real_mask)
            # print(dis_out_main1)
            loss_dis_main1 = adv_loss(dis_real, REAL_LABEL)
            # loss_dis_main = loss_dis_main / 2
            # loss_dis_main.backward()

            # train with target 在目标域的损失
            pseudo_output = beta * outputs_soft1 + (1.0-beta) * outputs_soft2
            pseudo_supervision = torch.argmax(pseudo_output, dim=1, keepdim=True).type(torch.FloatTensor).cuda()

            dis_fake_detach = dis_main(pseudo_supervision.detach())
            loss_dis_main2 = adv_loss(dis_fake_detach, FAKE_LABEL)

            loss_dis_main = lamda*0.5*(loss_dis_main1 + loss_dis_main2)
            # loss = lamda * loss_dis_main
            loss_dis_main.backward()
            optimizer_dis.step()

            # lr_ = lr_scheduler.get_last_lr()[0]
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_mae', loss_mae, iter_num)
            writer.add_scalar('info/loss_adv_trg_main', loss_adv_trg_main, iter_num)
            writer.add_scalar('info/loss_dis_main1', loss_dis_main1, iter_num)
            writer.add_scalar('info/loss_dis_main2', loss_dis_main2, iter_num)
            writer.add_scalar('info/loss_pse_sup', loss_pse_sup, iter_num)
            # writer.add_scalar('info/lamda', lamda, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_pse_sup: %f, loss_adv_trg_main: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_pse_sup.item(), loss_adv_trg_main.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                # image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                # masked_volume1 = masked_volume1[1, 0:1, :, :]
                # masked_volume1 = (masked_volume1 - masked_volume1.min()) / (masked_volume1.max() - masked_volume1.min())
                # writer.add_image('train/masked_volume1', masked_volume1, iter_num)
                restore_img = restore_img[1, 0:1, :, :]
                restore_img = (restore_img - restore_img.min()) / (restore_img.max() - restore_img.min())
                writer.add_image('train/restore_img', restore_img, iter_num)
                # outputs = torch.argmax(torch.softmax(
                #     pseudo_output, dim=1), dim=1, keepdim=True)
                outputs1 = torch.argmax(outputs_soft1, dim=1, keepdim=True)
                writer.add_image('train/outputs1',
                                 outputs1[1, ...] * 50, iter_num)
                outputs2 = torch.argmax(outputs_soft2, dim=1, keepdim=True)
                writer.add_image('train/real_mask',
                                 real_mask[1, ...] * 50, iter_num)
                writer.add_image('train/pseudo_supervision',
                                 pseudo_supervision[1, ...] * 50, iter_num)
                writer.add_image('train/zeros',
                                 zeros[1, ...] * 50, iter_num)
                outputs = torch.argmax(outputs_soft1, dim=1, keepdim=True)
                writer.add_image('train/Prediction1',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_mae_cct_feature_dropout(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:

                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    # torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                    print("Epoch {:04d}: DICE improved from {:.5f} to {:.5f}, best model saved in {}".format(epoch_num, best_performance, performance,save_mode_path))
                    
                    best_performance = performance
                    best_mean_hd95 = mean_hd95
                    best_iter_num = iter_num

                logging.info(
                    snapshot_path,'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            # if iter_num % 3000 == 0:
            #     save_mode_path = os.path.join(
            #         snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            
        # lr_scheduler.step()
        if iter_num >= max_iterations:
            iterator.close()
            break
    logging.info(
    '%s :best iteration %d : best_dice : %f best_hd95 : %f' % (snapshot_path,best_iter_num, best_performance, best_mean_hd95))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
