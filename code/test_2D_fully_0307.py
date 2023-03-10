import argparse
import os
import re
import shutil
import cv2
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders.dataset1119 import RandomGenerator
from torchvision import transforms
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from utils.util import BlockMaskingGenerator,resize

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/WeaklySeg_pCE_MumfordShah_Loss', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold5', help='fold')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--sup_type', type=str, default="scribble",
                    help='label')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')

def get_fold_ids(fold):
    all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
    fold1_testing_set = [
        "patient{:0>3}".format(i) for i in range(1, 21)]
    fold1_training_set = [
        i for i in all_cases_set if i not in fold1_testing_set]

    fold2_testing_set = [
        "patient{:0>3}".format(i) for i in range(21, 41)]
    fold2_training_set = [
        i for i in all_cases_set if i not in fold2_testing_set]

    fold3_testing_set = [
        "patient{:0>3}".format(i) for i in range(41, 61)]
    fold3_training_set = [
        i for i in all_cases_set if i not in fold3_testing_set]

    fold4_testing_set = [
        "patient{:0>3}".format(i) for i in range(61, 81)]
    fold4_training_set = [
        i for i in all_cases_set if i not in fold4_testing_set]

    fold5_testing_set = [
        "patient{:0>3}".format(i) for i in range(81, 101)]
    fold5_training_set = [
        i for i in all_cases_set if i not in fold5_testing_set]
    if fold == "fold1":
        return [fold1_training_set, fold1_testing_set]
    elif fold == "fold2":
        return [fold2_training_set, fold2_testing_set]
    elif fold == "fold3":
        return [fold3_training_set, fold3_testing_set]
    elif fold == "fold4":
        return [fold4_training_set, fold4_testing_set]
    elif fold == "fold5":
        return [fold5_training_set, fold5_testing_set]
    else:
        return "ERROR KEY"

def calculate_metric_percase_new(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
        asd = metric.binary.asd(pred, gt, voxelspacing=[10, 1, 1])
        return dice, hd95, asd
    else:
        return 0, 50, 10.

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice

def test_single_volume_mae(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    # transform=transforms.Compose([RandomGenerator(FLAGS.patch_size)])
    image2 = cv2.imread(FLAGS.image_path,cv2.IMREAD_GRAYSCALE)#[:, :, ::-1]#h5f['image'][:]
    image = h5f['image'][:]
    # label = h5f['label'][:]
    # scribble = h5f['scribble'][:]
    # sample = {'image': image, 'label': label, 'mask': label}
    # sample = transform(sample)
    # label = transform(label)
    # case = case.replace(".h5", "")


    image = torch.from_numpy(image).unsqueeze(1)
    image2 = torch.from_numpy(image2/255).unsqueeze(0).unsqueeze(0)
    image = resize(image)
    print(image2.shape)
    # mask = label[ind, :, :]
    # s=s+1
    # s[s==1]=0
    # s[s==5]=1
    masked_volume1,masked_volume2,mask1,mask2  =  BlockMaskingGenerator(image2, 0.5)
    print(masked_volume1.shape)
    net.eval()
    with torch.no_grad():
        # slice = image[0, :, :]
        # x, y = slice.shape[1], slice.shape[2]
        # print(x, y)
        # slice = zoom(slice, (256 / image.shape[1], 256 / image.shape[2]), order=0)
        input = masked_volume1.float().cuda()
        # mask =  zoom(mask, (256 / x, 256 / y), order=0)
        # s =  zoom(scribble[ind, :, :], (256 / x, 256 / y), order=0)
        out_main = net(masked_volume1.float().cuda())[1]
        # out = torch.argmax(torch.softmax(
        #     out_main, dim=1), dim=1).squeeze(0)
        # restore_img = out_main.cpu().detach().numpy()
        restore_img = out_main[0, :, :, :]
        restore_img = (restore_img - restore_img.min()) / (restore_img.max() - restore_img.min())
        print(restore_img.shape)
        plt.subplot(221)
        plt.imshow(input[0,0].cpu().detach().numpy(), cmap='Greys_r')
        plt.subplot(222)
        plt.imshow(restore_img[0].cpu().detach().numpy(), cmap='Greys_r')#
        plt.subplot(223)
        plt.imshow(image2[0,0], cmap='Greys_r')
        plt.subplot(224)
        plt.imshow(masked_volume1[0,0].cpu().detach().numpy()) # 显示图片)
        plt.show()
        # dice = metric.binary.dc(out, mask)
        # cv2.imwrite('restore_img.png', restore_img)
        plt.imsave("./restore_img.png", restore_img[0].cpu().detach().numpy(), cmap='Greys_r')
        # plt.imsave(slice_path+"dice"+"_pred.png", out)
        # plt.imsave(slice_path+"dice"+"_gt.png", mask)
        # plt.imsave(slice_path+"dice"+"_scribble.png", s)




def Inference_new(FLAGS,total_dice,total_hd95,class_dice,class_hd95):
    train_ids, test_ids = get_fold_ids(FLAGS.fold)
    all_volumes = os.listdir(
        FLAGS.root_path + "/ACDC_training_volumes")
    image_list = []
    for ids in test_ids:
        new_data_list = list(filter(lambda x: re.match(
            '{}.*'.format(ids), x) != None, all_volumes))
        image_list.extend(new_data_list)
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
    test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(torch.device("cuda"))
    # save_mode_path = os.path.join(
    #     snapshot_path, 'iter_3000.pth')
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path),strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    # print(FLAGS.model)
    for case in tqdm(image_list):
        test_single_volume_mae(case,net, test_save_path, FLAGS)
    #     dice_case=(first_metric[0]+second_metric[0]+third_metric[0])/3.0
    #     hd95_case=(first_metric[1]+second_metric[1]+third_metric[1])/3.0
    #     total_dice.append(dice_case)
    #     total_hd95.append(hd95_case)
    #     class_dice.append([first_metric[0], second_metric[0], third_metric[0]])
    #     class_hd95.append([first_metric[1], second_metric[1], third_metric[1]])
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
        # print(case,dice_case,hd95_case)
    # print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)  #dice, hd95, asd
    return total_dice,total_hd95,class_dice,class_hd95#,((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total_dice = []
    total_hd95 = []
    class_dice = []
    class_hd95 = []
    snapshot_path = "../model/{}_{}.txt".format(FLAGS.exp, FLAGS.sup_type)
    log_writter = open(snapshot_path, 'a+')
    FLAGS.image_path = '/home/lc/Study/Project/WSL4MIS-main/model/ACDC/WeaklySeg_pCE_mae_post_process__aux_feature_mask-60000_fold5/scribble/1.png'
    for i in [1, 2, 3, 4, 5]:
        FLAGS.fold = "fold{}".format(i)
        total_dice ,total_hd95,class_dice,class_hd95= Inference_new(FLAGS,total_dice,total_hd95,class_dice,class_hd95)      #dice, hd95, asd
        # print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
        # print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)

    #     print(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0))
    # print("Average Dice of MC,MYO,LV is {},std Dice is {}".format(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),),file=log_writter)
    # print("Average HD95 of MC,MYO,LV is {}, std HD95 is {}".format(np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0)),file=log_writter)
    # print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

    log_writter.flush()