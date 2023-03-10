import argparse
import os
import re
import shutil

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

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

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


def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    # print(case[0:10])
    org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def test_single_volume_mae(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    # print(case[0:10])
    org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()
    # plt.subplot(231)
    # plt.imshow(image[0])
    # # # plt.show()
    # plt.subplot(232)
    # plt.imshow(prediction[0])#, cmap='Greys_r') # 显示图片
    # plt.subplot(233)
    # plt.imshow(label[0])
    # plt.show()
    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

# def Inference(FLAGS):
#     train_ids, test_ids = get_fold_ids(FLAGS.fold)
#     all_volumes = os.listdir(
#         FLAGS.root_path + "/ACDC_training_volumes")
#     image_list = []
#     for ids in test_ids:
#         new_data_list = list(filter(lambda x: re.match(
#             '{}.*'.format(ids), x) != None, all_volumes))
#         image_list.extend(new_data_list)
#     snapshot_path = "../model/{}_{}/{}".format(
#         FLAGS.exp, FLAGS.fold, FLAGS.sup_type)
#     test_save_path = "../model/{}_{}/{}/{}_predictions/".format(
#         FLAGS.exp, FLAGS.fold, FLAGS.sup_type, FLAGS.model)
#     if os.path.exists(test_save_path):
#         shutil.rmtree(test_save_path)
#     os.makedirs(test_save_path)
#     net = net_factory(net_type=FLAGS.model, in_chns=1,
#                       class_num=FLAGS.num_classes)
    
#     if torch.cuda.device_count() > 1:
#         net = torch.nn.DataParallel(net)
#     net.to(torch.device("cuda"))
#     # save_mode_path = os.path.join(
#     #     snapshot_path, 'iter_3000.pth')
#     save_mode_path = os.path.join(
#         snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
#     net.load_state_dict(torch.load(save_mode_path))
#     print("init weight from {}".format(save_mode_path))
#     net.eval()

#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     for case in tqdm(image_list):
#         # print(case)
#         first_metric, second_metric, third_metric = test_single_volume_mae(
#             case, net, test_save_path, FLAGS)
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)
#     avg_metric = [first_total / len(image_list), second_total /
#                   len(image_list), third_total / len(image_list)]
#     print(avg_metric)
#     print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)  #dice, hd95, asd
#     return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)


# if __name__ == '__main__':
#     FLAGS = parser.parse_args()
#     total_dice = []
#     total_hd95 = []
#     snapshot_path = "../model/{}_{}.txt".format(FLAGS.exp, FLAGS.sup_type)
#     log_writter = open(snapshot_path, 'a+')
#     for i in [1, 2, 3, 4, 5]:
#         FLAGS.fold = "fold{}".format(i)
#         mean = Inference(FLAGS)      #dice, hd95, asd
#         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
#         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)
#         total_dice.append(mean[0])
#         total_hd95.append(mean[1])
#         print(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0))
#     print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

#     log_writter.flush()

def Inference_new(FLAGS,total_dice,total_hd95):
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
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        # print(case)
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        dice_case=(first_metric[0]+second_metric[0]+third_metric[0])/3.0
        hd95_case=(first_metric[1]+second_metric[1]+third_metric[1])/3.0
        total_dice.append(dice_case)
        total_hd95.append(hd95_case)
        # first_total += np.asarray(first_metric)
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
        # print(case,dice_case,hd95_case)
    # print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)  #dice, hd95, asd
    return total_dice,total_hd95#((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total_dice = []
    total_hd95 = []
    snapshot_path = "../model/{}_{}.txt".format(FLAGS.exp, FLAGS.sup_type)
    log_writter = open(snapshot_path, 'a+')
    for i in [1, 2, 3, 4, 5]:
        FLAGS.fold = "fold{}".format(i)
        total_dice ,total_hd95= Inference_new(FLAGS,total_dice,total_hd95)      #dice, hd95, asd
        # print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
        # print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)
        # total_dice.append(mean[0])
        # total_hd95.append(mean[1])
    print(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0),np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0))
    print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

    log_writter.flush()