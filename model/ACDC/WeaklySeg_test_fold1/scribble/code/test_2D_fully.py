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
    # transform=transforms.Compose([RandomGenerator(FLAGS.patch_size)])
    image = h5f['image'][:]
    label = h5f['label'][:]
    scribble = h5f['scribble'][:]
    # sample = {'image': image, 'label': label, 'mask': label}
    # sample = transform(sample)
    # label = transform(label)
    case = case.replace(".h5", "")
    case_path = test_save_path + case 
    prediction = np.zeros([image.shape[0],256,256])
    gt = np.zeros([image.shape[0],256,256])
    for ind in range(image.shape[0]):
        slice_path = case_path+"_{}".format(ind)
        # print(case_path,slice_path)
        slice = image[ind, :, :]
        mask = label[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # print(x, y)
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        mask =  zoom(mask, (256 / x, 256 / y), order=0)
        s =  zoom(scribble[ind, :, :], (256 / x, 256 / y), order=0)
        s=s+1
        s[s==1]=0
        s[s==5]=1
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = out#pred
            gt[ind] = mask#pred

    label = gt
    # print(case)
    org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.CopyInformation(org_img_itk)
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.CopyInformation(org_img_itk)
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def test_single_volume_unet(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path +
                    "/ACDC_training_volumes/{}".format(case), 'r')
    # transform=transforms.Compose([RandomGenerator(FLAGS.patch_size)])
    image = h5f['image'][:]
    label = h5f['label'][:]
    # sample = {'image': image, 'label': label, 'mask': label}
    # sample = transform(sample)
    # label = transform(label)
    case = case.replace(".h5", "")
    case_path = test_save_path + case 
    prediction = np.zeros([image.shape[0],256,256])
    gt = np.zeros([image.shape[0],256,256])
    for ind in range(image.shape[0]):
        slice_path = case_path+"_{}".format(ind)
        # print(case_path,slice_path)
        slice = image[ind, :, :]
        mask = label[ind, :, :]
        # print(slice.shape)
        x, y = slice.shape[0], slice.shape[1]
        # print(x, y)
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        mask =  zoom(mask, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)#[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = out#pred
            gt[ind] = mask#pred

            plt.imsave(slice_path+"_img.png", slice, cmap='Greys_r')
            plt.imsave(slice_path+"_pred.png", out)
            plt.imsave(slice_path+"_gt.png", mask)

    label = gt
    # print(case)
    org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    first_metric = calculate_metric_percase_new(
        prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
    second_metric = calculate_metric_percase_new(
        prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
    third_metric = calculate_metric_percase_new(
        prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.CopyInformation(org_img_itk)
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.CopyInformation(org_img_itk)
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.CopyInformation(org_img_itk)
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric
    
def test_single_volume_mae_cct(case, net, test_save_path, FLAGS):
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
            out_main = net(input,input)[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    case = case.replace(".h5", "")
    # print(case)
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
    return first_metric, second_metric, third_metric
  
def Inference_new(FLAGS,total_dice,total_hd95,class_dice,class_hd95,class_dice_0,class_dice_1,class_dice_2):
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
    print(FLAGS.model)
    for case in tqdm(image_list):
        if FLAGS.model == 'unet_cct':
            first_metric, second_metric, third_metric = test_single_volume_mae(
                case, net, test_save_path, FLAGS)
        elif FLAGS.model == 'unet':
            first_metric, second_metric, third_metric = test_single_volume_unet(
                case, net, test_save_path, FLAGS)
        elif FLAGS.model == 'unet_mae_cct':
            first_metric, second_metric, third_metric = test_single_volume_mae_cct(
                case, net, test_save_path, FLAGS)
        else:
            first_metric, second_metric, third_metric = test_single_volume_mae(
                case, net, test_save_path, FLAGS)
        dice_case=(first_metric[0]+second_metric[0]+third_metric[0])/3.0
        hd95_case=(first_metric[1]+second_metric[1]+third_metric[1])/3.0
        total_dice.append(dice_case)
        total_hd95.append(hd95_case)
        class_dice.append([first_metric[0], second_metric[0], third_metric[0]])
        class_dice_0.append(first_metric[1])
        class_dice_1.append(second_metric[1])
        class_dice_2.append(third_metric[1])
        class_hd95.append([first_metric[1], second_metric[1], third_metric[1]])
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]

    return total_dice,total_hd95,class_dice,class_hd95,((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    total_dice = []
    total_hd95 = []
    class_dice = []
    class_dice_0 = []
    class_dice_1 = []
    class_dice_2 = []
    class_hd95 = []
    snapshot_path = "../model/{}_{}.txt".format(FLAGS.exp, FLAGS.sup_type)
    log_writter = open(snapshot_path, 'a+')
    for i in [1, 2, 3, 4, 5]:
        FLAGS.fold = "fold{}".format(i)
        total_dice ,total_hd95,class_dice,class_hd95,mean= Inference_new(FLAGS,total_dice,total_hd95,class_dice,class_hd95)      #dice, hd95, asd
        print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
        print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)


        print(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0))
    print("Average Dice of MC,MYO,LV is {},std Dice is {}".format(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),),file=log_writter)
    print("Average HD95 of MC,MYO,LV is {}, std HD95 is {}".format(np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0)),file=log_writter)
    print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

    log_writter.flush()