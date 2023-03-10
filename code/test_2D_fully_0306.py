# import argparse
# import os
# import re
# import shutil
# # import cv2
# import h5py
# import nibabel as nib
# import numpy as np
# import SimpleITK as sitk
# import torch
# from medpy import metric
# from scipy.ndimage import zoom
# from scipy.ndimage.interpolation import zoom
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from dataloaders.dataset1119 import RandomGenerator
# from torchvision import transforms
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from networks.net_factory import net_factory

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/ACDC', help='Name of Experiment')
# parser.add_argument('--exp', type=str,
#                     default='ACDC/WeaklySeg_pCE_MumfordShah_Loss', help='experiment_name')
# parser.add_argument('--net', type=str,
#                     default='unet', help='model_name')
# parser.add_argument('--fold', type=str,
#                     default='fold5', help='fold')
# parser.add_argument('--num_classes', type=int,  default=4,
#                     help='output channel of network')
# parser.add_argument('--sup_type', type=str, default="scribble",
#                     help='label')
# parser.add_argument('--patch_size', type=list,  default=[256, 256],
#                     help='patch size of network input')

# def get_fold_ids(fold):
#     all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
#     fold1_testing_set = [
#         "patient{:0>3}".format(i) for i in range(1, 21)]
#     fold1_training_set = [
#         i for i in all_cases_set if i not in fold1_testing_set]

#     fold2_testing_set = [
#         "patient{:0>3}".format(i) for i in range(21, 41)]
#     fold2_training_set = [
#         i for i in all_cases_set if i not in fold2_testing_set]

#     fold3_testing_set = [
#         "patient{:0>3}".format(i) for i in range(41, 61)]
#     fold3_training_set = [
#         i for i in all_cases_set if i not in fold3_testing_set]

#     fold4_testing_set = [
#         "patient{:0>3}".format(i) for i in range(61, 81)]
#     fold4_training_set = [
#         i for i in all_cases_set if i not in fold4_testing_set]

#     fold5_testing_set = [
#         "patient{:0>3}".format(i) for i in range(81, 101)]
#     fold5_training_set = [
#         i for i in all_cases_set if i not in fold5_testing_set]
#     if fold == "fold1":
#         return [fold1_training_set, fold1_testing_set]
#     elif fold == "fold2":
#         return [fold2_training_set, fold2_testing_set]
#     elif fold == "fold3":
#         return [fold3_training_set, fold3_testing_set]
#     elif fold == "fold4":
#         return [fold4_training_set, fold4_testing_set]
#     elif fold == "fold5":
#         return [fold5_training_set, fold5_testing_set]
#     else:
#         return "ERROR KEY"

# def calculate_metric_percase_new(pred, gt, spacing):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
#         asd = metric.binary.asd(pred, gt, voxelspacing=[10, 1, 1])
#         return dice, hd95, asd
#     else:
#         return 0, 50, 10.

# def calculate_metric_percase(pred, gt, spacing):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     dice = metric.binary.dc(pred, gt)
#     asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
#     hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
#     return dice, hd95, asd

# def calculate_dice_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     dice = metric.binary.dc(pred, gt)
#     return dice

# def test_single_volume(case, net, test_save_path, args):
#     h5f = h5py.File(args.root_path +
#                     "/ACDC_training_volumes/{}".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out_main = net(input)[0]
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = pred
#     case = case.replace(".h5", "")
#     # print(case[0:10])
#     org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
#     org_img_itk = sitk.ReadImage(org_img_path)
#     spacing = org_img_itk.GetSpacing()

#     first_metric = calculate_metric_percase(
#         prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
#     second_metric = calculate_metric_percase(
#         prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
#     third_metric = calculate_metric_percase(
#         prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.CopyInformation(org_img_itk)
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.CopyInformation(org_img_itk)
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.CopyInformation(org_img_itk)
#     # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric

# def test_single_volume_mae(case, net, test_save_path, args):
#     h5f = h5py.File(args.root_path +
#                     "/ACDC_training_volumes/{}".format(case), 'r')
#     transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(args.patch_size,interpolation=transforms.InterpolationMode.BILINEAR),
#                                   transforms.Normalize([0.5], [0.5]),
#                                   ])
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     scribble = h5f['scribble'][:]
#     # sample = {'image': image, 'label': label, 'mask': label}
#     # sample = transform(sample)
#     image = transform(image)
#     print(image.shape)
#     case = case.replace(".h5", "")
#     case_path = test_save_path + case 
#     prediction = np.zeros([image.shape[0],256,256])
#     gt = np.zeros([image.shape[0],256,256])

#     target_layers = [net.encoder.down4]
#     # input_tensor = input
#     # Create an input tensor image for your net..
#     # Note: input_tensor can be a batch tensor with several images!

#     # Construct the CAM object once, and then re-use it on many images:
#     cam = GradCAM(net=net, target_layers=target_layers)

#     # You can also use it within a with statement, to make sure it is freed,
#     # In case you need to re-create it inside an outer loop:
#     # with GradCAM(net=net, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#     #   ...

#     # We have to specify the target we want to generate
#     # the Class Activation Maps for.
#     # If targets is None, the highest scoring category
#     # will be used for every image in the batch.
#     # Here we use ClassifierOutputTarget, but you can define your own custom targets
#     # That are, for example, combinations of categories, or specific outputs in a non standard net.

#     targets = None#out_main#[1]#
#     input_tensor1 = image.unsqueeze(1)[0:10]
#     print(input_tensor1.shape)
#     # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
#     grayscale_cam = cam(input_tensor=input_tensor1.cuda(), targets=targets)
#     # print(input_tensor1)
#     # # In this example grayscale_cam has only one image in the batch:
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(input[0], grayscale_cam, use_rgb=True)

#     for ind in range(image.shape[0]):
#         slice_path = case_path+"_{}".format(ind)
#         print(case_path,slice_path)
#         slice = image[ind, :, :]
#         mask = label[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         # print(x, y)
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         mask =  zoom(mask, (256 / x, 256 / y), order=0)
#         s =  zoom(scribble[ind, :, :], (256 / x, 256 / y), order=0)
#         s=s+1
#         s[s==1]=0
#         s[s==5]=1
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
        
#         with torch.no_grad():
#             out_main = net(input)[0]
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = out#pred
#             gt[ind] = mask#pred
#             # plt.subplot(221)
#             # plt.imshow(slice, cmap='Greys_r')
#             # plt.subplot(222)
#             # plt.imshow(out)#
#             # plt.subplot(223)
#             # plt.imshow(mask)
#             # plt.subplot(224)
#             # plt.imshow(s) # 显示图片)
#             # plt.show()

#             dice = metric.binary.dc(out, mask)
#             # plt.imsave(slice_path+"dice"+str(dice)+"_img.png", slice, cmap='Greys_r')
#             # plt.imsave(slice_path+"dice"+str(dice)+"_pred.png", out)
#             # plt.imsave(slice_path+"dice"+str(dice)+"_gt.png", mask)
#             # plt.imsave(slice_path+"dice"+str(dice)+"_scribble.png", s)

#             # cv2.imwrite(slice_path+"_img.png", slice)
#             # cv2.imwrite(slice_path+"_pred.png", pred)
#             # cv2.imwrite(slice_path+"_gt.png", label[ind, :, :])
#     label = gt
#     # print(case)
#     org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
#     org_img_itk = sitk.ReadImage(org_img_path)
#     spacing = org_img_itk.GetSpacing()

#     first_metric = calculate_metric_percase(
#         prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
#     second_metric = calculate_metric_percase(
#         prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
#     third_metric = calculate_metric_percase(
#         prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

#     # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     # img_itk.CopyInformation(org_img_itk)
#     # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     # prd_itk.CopyInformation(org_img_itk)
#     # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     # lab_itk.CopyInformation(org_img_itk)
#     # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric

# def test_single_volume_unet(case, net, test_save_path, args):
#     h5f = h5py.File(args.root_path +
#                     "/ACDC_training_volumes/{}".format(case), 'r')
#     # transform=transforms.Compose([RandomGenerator(args.patch_size)])
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     # sample = {'image': image, 'label': label, 'mask': label}
#     # sample = transform(sample)
#     # label = transform(label)
#     case = case.replace(".h5", "")
#     case_path = test_save_path + case 
#     prediction = np.zeros([image.shape[0],256,256])
#     gt = np.zeros([image.shape[0],256,256])
#     for ind in range(image.shape[0]):
#         slice_path = case_path+"_{}".format(ind)
#         # print(case_path,slice_path)
#         slice = image[ind, :, :]
#         mask = label[ind, :, :]
#         # print(slice.shape)
#         x, y = slice.shape[0], slice.shape[1]
#         # print(x, y)
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         mask =  zoom(mask, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out_main = net(input)#[0]
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = out#pred
#             gt[ind] = mask#pred
#             # plt.subplot(221)
#             # plt.imshow(slice, cmap='Greys_r')
#             # plt.subplot(222)
#             # plt.imshow(out)#, cmap='Greys_r') # 显示图片
#             # plt.subplot(223)
#             # plt.imshow(mask)

#             plt.imsave(slice_path+"_img.png", slice, cmap='Greys_r')
#             plt.imsave(slice_path+"_pred.png", out)
#             plt.imsave(slice_path+"_gt.png", mask)

#             # plt.show()
#             # cv2.imwrite(slice_path+"_img.png", slice)
#             # cv2.imwrite(slice_path+"_pred.png", pred)
#             # cv2.imwrite(slice_path+"_gt.png", label[ind, :, :])
#     label = gt
#     # print(case)
#     org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
#     org_img_itk = sitk.ReadImage(org_img_path)
#     spacing = org_img_itk.GetSpacing()

#     first_metric = calculate_metric_percase_new(
#         prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
#     second_metric = calculate_metric_percase_new(
#         prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
#     third_metric = calculate_metric_percase_new(
#         prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

#     # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     # img_itk.CopyInformation(org_img_itk)
#     # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     # prd_itk.CopyInformation(org_img_itk)
#     # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     # lab_itk.CopyInformation(org_img_itk)
#     # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric
    
# def test_single_volume_mae_cct(case, net, test_save_path, args):
#     h5f = h5py.File(args.root_path +
#                     "/ACDC_training_volumes/{}".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out_main = net(input,input)[0]
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = pred
#     case = case.replace(".h5", "")
#     # print(case)
#     org_img_path = "../data/ACDC_training/{}/{}.nii.gz".format(case[0:10],case)
#     org_img_itk = sitk.ReadImage(org_img_path)
#     spacing = org_img_itk.GetSpacing()
#     # plt.subplot(231)
#     # plt.imshow(image[0])
#     # # # plt.show()
#     # plt.subplot(232)
#     # plt.imshow(prediction[0])#, cmap='Greys_r') # 显示图片
#     # plt.subplot(233)
#     # plt.imshow(label[0])
#     # plt.show()
#     first_metric = calculate_metric_percase(
#         prediction == 1, label == 1, (spacing[2], spacing[0], spacing[1]))
#     second_metric = calculate_metric_percase(
#         prediction == 2, label == 2, (spacing[2], spacing[0], spacing[1]))
#     third_metric = calculate_metric_percase(
#         prediction == 3, label == 3, (spacing[2], spacing[0], spacing[1]))

#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.CopyInformation(org_img_itk)
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.CopyInformation(org_img_itk)
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.CopyInformation(org_img_itk)
#     # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric
  

# # def Inference(args):
# #     train_ids, test_ids = get_fold_ids(args.fold)
# #     all_volumes = os.listdir(
# #         args.root_path + "/ACDC_training_volumes")
# #     image_list = []
# #     for ids in test_ids:
# #         new_data_list = list(filter(lambda x: re.match(
# #             '{}.*'.format(ids), x) != None, all_volumes))
# #         image_list.extend(new_data_list)
# #     snapshot_path = "../net/{}_{}/{}".format(
# #         args.exp, args.fold, args.sup_type)
# #     test_save_path = "../net/{}_{}/{}/{}_predictions/".format(
# #         args.exp, args.fold, args.sup_type, args.net)
# #     if os.path.exists(test_save_path):
# #         shutil.rmtree(test_save_path)
# #     os.makedirs(test_save_path)
# #     net = net_factory(net_type=args.net, in_chns=1,
# #                       class_num=args.num_classes)
    
# #     if torch.cuda.device_count() > 1:
# #         net = torch.nn.DataParallel(net)
# #     net.to(torch.device("cuda"))
# #     # save_mode_path = os.path.join(
# #     #     snapshot_path, 'iter_3000.pth')
# #     save_mode_path = os.path.join(
# #         snapshot_path, '{}_best_model.pth'.format(args.net))
# #     net.load_state_dict(torch.load(save_mode_path))
# #     print("init weight from {}".format(save_mode_path))
# #     net.eval()

# #     first_total = 0.0
# #     second_total = 0.0
# #     third_total = 0.0
# #     for case in tqdm(image_list):
# #         # print(case)
# #         first_metric, second_metric, third_metric = test_single_volume_mae(
# #             case, net, test_save_path, args)
# #         first_total += np.asarray(first_metric)
# #         second_total += np.asarray(second_metric)
# #         third_total += np.asarray(third_metric)
# #     avg_metric = [first_total / len(image_list), second_total /
# #                   len(image_list), third_total / len(image_list)]
# #     print(avg_metric)
# #     print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)  #dice, hd95, asd
# #     return ((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)


# # if __name__ == '__main__':
# #     args = parser.parse_args()
# #     total_dice = []
# #     total_hd95 = []
# #     snapshot_path = "../net/{}_{}.txt".format(args.exp, args.sup_type)
# #     log_writter = open(snapshot_path, 'a+')
# #     for i in [1, 2, 3, 4, 5]:
# #         args.fold = "fold{}".format(i)
# #         mean = Inference(args)      #dice, hd95, asd
# #         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
# #         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)
# #         total_dice.append(mean[0])
# #         total_hd95.append(mean[1])
# #         print(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0))
# #     print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

# #     log_writter.flush()
# def Inference_new(args,total_dice,total_hd95,class_dice,class_hd95):
#     train_ids, test_ids = get_fold_ids(args.fold)
#     all_volumes = os.listdir(
#         args.root_path + "/ACDC_training_volumes")
#     image_list = []
#     for ids in test_ids:
#         new_data_list = list(filter(lambda x: re.match(
#             '{}.*'.format(ids), x) != None, all_volumes))
#         image_list.extend(new_data_list)
#     snapshot_path = "../net/{}_{}/{}".format(
#         args.exp, args.fold, args.sup_type)
#     test_save_path = "../net/{}_{}/{}/{}_predictions/".format(
#         args.exp, args.fold, args.sup_type, args.net)
#     if os.path.exists(test_save_path):
#         shutil.rmtree(test_save_path)
#     os.makedirs(test_save_path)
#     net = net_factory(net_type=args.net, in_chns=1,
#                       class_num=args.num_classes)
    
#     # if torch.cuda.device_count() > 1:
#     #     net = torch.nn.DataParallel(net)
#     net.to(torch.device("cuda"))
#     # save_mode_path = os.path.join(
#     #     snapshot_path, 'iter_3000.pth')
#     save_mode_path = os.path.join(
#         snapshot_path, '{}_best_model.pth'.format(args.net))
#     state = torch.load(save_mode_path)
#     state_dict = {k.replace("module.", ""): v for k, v in state.items()}
#     net.load_state_dict(state_dict)
#     print("init weight from {}".format(save_mode_path))
#     # print(net)
#     net.eval()


#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     print(args.net)
#     for case in tqdm(image_list):
#         if args.net == 'unet_cct':
#             first_metric, second_metric, third_metric = test_single_volume_mae(
#                 case, net, test_save_path, args)
#         elif args.net == 'unet':
#             first_metric, second_metric, third_metric = test_single_volume_unet(
#                 case, net, test_save_path, args)
#         elif args.net == 'unet_mae_cct':
#             first_metric, second_metric, third_metric = test_single_volume_mae_cct(
#                 case, net, test_save_path, args)
#         else:
#             first_metric, second_metric, third_metric = test_single_volume_mae(
#                 case, net, test_save_path, args)
#         dice_case=(first_metric[0]+second_metric[0]+third_metric[0])/3.0
#         hd95_case=(first_metric[1]+second_metric[1]+third_metric[1])/3.0
#         total_dice.append(dice_case)
#         total_hd95.append(hd95_case)
#         class_dice.append([first_metric[0], second_metric[0], third_metric[0]])
#         class_hd95.append([first_metric[1], second_metric[1], third_metric[1]])
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)
#     avg_metric = [first_total / len(image_list), second_total /
#                   len(image_list), third_total / len(image_list)]
#         # print(case,dice_case,hd95_case)
#     # print((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)  #dice, hd95, asd
#     return total_dice,total_hd95,class_dice,class_hd95,((avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3)

# if __name__ == '__main__':
#     args = parser.parse_args()
#     total_dice = []
#     total_hd95 = []
#     class_dice = []
#     class_hd95 = []
#     snapshot_path = "../net/{}_{}.txt".format(args.exp, args.sup_type)
#     log_writter = open(snapshot_path, 'a+')
#     for i in [1, 2, 3, 4, 5]:
#         args.fold = "fold{}".format(i)
#         total_dice ,total_hd95,class_dice,class_hd95,mean= Inference_new(args,total_dice,total_hd95,class_dice,class_hd95)      #dice, hd95, asd
#         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]))
#         print("Inference fold{}, Dice is {}, HD95 is {}, ASD is {}".format(i,mean[0],mean[1],mean[2]),file=log_writter)

#         print(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0))
#     print("Average Dice of MC,MYO,LV is {},std Dice is {}".format(np.asarray(class_dice).mean(axis = 0), np.asarray(class_dice).std(axis = 0),),file=log_writter)
#     print("Average HD95 of MC,MYO,LV is {}, std HD95 is {}".format(np.asarray(class_hd95).mean(axis = 0), np.asarray(class_hd95).std(axis = 0)),file=log_writter)
#     print("Total Average Dice is {:.4f},std Dice is {:.4f}, Average HD95 is {:.4f}, std HD95 is {:.4f}".format(np.asarray(total_dice).mean(axis = 0), np.asarray(total_dice).std(axis = 0), np.asarray(total_hd95).mean(axis = 0), np.asarray(total_hd95).std(axis = 0)),file=log_writter)

#     log_writter.flush()







import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image, standerlize_image

from networks.net_factory import net_factory
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 如果出现 OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='/home/lc/Study/Project/WSL4MIS-main/model/ACDC/WeaklySeg_pCE_mae_post_process__aux_feature_mask-60000_fold5/scribble/1.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')
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
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    net = net_factory(net_type=args.model, in_chns=1,
                      class_num=args.num_classes)
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    net.to(torch.device("cuda"))
    # save_mode_path = os.path.join(
    #     snapshot_path, 'iter_3000.pth')
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(args.model))
    state = torch.load(save_mode_path)
    state_dict = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state_dict)
    print("init weight from {}".format(save_mode_path))
    # print(net)
    net.eval()
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the net.
    # Some common choices can be:
    # Resnet18 and 50: net.layer4[-1]
    # VGG, densenet161: net.features[-1]
    # mnasnet1_0: net.layers[-1]
    # You can print the net to help chose the layer
    target_layer = [net.seg_decoder.up4]#
    target_layer = [net.encoder.down4]#

    cam = methods[args.method](model=net,
                               target_layers=target_layer,
                               use_cuda=args.use_cuda)

    gray_img = cv2.imread(args.image_path,cv2.IMREAD_GRAYSCALE)#[:, :, ::-1]

    gray_img = np.float32(gray_img) / 255
    input_tensor = preprocess_image(gray_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_1 = [ClassifierOutputTarget(1)]#None
    target_2 = [ClassifierOutputTarget(2)]#None
    target_3 = [ClassifierOutputTarget(3)]#None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor.cuda(),
                        targets=target_1,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam_1 = grayscale_cam[0, :]

    grayscale_cam = cam(input_tensor=input_tensor.cuda(),
                        targets=target_2,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam_2 = grayscale_cam[0, :]

    grayscale_cam = cam(input_tensor=input_tensor.cuda(),
                        targets=target_3,
                        aug_smooth=args.aug_smooth,
                        eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam_3 = grayscale_cam[0, :]

    grayscale_cam = (standerlize_image(grayscale_cam_1)+standerlize_image(grayscale_cam_2))
    print(grayscale_cam.shape)
    rgb_img = cv2.imread(args.image_path)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    print(rgb_img.shape)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # gb_model = GuidedBackpropReLUModel(model=net, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor.cuda(), target_category=target)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)
    plt.subplot(221)
    plt.imshow(show_cam_on_image(rgb_img, grayscale_cam_1))
    plt.subplot(222)
    plt.imshow(show_cam_on_image(rgb_img, grayscale_cam_2))
    plt.subplot(223)
    plt.imshow(show_cam_on_image(rgb_img, grayscale_cam_3))
    plt.subplot(224)
    plt.imshow(show_cam_on_image(rgb_img, grayscale_cam))
    plt.show()