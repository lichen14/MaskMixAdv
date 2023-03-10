"""
Script for generating point annotation.
"""

import argparse
import csv
import os
import sys

import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.measure import label
from joblib import Parallel, delayed
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import cv2


COLORS = (
    (0),
    (1),
    (2),
    (3),
)

def _sample_within_region(region_mask, class_label, num_samples=1):

    xs, ys = np.where(region_mask)

    if num_samples == 1:
        x_center, y_center = int(xs.mean().round()), int(ys.mean().round())

        retry = 0
        while True:
            # deviate from the center within a circle within radius 5
            x = x_center + np.random.randint(-5, 6)
            y = y_center + np.random.randint(-5, 6)

            # if the center point is inside the region, return it
            try:
                if region_mask[x, y]:
                    return np.c_[x, y, class_label]
            except IndexError:
                pass
            finally:
                retry += 1

            if retry > 5:
                break

    selected_indexes = np.random.permutation(len(xs))[:num_samples]
    xs, ys = xs[selected_indexes], ys[selected_indexes]

    return np.c_[xs, ys, np.full_like(xs, class_label)]


def _generate_points(mask, point_ratio=1e-4):
    points = []

    # loop over all class labels
    # (from 0 to n_classes, where 0 is background)
    for class_label in np.unique(mask):
        class_mask = mask == class_label
        if class_label == 0:
            # if background, randomly sample some points
            points.append(
                _sample_within_region(
                    class_mask, class_label,
                    # num_samples=int(class_mask.sum() * point_ratio)
                    num_samples=int(10)
                )
            )
        else:
            class_mask = label(class_mask)
            region_indexes = np.unique(class_mask)
            region_indexes = region_indexes[np.nonzero(region_indexes)]

            # iterate over all instances of this class
            for idx in np.unique(region_indexes):
                region_mask = class_mask == idx
                num_samples = max(1, int(point_ratio))
                # num_samples=max(1, int(region_mask.sum() * point_ratio))
                points.append(
                    _sample_within_region(
                        region_mask, class_label, num_samples=num_samples
                    )
                )

    return np.concatenate(points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dot annotation generator.')
    parser.add_argument('root_dir',default='/home/lc/Study/Project/WSL4MIS-main/data/ACDC',
                        help='Path to data root directory with mask-level annotation.')
    parser.add_argument('-p', '--point-ratio', type=int, default=1e-4,
                        help='Percentage of labeled objects (regions) for each class')
    parser.add_argument('-r', '--radius', type=int, default=3, help='Circle radius')
    args = parser.parse_args()

    mask_dir = os.path.join(args.root_dir, 'ACDC_training_slices') # mask dir
    if not os.path.exists(mask_dir):
        print('Cannot generate dot annotation without masks.')
        sys.exit(1)

    label_dir = os.path.join(args.root_dir, 'ACDC_training_slices_point_label_{}'.format(args.point_ratio))

    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    print('Generating point annotation ...')

    def para_func(fname):
        output_size=[256,256]
        basename = os.path.splitext(fname)[0]
        h5f = h5py.File(mask_dir + "/{}.h5".format(basename), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        mask  = h5f['label'][:]
        scribble  = h5f['scribble'][:]
        x, y = mask.shape
        image = zoom(
            image, (output_size[0] / x, output_size[1] / y), order=0)
        label = zoom(
            label, (output_size[0] / x, output_size[1] / y), order=0)
        scribble = zoom(
            scribble, (output_size[0] / x, output_size[1] / y), order=0)
        mask = zoom(
            mask, (output_size[0] / x, output_size[1] / y), order=0)
        # mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        zeros=np.ones_like(mask)*4
        points = _generate_points(mask, point_ratio=args.point_ratio)
        print(mask.shape,points.shape)
        plt.subplot(131)
        plt.imshow(mask)#, cmap='Greys_r') # 显示图片

        points[:, [0, 1]] = points[:, [1, 0]]
        for p in range(points.shape[0]):
            point = points[p]
            print(p,point,point[0], point[1],point[2])
            # point = [int(d) for d in point]
            cv2.circle(zeros, (point[0], point[1]), args.radius, COLORS[point[2]], -1)

        plt.subplot(132)
        plt.imshow(scribble)#, cmap='Greys_r') # 显示图片
        plt.subplot(133)
        plt.imshow(zeros)#, cmap='Greys_r') # 显示图片
        # plt.show()
        # conform to the xy format
        print(points)
        h5f_write = h5py.File(label_dir + "/{}.h5".format(basename), 'w')
        h5f_write['image']=image
        h5f_write['point']=zeros
        h5f_write['mask']=mask
        h5f_write['scribble']=scribble

        h5f_write.close()
        # with open(os.path.join(label_dir, f'{basename}.csv'), 'w') as fp:
        #     writer = csv.writer(fp)
        #     writer.writerows(points)

        return len(points)

    executor = Parallel(n_jobs=2)
    points_nums = executor(delayed(para_func)(fname) for fname in tqdm(os.listdir(mask_dir)))
    print(f'Average number of points: {np.mean(points_nums)}.')