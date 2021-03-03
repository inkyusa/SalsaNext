import os

#import matplotlib
import numpy as np
import pykitti

#matplotlib.use('Agg')

#import matplotlib.pyplot as plt

import yaml
import cv2
import cmapy
import math
#import concurrent.futures

from joblib import Parallel, delayed

kitti_basedir = '/home/sa001/workspace/dataset/semantic_kitti/dataset/'
#basedir = '/home/sa001/workspace/SalsaNext/prediction/second_trained_with_uncert/'
basedir = '/home/sa001/workspace/SalsaNext/prediction/first_trained_with_uncert/'
sequence = '08'
uncerts = 'uncert'
preds = 'predictions'
gt = 'labels'
projected_uncert = 'proj_uncert' # The name of folder to store projected images
#projected_label = 'proj_gt' # The name of folder to store projected images
projected_label = 'proj_label_' # The name of folder to store projected images
config_yaml = '/home/sa001/workspace/SalsaNext/train/tasks/semantic/config/labels/semantic-kitti.yaml'
dataset = pykitti.odometry(kitti_basedir, sequence)
uncert_cmap = 'jet' #viridis, hsv

EXTENSIONS_LABEL = ['.label']
EXTENSIONS_LIDAR = ['.bin']
EXTENSIONS_IMG = ['.png']

VIS_LABEL = True #set False to save uncertainty projected images
if VIS_LABEL == True:
    print("Label visualisation")
    projected_label_path = os.path.join(basedir, 'sequences', sequence, projected_label)
    if not os.path.exists(projected_label_path):
        os.mkdir(projected_label_path)
        print("create {}".format(projected_label_path))
    else:
        print("{} already exist".format(projected_label_path))

else:
    print("Uncertainty visualisation")
    projected_uncert_path = os.path.join(basedir, 'sequences', sequence, projected_uncert)
    if not os.path.exists(projected_uncert_path):
        os.mkdir(projected_uncert_path)
        print("create {}".format(projected_uncert_path))
    else:
        print("{} already exist".format(projected_uncert_path))

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_lidar(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LIDAR)


def is_img(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMG)

path = os.path.join(basedir, 'sequences', sequence, uncerts)


scan_uncert = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_uncert.sort()
path = os.path.join(basedir, 'sequences', sequence, preds)
scan_preds = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_preds.sort()

path = os.path.join(kitti_basedir, 'sequences', sequence, gt)
scan_gt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(path)) for f in fn if is_label(f)]
scan_gt.sort()

color_map_dict = yaml.safe_load(open(config_yaml))['color_map']
learning_map = yaml.safe_load(open(config_yaml))['learning_map']
color_map = {}
color_map_cv = {}
uncert_mean = np.zeros(20)
total_points_per_class = np.zeros(20)
for key, value in color_map_dict.items():
    color_map[key] = np.array(value, np.float32) / 255.0
    color_map_cv[key] = value


def plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name):
    labels = np.fromfile(label_name, dtype=np.int32).reshape((-1))
    uncerts = np.fromfile(label_uncert, dtype=np.float32).reshape((-1))
    velo_points = np.fromfile(lidar_name, dtype=np.float32).reshape(-1, 4)
    try:
        cam2_image = cv2.imread(cam2_image_name)
    except IOError:
        print('detect error img %s' % label_name)
    if True:

        # Project points to camera.
        cam2_points = dataset.calib.T_cam2_velo.dot(velo_points.T).T

        # Filter out points behind camera
        idx = cam2_points[:, 2] > 0
        #print(idx)
        # velo_points_projected = velo_points[idx]
        cam2_points = cam2_points[idx]
        labels_projected = labels[idx]
        uncert_projected = uncerts[idx]

        # Remove homogeneous z.
        cam2_points = cam2_points[:, :3] / cam2_points[:, 2:3]

        # Apply instrinsics.
        intrinsic_cam2 = dataset.calib.K_cam2
        cam2_points = intrinsic_cam2.dot(cam2_points.T).T[:, [1, 0]]
        cam2_points = cam2_points.astype(int)

        for i in range(0, cam2_points.shape[0]):
            u, v = cam2_points[i, :]
            label = labels_projected[i]
            uncert = uncert_projected[i]
            if math.isnan(uncert):
                uncert = 0
            if label >= 0 and label <= 1000 and v > 0 and v < 1241 and u > 0 and u < 376:
                uncert_mean[learning_map[label]] += uncert
                total_points_per_class[learning_map[label]] += 1
                if VIS_LABEL == True:
                    cv2.circle(cam2_image, (v, u), 3, color_map_cv[label], cv2.FILLED)
                else:
                    cv2.circle(cam2_image, (v, u), 3, cmapy.color(uncert_cmap, float(uncert)),  cv2.FILLED)

    if VIS_LABEL == True:
        path = os.path.join(basedir, 'sequences', sequence, projected_label, label_name.split('/')[-1].split('.')[0] + '.png')
    else:
        #uncertainty
        path = os.path.join(basedir, 'sequences', sequence, projected_uncert, label_name.split('/')[-1].split('.')[0] + '.png')
    cv2.imwrite(path, cam2_image)
    print("{} saved".format(path.split('/')[-1]))


#n_jobs = -1 to use all available CPUs
Parallel(n_jobs=1)(delayed(plot_and_save)(label_uncert, label_name, lidar_name, cam2_image_name)
                         for label_uncert, label_name, lidar_name, cam2_image_name in zip(scan_uncert, scan_gt, dataset.velo_files, dataset.cam2_files))

# for label_uncert, label_name, lidar_name, cam2_image_name in zip(scan_uncert, scan_gt, dataset.velo_files, dataset.cam2_files):
#     print(label_name.split('/')[-1])
#     plot_and_save(label_uncert, label_name, lidar_name, cam2_image_name)

print(total_points_per_class)
print(uncert_mean)
if __name__ == "__main__":
    pass
