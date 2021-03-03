#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import __init__ as booger

from common.laserscan import LaserScan, SemLaserScan
from common.laserscanvis_uncert import LaserScanVisUncert


import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/labels/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
             'Must point to directory containing the predictions in the proper format '
             ' (see readme)'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
             ', but if you are not done inferring this is not the case, so this disables'
             ' that safety.'
             'Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Sequence", FLAGS.sequence)
    print("Predictions", FLAGS.predictions)
    print("ignore_semantics", FLAGS.ignore_semantics)
    print("ignore_safety", FLAGS.ignore_safety)
    print("offset", FLAGS.offset)
    print("*" * 80)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()
    proj_pred_img_names = None
    # does sequence folder exist?
    if not FLAGS.ignore_semantics:
        if FLAGS.predictions is not None:
            pred_label_paths = os.path.join(FLAGS.predictions, "sequences",
                                       FLAGS.sequence, "predictions")
            gt_label_paths = os.path.join(FLAGS.dataset, "sequences",
                                       FLAGS.sequence, "labels")
        else:
            gt_label_paths = os.path.join(FLAGS.dataset, "sequences",
                                       FLAGS.sequence, "labels")
        if os.path.isdir(pred_label_paths):
            print("Labels folder exists! Using labels from %s" % pred_label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        pred_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_label_paths)) for f in fn]
        pred_label_names.sort()

        gt_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(gt_label_paths)) for f in fn]
        gt_label_names.sort()

        #get the list of prediction projected images
        proj_pred_img_names = glob.glob('/home/sa001/workspace/SalsaNext/prediction/second_trained_with_uncert/sequences/08/proj_label_with_uncert/*.png')
        proj_pred_img_names.sort()

        proj_uncert_img_names = glob.glob('/home/sa001/workspace/SalsaNext/prediction/second_trained_with_uncert/sequences/08/proj_uncert/*.png')
        proj_uncert_img_names.sort()




        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety:
            assert (len(pred_label_names) == len(scan_names))

    # create a scan
    if FLAGS.ignore_semantics:
        scan = LaserScan(project=True)  # project all opened scans to spheric proj
    else:
        color_dict = CFG["color_map"]
        scan = SemLaserScan(color_dict, project=True)

    # create a visualizer
    semantics = not FLAGS.ignore_semantics
    if not semantics:
        label_names = None
    vis = LaserScanVisUncert(scan=scan,
                       scan_names=scan_names,
                       pred_label_names=pred_label_names,
                       gt_label_names=gt_label_names,
                       proj_pred_img_names = proj_pred_img_names,
                       proj_uncert_img_names = proj_uncert_img_names,
                       offset=FLAGS.offset,
                       semantics=semantics,
                       instances=False)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tspace: toggle continous play")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
