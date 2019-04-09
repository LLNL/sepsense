#!/usr/bin/env python3

import os
import sys
import json
import math
import argparse
import collections
import torch.utils.data
import tifffile

import numpy as np
from pprint import pprint

import progressbar

# Local imports
from fmow import FMOW

# Parse user args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/data/fmow/full', help='Location of fMoW dataset on filesystem.')
parser.add_argument('--proc_op', type=str, choices=['crop', 'context'], default='context', help='')
parser.add_argument('--img_dir', type=str, default='/data/fmow/proc', help='')
parser.add_argument('--partition_dir', type=str, default='./partition', help='')
parser.add_argument('--modality', type=str, choices=['rgb', 'ms'], default='ms', help='')
parser.add_argument('--filetype', type=str, choices=['jpg', 'tif'], default='tif', help='')
parser.add_argument('--max_gsd', type=int, default=3.0, help='')
args = parser.parse_args()

# Num in train and test set for FMOW
NUM_IMG = {
    'train': 363572,
    'val': 100000,
    'test': 53473
}

NUM_INST = {
    'train': 1000,
    'val': 100,
    'test': 100
}

PLATFORM_LIST = [
    'GEOEYE01',
    'QUICKBIRD02',
    'WORLDVIEW02',
    'WORLDVIEW03_VNIR'
]

BAND_DICT = {
    4: ('GEOEYE01', 'QUICKBIRD02'),
    8: ('WORLDVIEW02', 'WORLDVIEW03_VNIR')
}

# Set up image path suffix to look for
suffix_str = '_{}.{}'.format(args.modality, args.filetype)

class Dataset(torch.utils.data.Dataset):
    """
    363,572 train images
     53,041 val   images
    """

    def __init__(self, dataset_dir, img_dir, label_dir, mode, num_img, inst_thresh=100):
        self.dataset_dir = dataset_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mode = mode
        self.num_img = num_img
        self.inst_thresh = inst_thresh
        if self.mode == 'train' or self.mode == 'val':
            self._prep_trainval()
        else:
            self._prep_test()

    def _prep_trainval(self):
        # Create image dir
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        # Extract train image paths
        partition_dir = os.path.join(self.dataset_dir, self.mode)
        self.path_list = []
        with progressbar.ProgressBar(max_value=self.num_img) as bar:
            i = 0
            for class_name in os.listdir(partition_dir):
                class_dir = os.path.join(partition_dir, class_name)
                inst_list = os.listdir(class_dir)
                #if len(inst_list) >= self.inst_thresh:
                #    inst_list = inst_list[:self.inst_thresh]
                for instance in inst_list:
                    instance_dir = os.path.join(class_dir, instance)
                    for fname in os.listdir(instance_dir):
                        if fname.endswith(suffix_str):
                            fbase = os.path.splitext(fname)[0]
                            old_img_path = os.path.join(instance_dir, fname)
                            json_path = os.path.join(instance_dir, '{}.json'.format(os.path.splitext(fname)[0]))
                            self.path_list.append((fbase, old_img_path, json_path))
                            i += 1
                            bar.update(i)
                            # Only take one image per instance
                            break

    def _prep_test(self):
        # Create image dir
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        # Extract train image paths
        partition_dir = os.path.join(self.dataset_dir, self.mode)
        self.path_list = []
        with progressbar.ProgressBar(max_value=self.num_img) as bar:
            i = 0
            for instance in os.listdir(partition_dir):
                instance_dir = os.path.join(partition_dir, instance)
                for fname in os.listdir(instance_dir):
                    if fname.endswith(suffix_str):
                        fbase = os.path.splitext(fname)[0]
                        old_img_path = os.path.join(instance_dir, fname)
                        json_path = os.path.join(instance_dir, '{}.json'.format(os.path.splitext(fname)[0]))
                        self.path_list.append((fbase, old_img_path, json_path))
                        i += 1
                        bar.update(i)

    def __getitem__(self, index):
        return self.path_list[index]

    def __len__(self):
        return len(self.path_list)

def extract(dataset_dir, img_dir, label_dir, mode, num_img, batch_size=64, num_workers=40, inst_thresh=100, max_gsd=1.0, num_folds=10):
    # Create dataset object
    dataset = Dataset(dataset_dir, img_dir, label_dir, mode, num_img, inst_thresh=inst_thresh)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    # Extract train image paths
    partition_dir = os.path.join(dataset_dir, mode)
    item_dict = collections.defaultdict(list)
    # Dict of counts
    count_dict = collections.defaultdict(lambda : collections.defaultdict(int))
    # Set of platforms seen
    platform_set = set()
    # Dict of abs_cal_factors
    cal_dict = collections.defaultdict(set) 
    with progressbar.ProgressBar(max_value=len(dataloader)) as bar:
        for batch_idx, (fbase_list, old_img_path_list, json_path_list) in enumerate(dataloader):
            for fbase, old_img_path, json_path in zip(fbase_list, old_img_path_list, json_path_list):
                # Open the JSON metadata and pull out the bbox
                with open(json_path, 'r') as fp:
                    md_dict = json.load(fp)
                gsd = md_dict['gsd']
                platform = md_dict['sensor_platform_name']
                platform_set.add(platform)
                if type(md_dict['bounding_boxes']) == dict:
                    bbox = md_dict['bounding_boxes']['box']
                    label = md_dict['bounding_boxes']['category']
                elif type(md_dict['bounding_boxes']) == list:
                    bbox = md_dict['bounding_boxes'][0]['box']
                    label = md_dict['bounding_boxes'][0]['category']
                if 'abs_cal_factors' in md_dict:
                    count_dict['cal']['yes'] += 1
                    if platform in PLATFORM_LIST:
                        if gsd <= max_gsd:
                            count_dict['gsd']['yes'] += 1
                            if len(item_dict[label]) < inst_thresh:
                                count_dict['thresh']['yes'] += 1
                                count_dict['platform'][platform] += 1
                                acf = md_dict['abs_cal_factors']
                                acf_list = []
                                band_set = set()
                                for x in acf:
                                    if x['band'] == 'red':
                                        acf_list.append(x['value'])
                                        band_set.add('red')
                                for x in acf:
                                    if x['band'] == 'green':
                                        acf_list.append(x['value'])
                                        band_set.add('green')
                                for x in acf:
                                    if x['band'] == 'blue':
                                        acf_list.append(x['value'])
                                        band_set.add('blue')
                                if len(acf_list) == 3:
                                    cal_dict[platform].add(tuple(acf_list))
                                if 'red' in band_set and 'green' in band_set and 'blue' in band_set:
                                    count_dict['bands']['yes'] += 1
                                    # Open the image file and pull out the band count
                                    tiff = tifffile.imread(old_img_path)
                                    if platform in BAND_DICT[tiff.shape[-1]]:
                                        count_dict['band_count']['yes'] += 1
                                        item_dict[label].append((old_img_path, bbox, label))
                                    else:
                                        count_dict['band_count']['no'] += 1
                                else:
                                    count_dict['bands']['no'] += 1
                            else:
                                count_dict['thresh']['no'] += 1
                        else:
                            count_dict['gsd']['no'] += 1
                    else:
                        count_dict['platform']['none'] += 1
                else:
                    count_dict['cal']['no'] += 1

            # Update the progresbar
            bar.update(batch_idx)

    # Print counts of each condition
    print('List of condition counts:')
    pprint(count_dict)

    # Print platforms seen
    print('List of platforms in data:')
    pprint(platform_set)

    # List of abs_cal_factors by platform
    print('List of abs_cal_factors by platform:')
    pprint(cal_dict)

    # Only classes with exactly inst_thresh instances
    thresh_dict = {}
    for k, v in item_dict.items():
        if len(v) == inst_thresh:
            thresh_dict[k] = v
    item_dict = thresh_dict

    # Split item dict into k-folds
    kfold_dict = collections.defaultdict(dict)
    np.random.seed(0)
    for k, v in item_dict.items():
        arr = np.array(v, dtype=object)
        np.random.shuffle(arr)
        folds = np.split(arr, num_folds)
        for i in range(num_folds):
            kfold_dict[i][k] = folds[i].tolist()

    # Create mapping between classes and indeces
    cat_list = list(item_dict.keys())
    fmow = FMOW(bg_flag=False, cat_list=cat_list)
    exp_dict = {}
    exp_dict['metadata'] = {
        'label_map': fmow.LABEL_MAP,
        'idx_map': fmow.IDX_MAP
    }

    # Store kfold_dict in exp_dict
    exp_dict['folds'] = kfold_dict

    # Save paths and labels in json format
    item_file = '{}_items_{}-fold.json'.format(mode, num_folds)
    item_path = os.path.join(label_dir, item_file)
    print('Item path: {}'.format(item_path))
    print('Num classes: {}'.format(len(item_dict)))
    print('Class list:', item_dict.keys())
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    with open(item_path, 'w') as fp:
        json.dump(exp_dict, fp)

if __name__=='__main__':
    partition_dir = os.path.join(args.partition_dir, args.modality)
    for mode in ['val', 'train']:
        img_dir = os.path.join(args.img_dir, args.proc_op, mode)
        extract(args.dataset_dir, img_dir, partition_dir, mode, NUM_IMG[mode], inst_thresh=NUM_INST[mode], max_gsd=args.max_gsd)
