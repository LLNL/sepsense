#!/usr/bin/env python3

import torch
import torch.utils.data
import json
import collections
import itertools
import numpy as np
import os
import sys
from PIL import Image
import tifffile
from pprint import pprint
from imsim import sensor_model_degrade_dg as sd    
from transform import SensorError
import shutil

def get_full_metadata(partition_dir, metadata_dir, num_folds, num_classes, origin='train', force_reload=False):
    # Path to find metadata
    metadata_path = os.path.join(metadata_dir, '{}_items_{}_{}-fold_full_md.json'.format(origin, num_classes, num_folds))
    # If data has not yet been loaded
    if not os.path.isfile(metadata_path) or force_reload:
        print('==> Computing metadata...')
        # Load image paths and labels
        item_path = os.path.join(partition_dir, '{}_items_{}_{}-fold.json'.format(origin, num_classes, num_folds))
        with open(item_path, 'r') as fp:
            item_dict = json.load(fp)['folds']
        # Get class-average of metadata for each fold
        md_dict = collections.defaultdict(lambda : collections.defaultdict(list))
        for fold in item_dict:
            for cls in item_dict[fold]:
                img_path_list = list(zip(*item_dict[fold][cls]))[0]
                md_path_list = [f.replace('.tif', '.json') for f in img_path_list]
                for md_path in md_path_list:
                    with open(md_path, 'r') as fp:
                        d = json.load(fp)
                        gsd = d['gsd']
                        md_dict['gsd'][cls].append(gsd)
                        md_dict['img_width'][cls].append(d['img_width'])
                        md_dict['img_height'][cls].append(d['img_height'])
                        md_dict['img_area'][cls].append(d['img_width']*d['img_height'])
                        _, _, box_width, box_height = d['bounding_boxes'][0]['box']
                        md_dict['box_width'][cls].append(box_width)
                        md_dict['box_height'][cls].append(box_height)
                        md_dict['box_area'][cls].append(box_width*box_height)
                        md_dict['width_dist'][cls].append(box_width*gsd)
                        md_dict['height_dist'][cls].append(box_height*gsd)
                        md_dict['area_dist'][cls].append(box_width*box_height*gsd**2)
                        # Other metadata
                        md_dict['sensor_platform_name'][cls].append(d['sensor_platform_name'])
                        md_dict['timestamp'][cls].append(d['timestamp'])
        print('==> Saving metadata...')
        print(metadata_path)
        with open(metadata_path, 'w') as fp:
            json.dump(md_dict, fp)
    else:
        print('==> Loading metadata...')
        with open(metadata_path, 'r') as fp:
            md_dict = json.load(fp)

    return md_dict
        
def get_metadata(partition_dir, metadata_dir, num_folds, num_classes, origin='train'):
    # Path to find metadata
    metadata_path = os.path.join(metadata_dir, '{}_items_{}_{}-fold_md.json'.format(origin, num_classes, num_folds))
    # If data has not yet been loaded
    if not os.path.isfile(metadata_path):
        # Load image paths and labels
        item_path = os.path.join(partition_dir, '{}_items_{}_{}-fold.json'.format(origin, num_classes, num_folds))
        with open(item_path, 'r') as fp:
            item_dict = json.load(fp)['folds']
        # Get class-average of metadata for each fold
        val_md_dict = collections.defaultdict(lambda : collections.defaultdict(dict))
        for fold in item_dict:
            for cls in item_dict[fold]:
                img_path_list = list(zip(*item_dict[fold][cls]))[0]
                md_path_list = [f.replace('.tif', '.json') for f in img_path_list]
                stat_dict = collections.defaultdict(list)
                for md_path in md_path_list:
                    with open(md_path, 'r') as fp:
                        d = json.load(fp)
                        gsd = d['gsd']
                        stat_dict['gsd'].append(gsd)
                        stat_dict['img_width'].append(d['img_width'])
                        stat_dict['img_height'].append(d['img_height'])
                        stat_dict['img_area'].append(d['img_width']*d['img_height'])
                        _, _, box_width, box_height = d['bounding_boxes'][0]['box']
                        stat_dict['box_width'].append(box_width)
                        stat_dict['box_height'].append(box_height)
                        stat_dict['box_area'].append(box_width*box_height)
                        stat_dict['width_dist'].append(box_width*gsd)
                        stat_dict['height_dist'].append(box_height*gsd)
                        stat_dict['area_dist'].append(box_width*box_height*gsd**2)
                for stat in stat_dict:
                    val_md_dict[fold][cls][stat] = np.mean(stat_dict[stat])
        print('==> Saving metadata...')
        print(metadata_path)
        with open(metadata_path, 'w') as fp:
            json.dump(val_md_dict, fp)
    else:
        print('==> Loading metadata...')
        with open(metadata_path, 'r') as fp:
            val_md_dict = json.load(fp)

    return val_md_dict
        

class Partition(torch.utils.data.Dataset):

    def __init__(self, partition_dir, num_folds, fold_index, origin='train', mode='train', transform=None, num_inst=100, preload=False, preload_dir=None, param_name=None, param_val=None, path_flag=False, cross_val='1x10', modality='rgb', num_class=None):
        # Modality
        self.modality = modality
        # Flags
        self.path_flag = path_flag
        # Load image paths and labels
        item_path = os.path.join(partition_dir, '{}_items_{}_{}-fold.json'.format(origin, num_class, cross_val))
        with open(item_path, 'r') as fp:
            exp_dict = json.load(fp)
        item_dict = exp_dict['folds']
        md_dict = exp_dict['metadata']
        self.label_map = md_dict['label_map']
        self.idx_map = md_dict['idx_map']
        # Take fold index if val mode, else combine all indeces != fold index for train mode 
        if cross_val == '1x10':
            if mode == 'val':
                self.item_dict = item_dict[str(fold_index)]
            elif mode == 'train':
                self.item_dict = collections.defaultdict(list)
                for i in range(10):
                    if i != fold_index:
                        for k, v in item_dict[str(i)].items():
                            self.item_dict[k].extend(v)
            elif mode == 'full':
                self.item_dict = collections.defaultdict(list)
                for i in range(10):
                    for k, v in item_dict[str(i)].items():
                        self.item_dict[k].extend(v)
            else:
                raise Exception('Invalid mode: {}'.format(mode))
        elif cross_val == '5x2':
            if mode == 'val':
                self.item_dict = item_dict[str(fold_index)]['val']
            elif mode == 'train':
                self.item_dict = item_dict[str(fold_index)]['train']
            elif mode == 'full':
                raise NotImplementedError
            else:
                raise Exception('Invalid mode: {}'.format(mode))
        else:
            raise Exception('Invalid cross_val mode: {}'.format(cross_val))
        # Determine how many images of the partition to use
        self.item_list = []
        for key in sorted(self.item_dict):
            items = self.item_dict[key][:num_inst]
            self.item_list.extend(items)
        # Store transform
        self.transform = transform
        # Store preload
        self.preload = preload
        self.preload_dir = preload_dir
        self.param_name = param_name
        self.param_val = param_val
        if self.param_val is None:
            self.param_val = 0
        # Make dir for the parameter
        if self.preload:
            param_dir = os.path.join(self.preload_dir, '{}_{:.3f}'.format(self.param_name, self.param_val))
            if not os.path.isdir(param_dir):
                try:
                    os.makedirs(param_dir)
                # Can occur with multithreading
                except FileExistsError:
                    pass

    def __getitem__(self, index):
        # Get image path, load with PIL, and convert to Tensor
        path, bbox, label = self.item_list[index]
        # Get label name and convert to index
        label_idx = self.label_map[label]
        label_tsr = torch.LongTensor([label_idx])
        # Pre-load image, or load original and transform
        if self.preload:
            file_name = os.path.basename(path)
            base_name, file_ext = os.path.splitext(file_name)
            preload_path = os.path.join(self.preload_dir, 
                '{}_{:.3f}'.format(self.param_name, self.param_val),
                '{}.t7'.format(base_name))
            if not os.path.isfile(preload_path):
                dgim = sd.readDG_tif(path, ms=True)
                try:
                    image_tsr = self.transform((dgim, bbox, path))
                except SensorError as e:
                    sensor_params = e.data
                    tb = e.traceback
                    print('Failed on: {}'.format(path))
                    print('Image size:', dgim.im.shape)
                    print('BBox:', bbox)
                    imsim_dir = './imsim'
                    rel_fail_dir = 'fail'
                    fail_dir = os.path.join(imsim_dir, rel_fail_dir)
                    img_dir = os.path.join(fail_dir, 'images')
                    new_path = os.path.join(img_dir, os.path.basename(path))
                    if not os.path.isdir(img_dir):
                        os.makedirs(img_dir)
                    print(path, new_path)
                    shutil.copy(path, new_path)
                    shutil.copy(path.replace('.tif', '.json'), new_path.replace('.tif', '.json'))
                    fail_path = os.path.join(fail_dir, '{}.json'.format(os.path.splitext(os.path.basename(path))[0]))
                    sensor_params = dict(sensor_params)
                    for k,v in sensor_params.items():
                        if type(v) == np.ndarray:
                            sensor_params[k] = v.tolist()
                    fail_dict = {
                        'params': dict(sensor_params),
                        'bbox': bbox,
                        'traceback': tb,
                    }
                    with open(fail_path, 'w') as fp:
                        json.dump(fail_dict, fp)
                    print(tb)
                    print('Failure package saved to: {}'.format(fail_path))
                    raise Exception('Failure complete.')
                print(preload_path, image_tsr.size())
                torch.save(image_tsr, preload_path)
            else:
                try:
                    image_tsr = torch.load(preload_path)
                except RuntimeError:
                    print(' RuntimeError for tensor path: {}'.format(preload_path))
                    raise
                except EOFError:
                    print(' EOFError for tensor path: {}'.format(preload_path))
                    raise
                except PermissionError:
                    raise Exception('Permission error on path: {}'.format(preload_path))
        else:
            dgim = sd.readDG_tif(path, ms=True)
            try:
                image_tsr = self.transform((dgim, bbox, path))
            except SensorError as e:
                sensor_params = e.data
                tb = e.traceback
                print('Failed on: {}'.format(path))
                print('Image size:', dgim.im.shape)
                print('BBox:', bbox)
                imsim_dir = './imsim'
                rel_fail_dir = 'fail'
                fail_dir = os.path.join(imsim_dir, rel_fail_dir)
                img_dir = os.path.join(fail_dir, 'images')
                new_path = os.path.join(img_dir, os.path.basename(path))
                if not os.path.isdir(img_dir):
                    os.makedirs(img_dir)
                print(path, new_path)
                shutil.copy(path, new_path)
                shutil.copy(path.replace('.tif', '.json'), new_path.replace('.tif', '.json'))
                fail_path = os.path.join(fail_dir, '{}.json'.format(os.path.splitext(os.path.basename(path))[0]))
                sensor_params = dict(sensor_params)
                for k,v in sensor_params.items():
                    if type(v) == np.ndarray:
                        sensor_params[k] = v.tolist()
                fail_dict = {
                    'params': dict(sensor_params),
                    'bbox': bbox,
                    'traceback': tb,
                }
                with open(fail_path, 'w') as fp:
                    json.dump(fail_dict, fp)
                print(tb)
                print('Failure package saved to: {}'.format(fail_path))
                raise Exception('Failure complete.')

        # If image is nan
        if (image_tsr != image_tsr).sum() > 0:
            raise Exception('nan path: {}'.format(preload_path))

        # Return image tensor and corresponding label
        if self.path_flag:
            return image_tsr, label_tsr, index
        else:
            return image_tsr, label_tsr

    def get(self, class_idx, inst_idx):
        inst_dict = self.item_dict[class_idx]
        path, bbox, label = inst_dict[inst_idx]
        if self.preload:
            file_name = os.path.basename(path)
            base_name, file_ext = os.path.splitext(file_name)
            preload_path = os.path.join(self.preload_dir, 
                '{}_{:.3f}'.format(self.param_name, self.param_val),
                '{}.jpg'.format(base_name))
            if not os.path.isfile(preload_path):
                image = Image.open(path)
                image = self.transform((image, bbox, path))
                param_dir = os.path.join(self.preload_dir, '{}_{:.3f}'.format(self.param_name, self.param_val))
                if not os.path.isdir(param_dir):
                    os.makedirs(param_dir)
                image.save(preload_path)
            else:
                image = Image.open(preload_path)
        else:
            if self.modality == 'rgb':
                image = sd.readDG_tif(path, ms=False)
            elif self.modality == 'ms3':
                image = sd.readDG_tif(path, ms=True)
            image = self.transform((image, bbox, path))
        return image

    def get_viz2(self, index, transform):
        # Get image path, load with PIL, and convert to Tensor
        path, bbox, label = self.item_list[index]
        # Pre-load image, or load original and transform
        #img = Image.open(path)
        img = sd.readDG_tif(path, ms=True)
        proc_img = transform((img, bbox, path))
        # Return image tensor and corresponding label
        return proc_img, label


    def __len__(self):
        return len(self.item_list)

class TripletPartition(Partition):

    #def __init__(self, partition_path, transform=None):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # now load the picked numpy arrays
        self.num_valid_classes = len(self.item_dict)

        self.idx_item_dict = {}
        self.item_idx_dict = {}

        self.cls_idx_dict = {}

        idx = 0
        for cls_idx, c in enumerate(self.item_dict):
            self.cls_idx_dict[c] = cls_idx
            for i in range(len(self.item_dict[c])):
                self.idx_item_dict[idx] = (c, i)
                self.item_idx_dict[(c, i)] = idx
                idx += 1

def class_cycle(queues):
    while True:
        classes = list(queues.keys())
        np.random.shuffle(classes)
        for c in classes:
            yield c

def get_inst(q, num_instances):
    for _ in range(num_instances):
        try:
            yield q.pop()
        except:
            break

def grouper(iterable, n):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

class TripletFullBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_class, num_inst):
        self.dataset = dataset
        self.num_class = num_class
        self.num_inst = num_inst

        # Num classes used per batch can't be greater than num in dataset
        if num_class > self.dataset.num_valid_classes:
            print("Num_classes reduced to {} from {}.".format(num_class, self.dataset.num_valid_classes))
            self.num_class = self.dataset.num_valid_classes
        else:
            self.num_class = num_class

        # If num batches -1, use the number of samples in the dataset (with replacement)
        num_items = self.num_class * self.num_inst
        tot_items = sum([len(v) for v in self.dataset.item_dict.values()])
        self.num_batches = tot_items // num_items
        print('Num batches (approximate): {}'.format(self.num_batches))

    def __iter__(self):
        train_queues = {}
        for k,v in self.dataset.item_dict.items():
            idx_arr = np.arange(len(v))
            np.random.shuffle(idx_arr)
            train_queues[k] = collections.deque(idx_arr)

        class_iterator = grouper(class_cycle(train_queues), self.num_class)

        # Train w/ semi-hard triplets
        while len(train_queues.keys()) >= 2:
            idx_list = []
            class_list = next(class_iterator)
            for c in class_list:
                try:
                    inst_list = list(get_inst(train_queues[c], self.num_inst))
                except KeyError:
                    continue
                if len(train_queues[c]) == 0:
                    del train_queues[c]
                for i in inst_list:
                    idx = self.dataset.item_idx_dict[(c, i)]
                    idx_list.append(idx)
            yield idx_list

    def __len__(self):
        return self.num_batches


if __name__=='__main__':
    """
    import os
    import time
    from pipeline import VizPipeline
    import pilutils
    from params import parse_params

    template_path = './params.yaml'
    profile_param_path = './profileviz.yaml'
    profile_params = parse_params(template_path, profile_param_path)

    category = 'oil_or_gas_facility'
    inst_idx = 5

    focal_length = 0.5
    pipeline = VizPipeline(profile_params, 'train', 0, param_name='focal_length', param_val=focal_length)
    
    t1 = time.time()
    img = pipeline.get(category, inst_idx)
    t2 = time.time()
    print('Load time: {:3f}'.format(t2 - t1))
    """
    val_md_dict = get_metadata('./partition/rgb', './metadata', 10, 'train')
    pprint(val_md_dict)
