# Global imports
import torch
import numpy as np
import torchvision
import progressbar
import collections
import itertools
import json
import os
import sys
import time
import scipy.misc
from PIL import Image

# Local imports
from dataset import Partition, TripletPartition, TripletFullBatchSampler
from transform import GaussianBlur, TargetCrop, TargetCrop2, TargetExtract, TargetResize, DrawBBox, DeTuple, Satellite, HOG, Normalize, Denormalize, ToReflectance, ToTensor, NormalizeImage, MinMax, CenterCrop
from models import Pass, Reshape, Reshape2
import models
import stats
from params import parse_params
import metrics
import brisque.brisque as brisque

class Pipeline(object):
    def __init__(self, params, mode, fold_index, param_name=None, param_val=None):
        # Save params object
        self.params = params
        # Save off some params
        self.mode = mode
        # Save param name and value to be manually altered
        self.param_name = param_name
        self.param_val = param_val

        # Save k-fold index to use
        self.fold_index = fold_index

        # Is CUDA available
        self.use_cuda = torch.cuda.is_available()

        if param_name is None:
            pc = 0
        else:
            pc = 25

        # Setup transform
        self.transform_list = []
        # Pick pre-processing mode
        if params.data.proc_mode == 'crop':
            self.transform_list.append(TargetCrop(params.data.resolution, params.data.modality, 'edge'))
        elif params.data.proc_mode == 'center_crop':
            self.transform_list.append(CenterCrop(params.data.resolution, params.data.modality, 'edge'))
        elif params.data.proc_mode == 'resize':
            self.transform_list.append(TargetExtract(params.data.modality, pc=pc))
        # Pick degrade mode
        if params.experiment.train_mode == 'orig' and mode == 'train' or param_name == None:
            pass
        else:
            if params.experiment.degrade_mode == 'blur':
                self.transform_list.append(GaussianBlur(param_val))
            elif params.experiment.degrade_mode == 'sat':
                sensor_params = params.sensor._asdict()
                sensor_params[param_name] = param_val
                sat = Satellite(sensor_params, params.data.modality, params.data.interp_type, params.data.interp_range, params.data.out_units)
                self.q = sat.sat.Q
                self.transform_list.append(sat)
            elif params.experiment.degrade_mode == 'none':
                pass

        if param_name is None and params.data.modality == 'ms3':
            self.transform_list.append(ToReflectance())

        if params.data.proc_mode == 'resize':
            self.transform_list.append(TargetResize(params.data.resolution, params.data.modality, params.data.interp_type, params.data.interp_range))
        elif params.data.proc_mode == 'crop':
            self.transform_list.append(TargetCrop2())
            self.transform_list.append(TargetCrop(params.data.resolution, params.data.modality, 'constant'))
            if params.data.modality == 'rgb':
                self.transform_list.append(lambda x: scipy.misc.imresize(x[0].im, (params.data.resolution, params.data.resolution)))
                
    def get_save_name(self, epoch_num):
        # Check for experiment type 3, in which test fold index is used
        if hasattr(self, 'train_fold_index') and self.train_fold_index is not None:
            fold_index = '{}_{}'.format(self.train_fold_index, self.fold_index)
        else:
            fold_index = self.fold_index

        # Try to create log for varying param first
        try:
            if self.params.experiment.train_mode == 'orig':
                raise TypeError
            save_name = '{}_{}_{}_{}_{}_{}_{}_{}_{:.3f}'.format(
                self.params.experiment.origin, 
                self.params.experiment.train_mode, 
                self.params.experiment.num_folds, fold_index, 
                self.params.data.num_class, 
                self.params.data.num_inst, 
                epoch_num, 
                self.param_name, self.param_val)
        except TypeError:
            save_name = '{}_{}_{}_{}_{}_{}_{}'.format(
                self.params.experiment.origin, 
                self.params.experiment.train_mode, 
                self.params.experiment.num_folds, fold_index, 
                self.params.data.num_class, 
                self.params.data.num_inst, 
                epoch_num)

        return save_name

    def get_log_name(self, epoch_num):
        # Check for experiment type 3, in which test fold index is used
        if hasattr(self, 'train_fold_index') and self.train_fold_index is not None:
            fold_index = '{}_{}'.format(self.train_fold_index, self.fold_index)
        else:
            fold_index = self.fold_index

        # Try to create log for varying param first
        try:
            if self.params.experiment.train_mode == 'orig':
                raise TypeError
            save_name = '{}_{}_{}_{}_{}_{}_{}_{}_{:.3f}'.format(
                self.params.experiment.origin, 
                self.params.experiment.train_mode, 
                self.params.experiment.num_folds, fold_index, 
                self.params.data.num_class, 
                self.params.data.num_inst, 
                epoch_num, 
                self.param_name, self.param_val)
        except TypeError:
            save_name = '{}_{}_{}_{}_{}_{}_{}'.format(
                self.params.experiment.origin, 
                self.params.experiment.train_mode, 
                self.params.experiment.num_folds, fold_index, 
                self.params.data.num_class, 
                self.params.data.num_inst, 
                epoch_num)

        return save_name

    def log(self, stats_dict):
        # Create save file name
        epoch_num = stats_dict['epoch_num']
        save_name = self.get_save_name(epoch_num)
        log_name = self.get_log_name(epoch_num)
        log_file = '{}.json'.format(self.mode)
        log_dir = os.path.join(self.params.logging.log_dir, save_name)

        # Create save dir if it does not exist
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        # Load current log dict
        if os.path.exists(log_path):
            with open(log_path, 'r') as fp:
                log_dict = json.load(fp)
        else:
            log_dict = {
                'metadata': {
                    'label_map': self.label_map,
                    'idx_map': self.idx_map
                },
                'entries': {}
            }

        # Append information to log dict
        log_dict['entries'][log_name] = stats_dict

        # Save log dict back to file path
        while True:
            num_attempt = 0
            try:
                with open(log_path, 'w') as fp:
                    json.dump(log_dict, fp)
                    os.fsync(fp.fileno())
            except OSError:
                num_attempt += 1
                print('Attempt {}: Failed to write file to disk: {}'.format(num_attempt, log_path))
                time.sleep(10)
            else:
                break

class NormPipeline(Pipeline):
    def __init__(self, params, mode, fold_index, param_name=None, param_val=None):
        super().__init__(params, mode, fold_index, param_name=param_name, param_val=param_val)

        if params.model.method == 'cnn':
            # Determine tensor transorm method
            if params.data.tensor_range == 'default':
                to_tensor = torchvision.transforms.ToTensor()
            elif params.data.tensor_range == 'noscale':
                to_tensor = ToTensor() 
            else:
                raise Exception('Invalid tensor_range: {}'.format(params.data.tensor_range))
            # Add transforms to list
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                to_tensor
            ])
        elif params.model.method == 'hog':
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                HOG()
            ])
        transform = torchvision.transforms.Compose(self.transform_list)

        # Instantiate Dataset
        train_set = Partition(params.data.partition_dir, params.experiment.num_folds, fold_index, 
            origin=params.experiment.origin, mode=self.mode, transform=transform, num_inst=params.data.num_inst,
            preload=params.data.preload, preload_dir=params.data.preload_dir,
            param_name=param_name, param_val=param_val, cross_val=params.experiment.cross_val,
            modality=params.data.modality, num_class=params.data.num_class)
        self.label_map = train_set.label_map
        self.idx_map = train_set.idx_map
        self.num_class = len(train_set.item_dict.keys())

        # Instantiate Dataloaders
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.learning.batch_size, shuffle=params.learning.shuffle, num_workers=params.learning.num_workers, pin_memory=params.learning.pin_memory)

    def norm(self):
        print('Normalizing...')
        # Extract features from images
        feat_list = []
        with progressbar.ProgressBar(max_value=len(self.train_loader)) as bar:
            for batch_idx, (image_tsr, _) in enumerate(self.train_loader):
                feat_list.append(image_tsr)
                # Update progressbar
                bar.update(batch_idx)
                # Compute mean, std of features
                feat_arr = torch.cat(feat_list).numpy()
                feat_mean = np.mean(feat_arr, axis=(0, 1, 2))
                feat_std = np.std(feat_arr, axis=(0, 1, 2))
                feat_min = np.min(feat_arr, axis=(0, 1, 2))
                feat_max = np.max(feat_arr, axis=(0, 1, 2))
                print('Stats:', feat_mean, feat_std, feat_min, feat_max)

        # Compute mean, std of features
        feat_tsr = torch.cat(feat_list)
        feat_mean = torch.mean(feat_tsr, dim=0)
        feat_std = torch.std(feat_tsr, dim=0)

        return feat_mean, feat_std


class VizPipeline(Pipeline):
    def __init__(self, params, mode, fold_index, param_name=None, param_val=None):
        super().__init__(params, mode, fold_index, param_name=param_name, param_val=param_val)

        # Draw the target bbox on the image if needed
        #if params.display.bbox_flag:
        #    self.transform_list.append(DrawBBox(3))

        # Fail-safe operation to make sure output is just image, not (image, bbox)
        self.transform_list.append(DeTuple())

        self.transform_list.append(lambda x: x if type(x) == np.ndarray else x.im)

        # Convert output to PIL image for display
        if params.data.modality == 'ms3':
            #self.transform_list.append(lambda x: x-x.min())
            #self.transform_list.append(lambda x: x/x.max())
            #self.transform_list.append(lambda x: (x*255).astype(np.uint8))
            pass
        elif params.data.modality == 'rgb':
            #self.transform_list.append(lambda x: x.astype(np.uint8))
            pass
        #self.transform_list.append(torchvision.transforms.ToPILImage())

        transform = torchvision.transforms.Compose(self.transform_list)

        # Instantiate Dataset
        self.data_set = Partition(params.data.partition_dir, params.experiment.num_folds, fold_index, 
            origin=params.experiment.origin, mode=mode, transform=transform, num_inst=params.data.num_inst, 
            preload=params.data.preload, preload_dir=params.data.preload_dir,
            param_name=param_name, param_val=param_val, cross_val=params.experiment.cross_val,
            modality=params.data.modality, num_class=params.data.num_class)
        self.label_map = self.data_set.label_map
        self.idx_map = self.data_set.idx_map


    def get(self, class_idx, inst_idx):
        return self.data_set.get(class_idx, inst_idx)

class VizGroup:
    def __init__(self, params, mode, fold_index, param_name_list, param_val_list):
        self.viz_pipeline_list = []
        self.img_list = []
        for param_name, param_val in zip(param_name_list, param_val_list):
            viz_pipeline = VizPipeline(params, mode, fold_index, 
                param_name=param_name, param_val=param_val)
            self.viz_pipeline_list.append(viz_pipeline)
            
    def __call__(self, category, inst_idx):
        arr_list = []
        min_list, max_list = [], []
        for viz_pipeline in self.viz_pipeline_list:
            arr = viz_pipeline.get(category, inst_idx)         
            arr_list.append(arr)
            min_list.append(arr.min())
            max_list.append((arr-arr.min()).max())
        min_mean = np.mean(min_list)
        max_mean = np.mean(max_list)
        img_list = []
        for arr in arr_list:
            norm_arr = (arr - min_mean) / max_mean
            clip_arr = np.clip(norm_arr, 0, 1)
            uint_arr = (clip_arr * 255).astype(np.uint8)
            img = Image.fromarray(uint_arr)
            img_list.append(img)
        return img_list

class TrainPipeline(Pipeline):
    def __init__(self, params, mode, fold_index, param_name=None, param_val=None, feat_mean=None, feat_std=None):
        if params.experiment.train_mode == 'degrade':
            super().__init__(params, mode, fold_index, param_name=param_name, param_val=param_val)
        else:
            super().__init__(params, mode, fold_index)

        # Number of epochs trained for is hidden within this class
        self.epoch_num = 0

        if params.model.method == 'cnn':
            # Determine tensor transorm method
            if params.data.tensor_range == 'default':
                to_tensor = torchvision.transforms.ToTensor()
            elif params.data.tensor_range == 'noscale':
                to_tensor = ToTensor() 
            else:
                raise Exception('Invalid tensor_range: {}'.format(params.data.tensor_range))
            # Add transforms to list
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                to_tensor,
            ])
            if params.data.normalize:
                if params.data.minmax:
                    self.transform_list.append(MinMax(params.data.minmax_channel))
                self.transform_list.append(NormalizeImage(params.data.data_mean, params.data.data_std))
        elif params.model.method == 'hog':
            self.num_feat = feat_mean.size(0)
            print('Num features: {}'.format(self.num_feat))
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                HOG(),
            ])
            if params.data.normalize:
                self.transform_list.append(Normalize(feat_mean=feat_mean, feat_std=feat_std))
        transform = torchvision.transforms.Compose(self.transform_list)

        # Setup CNN and CUDA
        if params.model.method == 'cnn':
            if params.model.cnn_arch == 'densenet161':
                self.net = torchvision.models.densenet161(pretrained=params.model.pretrained)
                self.net.classifier = Pass()
                self.classifier = torch.nn.Linear(2208, self.params.data.num_class)
            elif params.model.cnn_arch == 'densenet161_alt1':
                self.net = torchvision.models.densenet161(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Linear(2208, 128)
                self.classifier = torch.nn.Linear(128, self.params.data.num_class)
            elif params.model.cnn_arch == 'resnet18':
                self.net = torchvision.models.resnet18(pretrained=params.model.pretrained)
                self.net.fc = Pass()
                self.classifier = torch.nn.Linear(512, self.params.data.num_class)
            elif params.model.cnn_arch == 'resnet18_alt1':
                self.net = torchvision.models.resnet18(pretrained=params.model.pretrained)
                self.net.fc = torch.nn.Linear(512, 128)
                self.classifier = torch.nn.Linear(128, self.params.data.num_class)
            elif params.model.cnn_arch == 'resnet152':
                self.net = torchvision.models.resnet152(pretrained=params.model.pretrained)
                self.net.fc = Pass()
                self.classifier = torch.nn.Linear(2048, self.params.data.num_class)
            elif params.model.cnn_arch == 'resnet152_alt1':
                self.net = torchvision.models.resnet152(pretrained=params.model.pretrained)
                self.net.fc = torch.nn.Linear(2048, 128)
                self.classifier = torch.nn.Linear(128, self.params.data.num_class)
            elif params.model.cnn_arch == 'squeezenet1_1_alt':
                self.net = torchvision.models.squeezenet1_1(pretrained=params.model.pretrained)
                self.net.num_classes = 128
                self.net.classifier = torch.nn.Sequential(
                    Reshape2(),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Conv2d(512, 128, kernel_size=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AvgPool2d(13, stride=1)
                )
                self.classifier = torch.nn.Linear(128, params.data.num_class)
            elif params.model.cnn_arch == 'vgg11':
                self.net = torchvision.models.vgg11_bn(pretrained=params.model.pretrained)
                self.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:-1], 
                        torch.nn.Linear(4096, self.params.data.num_class)
                )
                self.net.classifier = Pass()
                print('Loaded VGG11!')
            elif params.model.cnn_arch == 'vgg16':
                self.net = torchvision.models.vgg16_bn(pretrained=params.model.pretrained)
                self.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:-1], 
                        torch.nn.Linear(4096, self.params.data.num_class)
                )
                self.net.classifier = Pass()
                print('Loaded VGG16!')
            elif params.model.cnn_arch == 'vgg16_alt':
                self.net = torchvision.models.vgg16_bn(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Sequential(*list(self.net.classifier.modules())[1:-1])
                self.classifier = torch.nn.Linear(4096, self.params.data.num_class)
                print('Loaded VGG16 alt!')
            elif params.model.cnn_arch == 'vgg16_alt2':
                self.net = torchvision.models.vgg16_bn(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:-1],
                    torch.nn.Linear(4096, 128)
                )
                self.classifier = torch.nn.Linear(128, self.params.data.num_class)
                print('Loaded VGG16 alt2!')
            elif params.model.cnn_arch == 'alexnet':
                self.net = torchvision.models.alexnet(pretrained=params.model.pretrained)
                self.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:-1], 
                        torch.nn.Linear(4096, self.params.data.num_class)
                )
                self.net.classifier = Pass()
                print('Loaded Alexnet!')
            elif params.model.cnn_arch == 'alexnet_alt1':
                self.net = torchvision.models.alexnet(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Sequential(*list(self.net.classifier.modules())[1:-1])
                self.classifier = torch.nn.Linear(4096, self.params.data.num_class)
                print('Loaded Alexnet alt1!')
            elif params.model.cnn_arch == 'alexnet_alt2':
                self.net = torchvision.models.alexnet(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:-1],
                    torch.nn.Linear(4096, 128)
                )
                self.classifier = torch.nn.Linear(128, self.params.data.num_class)
                print('Loaded Alexnet alt2!')
            elif params.model.cnn_arch == 'alexnet_alt3':
                self.net = torchvision.models.alexnet(pretrained=params.model.pretrained)
                self.net.classifier = torch.nn.Sequential(
                    *list(self.net.classifier.modules())[1:],
                )
                self.classifier = torch.nn.Linear(1000, self.params.data.num_class)
                print('Loaded Alexnet alt3!')
            else:
                raise Exception('Unsupported cnn_arch: {}'.format(params.mode.cnn_arch))
        elif params.model.method == 'hog':
            self.net = Pass()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.num_feat, self.num_feat),
                torch.nn.ReLU(),
                torch.nn.Linear(self.num_feat, self.data.params.num_class),
            )

        # Convert net to GPU
        if self.use_cuda:
            self.net.cuda()
            self.classifier.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = True

        # Determine type of dataset needed
        if params.problem.objective == 'classifier':
            partition_class = Partition
        elif params.problem.objective == 'triplet':
            partition_class = TripletPartition
        else:
            raise Exception('Unsupported objective: {}'.format(params.problem.objective))

        # Instantiate Dataset
        train_set = partition_class(params.data.partition_dir, params.experiment.num_folds, fold_index, 
            origin=params.experiment.origin, mode=self.mode, transform=transform, num_inst=params.data.num_inst,
            preload=params.data.preload, preload_dir=params.data.preload_dir,
            param_name=param_name, param_val=param_val, cross_val=params.experiment.cross_val,
            modality=params.data.modality, num_class=params.data.num_class)

        # Instantiate Dataloaders and Criterions
        if params.problem.objective == 'classifier':
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.learning.batch_size, shuffle=params.learning.shuffle, num_workers=params.learning.num_workers, pin_memory=params.learning.pin_memory)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.classifier.parameters()), lr=params.learning.learning_rate)
            self.train = self._train_classifier
        elif params.problem.objective == 'triplet':
            triplet_sampler = TripletFullBatchSampler(train_set, params.problem.num_classes, params.problem.num_instances)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=triplet_sampler, num_workers=params.learning.num_workers, pin_memory=params.learning.pin_memory)

            self.criterion = torch.nn.TripletMarginLoss(margin=params.problem.margin)
            self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=params.learning.learning_rate)
            self.train = self._train_triplet
        else:
            raise Exception('Unsupported objective: {}'.format(params.problem.objective))

        # Store some other info from the train set
        self.label_map = train_set.label_map
        self.idx_map = train_set.idx_map
        self.num_class = len(train_set.item_dict.keys())

    def save(self):
        # Create state
        train_state = {
            'net': self.net.module if self.use_cuda else self.net,
            'classifier': self.classifier.module if self.use_cuda else self.classifier
        }

        # Create save file name
        save_name = self.get_save_name(self.epoch_num)
        save_file = '{}.t7'.format(save_name)

        # Create save dir if it does not exist
        if not os.path.isdir(self.params.experiment.save_dir):
            os.makedirs(self.params.experiment.save_dir)
        save_path = os.path.join(self.params.experiment.save_dir, save_file)

        # Save off the train state
        while True:
            num_attempt = 0
            # Attempt to save the model
            try:
                torch.save(train_state, save_path)
            # If there is "No space left on device", sleep for 10 seconds and try again
            # this will give time for the ceph rebalancer to create needed space
            except OSError:
                num_attempt += 1
                print('Attempt {}: Failed to write file to disk: {}'.format(num_attempt, save_path))
                time.sleep(10)
            # Break out of the loop if the save if successful
            else:
                break


    def exists(self):
        # Create save file name
        save_name = self.get_save_name(self.epoch_num)
        save_file = '{}.t7'.format(save_name)

        # Create save dir if it does not exist
        if not os.path.isdir(self.params.experiment.save_dir):
            return False

        # Form path
        save_path = os.path.join(self.params.experiment.save_dir, save_file)

        # Check if path exists
        if os.path.isfile(save_path):
            return True
        else:
            return False

    def _train_classifier(self, save_flag=False):
        print('Training...')
        # Bump the epoch num before training the epoch
        self.epoch_num += 1
        # If this net already exists, no need to train it again
        if self.exists():
            print('Train trial net already exists!')
            return self.epoch_num
        # Train the network
        self.net.train()
        self.classifier.train()
        train_loss = 0.0
        correct, total = 0, 0
        feat_dict = collections.defaultdict(list)
        with progressbar.ProgressBar(max_value=len(self.train_loader), redirect_stdout=False, widgets=[
            'Train {} ['.format(self.epoch_num), progressbar.Percentage(), '] ',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
            ' [', progressbar.DynamicMessage('loss'), '] ',
            ' [', progressbar.DynamicMessage('acc'), '] ',
            ]) as bar:
            for batch_idx, (image_tsr, label_tsr) in enumerate(self.train_loader):
                # Squeeze labels
                label_tsr = label_tsr.squeeze()
                # Copy images to GPU
                if self.use_cuda:
                    image_tsr, label_tsr = image_tsr.cuda(), label_tsr.cuda()
                # Zero grads
                self.optimizer.zero_grad()
                # Create variable object
                image_var, label_var = image_tsr, label_tsr
                # Run variable through network
                feat_var = self.net(image_var)
                out_var = self.classifier(feat_var)
                # Compute loss
                loss_var = self.criterion(out_var, label_var)
                ### TEMP: measure memory used
                #print('d0={}/{}, d1={}/{}'.format(
                #    torch.cuda.memory_allocated(device=0), 
                #    torch.cuda.max_memory_allocated(device=0), 
                #    torch.cuda.memory_allocated(device=1), 
                #    torch.cuda.max_memory_allocated(device=1)
                #)) 
                # Backpropgate gradients
                loss_var.backward()
                # Step the optimizer
                self.optimizer.step()
                # Compute stats
                train_loss += loss_var.item()
                _, pred_var = torch.max(out_var, 1)
                total += label_var.size(0)
                correct += pred_var.eq(label_var).cpu().sum().item()
                train_acc = correct/total
                # Convert feats and save them off?
                feat_tsr = feat_var.cpu().detach()
                label_tsr = label_var.cpu().detach()
                for feat, label in zip(feat_tsr, label_tsr):
                    feat_dict[label.item()].append(feat)
                # Update progressbar
                bar.update(batch_idx, acc=train_acc, loss=train_loss/(batch_idx+1))

        # Train loss
        train_loss = train_loss/(batch_idx+1)
        print('train loss: {:2.3f}'.format(train_loss))

        # Train acc
        train_acc = correct/total
        print('train acc: {:2.3f}'.format(train_acc))

        # Log basic results from training
        if self.params.logging.log_flag:
            stats_dict = {
                'fold_index': self.fold_index,
                'epoch_num': self.epoch_num,
                'loss': train_loss,
                'acc': train_acc,
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                try:
                    stats_dict['q'] = self.q.tolist()
                except AttributeError:
                    pass
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)

        # Optional save net after training epoch
        if save_flag:
            self.save()

        # Return the epoch num for easy loading/testing
        return self.epoch_num

    def _train_triplet(self, save_flag=False):
        print('Training...')
        # Bump the epoch num before training the epoch
        self.epoch_num += 1
        # If this net already exists, no need to train it again
        if self.exists():
            print('Train trial net already exists!')
            return self.epoch_num
        # Train the network
        total_loss = 0.0
        with progressbar.ProgressBar(max_value=len(self.train_loader), redirect_stdout=False, widgets=[
            'Train {} ['.format(self.epoch_num), progressbar.Percentage(), '] ',
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
            ' [', progressbar.DynamicMessage('loss'), '] ',
            ]) as bar:
            # Iterate throught the dataset in batches
            for batch_idx, (img_tsr, cls_list) in enumerate(self.train_loader):

                # Compute dict of pairs and instances to create triplets
                cls_dict = collections.defaultdict(list)
                for elem_idx, cls in enumerate(cls_list.squeeze().tolist()):
                    cls_dict[cls].append(elem_idx)
                pair_dict = {}
                inst_dict = {}
                for cls in cls_dict:
                    inst_list = cls_dict[cls]
                    pair_list = itertools.combinations(inst_list, 2)
                    pair_dict[cls] = pair_list
                    inst_dict[cls] = inst_list

                # Compute list of triplet indeces
                triplet_idx_list = []
                for c1 in inst_dict.keys():
                    for i1, i2 in pair_dict[c1]:
                        for c2 in [c for c in inst_dict.keys() if c != c1]:
                            for i3 in inst_dict[c2]:
                                triplet_idx_list.append((i1, i2, i3))

                # Put triplet indeces on GPU and detach from computation graph
                triplet_idx_tsr = torch.LongTensor(list(zip(*triplet_idx_list))).cuda().detach()

                # Put image on GPU
                img_tsr = img_tsr.cuda()

                ### Start training
                # Zero gradients before putting anything through the network
                self.optimizer.zero_grad()

                # Put the images through the network
                emb_tsr = self.net(img_tsr)

                # Index embeddings with triplet indeces
                emb_a, emb_p, emb_n = [torch.index_select(emb_tsr, 0, idx_tsr) for idx_tsr in triplet_idx_tsr]

                # Compute triplet loss
                loss = self.criterion(emb_a, emb_p, emb_n)

                # Backpropagate gradients 
                loss.backward()

                # Stop the optimizer
                self.optimizer.step()

                # Store loss
                batch_loss = loss.item()
                total_loss += batch_loss

                # Update the progress bar
                bar.update(batch_idx, loss=total_loss / (batch_idx + 1))

        # Compute average loss for the epoch
        train_loss = total_loss / (batch_idx + 1)
        print('train loss: {:2.3f}'.format(train_loss))

        # Log basic results from training
        if self.params.logging.log_flag:
            stats_dict = {
                'fold_index': self.fold_index,
                'epoch_num': self.epoch_num,
                'loss': train_loss,
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                try:
                    stats_dict['q'] = self.q.tolist()
                except AttributeError:
                    pass
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)

        # Optional save net after training epoch
        if save_flag:
            self.save()

        # Return the epoch num for easy loading/testing
        return self.epoch_num

class TestPipeline(Pipeline):
    def __init__(self, params, mode, fold_index, param_name=None, param_val=None, feat_mean=None, feat_std=None, 
        train_fold_index=None):
        super().__init__(params, mode, fold_index, param_name=param_name, param_val=param_val)

        self.train_fold_index = train_fold_index

        if params.model.method == 'cnn':
            # Determine tensor transorm method
            if params.data.tensor_range == 'default':
                to_tensor = torchvision.transforms.ToTensor()
            elif params.data.tensor_range == 'noscale':
                to_tensor = ToTensor() 
            else:
                raise Exception('Invalid tensor_range: {}'.format(params.data.tensor_range))
            # Add transforms to list
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                to_tensor,
            ])
            if params.data.normalize:
                if params.data.minmax:
                    self.transform_list.append(MinMax(params.data.minmax_channel))
                self.transform_list.append(NormalizeImage(params.data.data_mean, params.data.data_std))
        elif params.model.method == 'hog':
            self.transform_list.extend([
                DeTuple(),
                lambda x: x if type(x) == np.ndarray else x.im,
                HOG(),
            ])
            if params.data.normalize:
                self.transform_list.append(Normalize(feat_mean=feat_mean, feat_std=feat_std))
        transform = torchvision.transforms.Compose(self.transform_list)

        # Instantiate Dataset
        val_set = Partition(params.data.partition_dir, params.experiment.num_folds, fold_index, 
            origin=params.experiment.origin, mode=mode, transform=transform, num_inst=params.data.num_inst,
            preload=params.data.preload, preload_dir=params.data.preload_dir,
            param_name=param_name, param_val=param_val,
            path_flag=params.display.path_flag, cross_val=params.experiment.cross_val,
            modality=params.data.modality, num_class=params.data.num_class)
        self.val_set = val_set
        self.label_map = val_set.label_map
        self.idx_map = val_set.idx_map
        self.num_class = len(val_set.item_dict.keys())

        # Instantiate Dataloaders
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=params.learning.batch_size, shuffle=params.learning.shuffle, num_workers=params.learning.num_workers, pin_memory=params.learning.pin_memory)

        # Prep criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # Determine type of dataset needed
        if params.problem.objective == 'classifier':
            partition_class = Partition
            self.test = self._test_classifier
        elif params.problem.objective == 'triplet':
            partition_class = TripletPartition
            self.test = self._test_triplet
        else:
            raise Exception('Unsupported objective: {}'.format(params.problem.objective))

    def get_path(self, epoch_num):
        # Create save file name
        save_name = self.get_save_name(epoch_num)
        save_file = '{}.t7'.format(save_name)

        # Load from save dir
        save_path = os.path.join(self.params.experiment.save_dir, save_file)
        if not os.path.isfile(save_path):
            raise Exception('Load path does not exist: {}'.format(save_path))

        return save_path

    def load(self, epoch_num):
        # Create save file name
        save_name = self.get_save_name(epoch_num)
        save_file = '{}.t7'.format(save_name)

        # Load from save dir
        save_path = os.path.join(self.params.experiment.save_dir, save_file)
        if not os.path.isfile(save_path):
            raise Exception('Load path does not exist: {}'.format(save_path))

        # Load net
        try:
            train_state = torch.load(save_path)
        except (RuntimeError, EOFError) as e:
            print('Failed to load net: {}'.format(save_path))
            raise
        except KeyError:
            print('File corrupted: {}'.format(save_path))
            raise
            
        self.net = train_state['net']
        self.classifier = train_state['classifier']

        # Use GPU
        if self.use_cuda:
            self.net.cuda()
            self.classifier.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = True

    def load_model(self, model_path):
        # Load from save dir
        if not os.path.isfile(model_path):
            raise Exception('Load path does not exist: {}'.format(model_path))

        # Load net
        try:
            train_state = torch.load(model_path)
        except (RuntimeError, EOFError) as e:
            print('Failed to load net: {}'.format(model_path))
            raise
            
        self.net = train_state['net']
        self.classifier = train_state['classifier']

        # Use GPU
        if self.use_cuda:
            self.net.cuda()
            self.classifier.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=range(torch.cuda.device_count()))
            torch.backends.cudnn.benchmark = True

    def exists(self, epoch_num):
        # Create save file name
        save_name = self.get_save_name(epoch_num)
        log_name = self.get_log_name(epoch_num)
        log_file = '{}.json'.format(self.mode)
        log_dir = os.path.join(self.params.logging.log_dir, save_name)

        # Create save dir if it does not exist
        #print('Check log dir: {}'.format(log_dir))
        if not os.path.isdir(log_dir):
            return False
        else:
            log_path = os.path.join(log_dir, log_file)

        # Load current log dict
        #print('Check path: {}'.format(log_path))
        if os.path.exists(log_path):
            with open(log_path, 'r') as fp:
                try:
                    log_dict = json.load(fp)
                except json.decoder.JSONDecodeError:
                    print('Couldnt load JSON file: {}'.format(log_path))
                    raise
                
        else:
            log_dict = {
                'metadata': {
                    'label_map': self.label_map,
                    'idx_map': self.idx_map
                },
                'entries': {}
            }

        if log_name in log_dict['entries']:
            return True
        else:
            return False

    @torch.no_grad()
    def _test_classifier(self, epoch_num, model_path=None):
        print('Testing...')
        # Check if this trial already exists
        if self.exists(epoch_num):
            print('Test trial log already exists!')
            return
        if model_path is not None:
            # Load the network using given path
            print('Loading model from path...')
            self.load_model(model_path)
        else:
            # Load the network using internal params
            self.load(epoch_num)
        # Extract features from images
        self.net.eval()
        self.classifier.eval()
        val_loss = 0.0
        correct, total = 0, 0
        topk_correct = 0
        top_k = self.params.metric.top_k
        feat_dict = collections.defaultdict(list)
        # Confusion matrix
        conf_arr = np.zeros((self.params.data.num_class, self.params.data.num_class))
        # Top-k per class accuracy array
        topk_correct_arr = np.zeros(self.params.data.num_class)
        topk_tot_arr = np.zeros(self.params.data.num_class)
        # Score list for AUC
        score_list, label_list = [], []
        with progressbar.ProgressBar(max_value=len(self.val_loader)) as bar:
            for batch_idx, (image_tsr, label_tsr) in enumerate(self.val_loader):
                # Squeeze labels
                label_tsr = label_tsr.squeeze()
                # Copy images to GPU
                if self.use_cuda:
                    image_tsr, label_tsr = image_tsr.cuda(), label_tsr.cuda()
                # Create variable object
                image_var, label_var = image_tsr, label_tsr
                # Run variable through network
                feat_var = self.net(image_var)
                out_var = self.classifier(feat_var)
                # Compute loss
                loss_var = self.criterion(out_var, label_var)
                # Compute stats
                val_loss += loss_var.item()
                _, pred_tsr = torch.max(out_var, 1)
                total += label_var.size(0)
                correct += pred_tsr.eq(label_var).cpu().sum().item()
                np.add.at(conf_arr, [label_var.cpu().numpy(), pred_tsr.cpu().numpy()], 1)
                val_acc = correct/total
                # Compute top-k accuracy
                pred_topk = out_var.cpu().numpy()
                true_topk = label_var.cpu().numpy()
                
                topk_axis_sum = (np.argpartition(-pred_topk, top_k, axis=1)[:, :top_k] == true_topk[:, np.newaxis].repeat(top_k, axis=1)).sum(axis=1)
                topk_correct += topk_axis_sum.sum()
                np.add.at(topk_tot_arr, true_topk, 1)
                np.add.at(topk_correct_arr, true_topk, topk_axis_sum)
                # Store info for AUC
                score_list.append(pred_topk)
                label_list.append(true_topk)
                # Convert feats and save them off?
                feat_tsr = feat_var.cpu().detach()
                label_tsr = label_var.cpu().detach()
                for feat, label in zip(feat_tsr, label_tsr):
                    feat_dict[label.item()].append(feat)
                # Update progressbar
                bar.update(batch_idx)

        # Val loss
        val_loss = val_loss/(batch_idx+1)
        print('val loss: {:2.3f}'.format(val_loss))

        # Val acc (top-1)
        val_acc = correct / total
        print('val acc: {:2.3f}'.format(val_acc))

        # Val acc (top-k)
        val_topk = topk_correct/total
        print('val top-{}: {:2.3f}'.format(top_k, val_topk))

        # Val acc (top-k per class)
        val_class_topk = topk_correct_arr / topk_tot_arr

        # Val MAP
        val_map, val_map_per_class = stats.map(feat_dict, min_gallery=0, batch_size=1, use_gpu=True, print_time=False, per_class=True) 
        print('val MAP: {:2.3f}'.format(val_map))

        # Val AUC, AP, F1, F2
        score_arr = np.concatenate(score_list)
        label_arr = np.concatenate(label_list)
        val_auc, val_auc_list = metrics.classwise_auc(score_arr, label_arr)
        val_ap, val_ap_list = metrics.classwise_ap(score_arr, label_arr)
        val_f1, val_f1_list = metrics.classwise_fbeta(score_arr, label_arr, 1.0)
        val_f2, val_f2_list = metrics.classwise_fbeta(score_arr, label_arr, 2.0)
        print('val AUC: {:2.3f}'.format(val_auc))
        print('val AP: {:2.3f}'.format(val_ap))
        print('val F1: {:2.3f}'.format(val_f1))
        print('val F2: {:2.3f}'.format(val_f2))

        if self.params.logging.log_flag:
            stats_dict = {
                'fold_index': self.fold_index,
                'epoch_num': epoch_num,
                'loss': val_loss,
                'acc': val_acc, 
                'map': val_map,
                'conf': conf_arr.tolist(),
                'cmap': val_map_per_class.numpy().tolist(),
                'top{}'.format(top_k): val_topk,
                'class_top{}'.format(top_k): val_class_topk.tolist(),
                'auc': val_auc,
                'class_auc': val_auc_list,
                'ap': val_ap,
                'class_ap': val_ap_list,
                'f1': val_f1,
                'class_f1': val_f1_list,
                'f2': val_f2,
                'class_f2': val_f2_list,
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                stats_dict['q'] = self.q.tolist()
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)

    @torch.no_grad()
    def _test_triplet(self, epoch_num, model_path=None):
        print('Testing...')
        # Check if this trial already exists
        if self.exists(epoch_num):
            print('Test trial log already exists!')
            return
        if model_path is not None:
            # Load the network using given path
            print('Loading model from path...')
            self.load_model(model_path)
        else:
            # Load the network using internal params
            self.load(epoch_num)
        # Extract features from images
        self.net.eval()
        feat_dict = collections.defaultdict(list)
        with progressbar.ProgressBar(max_value=len(self.val_loader)) as bar:
            for batch_idx, (image_tsr, label_tsr) in enumerate(self.val_loader):
                # Squeeze labels
                label_tsr = label_tsr.squeeze()
                # Copy images to GPU
                if self.use_cuda:
                    image_tsr, label_tsr = image_tsr.cuda(), label_tsr.cuda()
                # Create variable object
                image_var, label_var = image_tsr, label_tsr
                # Run variable through network
                feat_var = self.net(image_var)
                # Convert feats and save them off?
                feat_tsr = feat_var.cpu().detach()
                label_tsr = label_var.cpu().detach()
                for feat, label in zip(feat_tsr, label_tsr):
                    feat_dict[label.item()].append(feat)
                # Update progressbar
                bar.update(batch_idx)

        # Val MAP
        val_map, val_map_per_class = stats.map(feat_dict, min_gallery=0, batch_size=1, use_gpu=True, print_time=False, per_class=True) 
        print('val MAP: {:2.3f}'.format(val_map))

        if self.params.logging.log_flag:
            stats_dict = {
                'fold_index': self.fold_index,
                'epoch_num': epoch_num,
                'map': val_map,
                'cmap': val_map_per_class.numpy().tolist(),
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                stats_dict['q'] = self.q.tolist()
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)

    @torch.no_grad()
    def test_map(self, epoch_num, model_path=None):
        print('Testing...')
        # Check if this trial already exists
        if self.exists(epoch_num):
            print('Test trial log already exists!')
            return
        if model_path is not None:
            # Load the network using given path
            print('Loading model from path...')
            self.load_model(model_path)
        else:
            # Load the network using internal params
            self.load(epoch_num)
        # Extract features from images
        self.net.eval()
        feat_dict = collections.defaultdict(list)
        with progressbar.ProgressBar(max_value=len(self.val_loader)) as bar:
            for batch_idx, (image_tsr, label_tsr) in enumerate(self.val_loader):
                # Squeeze labels
                label_tsr = label_tsr.squeeze()
                # Copy images to GPU
                if self.use_cuda:
                    image_tsr = image_tsr.cuda()
                # Create variable object
                image_var = image_tsr
                # Run variable through network
                feat_var = self.net(image_var)
                # Convert feats and save them off?
                feat_tsr = feat_var.cpu()
                for feat, label in zip(feat_tsr, label_tsr):
                    feat_dict[label].append(feat)
                # Update progressbar
                bar.update(batch_idx)

        # Val MAP
        val_map, val_map_per_class = stats.map(feat_dict, min_gallery=0, batch_size=1, use_gpu=True, print_time=False, per_class=True) 
        print('val MAP: {:2.3f}'.format(val_map))

        if self.params.logging.log_flag:
            stats_dict = {
                'fold_index': self.fold_index,
                'epoch_num': epoch_num,
                'map': val_map,
                'cmap': val_map_per_class.numpy().tolist(),
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                stats_dict['q'] = self.q.tolist()
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)

    @torch.no_grad()
    def viz_test(self, epoch_num):
        print('Viz...')
        # Load the network
        self.load(epoch_num)
        # Extract features from images
        self.net.eval()
        self.classifier.eval()
        val_loss = 0.0
        correct, total = 0, 0
        feat_dict = collections.defaultdict(list)
        idx_dict = collections.defaultdict(list)
        # Confusion matrix
        conf_arr = np.zeros((self.params.data.num_class, self.params.data.num_class))
        with progressbar.ProgressBar(max_value=len(self.val_loader)) as bar:
            for batch_idx, (image_tsr, label_tsr, idx_tsr) in enumerate(self.val_loader):
                # Squeeze labels
                label_tsr = label_tsr.squeeze()
                # Copy images to GPU
                if self.use_cuda:
                    image_tsr, label_tsr = image_tsr.cuda(), label_tsr.cuda()
                # Create variable object
                image_var, label_var = image_tsr, label_tsr
                # Run variable through network
                feat_var = self.net(image_var)
                out_var = self.classifier(feat_var)
                # Compute loss
                loss_var = self.criterion(out_var, label_var)
                # Compute stats
                val_loss += loss_var.item()
                _, pred_tsr = torch.max(out_var, 1)
                total += label_var.size(0)
                correct += pred_tsr.eq(label_var).cpu().sum()
                np.add.at(conf_arr, [label_var.cpu().numpy(), pred_tsr.cpu().numpy()], 1)
                val_acc = correct/total
                # Convert feats and save them off?
                feat_tsr = feat_var.cpu().detach()
                label_tsr = label_tsr.cpu().detach()
                idx_tsr = idx_tsr.cpu().detach()
                for feat, label, idx in zip(feat_tsr, label_tsr, idx_tsr):
                    feat_dict[label.item()].append(feat.numpy())
                    idx_dict[label.item()].append(idx.item())
                # Update progressbar
                bar.update(batch_idx)

        # Val loss
        val_loss = val_loss/(batch_idx+1)
        print('val loss: {:2.3f}'.format(val_loss))

        # Val acc
        val_acc = correct/total
        print('val acc: {:2.3f}'.format(val_acc))

        return feat_dict, idx_dict
        
    def viz_prep2(self, feat_dict, idx_dict, cls_list, min_gallery=0, top_n=3, num_probes=1):
        # Val MAP
        probe_idx_list, gallery_idx_list, dist_list, match_idx_arr = stats.top_n_pick(feat_dict, idx_dict, cls_list,
            min_gallery=min_gallery, top_n=top_n, num_probes=num_probes)

        new_transform_list = self.transform_list[:-2]
        new_transform_list.extend([
            lambda x: x if type(x) == np.ndarray else x.im,
            lambda x: x-x.min(),
            lambda x: x/x.max(),
            lambda x: (x*255).astype(np.uint8),
            torchvision.transforms.ToPILImage(),        
            torchvision.transforms.Resize((224, 224)),
        ])
        transform = torchvision.transforms.Compose(new_transform_list)

        # Denormalize images
        probe_img_list = [self.val_loader.dataset.get_viz2(p, transform) for p in probe_idx_list]
        gallery_img_list = [[self.val_loader.dataset.get_viz2(p, transform) for p in pl] for pl in gallery_idx_list]

        return probe_img_list, gallery_img_list, match_idx_arr, dist_list

    def get(self, class_idx, inst_idx):
        return self.val_set.get(class_idx, inst_idx)

class QualityPipeline(TestPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_save_name(self):
        save_name = '{}_{}_{}_{}_{}_{:.3f}'.format(
            self.params.experiment.origin, 
            self.params.experiment.train_mode, 
            self.params.data.num_class, 
            self.params.data.num_inst, 
            self.param_name, self.param_val)
        return save_name

    def log(self, stats_dict):
        # Create save file name
        save_name = self.get_save_name()
        log_name = save_name
        log_file = '{}.json'.format(self.mode)
        log_dir = os.path.join(self.params.logging.log_dir, save_name)

        # Create save dir if it does not exist
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        # Load current log dict
        if os.path.exists(log_path):
            with open(log_path, 'r') as fp:
                log_dict = json.load(fp)
        else:
            log_dict = {
                'metadata': {
                    'label_map': self.label_map,
                    'idx_map': self.idx_map
                },
                'entries': {}
            }

        # Append information to log dict
        log_dict['entries'][log_name] = stats_dict

        # Save log dict back to file path
        while True:
            num_attempt = 0
            try:
                with open(log_path, 'w') as fp:
                    json.dump(log_dict, fp)
                    os.fsync(fp.fileno())
            except OSError:
                num_attempt += 1
                print('Attempt {}: Failed to write file to disk: {}'.format(num_attempt, log_path))
                time.sleep(10)
            else:
                break

    def test(self):
        print('Testing...')
        # Extract features from images
        q_list = []
        label_list = []
        with progressbar.ProgressBar(max_value=len(self.val_loader)) as bar:
            for batch_idx, (img_tsr, label) in enumerate(self.val_loader):
                x = img_tsr.numpy()[0]
                x = x-x.min()
                x = x/x.max()
                x = (x*255).astype(np.uint8)
                x = np.rollaxis(x, 0, 3)
                x = Image.fromarray(x)
                x = x.convert('LA')
                x = np.array(x)
                img_arr = x
                q = brisque.measure(img_arr)
                q_list.append(q)
                label_list.append(int(label.numpy()[0][0]))
                # Update progressbar
                bar.update(batch_idx)

        if self.params.logging.log_flag:
            stats_dict = {
                'brisque': q_list,
                'labels': label_list,
            }
            if self.params.experiment.degrade_mode == 'blur':
                stats_dict['blur_radius'] = self.param_val
            elif self.params.experiment.degrade_mode == 'sat':
                stats_dict[self.param_name] = self.param_val
                stats_dict['q'] = self.q.tolist()
            elif self.params.experiment.degrade_mode == 'none':
                stats_dict['original'] = 1
            self.log(stats_dict)
        

if __name__=='__main__':
    params = parse_params('params.yaml', './expx.yaml')
    fold_index = 0
    param_val = 2.0
    epoch_num = 5
    test_pipeline = TestPipeline(params, 'val', fold_index, param_name=params.experiment.sat_param, param_val=param_val,
        feat_mean=None, feat_std=None)
    print(TestPipeline.dataset)
    #test_pipeline.viz_test(epoch_num)
