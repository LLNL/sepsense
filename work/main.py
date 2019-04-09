#!/usr/bin/env python3

# Ignore matplotlib warning from imsim
import sys
import warnings
warnings.simplefilter("ignore")

# Local imports
from pipeline import TrainPipeline, TestPipeline, NormPipeline, QualityPipeline
from params import parse_cmdline, parse_params

def main():
    # Parse args
    args = parse_cmdline()
    params = parse_params(args.template_path, args.param_path)

    # TODO: Check if number of available GPUs is same as number specified

    # No-reference quality metrics
    if params.experiment.train_mode == 'noref':
        print('### Experiment Type x (no ref) ###')
        # Get metadata for HOG, None for CNN
        for param_val in getattr(params.sensor, params.experiment.sat_param):
            qual_pipeline = QualityPipeline(params, 'full', None, param_name=params.experiment.sat_param, param_val=param_val)
            qual_pipeline.test()

    # Normalization
    elif params.experiment.train_mode == 'norm':
        print('### Experiment Type 2 (norm) ###')
        for fold_index in range(params.experiment.num_folds):
            # Get metadata for HOG, None for CNN
            for param_val in getattr(params.sensor, params.experiment.sat_param):
                norm_pipeline = NormPipeline(params, 'full', fold_index, param_name=params.experiment.sat_param, param_val=param_val)
                feat_mean, feat_std = norm_pipeline.norm()

    # Experiment type 0
    elif params.experiment.train_mode == 'basic':
        print('### Experiment Type 0 ###')
        for fold_index in range(params.experiment.num_folds):
            if params.model.method == 'hog':
                norm_pipeline = NormPipeline(params, 'train', fold_index)
                feat_mean, feat_std = norm_pipeline.norm()
            else:
                feat_mean, feat_std = None, None
            train_pipeline = TrainPipeline(params, 'train', fold_index, feat_mean=feat_mean, feat_std=feat_std)
            test_pipeline = TestPipeline(params, 'val', fold_index, feat_mean=feat_mean, feat_std=feat_std)
            train_pipeline.save()
            test_pipeline.test(0)
            for epoch_num in range(1, params.learning.num_epochs+1):
                train_pipeline.train(save_flag=True)
                test_pipeline.test(epoch_num)

    # Experiment type 1
    elif params.experiment.train_mode == 'orig':
        print('### Experiment Type 1 ###')
        mean_list, std_list = [], []
        for fold_index in range(params.experiment.num_folds):
            # Get metadata for HOG, None for CNN
            if params.model.method == 'hog':
                norm_pipeline = NormPipeline(params, 'train', fold_index)
                feat_mean, feat_std = norm_pipeline.norm()
                mean_list.append(feat_mean)
                std_list.append(feat_std)
            else:
                mean_list.append(None), std_list.append(None)
            # Create train pipeline
            train_pipeline = TrainPipeline(params, 'train', fold_index, 
                feat_mean=mean_list[fold_index], feat_std=std_list[fold_index])
            train_pipeline.save()
            for epoch_num in range(params.learning.num_epochs):
                train_pipeline.train(save_flag=True)

        for fold_index in range(params.experiment.num_folds):
            for param_val in getattr(params.sensor, params.experiment.sat_param):
                test_pipeline = TestPipeline(params, 'val', fold_index, 
                    param_name=params.experiment.sat_param, param_val=param_val, 
                    feat_mean=mean_list[fold_index], feat_std=std_list[fold_index])
                for epoch_num in range(params.learning.num_epochs+1):
                    test_pipeline.test(epoch_num)

    # Experiment type 3
    elif args.test_param_path is not None:
        feat_mean, feat_std = None, None
        test_params = parse_params(args.template_path, args.test_param_path)
        print('### Experiment Type 3 ###')
        # Get metadata for HOG, None for CNN
        for param_val in getattr(params.sensor, params.experiment.sat_param):
            for train_fold_index in range(params.experiment.num_folds):
                load_pipeline = TestPipeline(params, 'val', train_fold_index, 
                    param_name=params.experiment.sat_param, param_val=param_val,
                    feat_mean=feat_mean, feat_std=feat_std)
                model_path = load_pipeline.get_path(5)
                for test_fold_index in range(test_params.experiment.num_folds):
                    test_pipeline = TestPipeline(test_params, 'val', test_fold_index, 
                        param_name=params.experiment.sat_param, param_val=param_val,
                        feat_mean=feat_mean, feat_std=feat_std, train_fold_index=train_fold_index)
                    test_pipeline.test_map(0, model_path=model_path)

    # Experiment type 2
    elif params.experiment.train_mode == 'degrade':
        print('### Experiment Type 2 ###')
        for fold_index in range(params.experiment.num_folds):
            # Get metadata for HOG, None for CNN
            for param_val in getattr(params.sensor, params.experiment.sat_param):
                if params.model.method == 'hog':
                    norm_pipeline = NormPipeline(params, 'train', fold_index, param_name=params.experiment.sat_param, param_val=param_val)
                    feat_mean, feat_std = norm_pipeline.norm()
                else:
                    feat_mean, feat_std = None, None
                train_pipeline = TrainPipeline(params, 'train', fold_index, param_name=params.experiment.sat_param, param_val=param_val,
                    feat_mean=feat_mean, feat_std=feat_std)
                test_pipeline = TestPipeline(params, 'val', fold_index, param_name=params.experiment.sat_param, param_val=param_val,
                    feat_mean=feat_mean, feat_std=feat_std)
                train_pipeline.save()
                test_pipeline.test(0)
                # Train for num_epochs
                for _ in range(1, params.learning.num_epochs+1):
                    epoch_num = train_pipeline.train(save_flag=True)
                    test_pipeline.test(epoch_num)

    # Invalid experiment type
    else:
        raise Exception('Invalid experiment train_mode: {}'.format(params.experiment.train_mode))

if __name__ == '__main__':
    main()
