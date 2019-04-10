#!/usr/bin/env python3

# Global imports
import os
import sys
import json
import math
import collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
# Show backend
mpl.rcParams['axes.xmargin'] = 0.0
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['xtick.labelsize'] = 12.0
mpl.rcParams['ytick.labelsize'] = 12.0
#print(mpl.rcParams)
#sys.exit()
print(mpl.get_backend())
print(mpl.matplotlib_fname())

import torch
import torch.utils.data

from scipy.stats.stats import pearsonr, spearmanr

# Set up local imports from ../
import sys
sys.path.append('../')
sys.argv=['']; del sys

# Local imports
from dataset import get_metadata
from scipy.signal import savgol_filter as savitzky_golay

# Global constants
FIG_DIR = '../paper/figures'

FNUMBER_FACTOR = 6e-6/4.5e-7

def _load_basic_logs(log_dir):
    fold_set = set()
    # Load full logs for a trial
    log_dir_list = os.listdir(log_dir)
    epoch_dict = collections.defaultdict(list)
    for ld in log_dir_list:
        try:
            _, _, _, _, _, _, epoch_num = ld.split('_')
        except:
            print(log_dir)
            print(ld)
            raise
        log_path = os.path.join(log_dir, ld, 'val.json')
        if not os.path.isfile(log_path):
            continue
        with open(log_path, 'r') as fp:
            log_dict = json.load(fp)
        epoch_dict[epoch_num].extend(log_dict['entries'].values())

    return epoch_dict, log_dict, fold_set
        
def load_basic_logs(log_dir, stat_list):
    epoch_dict, log_dict, fold_set = _load_basic_logs(log_dir)

    # Get averages for a parameter across k-folds
    avg_dict = {}
    for stat_name in stat_list:
        entry_dict = collections.defaultdict(list)
        for epoch_num in sorted(epoch_dict):
            for entry in epoch_dict[epoch_num]:
                fold_index = entry['fold_index']
                fold_set.add(fold_index)
                entry_dict[epoch_num].append(entry[stat_name])
        stat_mean_list, stat_std_list = [], []
        for epoch_num in sorted(entry_dict):
            stat_mean = np.mean(entry_dict[epoch_num])
            stat_std = np.std(entry_dict[epoch_num])
            stat_mean_list.append(stat_mean)
            stat_std_list.append(stat_std)
        max_idx = np.argmax(stat_mean_list)
        avg_dict[stat_name] = (max_idx, stat_mean_list[max_idx])
                
    print('Num folds found: {}'.format(len(fold_set)))
        
    return avg_dict

def compare_basic(name_list, dir_list, epoch_list, stat_list, 
                   std=True, norm_list=[False], 
                   smooth=True, resample=False, niirs_list=False, niirs_label_list=None,
                   save_file=None):

    trial_dict = {}
    for name, log_dir in zip(name_list, dir_list):
        trial_dict[name] = load_basic_logs(log_dir, stat_list)

    return trial_dict

def _load_logs(log_dir, interval=0.1):
    fold_set = set()
    # Load full logs for a trial
    log_dir_list = os.listdir(log_dir)
    epoch_dict = collections.defaultdict(list)
    for ld in log_dir_list:
        # XXX: Deprecated
        #_, _, _, _, _, _, epoch_num = ld.split('_')
        try:
            _, _, _, _, _, _, epoch_num, _, _, fl_str = ld.split('_')
            fl_float = float(fl_str)
            if (fl_float+1e-8)%interval > 1e-5:
                continue
        except:
            _, _, _, _, _, _, _, epoch_num, _, _, fl_str = ld.split('_')
            fl_float = float(fl_str)
            if (fl_float+1e-8)%interval > 1e-5:
                continue
        log_path = os.path.join(log_dir, ld, 'val.json')
        if not os.path.isfile(log_path):
            continue
        with open(log_path, 'r') as fp:
            try:
                log_dict = json.load(fp)
            except json.JSONDecodeError:
                print('Could not load log: {}'.format(log_path))
                sys.exit(1)
        epoch_dict[epoch_num].extend(log_dict['entries'].values())

    return epoch_dict, log_dict, fold_set
        

def load_logs(log_dir, param_name, stat_name, interval=0.1):
    epoch_dict, log_dict, fold_set = _load_logs(log_dir, interval=interval)

    # Get averages for a parameter across k-folds
    fold_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    avg_dict = collections.defaultdict(list)
    for epoch_num in epoch_dict:
        entry_dict = collections.defaultdict(list)
        for entry in epoch_dict[epoch_num]:
            fold_index = entry['fold_index']
            fold_set.add(fold_index)
            if param_name == 'focal_length':
                # Floating point issue caused different dict keys, this is effectively rounding
                param_val = float('{:.5f}'.format(entry[param_name]))
                fold_dict[epoch_num][fold_index].append((param_val, entry[stat_name]))
                entry_dict[param_val].append(entry[stat_name])
            elif param_name == 'q' or param_name == 'f#':
                # Floating point issue caused different dict keys, this is effectively rounding
                param_val = float('{:.5f}'.format(entry['q'][0]))
                if param_name == 'f#':
                    param_val = param_val * FNUMBER_FACTOR
                fold_dict[epoch_num][fold_index].append((param_val, entry[stat_name]))
                entry_dict[param_val].append(entry[stat_name])
        for p in entry_dict:
            #print('min/max: {:.3f}, {:.3f}'.format(np.min(entry_dict[p]), np.max(entry_dict[p])))
            stat_mean = np.mean(entry_dict[p])
            stat_std = np.std(entry_dict[p])
            avg_dict[epoch_num].append((p, stat_mean, stat_std))
            
    print('Num folds found: {}'.format(len(fold_set)))
        
    return avg_dict

def load_conf(log_dir, param_name, avg=True, interval=0.1):
    epoch_dict, log_dict, fold_set = _load_logs(log_dir, interval=interval)

    # Get averages for a parameter across k-folds
    fold_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    avg_dict = collections.defaultdict(list)
    for epoch_num in epoch_dict:
        entry_dict = collections.defaultdict(list)
        for entry in epoch_dict[epoch_num]:
            fold_index = entry['fold_index']
            fold_set.add(fold_index)
            
            conf_arr = np.array(entry['conf'])
            d = conf_arr.diagonal()
            t = conf_arr.sum(axis=1)
            m = d/t
            
            param_val = float('{:.5f}'.format(entry[param_name]))
            fold_dict[epoch_num][fold_index].append((param_val, m))
            entry_dict[param_val].append(entry['conf'])
        for p in entry_dict:
            conf_arr = np.array(entry_dict[p])
            sum_conf_arr = conf_arr.sum(axis=0)
            d = sum_conf_arr.diagonal()
            t = sum_conf_arr.sum(axis=1)
            m = d/t
            avg_dict[epoch_num].append((p, m))
            
    print('Num folds found: {}'.format(len(fold_set)))
    
    if avg:
        return avg_dict, log_dict['metadata']
    else:
        return fold_dict, log_dict['metadata']

def load_cmap(log_dir, param_name, stat_name, avg=True, interval=0.1):
    epoch_dict, log_dict, fold_set = _load_logs(log_dir, interval=interval)

    # Get averages for a parameter across k-folds
    fold_dict = collections.defaultdict(lambda : collections.defaultdict(list))
    avg_dict = collections.defaultdict(list)
    for epoch_num in epoch_dict:
        entry_dict = collections.defaultdict(list)
        for entry in epoch_dict[epoch_num]:
            fold_index = entry['fold_index']
            fold_set.add(fold_index)
            param_val = float('{:.5f}'.format(entry[param_name]))
            fold_dict[epoch_num][fold_index].append((param_val, entry[stat_name]))
            entry_dict[param_val].append(entry[stat_name])
        for p in entry_dict:
            conf_arr = np.array(entry_dict[p])
            mean_conf_arr = np.mean(conf_arr, axis=0)
            avg_dict[epoch_num].append((p, mean_conf_arr))
            
    print('Num folds found: {}'.format(len(fold_set)))
        
    if avg:
        return avg_dict, log_dict['metadata']
    else:
        return fold_dict, log_dict['metadata']

def per_class_plot(log_dir_list, stat_name_list, param_name, resample=True, smooth=True, norm=True, avg=False, param_idx=10, plot_max=True, epoch_num_list=['5'], scale_factor=0.7, save_file=None, num_plot=35, num_plot_per_row=5,
    axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16):

    num_plot_per_col = math.ceil(num_plot/num_plot_per_row)
    plt.figure(figsize=(num_plot_per_row*2.5, num_plot_per_col*2.5))
    sorted_idx = None
    for log_dir_idx, (log_dir, stat_name, epoch_num) in enumerate(zip(log_dir_list, stat_name_list, epoch_num_list)):
        if stat_name == 'map':
            lower_mean = 0.030443216526675167
            lower_sd = 0.004069010021403916
        elif stat_name == 'acc':
            lower_mean = 1.0/35.0
            lower_var = (((2.0/35.0) + 1.0)**2 - 1.0) / 12.0
            lower_sd = np.sqrt(lower_var)
        # XXX: compute actual lower_mean, lower_sd
        else:
            lower_mean = 0
            lower_sd = 0
        if stat_name == 'acc':
            avg_dict, metadata = load_conf(log_dir, param_name)
        elif stat_name == 'map':
            avg_dict, metadata = load_cmap(log_dir, param_name, 'cmap')
        else:
            avg_dict, metadata = load_cmap(log_dir, param_name, 'class_{}'.format(stat_name))

        sorted_elem = sorted(avg_dict[epoch_num])
        p, m = zip(*sorted_elem)
        m_arr = np.array(m).T
        # Sort
        if log_dir_idx == 0:
            sorted_idx = np.array(list(zip(*sorted(enumerate(m_arr), key=lambda ea: ea[1].mean(), reverse=True)))[0])

        tonicity_list = []
        weighted_tonicity_list = []
        if avg:
            if smooth:
                plt.plot(p, savitzky_golay(m_arr.mean(axis=0), 15, 2))
            else:
                plt.plot(p, m_arr.mean(axis=0))
        else:
            for j, (i, acc) in enumerate(zip(sorted_idx, m_arr[sorted_idx])):
                if j >= num_plot:
                    break
                cls = metadata['idx_map'][str(i)]
                cls = cls.replace('_', ' ').title()
                cls = cls[:12]+'...' if len(cls) > 12 else cls
                if resample:
                    pp = np.linspace(p[0], p[-1], 4*len(p))
                    acc = np.interp(pp, p, acc)
                else:
                    pp = p

                if smooth:
                    acc = savitzky_golay(acc, len(acc)-1, 2)
                # Optional normalization
                if norm:
                    acc = np.array(acc)
                    acc = acc - acc.min()
                    acc = acc / acc.max()

                # Plot
                plt.subplot(math.ceil(num_plot/num_plot_per_row), num_plot_per_row, j+1)
                plt.plot(pp, acc, label=(log_dir+stat_name))
                tonicity_list.append(tonicity.compute(acc))
                weighted_tonicity_list.append(tonicity.weighted_compute(acc))
                if plot_max:
                    plt.plot(pp[acc.argmax()], [acc.max()], 'v', color='black')#, mfc='none')
                #plt.axis('off')
                plt.xticks(np.arange(0.1, 1.01, 0.3))
                #plt.yticks([])
                plt.title(cls, fontsize=axis_label_size)
                plt.axhline(y=lower_mean, color='black', linestyle='--')
                #plt.title('{} vs. {} for {}'.format(stat_name, param_name, cls))
                #plt.xlabel('{} (m)'.format(param_name))
                #plt.ylabel('{}'.format(stat_name))
                if True:#norm:
                    plt.ylim(0.0, 1.0)
        print('Tonicity:', stat_name, np.mean(tonicity_list))
        print('Weighted Tonicity:', stat_name, np.mean(weighted_tonicity_list))
    #plt.legend()
                        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('rAP vs. Focal Length for {} Classes'.format(num_plot), fontsize=title_size, y=1.05)
    #plt.legend(bbox_to_anchor=(1.0, 10.0))
    #plt.show()
    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')

metric_title_dict = {
    'area_dist': 'Average Object Area'
}

metric_axis_dict = {
    'area_dist': r'Average Object Area ($m^2$)'
}

def per_class_scatter(fold_dict, metadata, stat_name, md_name, md_dict, param_val=0.1, smooth=True, avg=False,
                   axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16, save_file=None):
    data_list = []
    for epoch_num in sorted(fold_dict):
        if epoch_num == '5':
            for fold in fold_dict[epoch_num]:
                for entry in fold_dict[epoch_num][fold]:
                    param_val_i, stat_list = entry
                    if param_val_i == param_val:
                        for cls_idx, stat_val in enumerate(stat_list):
                            cls = metadata['idx_map'][str(cls_idx)]
                            md_val = md_dict[str(fold)][cls][md_name]
                            data_list.append((md_val, stat_val))
                    
    print('Num points: {}'.format(len(data_list)))
    arr = np.array(data_list).T
    cc, pv = pearsonr(*arr)
    sc, sv = spearmanr(*arr)
    print('Pearson Correlation: {:.3f}, {:.5f}'.format(cc, pv))
    print('Spearman Correlation: {:.3f}, {:.5f}'.format(sc, sv))
    plt.figure()
    plt.scatter(*arr, edgecolors='black')
    plt.xscale('log')
    plt.title('{} vs. {} for 35 Classes'.format(stat_name, metric_title_dict[md_name]), fontsize=title_size)
    plt.xlabel('{}'.format(metric_axis_dict[md_name]), fontsize=axis_label_size)
    plt.ylabel('{}'.format(stat_name), fontsize=axis_label_size)
    
    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, bbox_inches="tight", format='png', dpi=300)
        
def per_peak_scatter(fold_dict, metadata, param_name, md_name, md_dict, smooth=True, avg=False):
    data_list = []
    for epoch_num in sorted(fold_dict):
        if epoch_num == '5':
            for fold in sorted(fold_dict[epoch_num])[:-1]:
                entry_list = sorted(fold_dict[epoch_num][fold], key=lambda x: x[0])
                param_arr = np.array(list(zip(*entry_list))[0])
                entry_arr = np.array(list(zip(*entry_list))[1])
                max_idx = np.argmax(entry_arr, axis=0)
                max_list = param_arr[max_idx]
                for cls_idx, max_val in enumerate(max_list):
                    cls = metadata['idx_map'][str(cls_idx)]
                    md_val = md_dict[str(fold)][cls][md_name]
                    data_list.append((md_val, max_val))
                    
    print('Num points: {}'.format(len(data_list)))
    arr = np.array(data_list).T
    cc, pv = pearsonr(*arr)
    sc, sv = spearmanr(*arr)
    print('Pearson Correlation: {:.3f}, {:.5f}'.format(cc, pv))
    print('Spearman Correlation: {:.3f}, {:.5f}'.format(sc, sv))
    plt.figure()
    plt.scatter(*arr)
    plt.xscale('log')
    plt.title('{} vs. {} for 35 Classes'.format(param_name, md_name))
    plt.xlabel('{}'.format(md_name))
    plt.ylabel('{}'.format(param_name))
    
def per_class_corr(fold_dict, metadata, param_name, stat_name, md_name, md_dict, smooth=True, avg=False,
                   axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16, save_file=None):
    plt.figure()
    data_dict = collections.defaultdict(list)
    for epoch_num in sorted(fold_dict):
        if epoch_num == '5':
            for fold in fold_dict[epoch_num]:
                for entry in fold_dict[epoch_num][fold]:
                    param_val, stat_list = entry
                    for cls_idx, stat_val in enumerate(stat_list):
                        cls = metadata['idx_map'][str(cls_idx)]
                        md_val = md_dict[str(fold)][cls][md_name]
                        data_dict[param_val].append((md_val, stat_val))
    
    param_list = []
    pearson_list = []
    #spearman_list = []
    for param_val, data_list in sorted(data_dict.items()):
        arr = np.array(data_list).T
        cc, pv = pearsonr(*arr)
        #sc, sv = spearmanr(*arr)
        param_list.append(param_val)
        pearson_list.append(cc)
        #spearman_list.append(sc)
        
    plt.figure()
    plt.plot(param_list, pearson_list)
    #plt.figure()
    #plt.plot(param_list, spearman_list)
    plt.title('{} vs. {} for 35 Classes'.format('rAP/Area Correlation', title_dict[param_name]), fontsize=title_size)
    plt.xlabel('{}'.format(axis_dict[param_name]), fontsize=axis_label_size)
    plt.ylabel('{}'.format('rAP/Area Correlation'), fontsize=axis_label_size)

    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, bbox_inches="tight", format='png', dpi=300)
        
title_dict = {
    'q': 'Optical Q',
    'f#': 'F-Number',
    'focal_length': 'Focal Length',
    'acc': 'Top-1 Accuracy',
    'top3': 'Top-3 Accuracy',
    'map': 'rAP',  
    'auc': 'AUC',
    'ap': 'cAP',
    'f1': 'F1',
    'f2': 'F2',
}

axis_dict = {
    'q': 'Optical Q',
    'f#': 'F-Number',
    'focal_length': 'Focal Length (m)',
    'acc': 'Top-1 Accuracy',
    'top3': 'Top-3 Accuracy',
    'map': 'rAP',  
    'auc': 'AUC',
    'ap': 'cAP',
    'f1': 'F1',
    'f2': 'F2',
}
    
def per_stat_plots(log_dir, num_folds, param_name, stat_list, save_file=None,
    legend_pos=(1.0, 0.55), text_x=0.84, text_y_sub=0.01, figsize=None, show_max=True,
    epoch_num=0, norm_list=[False, True],
    axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16):
    
    # Load logs
    dict_list = []
    for stat_name in stat_list:
        avg_dict = load_logs(log_dir, param_name, stat_name)
        dict_list.append((stat_name, avg_dict))
    
    fig, ax = plt.subplots(nrows=1, ncols=len(norm_list), figsize=figsize)
    if len(norm_list) == 1:
        ax = [ax]
    #plt.figure(figsize=figsize)
    print('Fig size:', plt.rcParams['figure.figsize'])
    for norm_idx, norm in enumerate(norm_list):
        for stat_name, avg_dict in dict_list:
            sorted_elem = sorted(avg_dict[str(epoch_num)])
            x, y, z = zip(*sorted_elem)
            x, y = np.array(x), np.array(y)
            if norm:
                y = y - y.min()
                y = y / y.max()
                ax[norm_idx].set_ylim(0, 1.1)
                ax[norm_idx].set_ylabel('Normalized Metric Values', fontsize=axis_label_size)
                ax[norm_idx].set_title('{} vs. {}'.format('Normalized Metric Values', title_dict[param_name]), fontsize=title_size)
            else:
                ax[norm_idx].set_ylabel('Metric Values', fontsize=axis_label_size)
                ax[norm_idx].set_title('{} vs. {}'.format('Metric Values', title_dict[param_name]), fontsize=title_size)
            ax[norm_idx].plot(x, y, label=title_dict[stat_name])
            # Compute tonicity
            print('Tonicity {}: {}'.format(stat_name, tonicity.compute(y)))
            # Show max
            if show_max:
                ax[norm_idx].plot(x[y.argmax()], [y.max()], 'v', color='black')#, mfc='none')
        #plt.axhline(y=lower_mean, color='black')
        #plt.text(text_x, lower_mean-text_y_sub, 'Random Guess')
        ax[norm_idx].set_xlabel(axis_dict[param_name], fontsize=axis_label_size)
        if norm_idx == len(norm_list)-1:
            legend = ax[norm_idx].legend(title='Metric Name', frameon=True, fontsize=legend_label_size, loc='lower right')
            legend.set_title('Metric Name', prop={'size': legend_title_size})

    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, bbox_inches="tight", format='png', dpi=300)
        
def per_epoch_plots(log_dir, num_folds, param_name, stat_name, save_file=None,
    legend_pos=(1.0, 0.55), text_x=0.84, text_y_sub=0.01, figsize=None, show_max=True,
    start_epoch=0, show_random=False, plot_title_add=''):

    # Load logs
    avg_dict = load_logs(log_dir, param_name, stat_name)
    
    if stat_name == 'map':
        lower_mean = 0.030443216526675167
        lower_sd = 0.004069010021403916
    elif stat_name == 'acc':
        lower_mean = 1.0/35.0
        lower_var = (((2.0/35.0) + 1.0)**2 - 1.0) / 12.0
        lower_sd = np.sqrt(lower_var)
    elif stat_name == 'auc':
        # XXX: Add real vals
        lower_mean = 0
        lower_sd = 0
    elif stat_name == 'top3':
        # XXX: Add real vals
        lower_mean = 0
        lower_sd = 0
    
    plt.figure(figsize=figsize)
    print('Fig size:', plt.rcParams['figure.figsize'])
    for epoch_num in sorted(avg_dict, key=lambda x: int(x)):
        if int(epoch_num) < start_epoch:
            continue
        sorted_elem = sorted(avg_dict[epoch_num])
        x, y, z = zip(*sorted_elem)
        x, y = np.array(x), np.array(y)
        print(epoch_num, ':', np.mean(y))
        plt.plot(x, y, label=epoch_num)
        # Show max
        if show_max:
            plt.plot(x[y.argmax()], [y.max()], 'v', color='black')#, mfc='none')
    if show_random:
        plt.axhline(y=lower_mean, color='black', linestyle='--')
        plt.text(text_x, lower_mean-text_y_sub, 'Lower Bound')
    plt.title('{} vs. {}\nfor {} Epochs {}'.format(title_dict[stat_name], title_dict[param_name], len(avg_dict)-1, plot_title_add))
    plt.xlabel(axis_dict[param_name])
    plt.ylabel(axis_dict[stat_name])
    plt.legend(title='Epochs Trained', frameon=True, bbox_to_anchor=legend_pos)

    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, bbox_inches="tight", format='png', dpi=300)
        
def per_epoch_split_plots(avg_dict, num_folds, param_name, stat_name, save_file=None):
    if stat_name == 'map':
        lower_mean = 0.030443216526675167
        lower_sd = 0.004069010021403916
    elif stat_name == 'acc':
        lower_mean = 1.0/35.0
        lower_var = (((2.0/35.0) + 1.0)**2 - 1.0) / 12.0
        lower_sd = np.sqrt(lower_var)
    
    for epoch_num in sorted(avg_dict, key=lambda x: int(x)):
        plt.figure()
        sorted_elem = sorted(avg_dict[epoch_num])
        x, y, z = zip(*sorted_elem)
        y_u = np.array(y) + np.array(z)
        y_l = np.array(y) - np.array(z)
        plt.fill_between(x, y_u, y_l, color=[1.0, 0.0, 0.0, 0.5])
        plt.plot(x, y, label=epoch_num)
        if epoch_num:
            #plt.axhline(y=lower_mean, color='black')
            x = np.arange(0, 4.6, 0.1)
            by = np.repeat([lower_mean], len(x))
            bsd = np.repeat([lower_sd], len(x))
            byl = by - bsd
            byl[byl<0] = 0
            byu = by + bsd
            ###
            #plt.fill_between(x, byu, byl, color=[1.0, 0.0, 0.0, 0.5])
            #plt.plot(x, by, color='black')
            plt.axhline(y=lower_mean, color='black')
        plt.title('{} vs. {}'.format(title_dict[stat_name], title_dict[param_name]))
        plt.xlabel(axis_dict[param_name])
        plt.ylabel(axis_dict[stat_name])
        plt.legend(title='Epochs Trained', bbox_to_anchor=(1.3, 1))
        plt.text(1.05, lower_mean, 'Random Guess')

def compare_models(name_list, dir_list, epoch_list, param_name, stat_name, 
                   std=True, norm_list=[False], 
                   smooth=True, resample=False, niirs_list=False, niirs_label_list=None,
                   log_interval=0.1,
                   legend_pos=(1.45, 1.05), 
                   legend_title='default',
                   show_max=True,
                   filter_len=5, 
                   figsize=None,
                   show_random=False,
                   show_random_x=0.825,
                   show_random_sub=0.01,
                   show_random_text='Lower Bound',
                   show_random_classes=35,
                   save_file=None,
                   limit_yrange=False,
                   axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16):

    log_list = []
    for name, log_dir in zip(name_list, dir_list):
        log_list.append((name, load_logs(log_dir, param_name, stat_name, interval=log_interval)))
        
    if stat_name == 'map':
        #lower_mean = 0.030443216526675167
        lower_mean = prob.get_lower_bound(num_classes=show_random_classes)
        #lower_sd = 0.004069010021403916
    elif stat_name == 'acc':
        lower_mean = 1.0/35.0
        #lower_var = (((2.0/35.0) + 1.0)**2 - 1.0) / 12.0
        #lower_sd = np.sqrt(lower_var)

    fig, ax = plt.subplots(nrows=1, ncols=len(norm_list), figsize=figsize)
    if len(norm_list) == 1:
        ax = [ax]
    max_val = None
    val_buf = 0.1
    for norm_idx, norm in enumerate(norm_list):
        print('Fig size:', plt.rcParams['figure.figsize'])
        labels = []
        for i, (log_name, avg_dict) in enumerate(log_list):
            epoch_num = str(epoch_list[i])
            sorted_elem = sorted(avg_dict[epoch_num])
            x, y, z = zip(*sorted_elem)
            if resample:
                xp = np.linspace(x[0], x[-1], 4*len(x))
                y = np.interp(xp, x, y)
                z = np.interp(xp, x, z)
                x = xp
            else:
                x, y, z = np.array(x), np.array(y), np.array(z)
            # Optional smoothing
            if smooth:
                y = savitzky_golay(y, filter_len, 2)
                z = savitzky_golay(z, filter_len, 2)
            y_u = np.array(y) + np.array(z)
            y_l = np.array(y) - np.array(z)
            # Optional normalization
            if norm:
                y = np.array(y)
                y_u = y_u - y.min()
                y_l = y_l - y.min()
                y = y - y.min()
                y_u = y_u / y.max()
                y_l = y_l / y.max()
                y = y / y.max()
            if std:
                ax[norm_idx].fill_between(x, y_u, y_l, color=[1.0, 0.0, 0.0, 0.5])
            # Plot vals as line plot
            l = ax[norm_idx].plot(x, y, label=log_name)
            # Show max
            if show_max:
                ax[norm_idx].plot(x[y.argmax()], [y.max()], 'v', color='black')#, mfc='none')
            labels.extend(l)
            # Get max value to use for NIIRS plotting
            max_val = max(y)
            
        # Plot NIIRS
        if niirs_list:
            for i, niirs in enumerate(niirs_list):
                if i == 0:
                    nax = ax[norm_idx].twinx()
                    nax.set_ylabel(niirs_label_list[i], fontsize=axis_label_size)
                    nax.yaxis.set_label_coords(1.1, 0.3)
                if norm:
                    if param_name == 'focal_length':
                        nx = niirs[param_name]
                    elif param_name == 'f#':
                        nx = niirs[param_name]
                    elif param_name == 'q':
                        nx = niirs[param_name][:, 0]
                    ny = niirs['niirs']
                    #ny = np.array(ny)
                    #ny = ny - ny.min()
                    #ny = ny / ny.max()
                    l = nax.plot(nx, ny, label=niirs_label_list[i], color='black')
                    labels.extend(l)
                else:
                    if param_name == 'focal_length':
                        nx = niirs[param_name]
                    elif param_name == 'f#':
                        nx = niirs[param_name]
                    elif param_name == 'q':
                        nx = niirs[param_name][:, 0]
                    ny = niirs['niirs']
                    
                    l = nax.plot(nx, ny, label=niirs_label_list[i], color='black') 
                    labels.extend(l)

                if show_max:
                    nax.plot(nx[np.argmax(ny)], [max(ny)], 'v', color='black')#, mfc='none')

                if True:
                    # Compute scaled max value
                    smax = ((max_val+val_buf)/max_val) * max(ny)
                    nax.set_ylim(0, smax)
            
        if norm:
            ax[norm_idx].set_title('Normalized {} vs. {}'.format(title_dict[stat_name], title_dict[param_name]), fontsize=title_size)
            ax[norm_idx].set_ylabel('Normalized {}'.format(axis_dict[stat_name]), fontsize=axis_label_size)
        else:
            ax[norm_idx].set_title('{} vs. {}'.format(title_dict[stat_name], title_dict[param_name]), fontsize=title_size)
            ax[norm_idx].set_ylabel(axis_dict[stat_name], fontsize=axis_label_size)
        if show_random and norm_idx == 0:
            ax[norm_idx].axhline(y=lower_mean, color='black', linestyle='--')
            ax[norm_idx].text(show_random_x, lower_mean-show_random_sub, show_random_text)
        ax[norm_idx].set_xlabel(axis_dict[param_name], fontsize=axis_label_size)
            
        labs = [l.get_label() for l in labels]
        if norm_idx == (len(norm_list) - 1):
            #ax[norm_idx].legend(labels, labs, title=legend_title, bbox_to_anchor=legend_pos, frameon=True)
            legend = ax[norm_idx].legend(labels, labs, title=legend_title, loc='lower right', frameon=True, fontsize=legend_label_size)
            legend.set_title(legend_title, prop={'size': legend_title_size})
        #plt.text(1.05, lower_mean, 'Random Guess')
        if limit_yrange:
            ax[norm_idx].set_ylim(0, max_val+val_buf)
        #else:
        #    ax[norm_idx].set_ylim(0, 1.0)
    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')

    return fig, ax
    
def compare_params(name_list, dir_list, epoch_list, param_list, stat_name, 
                   std=True, norm=False, 
                   smooth=True, resample=False, niirs_list=False, niirs_label_list=None,
                   log_interval=0.1,
                   legend_pos=(1.45, 1.05), 
                   legend_title='default',
                   show_max=True,
                   filter_len=5, 
                   figsize=None,
                   show_random=False,
                   save_file=None,
                   scale_minmax=False,
                   axis_label_size=14, legend_title_size=14, legend_label_size=12, title_size=16):

    log_list_list = []
    for param_name in param_list:
        log_list = []
        for name, log_dir in zip(name_list, dir_list):
            log_list.append((name, load_logs(log_dir, param_name, stat_name, interval=log_interval)))
        log_list_list.append(log_list)
        
    if stat_name == 'map':
        lower_mean = 0.030443216526675167
        lower_sd = 0.004069010021403916
    elif stat_name == 'acc':
        lower_mean = 1.0/35.0
        lower_var = (((2.0/35.0) + 1.0)**2 - 1.0) / 12.0
        lower_sd = np.sqrt(lower_var)

    fig, ax = plt.subplots(nrows=1, ncols=len(param_list), figsize=figsize)
    if len(param_list) == 1:
        ax = [ax]
    for param_idx, param_name in enumerate(param_list):
        print('Fig size:', plt.rcParams['figure.figsize'])
        labels = []
        for i, (log_name, avg_dict) in enumerate(log_list_list[param_idx]):
            epoch_num = str(epoch_list[i])
            sorted_elem = sorted(avg_dict[epoch_num])
            x, y, z = zip(*sorted_elem)
            if resample:
                xp = np.linspace(x[0], x[-1], 4*len(x))
                y = np.interp(xp, x, y)
                z = np.interp(xp, x, z)
                x = xp
            # Optional smoothing
            if smooth:
                y = savitzky_golay(y, filter_len, 2)
                z = savitzky_golay(z, filter_len, 2)
            y_u = np.array(y) + np.array(z)
            y_l = np.array(y) - np.array(z)
            # Optional normalization
            if norm:
                y = np.array(y)
                y_u = y_u - y.min()
                y_l = y_l - y.min()
                y = y - y.min()
                y_u = y_u / y.max()
                y_l = y_l / y.max()
                y = y / y.max()
            if std:
                ax[param_idx].fill_between(x, y_u, y_l, color=[1.0, 0.0, 0.0, 0.5])
            # Plot vals as line plot
            l = ax[param_idx].plot(x, y, label=log_name)
            # Show max
            if show_max:
                ax[param_idx].plot(x[y.argmax()], [y.max()], 'v', color='black')#, mfc='none')
            labels.extend(l)
            
        # Plot NIIRS
        plt.gca().set_prop_cycle(None)
        if niirs_list:
            for i, niirs in enumerate(niirs_list):
                if i == 0:
                    nax = ax[param_idx].twinx()
                    nax.set_ylabel('GIQE5 NIIRS', fontsize=axis_label_size)
                    nax.yaxis.set_label_coords(1.1, 0.3)
                if norm:
                    if param_name == 'focal_length':
                        nx = niirs[param_name]
                    elif param_name == 'f#':
                        nx = niirs[param_name]
                    elif param_name == 'q':
                        nx = niirs[param_name][:, 0]
                    ny = niirs['niirs']
                    #ny = np.array(ny)
                    #ny = ny - ny.min()
                    #ny = ny / ny.max()
                    l = nax.plot(nx, ny, label=niirs_label_list[i], color='black')
                    labels.extend(l)
                else:
                    if param_name == 'focal_length':
                        nx = niirs[param_name]
                        if True:
                            nax.set_ylim(0, 1.6)
                    elif param_name == 'f#':
                        nx = niirs[param_name]
                        if True:
                            nax.set_ylim(0, 1.6)
                    elif param_name == 'q':
                        nx = niirs[param_name][:, 0]
                        if True:
                            nax.set_ylim(0, 1.6)
                    ny = niirs['niirs']
                    
                    l = nax.plot(nx, ny, label=niirs_label_list[i], linestyle='--') 
                    #labels.extend(l)

                if show_max:
                    nax.plot(nx[np.argmax(ny)], [max(ny)], 'v', color='black')#, mfc='none')
            
        if norm:
            ax[param_idx].set_title('Normalized {} vs. {}'.format(title_dict[stat_name], title_dict[param_name]), fontsize=title_size)
            ax[param_idx].set_ylabel('Normalized {}'.format(axis_dict[stat_name]), fontsize=axis_label_size)
        else:
            print(param_name, title_dict[param_name])
            ax[param_idx].set_title('{} vs. {}'.format(title_dict[stat_name], title_dict[param_name]), fontsize=title_size)
            ax[param_idx].set_ylabel(axis_dict[stat_name], fontsize=axis_label_size)
            if show_random:
                ax[param_idx].axhline(y=lower_mean, color='black')
                ax[param_idx].text(0.825, lower_mean-0.01, 'Random Guess')
        ax[param_idx].set_xlabel(axis_dict[param_name], fontsize=axis_label_size)
            
        if param_name == 'f#':
            ax[param_idx].set_xlim(0, 25)
        if param_name == 'q':
            ax[param_idx].set_xlim(0, 1.5)
        elif param_name == 'focal_length':
            ax[param_idx].set_xlim(0, 1.4)

        labs = [l.get_label() for l in labels]
        if param_idx == (len(param_list) - 1):
            #ax[param_idx].legend(labels, labs, title=legend_title, bbox_to_anchor=legend_pos, frameon=True)
            legend = ax[param_idx].legend(labels, labs, title=legend_title, loc='lower right', frameon=True, fontsize=legend_label_size)
            legend.set_title(legend_title, prop={'size': legend_title_size})
        #plt.text(1.05, lower_mean, 'Random Guess')
        if norm:
            ax[param_idx].set_ylim(0, 1.1)
    
    plt.tight_layout()

    if save_file is not None:
        fig_path = os.path.join(FIG_DIR, save_file)
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')

    return fig, ax
    
label_dict = {
    3: '35 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    4: '1x10CV Pre Degrade Resize Dense D=0.05', 
    5: '1x10CV Pre Orig Crop Dense D=0.05',
    6: '1x10CV Pre Orig Resize Dense D=0.05',
    7: 'Orig Norm 1x10CV Scratch Degrade Crop Dense D=0.05',
    8: '1x10CV Scratch Degrade Resize Dense D=0.05',
    9: '1x10CV Pre Degrade Crop Squeeze D=0.05',
    10: '1x10CV Pre Degrade Resize Squeeze D=0.05',
    11: '5x2CV Pre Degrade Crop Dense D=0.05',
    12: '5x2CV Pre Degrade Resize Dense D=0.05',
    13: 'Ref Nearest 1x10CV Pre Degrade Crop Dense D=0.075',
    14: 'Ref Nearest 1x10CV Pre Degrade Resize Dense D=0.075',
    15: 'Rad Nearest 1x10CV Pre Degrade Resize Dense D=0.05',
    16: 'Rad Bilinear 1x10CV Pre Degrade Resize Dense D=0.05',
    17: 'Ref Bilinear 1x10CV Pre Degrade Resize Dense D=0.06',
    18: 'Ref Bilinear 1x10CV Pre Degrade Resize Dense D=0.065',
    19: 'Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.075',
    20: 'Ref Bilinear 1x10CV Pre Degrade Resize Dense D=0.075',
    21: 'Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.055',
    22: 'Ref Bilinear 1x10CV Pre Degrade Resize Dense D=0.07',
    23: 'ImageNet Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    24: 'Self Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    25: 'Imagenet Norm Ref Bilinear 1x10CV Scratch Degrade Crop Dense D=0.05',
    26: 'Self Norm Ref Bilinear 1x10CV Scratch Degrade Crop Dense D=0.05',
    27: '2 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    28: '3 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    29: '5 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    30: '10 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    31: '15 Orig Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    32: 'D=0.5',#'No Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    33: 'No Norm Ref Bilinear 1x10CV Scratch Degrade Crop Dense D=0.05 (20 Epochs)',
    34: '(No Noise) Self Norm Ref Bilinear 1x10CV Pre Degrade Crop Dense D=0.05',
    35: '2',
    36: '3', 
    37: '5',
    38: '10',
    39: '15',
    40: '2 5x2',
    41: '3 5x2', 
    42: '5 5x2',
    43: '10 5x2',
    44: '15 5x2',
    
    45: '2 5x2',
    
    46: 'D=0.55', 
    47: 'D=0.6',
    48: 'D=0.65',
    49: 'D=0.7',
    50: 'D=0.75',
}

if __name__=='__main__':
    if False:
        param_name = 'focal_length'
        stat_name = 'map'
        log_dir_list = ['/data/sepsense/experiments/exp3/logs', '/data/sepsense/experiments/exp32/logs']
        per_class_plot(log_dir_list, stat_name, param_name, norm=False, smooth=False, resample=False, save_file=None)
    if False:
        log_dir = '/data/sepsense/experiments/exp50/logs'
        num_folds = 10
        param_name = 'focal_length'
        stat_name = 'map'
        save_file = 'per_epoch_plots.png'
        epoch_dict = load_logs(log_dir, param_name, stat_name)
        per_epoch_plots(epoch_dict, num_folds, param_name, stat_name, save_file=None)
    if False:
        param_name = 'focal_length'
        stat_name = 'map'
        idx_list = [3, 4]
        label_dict = {
            3: 'Baseline Crop', 
            4: 'Baseline Resize'
        }
        save_file = 'crop_resize.png'
        compare_models(
            [label_dict[i] for i in idx_list], 
            ['/data/sepsense/experiments/exp{}/logs'.format(i) for i in idx_list], 
            [5, 5], 
            param_name, stat_name, std=False, resample=True, smooth=False, norm=False, niirs_list=False, 
            figsize=(8, 6),
            legend_pos=(1.0, 1.0),
            save_file=save_file)
    if False:
        # Scratch vs. Pretrained
        param_name = 'q'
        stat_name = 'map'
        idx_list = [32]
        label_dict = {
            32: 'Pre-Trained', 
            33: 'Scratch'
        }
        path_list = [
            '/data/sepsense/experiments/exp32/logs',
            '..//experiments/exp33/logs'
        ]
        save_file = 'crop_resize.png'
        compare_models(
            [label_dict[i] for i in idx_list], 
            path_list, 
            [5, 14], 
            param_name, stat_name, std=False, resample=True, smooth=False, norm=False, niirs_list=False,
            figsize=(8, 6),
            legend_pos=(1.0, 1.0),
            save_file='scratch_pretrained.png')
    if False:
        # Vary Aperture Diameter
        param_name = 'q'
        stat_name = 'map'
        idx_list = [32, 46, 47, 48, 49, 50]
        label_dict = {
            32: 'D=0.050',
            46: 'D=0.055',
            47: 'D=0.060',
            48: 'D=0.065',
            49: 'D=0.070',
            50: 'D=0.075', 
        }
        compare_models(
            [label_dict[i] for i in idx_list], 
            ['/data/sepsense/experiments/exp{}/logs'.format(i) for i in idx_list], 
            [5, 5, 5, 5, 5, 5],
            param_name, stat_name, std=False, resample=True, smooth=False, norm=False, niirs_list=False,
            log_interval=0.05,
            figsize=(8, 6),
            legend_pos=(1.0, 1.0),
            save_file='vary_diameter_q.png')
    if False:
        # Rad vs Ref
        param_name = 'focal_length'
        stat_name = 'map'
        idx_list = [0, 1]
        label_dict = {
            0: 'Radiance',
            1: 'Reflectance', 
        }
        save_file = 'crop_resize.png'
        compare_models(
            [label_dict[i] for i in idx_list], 
            ['/data/sepsense/experiments/exp{}/logs'.format(i) for i in idx_list],
            [5, 5], 
            param_name, stat_name, std=False, resample=True, smooth=False, norm=False, niirs_list=False,
            figsize=(8, 6),
            legend_pos=(1.0, 1.0),
            save_file=None)
    if False:
        # Baseline per-epoch for rAP
        log_dir = '/data/sepsense/experiments/exp55/logs'
        num_folds = 10
        param_name = 'focal_length'
        stat_name = 'map'
        per_epoch_plots(log_dir, num_folds, param_name, stat_name, save_file=None, 
            figsize=(8, 6), legend_pos=(1.3, 1.0), start_epoch=1)
    if False:
        # Scratch vs. Pretrained
        param_name = 'focal_length'
        stat_name = 'map'
        idx_list = [57, 56, '56_3', 56]
        label_list = [
            'Train 20', 
            'Test 15', 
            'General 15',
            'ImageNet 15'
        ]
        save_file = 'crop_resize.png'
        compare_models(
            [label_list[i] for i in range(len(idx_list))], 
            ['/data/sepsense/experiments/exp{}/logs'.format(i) for i in idx_list],
            [5, 5, 0, 0], 
            param_name, stat_name, std=False, resample=True, smooth=False, norm_list=[False, True], niirs_list=False,
            figsize=(2*8, 6),
            legend_pos=(1.0, 0.3),
            save_file=None)
    if False:
        # Scratch vs. Pretrained
        param_name = 'focal_length'
        stat_name = 'map'
        idx_list = ['56_3', '56_3']
        label_list = [
            'General 15',
            'General 15 old',
        ]
        dir_list = [
            '/data/sepsense/experiments/exp{}/logs'.format(idx_list[0]),
            '/data/sepsense/experiments/exp{}/.old_logs'.format(idx_list[1]),
        ]
        save_file = 'crop_resize.png'
        compare_models(
            [label_list[i] for i in range(len(idx_list))], 
            dir_list,
            [0, 0], 
            param_name, stat_name, std=False, resample=True, smooth=False, norm_list=[False, True], niirs_list=False,
            figsize=(2*8, 6),
            legend_pos=(1.0, 0.3),
            save_file=None)
    if True:
        # Baseline per-epoch for rAP
        log_dir = '/data/sepsense/experiments/exp55/logs'
        num_folds = 10
        param_name = 'focal_length'
        stat_list = ['acc', 'top3', 'map', 'auc']
        per_stat_plots(log_dir, num_folds, param_name, stat_list, save_file=None, 
            figsize=(8, 6), legend_pos=(1.0, 0.3), epoch_num=5, norm_list=[False])
    if False:
        # Scratch vs. Pretrained
        stat_name = 'map'
        idx_list = ['_dense_base', '_res_base', '_squeeze_base']
        label_list = [
            'DenseNet', 
            'ResNet', 
            'SqueezeNet',
        ]
        save_file = 'basic_plot.png'
        ### run code
        trial_dict = compare_basic(
            [label_list[i] for i in range(len(idx_list))], 
            ['/data/sepsense/experiments/exp{}/logs'.format(i) for i in idx_list],
            [5, 5, 5], 
            stat_list=['acc', 'top3', 'map', 'auc'], std=False, resample=True, smooth=False, norm_list=[False, True], niirs_list=False,
            save_file=save_file)
        print(trial_dict)
                
    # Show
    plt.show()
