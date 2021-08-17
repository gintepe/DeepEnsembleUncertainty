from configuration import Configuration
from util import *

import numpy as np
import torch
import matplotlib.pyplot as plt
import re

def get_nums(name):
    r = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
    
    return re.findall(r, name)

def convert_name(name, param_only=False):
    if "single" in name:
        return 'Single'
    if 'ens-5' in name:
        return 'DE'
    if 'mcdrop' in name:
        return 'MC Dropout'
    nums = get_nums(name)
    if 'ncens' in name:
        if 'd' not in name:
            param = f'$\lambda$={nums[0]}'
        else:
            # param = f'$\lambda$={nums[0]}, d={nums[1]}'
            param = f'$\lambda$={nums[0]}\nd={nums[1]}'
        return param if param_only else f'NCL-DE ({param})'
    if 'ceens' in name:
        if 'd' not in name:
            param = f'$\lambda$={nums[0]}'
        else:
            # param = f'$\lambda$={nums[0]}, d={nums[1]}'
            param = f'$\lambda$={nums[0]}\nd={nums[1]}'
        return param if param_only else f'CE-DE ({param})'
    ans = ''
    if 'ens-loss' in name:
        ans += '$L_{ENS}$'
    elif 'sum-loss' in name:
        ans += '$L_{SUM}$'
    # if 'sparse' in name:
    #     ans += 'Sparse'
    # else: 
    #     ans += 'Dense'
    if 'mcd' not in name:
        if 'gate-same' in name:
            ans += ', gate: Exp.'
        elif 'gate-conv' in name:
            ans += ', gate: Conv.'
        elif 'gate-simple' in name:
            ans += ', gate: MLP'
    else:
        if 'gate-mcd_simple' in name:
            ans += ', gate: MLP'
        elif 'gate-mcd_lenet' in name or 'gate-mcd_resnet' in name:
            ans += ', gate: Exp.'
        elif 'gate-mcd_conv' in name:
            ans += ', gate: Conv.'
        ans += f', $p=${float(get_nums(name)[-1])*-1}'
    return ans



METRIC_NAMES = [('accuracies', 'accuracy'), ('confidences', 'confidence'), ('disagreements', 'disagreement'), ('eces', 'ECE'), ('nlls', 'NLL'), ('briers', 'Brier')]
METRIC_LABELS = ['Accuracy', 'Confidence', 'Mean Disagreement', 'ECE', 'NLL', 'Brier Score']

class MNISTRunData:
    def __init__(self,path):

        res_dict, rot_np, tran_np = load_res_mnist(path)
        self.args = Configuration.from_json(f'{path}/args.json')

        self.name = self.args.run_name
        self.path = path

        self.res_dict = res_dict
        self.rot_np = rot_np
        self.tran_np = tran_np

    def get_metric(self, metric_name_shifted, metric_name_val, shift='rot', include_val=True):
        if shift == 'rot':
            vals = self.rot_np[metric_name_shifted]
            shifts = list(np.arange(0, 181, 15))
        else:
            vals = self.tran_np[metric_name_shifted]
            shifts = list(np.arange(0, 29, 2))

        if include_val:
            res = np.zeros(vals.shape[0] + 1)
            res[0] = self.res_dict[f'val_{metric_name_val}']
            res[1:] = vals
        else:
            res = vals

        return res, np.zeros_like(res), shifts

class MNISTMultiRunData:
    def __init__(self, runs):

        self.name = runs[0].name
        self.runs = runs

    def get_metric(self, metric_name_shifted, metric_name_val, shift='rot', include_val=True):
        
        ress = np.stack([run.get_metric(metric_name_shifted, metric_name_val, shift, include_val)[0] for run in self.runs], axis=0)
        means = np.mean(ress, axis=0)
        stds = np.std(ress, axis=0)
        shifts = self.runs[0].get_metric(metric_name_shifted, metric_name_val, shift, include_val)[2]
        return means, stds, shifts

colors = ['#2a9d8f', '#e9c46a', '#e76f51', '#9b2226', '#540d6e', '#b6465f', '#005f73', '#264653']
# colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51', '#9b2226', '#540d6e', '#b6465f', '#005f73']
# colors = [ '#2a9d8f', '#e76f51', '#6a040f', '#540d6e', '#b6465f', '#005f73', '#264653', '#e9c46a', 'green']
# colors = [ '#2a9d8f', '#e76f51', '#6a040f', '#264653', '#e9c46a', '#540d6e', '#b6465f', '#014f86']
light_green = '#2a9d8f'
orange = '#e76f51'
dark_greeen = '#264653'

def set_axis_props(ax, x_label, y_label, fontsize):
    ax.set_xlabel(x_label, fontsize=fontsize, color=dark_greeen)
    ax.set_ylabel(y_label, fontsize=fontsize, color=dark_greeen)
    plt.setp(ax.spines.values(), color=dark_greeen)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.tick_params(axis='x', labelsize=fontsize-2)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=dark_greeen) 
    plt.setp(ax.spines.values(), color=dark_greeen)

def plot_single_metric_mnist(ax, rds, metric_name_shifted, metric_name_val, shift='rot', include_val=True, x_label='Shift', y_label='metric', fontsize=14, legend=False):
    for rd, col in zip(rds, colors[1:]):
        vals, _, shifts = rd.get_metric(metric_name_shifted, metric_name_val, shift, include_val)
        labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
        proxy_vals = np.arange(len(labels))
        ax.plot(proxy_vals, vals, color=col, label=rd.name)

    ax.set_xticks(proxy_vals)
    ax.set_xticklabels(labels, fontsize=fontsize,color=dark_greeen)
    ax.set_xlabel(x_label, fontsize=fontsize, color=dark_greeen)
    ax.set_ylabel(y_label, fontsize=fontsize, color=dark_greeen)
    if legend:
        l = ax.legend(fontsize=fontsize)
        plt.setp(l.get_texts(), color=dark_greeen)

    

def plot_single_metric_error_mnist(
    ax, 
    rds, 
    metric_name_shifted, 
    metric_name_val, 
    shift='rot', 
    include_val=True, 
    x_label='Shift', 
    y_label='metric', 
    fontsize=14, 
    legend=False, 
    legend_param_only=False,
    legend_in_layout=True,
    legend_bbox=(1,1),
    legend_col=1,
    error_alpha=0.2,
    legend_fontsize=14,
    sparse_ticks=None,
    legend_title=None,
    extra_rd = None,
    leg_names=None,
):
    for rd, col, i in zip(rds, colors, range(len(rds))):
        means, stds, shifts = rd.get_metric(metric_name_shifted, metric_name_val, shift, include_val)
        labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
        proxy_vals = np.arange(len(labels))

        ax.plot(proxy_vals, means, color=col, label=convert_name(rd.name, legend_param_only) if leg_names is None else leg_names[i])
        ax.fill_between(proxy_vals, means - 2*stds, means + 2*stds, color=col, alpha=error_alpha)
    
    if extra_rd:
        ax.plot(proxy_vals, extra_rd.get_metric(metric_name_shifted, metric_name_val, shift, include_val)[0], color=dark_greeen, ls='--', lw=2)

    if legend:
        l = ax.legend(fontsize=legend_fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col, title=legend_title)
        
        plt.setp(l.get_texts(), color=dark_greeen)
        l.set_in_layout(legend_in_layout)

    ax.set_xticks(proxy_vals)
    if sparse_ticks is not None:
        ax.set_xticklabels([label if i%sparse_ticks==0 else '' for i, label in enumerate(labels)], fontsize=fontsize-2, color=dark_greeen)
    else:
        ax.set_xticklabels(labels, fontsize=fontsize-2, color=dark_greeen)  
    set_axis_props(ax, x_label, y_label, fontsize)

def plot_single_metric_error_bar_mnist(
    ax, 
    rds, 
    metric_name_shifted, 
    metric_name_val, 
    shift='rot', 
    include_val=True, 
    x_label='method', 
    y_label='metric', 
    fontsize=14, 
    legend=False, 
    legend_col=1, 
    legend_bbox=(1.05, 1), 
    legend_in_layout=False, 
    picks=[0], 
    fig_for_legend=None,
    title=None,
    legend_param_only=False,
    alpha=1,
    leg_names=None,
    ):
    bar_level, err, cols, methods = [], [], [], []
    for pick in picks:
        for rd, col in zip(rds, colors):
            means, stds, shifts = rd.get_metric(metric_name_shifted, metric_name_val, shift, include_val)
            labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
            bar_level.append(means[pick])
            err.append(2*stds[pick])
            cols.append(col)
            if len(methods) < len(rds):
                methods.append(rd.name)
    n = len(methods)

    # proxy_loc = np.concatenate([np.arange((n+1)*i + 1, (n+1)*i + n+1) for i, pick in enumerate(picks)])
    proxy_loc = np.concatenate([np.arange((n+1)*i, (n+1)*i + n) for i, pick in enumerate(picks)])

    ax.bar(proxy_loc, bar_level, color=cols, width=1, label=methods,edgecolor='white', alpha=alpha)
    ax.errorbar(proxy_loc, bar_level, yerr=err,
             fmt='o', capsize=2, ms=2, color=dark_greeen)

    ax.set_xticks(proxy_loc)
    # ax.set_xticklabels(methods, fontsize=fontsize,color=dark_greeen)
    x_labels=[]
    if len(x_label) > 0:
        x_labels = [""]*len(proxy_loc)
        for i, pick in enumerate(picks):
            x_labels[(n)*i + n // 2] = labels[pick]
    ax.set_xticklabels(x_labels, fontsize=fontsize,color=dark_greeen)

    l=None
    if legend:
        handles = [plt.Rectangle((0,0),1,1, color=cols[i], alpha=alpha) for i in range(len(methods))]
        names = [convert_name(m, param_only=legend_param_only) for m in methods] if leg_names is None else leg_names
        if fig_for_legend is not None:
            l = fig_for_legend.legend(handles, names, fontsize=fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col)
        else:
            l = ax.legend(handles, names, fontsize=fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col)
        
        plt.setp(l.get_texts(), color=dark_greeen)
        l.set_in_layout(legend_in_layout)

    set_axis_props(ax, x_label, y_label, fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title is not None:
        ax.set_title(title, color=dark_greeen, fontsize = fontsize+2)
    
    return labels, l

        

class CIFARRunData:
    def __init__(self,path):

        res_dict, res_np = load_res_cifar(path)
        self.args = Configuration.from_json(f'{path}/args.json')
        self.path = path
        self.name = self.args.run_name

        self.res_dict = res_dict
        self.res_np = res_np

    def get_metric_avg(self, metric_name_shifted, metric_name_val, include_val=True):
        vals = np.mean(self.res_np[metric_name_shifted], axis=0).flatten()
        shifts = list(np.arange(0, 6))
        res1 = np.zeros(vals.shape[0] + 1)
        res1[0] = self.res_dict[f'test_{metric_name_val}']
        res1[1:] = vals

        if include_val:
            res = np.zeros(res1.shape[0] + 1)
            res[0] = self.res_dict[f'val_{metric_name_val}']
            res[1:] = res1
        else:
            res = res1

        return res, shifts

    def get_metric_shifted(self, metric_name_shifted, metric_name_val):
        vals = self.res_np[metric_name_shifted]
        return vals

class CIFARMultiRunData:
    def __init__(self, runs):

        self.name = runs[0].name
        self.runs = runs

    def get_metric_avg(self, metric_name_shifted, metric_name_val, include_val=True):
        
        ress = np.stack([run.get_metric_avg(metric_name_shifted, metric_name_val, include_val)[0] for run in self.runs], axis=0)
        means = np.mean(ress, axis=0)
        stds = np.std(ress, axis=0)
        shifts = self.runs[0].get_metric_avg(metric_name_shifted, metric_name_val, include_val)[1]
        return means, stds, shifts

    def get_metric_shifted(self, metric_name_shifted, metric_name_val):
        vals = self.runs[0].res_np[metric_name_shifted]
        return vals

def plot_single_metric_cifar(ax, rds, metric_name_shifted, metric_name_val, include_val=True, x_label='Shift', y_label='metric', fontsize=14, legend=False):
    for rd, col in zip(rds, colors):
        vals, shifts = rd.get_metric_avg(metric_name_shifted, metric_name_val, include_val)
        labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
        proxy_vals = np.arange(len(labels))
        ax.plot(proxy_vals, vals, color=col, label=rd.name)

    ax.set_xticks(proxy_vals)
    ax.set_xticklabels(labels, fontsize=fontsize,color=dark_greeen)
    set_axis_props(ax, x_label, y_label, fontsize)
    if legend:
        l = ax.legend(fontsize=fontsize)
        plt.setp(l.get_texts(), color=dark_greeen)

def plot_single_metric_error_cifar(
    ax, 
    rds, 
    metric_name_shifted, 
    metric_name_val, 
    include_val=True, 
    x_label='Shift Intensity', 
    y_label='metric', 
    fontsize=14, 
    legend=False, 
    legend_param_only=False,
    legend_in_layout=True,
    legend_bbox=(1,1),
    legend_col=1,
    error_alpha=0.2,
    legend_fontsize=14,
    sparse_ticks=None,
    legend_title=None,
    extra_rd = None,
    leg_names=None,
):
    for rd, col, i in zip(rds, colors, range(len(rds))):
        means, stds, shifts = rd.get_metric_avg(metric_name_shifted, metric_name_val, include_val)
        labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
        proxy_vals = np.arange(len(labels))

        ax.plot(proxy_vals, means, color=col, label=convert_name(rd.name, legend_param_only) if leg_names is None else leg_names[i])
        ax.fill_between(proxy_vals, means - 2*stds, means + 2*stds, color=col, alpha=error_alpha)

    if extra_rd:
        ax.plot(proxy_vals, extra_rd.get_metric_avg(metric_name_shifted, metric_name_val, include_val)[0], color=dark_greeen, ls='--', lw=2)

    if legend:

        l = ax.legend(fontsize=legend_fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col, title=legend_title)
        
        plt.setp(l.get_texts(), color=dark_greeen)
        l.set_in_layout(legend_in_layout)

    ax.set_xticks(proxy_vals)
    if sparse_ticks is not None:
        ax.set_xticklabels([label if i%sparse_ticks==0 else '' for i, label in enumerate(labels)], fontsize=fontsize-2, color=dark_greeen)
    else:
        ax.set_xticklabels(labels, fontsize=fontsize-2, color=dark_greeen)  
    set_axis_props(ax, x_label, y_label, fontsize)


def plot_single_metric_error_bar_cifar(
    ax, 
    rds, 
    metric_name_shifted, 
    metric_name_val, 
    include_val=True, 
    x_label='Shift', 
    y_label='metric', 
    fontsize=14, 
    legend=False, 
    legend_col=1, 
    legend_bbox=(1.05, 1), 
    legend_in_layout=False, 
    picks=range(0, 7),
    alpha=1,
    leg_names = None,
    ):
    bar_level, err, cols, methods = [], [], [], []
    for pick in picks:
        for rd, col in zip(rds, colors):
            means, stds, shifts = rd.get_metric_avg(metric_name_shifted, metric_name_val, include_val)
            labels = (['Val'] if include_val else []) + ['Test'] + shifts[1:]
            bar_level.append(means[pick])
            err.append(2*stds[pick])
            cols.append(col)
            if len(methods) < len(rds):
                methods.append(rd.name)
    n = len(methods)

    # proxy_loc = np.concatenate([np.arange((n+1)*i + 1, (n+1)*i + n+1) for i, pick in enumerate(picks)])
    proxy_loc = np.concatenate([np.arange((n+1)*i, (n+1)*i + n) for i, pick in enumerate(picks)])

    ax.bar(proxy_loc, bar_level, color=cols, width=1, label=methods,edgecolor='white', alpha=alpha)
    ax.errorbar(proxy_loc, bar_level, yerr=err,
             fmt='o', capsize=2, ms=2, color=dark_greeen)

    ax.set_xticks(proxy_loc)
    # ax.set_xticklabels(methods, fontsize=fontsize,color=dark_greeen)
    x_labels=[]
    if len(x_label) > 0:
        x_labels = [""]*len(proxy_loc)
        for i, pick in enumerate(picks):
            x_labels[(n)*i + n // 2] = labels[pick]
    ax.set_xticklabels(x_labels, fontsize=fontsize,color=dark_greeen)


    if legend:
        handles = [plt.Rectangle((0,0),1,1, color=cols[i], alpha=alpha) for i in range(len(methods))]
        names = [convert_name(m) for m in methods] if leg_names is None else leg_names
        l = ax.legend(handles, names, fontsize=fontsize-1, bbox_to_anchor=legend_bbox, ncol=legend_col)
        plt.setp(l.get_texts(), color=dark_greeen)
        l.set_in_layout(legend_in_layout)
    
    set_axis_props(ax, x_label, y_label, fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    return labels

def plot_single_metric_boxplot_cifar(
    ax, 
    rds, 
    metric_name_shifted, 
    metric_name_val, 
    include_id=True, 
    x_label='Shift', 
    y_label='metric', 
    fontsize=14, 
    legend=False, 
    legend_col=1, 
    legend_bbox=(1.05, 1), 
    alpha=0.9, 
    legend_in_layout=False,
    fig=None,
    picks=range(0, 5),
    leg_names=None,
    ):
    metric_data, err, cols, methods = [], [], [], []

    ax.grid(axis='y', color=dark_greeen, alpha=0.35, linestyle='--')

    
    # ----------------v1-------------
    # for pick in picks:
    #     for rd, col in zip(rds, colors[1:]):
    #         vals = rd.get_metric_shifted(metric_name_shifted, metric_name_val)
    #         labels = range(1, 6)
    #         metric_data.append(vals[:, pick])
    #         cols.append(col)
    #         if len(methods) < len(rds):
    #             methods.append(rd.name)
    # n = len(methods)
    # proxy_loc = np.concatenate([np.arange((n+1)*i + 1, (n+1)*i + n+1) for i, pick in enumerate(picks)])
    # for val, col, loc in zip(metric_data, cols, proxy_loc):
    #     bp = ax.boxplot(val, patch_artist=True, widths=0.75, positions=[loc], boxprops={'color':dark_greeen, 'facecolor':col, 'alpha':0.6}, medianprops={'color':col}, whiskerprops={'color':dark_greeen}, capprops={'color':dark_greeen}, showfliers=False)
    
    # ------------------v2-------------
    for rd, col in zip(rds, colors):
        vals = rd.get_metric_shifted(metric_name_shifted, metric_name_val)
        if include_id:
            id_vals = list(rd.get_metric_avg(metric_name_shifted, metric_name_val, include_val=True)[0][:2])
            vals = id_vals + list([vals[:, pick] for pick in picks])
            metric_data.append(vals)
        else:
            metric_data.append(vals[:, picks])

        labels = (["Val", "Test"] if include_id else []) + list([pick+1 for pick in picks ])
        cols.append(col)
        if len(methods) < len(rds):
            methods.append(rd.name)
    n = len(methods)

    proxy_loc = [np.arange((n+1)*i + 1, (n+1)*i + n + 1) for i in range(len(picks)+2 if include_id else len(picks))]
    
    for val, col, i in zip(metric_data, cols, range(n)):
        bp = ax.boxplot(
            val, 
            patch_artist=True, 
            widths=0.9, 
            whis=(0, 100),
            positions=[loc[i] for loc in proxy_loc], 
            boxprops={'color':dark_greeen, 'facecolor':col, 'alpha':alpha}, 
            medianprops={'color':col, 'lw':3} if include_id else {'color':dark_greeen}, 
            whiskerprops={'color':dark_greeen}, 
            capprops={'color':dark_greeen}, 
            showfliers=False)
    proxy_loc = np.concatenate(proxy_loc)

    # -----------------shared---------------------
    ax.set_xticks(proxy_loc)
    # ax.set_xticklabels(methods, fontsize=fontsize,color=dark_greeen)
    x_labels=[]
    if len(x_label) > 0:
        x_labels = [""]*len(proxy_loc)
        for i, l in enumerate(labels):
            x_labels[(n)*i + n // 2] = l
    ax.set_xticklabels(x_labels, fontsize=fontsize-1,color=dark_greeen)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    l=None
    if legend:
        handles = [plt.Rectangle((0,0),1,1, color=cols[i], alpha=alpha) for i in range(len(methods))]
        names = [convert_name(m) for m in methods] if leg_names is None else leg_names
        if fig is None:
            l = ax.legend(handles, names, fontsize=fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col)
        else:
            l = fig.legend(handles, names, fontsize=fontsize, bbox_to_anchor=legend_bbox, ncol=legend_col)
        plt.setp(l.get_texts(), color=dark_greeen)
        l.set_in_layout(legend_in_layout)

    set_axis_props(ax, x_label, y_label, fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    return labels, l
