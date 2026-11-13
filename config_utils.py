"""训练与测试共用的配置、设备及数据参数。"""
import argparse
import math
import os
from pathlib import Path

import yaml


RENDER_MODEL = "normalized_slice_v1"


def load_eanox_config(path):
    """读取 EANOX 配置，不触发渲染工程的参数检查，保持字典接口。"""
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f'配置键重复：{key}')
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(values, dict):
        raise ValueError('配置必须为字典')
    for key in ('runtime', 'data', 'model', 'sequence', 'losses', 'training'):
        if not isinstance(values.get(key), dict):
            raise ValueError(f'缺少 EANOX 配置分组：{key}')
    name = values.setdefault('dataset_name', 'acdc').lower()
    name = {'mm': 'mnms', 'mms': 'mnms', 'm&ms': 'mnms'}.get(name, name)
    if name not in ('acdc', 'mnms'):
        raise ValueError('dataset_name 仅支持 acdc 或 mnms')
    values['dataset_name'] = name
    values.setdefault('train_fraction', 0.7)
    values.setdefault('normalization_epsilon', 1e-4)
    if not 0 < values['train_fraction'] < 1:
        raise ValueError('训练和验证均需非空，train_fraction 必须在 (0,1) 内')
    if not math.isfinite(values['normalization_epsilon']) or values['normalization_epsilon'] <= 0:
        raise ValueError('normalization_epsilon 必须为有限正数')
    data = values['data']
    data.setdefault('dim', '2D')
    data.setdefault('augmentation', False)
    data.setdefault('pin_memory', True)
    if data['dim'] not in ('2D', '3D'):
        raise ValueError('data.dim 必须为 2D 或 3D')
    if data['augmentation']:
        raise ValueError('当前 loader 未执行实际数据增强，data.augmentation 必须为 false')
    if data['batch_size'] < 1 or data['num_workers'] < 0:
        raise ValueError('batch_size 必须为正数，num_workers 必须非负')
    paths = values.get('dataset_paths', {})
    if not isinstance(paths, dict):
        raise ValueError('dataset_paths 必须为数据集到目录的映射')
    values['dataset_paths'] = {
        key: str((config_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve())
        for key, value in paths.items()
    }
    return values


def load_config(path):
    # 拒绝重复键，避免同一实验参数被后面的值静默覆盖。
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f'配置键重复：{key}')
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config_path = Path(path).resolve()
    with config_path.open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=UniqueLoader)
    values.setdefault('dataset_name', 'acdc')
    values.setdefault('render_model', RENDER_MODEL)
    if not 0 <= values['train_fraction'] <= 1 or values['reference_phase'] < 0:
        raise ValueError('病例划分比例或参考相位非法')
    if len(values['bounds']) != 3 or any(value <= 0 for value in values['bounds']):
        raise ValueError('bounds 必须为三个正半边长')
    if 'iters' in values:
        if values['render_model'] != RENDER_MODEL:
            raise ValueError('Stage1 仅支持 normalized_slice_v1 渲染模型')
        obsolete = set(values) & {'response_mode', 'psf_sigma', 'psf_std_vector',
                                  'kernel_cutoff', 'cutoff_transition', 'covariance_jitter'}
        if obsolete:
            raise ValueError(f'新库不支持这些旧渲染配置，请移除：{sorted(obsolete)}')
        for key in ('iters', 'checkpoint_interval', 'loss_plot_interval', 'status_interval',
                    'real_slices_per_iter', 'num_pts', 'interp_step_size',
                    'stage2_iters', 'stage2_save_interval', 'stage2_loss_plot_interval',
                    'init_edge_sigma', 'init_edge_truncate', 'init_fine_sigma_pixels', 'init_coarse_sigma_pixels',
                    'init_normal_sigma_spacing', 'init_normal_floor_pixels', 'init_ridge_relative'):
            if values[key] <= 0:
                raise ValueError(f'{key} 必须大于零')
        if not (0 <= values['init_uniform_fraction'] <= 1
                and 0 <= values['init_foreground_fraction'] <= 1-values['init_uniform_fraction']):
            raise ValueError('初始化均匀/前景比例非负且总和不能超过 1')
        quantiles = values['init_sampling_quantiles']
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError('初始化采样分位数必须为 [low,high]，满足 0 <= low < high <= 1')
        for key in ('init_foreground_threshold','init_edge_quantile','init_coarse_fraction'):
            if not 0 <= values[key] <= 1:
                raise ValueError(f'{key} 必须位于 [0,1]')
        for key in ('init_foreground_dilation','init_fit_steps','init_support_retries'):
            if not isinstance(values[key],int) or values[key] < 0:
                raise ValueError(f'{key} 必须为非负整数')
        if not (0 <= values['init_grid_jitter'] < .5 and 0 < values['init_normal_jitter'] <= .5
                and 0 < values['init_cg_rtol'] < 1 and values['init_max_axis_ratio'] >= 1):
            raise ValueError('初始化抖动、CG 容差或轴比配置非法')
        if not 1 <= values['structural_min_points'] <= values['num_pts'] <= values['max_num_pts']:
            raise ValueError('结构容量需要 1 <= min_points <= num_pts <= max_num_pts')
        if not 0 < values['structural_split_fraction'] < 1:
            raise ValueError('structural_split_fraction 必须在 (0,1) 内')
        if values['structural_algorithm'] != 'batch_normalized_v1':
            raise ValueError('当前 Stage1 仅支持 structural_algorithm=batch_normalized_v1')
        if values['structural_prune_peak'] != 0:
            raise ValueError('归一化平均不能按低强度剪枝，structural_prune_peak 必须为 0')
        for key in ('structural_grad_threshold', 'structural_grad_multiplier',
                    'structural_clone_jitter', 'structural_prune_peak'):
            if not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f'{key} 必须为有限非负数')
        if not (isinstance(values['structural_min_observations'], int)
                and values['structural_min_observations'] > 0):
            raise ValueError('structural_min_observations 必须为正整数')
        if not math.isfinite(values['structural_split_sigma_pixels']) or values['structural_split_sigma_pixels'] <= 0:
            raise ValueError('structural_split_sigma_pixels 必须为有限正数')
        if not 0 <= values['structural_growth_fraction'] <= 1:
            raise ValueError('structural_growth_fraction 必须位于 [0,1]')
        if (values['lambda_l1'] < 0 or values['lambda_ssim'] < 0
                or values['lambda_l1']+values['lambda_ssim'] <= 0):
            raise ValueError('训练需要非负且不全为零的 L1/SSIM 权重')
        if values['ssim_window_size'] <= 0 or values['ssim_window_size'] % 2 != 1:
            raise ValueError('SSIM 需要正奇数窗口')
        if not (0 <= values['interpolation_probability'] <= 1
                and 0 <= values['interpolation_alpha_min'] <= values['interpolation_alpha_max'] <= 1):
            raise ValueError('插层概率或 alpha 范围非法')
    # 相对路径以配置文件为基准，不依赖启动命令所在目录。
    for key, value in values.items():
        if key.endswith(('_path', '_dir', '_root')) and isinstance(value, str) and value:
            value = os.path.expandvars(os.path.expanduser(value))
            values[key] = str((config_path.parent / value).resolve())
    values['dataset_paths'] = {
        name: str((config_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve())
        for name, value in values['dataset_paths'].items()
    }
    return argparse.Namespace(**values)


def setup_runtime(args):
    import random
    import numpy as np
    import torch

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('当前渲染扩展要求 CUDA 设备')
    torch.cuda.set_device(device)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def dataset_options(args):
    # 主训练/重建按完整病例工作；固定使用加载器的 [S,T,H,W] 输出。
    name = getattr(args, 'dataset_name', 'acdc').lower()
    name = {'mm':'mnms','mms':'mnms','m&ms':'mnms'}.get(name,name)
    return dict(
        datasets=name, data_path=args.dataset_paths[name], dim='3D',
        train_fraction=args.train_fraction,
        normalization_epsilon=args.normalization_epsilon, config=args,
    )


def render_options(args):
    return dict(bounds=args.bounds, raster_options={
        'background': args.background,
        'debug': args.raster_debug,
    })


def ssim_options(args):
    return dict(window_size=args.ssim_window_size, sigma=args.ssim_sigma,
                k1=args.ssim_k1, k2=args.ssim_k2, data_range=args.data_range)
