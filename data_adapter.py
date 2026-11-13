"""将用户提供的 NPZ 加载器连接到 EANOX，不修改其读取和预处理逻辑。"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def collate_records(records):
    """保留变长病例与稀疏标签元数据，不强制拼接时间轴。"""
    return records


def seed_worker(worker_id):
    """在工作进程中同步 Python 与 NumPy 的随机状态。"""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(config, split):
    """默认按切片采样；完整病例模式仍逐切片运行模型。"""
    from dataloader import dataset
    if tuple(config['model']['input_size']) != (128, 128):
        raise ValueError('当前真实 loader 固定输出 128×128，请设置 model.input_size: [128,128]')
    name = config['dataset_name']
    if name not in config['dataset_paths']:
        raise ValueError(f'dataset_paths 中未配置 {name} 的目录')
    data = config['data']
    current = dataset(mode=split, datasets=name, dim=data['dim'],
                      data_path=config['dataset_paths'][name],
                      train_fraction=config['train_fraction'],
                      normalization_epsilon=config['normalization_epsilon'],
                      align=False, augmentation=False, require_labels=False)
    if not len(current):
        raise ValueError(f'{split} 数据为空，请检查目录、病例数和 train_fraction')
    return DataLoader(current, batch_size=data['batch_size'], shuffle=split == 'train',
                      num_workers=data['num_workers'], pin_memory=data['pin_memory'],
                      drop_last=False, collate_fn=collate_records, worker_init_fn=seed_worker)


def get_images(batch, config, device):
    """返回 [切片数,1,T,H,W]，不再次缩放、归一化或筛选标注帧。"""
    if not isinstance(batch, dict):
        raise ValueError('单个样本必须为字典')
    key = config['data']['image_key']
    # 原模型格式优先；同时支持 loader 原生的 cmr 键。
    if key in batch:
        images = batch[key]
        if key != 'cmr' or (images.ndim == 5 and images.shape[1] == 1 and 'info' not in batch):
            if images.ndim != 5 or images.shape[1] != 1:
                raise ValueError('模型格式图像必须为 [B,1,T,H,W]')
            return images.to(device=device, dtype=torch.float32)
    elif 'cmr' in batch:
        images = batch['cmr']
    else:
        raise ValueError(f'样本需包含 {key} 或 loader 原生的 cmr')
    info = batch.get('info', {})
    layout = info.get('cmr_layout')
    if isinstance(layout, (list, tuple)):
        if len(set(layout)) != 1:
            raise ValueError('同批次图像布局不一致')
        layout = layout[0]
    if layout is None:
        raise ValueError('原生 cmr 输入需要 info.cmr_layout，避免混淆时间与切片轴')
    if layout == 'THW':
        if images.ndim == 3:
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.ndim == 4:
            images = images.unsqueeze(1)
        else:
            raise ValueError('THW 输入必须为 [T,H,W] 或 [B,T,H,W]')
    elif layout == 'STHW':
        if images.ndim == 5 and images.shape[0] == 1:
            images = images[0]
        if images.ndim != 4:
            raise ValueError('完整病例输入必须为 [S,T,H,W]；请勿强制拼接多个病例')
        images = images.unsqueeze(1)
    else:
        raise ValueError(f'未知 cmr_layout：{layout}')
    if images.shape[0] < 1 or images.shape[2] < 2:
        raise ValueError('病例至少需要一个切片和两个时间帧')
    return images.to(device=device, dtype=torch.float32)


def data_protocol(config, loader):
    """记录患者集合及预处理协议，恢复训练时防止误换数据集或划分。"""
    current = getattr(loader, 'dataset', None)
    if current is None:
        return None
    return {'dataset': current.datasets, 'data_path': str(current.data_pth),
            'patients': list(current.patients), 'dim': current.dim,
            'train_fraction': config['train_fraction'],
            'normalization_epsilon': current.normalization_epsilon,
            'preprocessing': '用户 loader：完整病例归一化，128×128 面内缩放'}
