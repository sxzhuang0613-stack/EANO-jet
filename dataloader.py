import json
import os
import random
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

class data_augmentation():

    def init(self,):
        self.prob = random.random()
        self.random_rotation = transforms.RandomRotation(degrees=(-180, 180))

    def RandomFlip(self, images, labels=None):
        
        if random.random() < 0.3:
            images = torch.flipud(images)
            labels = torch.flipud(labels) if labels is not None else None
        elif random.random() > 0.7:
            images = torch.fliplr(images)
            labels = torch.fliplr(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images
    
    def RandomRotate(self, images, labels=None):

        if self.prob < 0.8:
            images = self.random_rotation(images)
            labels = self.random_rotation(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images

class dataset(Dataset):

    def __init__(self, mode='train', datasets='acdc', align=False,
                 augmentation=False, data_path=None, train_fraction=0.7,
                 normalization_epsilon=1e-4, config=None, dim='3D', require_labels=False):
        # 数据集差异集中在此处；后续可增加字段、轴序和类别映射的适配。
        datasets = datasets.lower()
        datasets = {'mm': 'mnms', 'mms': 'mnms', 'm&ms': 'mnms'}.get(datasets, datasets)
        label_maps = {'acdc': [0, 1, 2, 3], 'mnms': [0, 3, 2, 1]}
        if datasets not in label_maps:
            raise ValueError('datasets 当前仅支持 acdc 和 mnms（或 mm、mms、m&ms）')
        if dim not in ('2D', '3D'):
            raise ValueError('dim 必须为 2D 或 3D')
        if not 0 <= train_fraction <= 1 or normalization_epsilon <= 0:
            raise ValueError('train_fraction 必须在 [0,1] 内，normalization_epsilon 必须大于零')
        self.augmentation = augmentation
        self.datasets = datasets
        self.dim = dim
        self.label_map = torch.tensor(label_maps[datasets], dtype=torch.long)
        self.normalization_epsilon = normalization_epsilon
        if data_path is None:
            from config_utils import load_config
            config = config or load_config('config.yaml')
            data_path = config.dataset_paths[datasets]
        self.data_pth = data_path
        self.config = config
        patients = sorted(f for f in os.listdir(data_path) if f.endswith('.npz'))
        self.case_info = {}
        for patient in patients:
            with np.load(os.path.join(data_path, patient), allow_pickle=False) as arrays:
                info = json.loads(arrays['info'].item())
            if info['schema']['array_axes'] != {
                'cmr': ['T', 'Z', 'Y', 'X'], 
                'labels': ['phase', 'Z', 'Y', 'X'],
            }:
                raise ValueError(f'{patient}: 不支持的 NPZ 轴序')
            self.case_info[patient] = info
        split = int(len(patients) * train_fraction)
        if mode == 'train':
            self.patients = patients[:split]
        elif mode == 'val':
            self.patients = patients[split:]
        elif mode == 'all':
            self.patients = patients
        elif mode == 'test':
            self.patients = [
                p for p in patients
                if self.case_info[p]['case']['original_split'].lower() in ('testing', 'test')
            ]
        else:
            raise ValueError('mode 必须为 train、val、test 或 all')
        # 先划分再过滤，不让标签缺失改变其余病例的集合归属。
        if require_labels or mode == 'test':
            self.patients = [p for p in self.patients if self.has_valid_labels(p)]
        self.samples = []
        for patient in self.patients:
            if dim == '2D':
                slices = self.case_info[patient]['processing']['output']['cmr_shape'][1]
                self.samples.extend((patient, s) for s in range(slices))
            else:
                self.samples.append((patient, None))
        print(f'{len(self.patients)} cardiac cases.')

    def has_valid_labels(self, patient):
        info = self.case_info[patient]
        annotation, case = info['annotation'], info['case']
        phases = info['processing']['output']['labels_shape'][0]
        return (phases > 0 and annotation.get('has_labels', True)
                and case.get('official_label_status', 'labeled') == 'labeled'
                and annotation.get('phase_metadata_valid', True))

    def normalize_data(self, img_np, mode='cmr'):

        cmin = torch.min(img_np)
        cmax = torch.max(img_np)
        img_np = (img_np - cmin) / (cmax - cmin + self.normalization_epsilon)

        if mode == 'tmr':
            if len(img_np.shape) == 3:
                meanv = torch.mean(img_np, dim=(1,2))
                meanv = meanv.view(-1,1,1)
            if len(img_np.shape) == 4:
                s, t, h, w = img_np.shape
                meanv = torch.mean(img_np, dim=(2,3))
                meanv = meanv.view(-1,1,1).reshape(s, t, 1, 1)
            img_np = img_np / (2 * meanv)
            img_np = torch.clamp(img_np, -1, 1)
        return img_np

    def split_masks(self, masks):
        unique = [x for x in torch.unique(masks) if x != 0]
        Masks = []
        for u in unique:
            mask = torch.zeros_like(masks)
            mask[masks == u] = 1
            Masks.append(mask)
        return Masks

    def __getitem__(self, idx):
        
        if self.augmentation:
            self.angle = random.uniform(-45, 45)
            self.prob = random.random()>0.5
        Data = self.get_data(idx)
        

        return Data

    def __len__(self):
        return len(self.samples)


    def get_data(self, idx):
        patient, slice_index = self.samples[idx]
        metadata = self.case_info[patient]
        annotation = metadata['annotation']
        with np.load(os.path.join(self.data_pth, patient), allow_pickle=False) as arrays:
            cmr = torch.from_numpy(arrays['cmr']).float()
            label = torch.from_numpy(arrays['labels'])
        time_indices = torch.tensor(annotation['time_indices'], dtype=torch.long)
        if (cmr.ndim != 4 or label.ndim != 4 or cmr.shape[1:] != label.shape[1:]
                or label.shape[0] != len(time_indices)
                or label.shape[0] != len(annotation['phase_names'])):
            raise ValueError(f'{patient}: 图像、标签形状或标注相位信息不一致')
        if not torch.isfinite(cmr).all() or cmr.numel() == 0:
            raise ValueError(f'{patient}: cmr 为空或含非有限值')
        if ((time_indices < 0) | (time_indices >= cmr.shape[0])).any():
            raise ValueError(f'{patient}: 标注时间索引超出图像帧范围')
        if label.numel() and ((label < 0) | (label > 3) | (label != label.long())).any():
            raise ValueError(f'{patient}: 标签必须为 0、1、2、3')
        source_spatial_shape = tuple(cmr.shape[-2:])
        output_size = (128, 128)
        # crop 不改变 spacing；仅按实际输入 H/W 修正 resize 后的面内间距。
        source_spacing = torch.tensor(
            metadata['source']['files']['cmr_4d']['zooms'][:3], dtype=torch.float64)
        if source_spacing.shape != (3,) or not torch.isfinite(source_spacing).all() or (source_spacing <= 0).any():
            raise ValueError(f'{patient}: CMR spacing 必须为三个有限正数，顺序为 XYZ')
        spacing = source_spacing * source_spacing.new_tensor([
            source_spatial_shape[1]/output_size[1],
            source_spatial_shape[0]/output_size[0], 1.])
        # 仅缩放 H/W；先处理完整病例，使 2D 与 3D 的对应像素一致。
        cmr = self.normalize_data(cmr)
        label = self.label_map[label.long()]
        if source_spatial_shape != output_size:
            cmr = TF.interpolate(cmr, size=output_size, mode='bilinear', align_corners=False)
            # 最近邻保持离散类别；空相位轴单独处理，不生成伪标签。
            if label.shape[0]:
                label = TF.interpolate(label.float(), size=output_size, mode='nearest').long()
            else:
                label = label.new_empty((0, label.shape[1], *output_size))
        cmr = cmr.clamp(0, 1).permute(1, 0, 2, 3)
        label = label.permute(1, 0, 2, 3)
        if slice_index is not None:
            cmr, label = cmr[slice_index], label[slice_index]
        info = {
            'patient_id': metadata['case'].get(
                'patient_id', metadata['case'].get('case_id', os.path.splitext(patient)[0])),
            'filename': patient,
            'dataset': self.datasets,
            'original_split': metadata['case']['original_split'],
            'slice_index': -1 if slice_index is None else slice_index,
            'cmr_layout': 'THW' if self.dim == '2D' else 'STHW',
            'label_layout': 'PHW' if self.dim == '2D' else 'SPHW',
            'source_spatial_shape': torch.tensor(source_spatial_shape),
            'spatial_shape': torch.tensor(output_size),
            'source_spacing_xyz': source_spacing,
            'spacing_xyz': spacing,
            'time_indices': time_indices,
            'phase_names': annotation['phase_names'],
            'has_labels': label.shape[-3] > 0,
            'valid_labels': self.has_valid_labels(patient),
            'label_definitions': {'0': 'background', '1': 'right_ventricular_cavity',
                                  '2': 'left_ventricular_myocardium',
                                  '3': 'left_ventricular_cavity'},
            'label_mapping': self.label_map.clone(),
            # 原始 JSON 含异构字段和 null；保留为字符串，兼容默认 batch 拼接。
            'metadata': json.dumps(metadata, ensure_ascii=False),
        }
        return {'cmr': cmr.contiguous(), 'label': label.contiguous(), 'info': info}

def fix_len(arr):
    """把 array 的第 0 维统一到 target 长度。
    不足：复制最后一帧；超出：截断。
    """
    n = arr.shape[0]
    if n == 40:
        return arr
    if n > 40:
        return arr[:40]
    # n < target: 复制最后一帧补齐
    pad = np.repeat(arr[-1:], 40 - n, axis=0)
    return np.concatenate([arr, pad], axis=0)

if __name__ == '__main__':
    # 测试loader调用和循环
    from config_utils import load_config
    config = load_config('config.yaml')
    loader = torch.utils.data.DataLoader(dataset())

    for i, data in enumerate(loader):
        a = 1
        # print(i, data['cmr'].shape, data['label'].shape, data['info'])