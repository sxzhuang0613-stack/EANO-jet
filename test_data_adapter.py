"""验证 NPZ 接口、病例划分、变长序列和两种训练单位。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from EANO.config_utils import load_eanox_config
from EANO.data_adapter import build_dataloader, get_images, data_protocol
from EANO.main import run_epoch
from EANO.test_eanox import small_config


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for index, (frames, slices, phases) in enumerate(((3, 2, 2), (5, 1, 0), (4, 3, 2), (6, 1, 2))):
            cmr = np.arange(frames * slices * 8 * 10, dtype=np.float32).reshape(frames, slices, 8, 10)
            labels = np.zeros((phases, slices, 8, 10), dtype=np.int64)
            metadata = {
                'schema': {'array_axes': {'cmr': ['T', 'Z', 'Y', 'X'], 'labels': ['phase', 'Z', 'Y', 'X']}},
                'case': {'patient_id': str(index), 'original_split': 'training'},
                'annotation': {'time_indices': [0, frames - 1] if phases else [],
                               'phase_names': ['ED', 'ES'] if phases else []},
                'processing': {'output': {'cmr_shape': list(cmr.shape), 'labels_shape': list(labels.shape)}},
                'source': {'files': {'cmr_4d': {'zooms': [1.5, 2.0, 8.0, 1.0]}}},
            }
            np.savez(self.root / f'case{index}.npz', cmr=cmr, labels=labels, info=json.dumps(metadata))
        self.config = small_config()
        self.config['model']['input_size'] = [128, 128]
        self.config['dataset_paths'] = {'acdc': str(self.root)}
        self.config['train_fraction'] = 0.5
        self.config['data'].update(dim='2D', batch_size=3, num_workers=0)

    def tearDown(self):
        self.temporary.cleanup()

    def test_npz_split_shapes_and_unlabeled_cases(self):
        loader = build_dataloader(self.config, 'train')
        validation = build_dataloader(self.config, 'val')
        self.assertEqual(set(loader.dataset.patients), {'case0.npz', 'case1.npz'})
        self.assertFalse(set(loader.dataset.patients) & set(validation.dataset.patients))
        records = next(iter(loader))
        self.assertEqual(len(records), 3)
        lengths = []
        for record in records:
            images = get_images(record, self.config, 'cpu')
            lengths.append(images.shape[2])
            self.assertEqual(images.shape[:2], (1, 1))
            self.assertEqual(images.shape[-2:], (128, 128))
            torch.testing.assert_close(images[0, 0], record['cmr'])
        self.assertEqual(sorted(lengths), [3, 3, 5])
        self.assertEqual(data_protocol(self.config, loader)['dim'], '2D')

    def test_complete_case_matches_slice_and_spacing(self):
        slices = build_dataloader(self.config, 'train').dataset
        config = copy.deepcopy(self.config)
        config['data']['dim'] = '3D'
        cases = build_dataloader(config, 'train').dataset
        whole = cases[0]
        images = get_images(whole, config, 'cpu')
        self.assertEqual(images.shape, (2, 1, 3, 128, 128))
        torch.testing.assert_close(images[0, 0], slices[0]['cmr'])
        torch.testing.assert_close(images[1, 0], slices[1]['cmr'])
        expected = torch.tensor([1.5 * 10 / 128, 2 * 8 / 128, 8.0], dtype=torch.float64)
        torch.testing.assert_close(whole['info']['spacing_xyz'], expected)

    def test_configuration_is_independent_of_renderer_and_working_directory(self):
        path = self.root / 'eanox.yaml'
        config = copy.deepcopy(self.config)
        config['dataset_paths'] = {'acdc': '.'}
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        loaded = load_eanox_config(path)
        self.assertEqual(loaded['dataset_paths']['acdc'], str(self.root))
        self.assertNotIn('bounds', loaded)
        with path.open('a') as stream:
            stream.write('dataset_name: mnms\n')
        with self.assertRaisesRegex(ValueError, '配置键重复'):
            load_eanox_config(path)

    def test_epoch_averages_all_case_slices_and_handles_variable_lengths(self):
        # 使用可解析梯度的简化组件，验证完整病例没有漏切片或按总数错误缩放。
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                return self.scale * images

        class Loss(torch.nn.Module):
            def forward(self, images, prediction):
                return {'total': prediction.square().mean(), 'terms': {}, 'details': {},
                        'contributions': {}, 'diagnostics': {}}

        config = small_config()
        config['training'].update(accumulation_steps=1, gradient_clip_norm=None)
        first = {'cmr': torch.ones(2, 3, 8, 8), 'info': {'cmr_layout': 'STHW'}}
        first['cmr'][1] = 3
        second = {'cmr': torch.ones(1, 5, 8, 8) * 2, 'info': {'cmr_layout': 'STHW'}}
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = run_epoch(model, Loss(), [[first, second]], config, 'cpu', optimizer)
        self.assertAlmostEqual(result['total'], 4.5)
        self.assertAlmostEqual(float(model.scale), 0.1, places=6)


if __name__ == '__main__':
    unittest.main()
