"""检查白底颜色、全帧绘图、覆盖写入以及训练随机状态隔离。"""
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from EANO.main import synthetic_batches
from EANO.test_eanox import small_config
from EANO.ulits import flow_to_hsv, save_epoch_visualization


class VisualizationTests(unittest.TestCase):
    def test_zero_displacement_is_white_and_directions_differ(self):
        zero = np.zeros((2, 4, 4), dtype=np.float32)
        np.testing.assert_array_equal(flow_to_hsv(zero, 0), np.ones((4, 4, 3)))
        horizontal = zero.copy()
        vertical = zero.copy()
        horizontal[0] = 1
        vertical[1] = 1
        self.assertFalse(np.allclose(flow_to_hsv(horizontal, 1), flow_to_hsv(vertical, 1)))

    def test_all_frames_overwrite_and_preserve_training_state(self):
        class Model(torch.nn.Module):
            def forward(self, images):
                self.grad_enabled = torch.is_grad_enabled()
                # 模拟组件内部的随机操作，确认不会影响训练随机流。
                torch.rand(1)
                random.random()
                np.random.rand()
                frames = images.shape[2]
                pairs = frames if self.cyclic else frames - 1
                flow = images.new_zeros(1, 2, pairs, *images.shape[-2:])
                flow[:, 0] = torch.arange(1, pairs + 1).view(1, pairs, 1, 1)
                return {'inf_lag_flow': flow}

        config = small_config()
        config['visualization'].update(dpi=40)
        loader = synthetic_batches(config, count=2)
        model = Model().train()
        with tempfile.TemporaryDirectory() as directory:
            for cyclic in (True, False):
                config['sequence']['cyclic'] = model.cyclic = cyclic
                python_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state().clone()
                result = save_epoch_visualization(model, loader, config, 'cpu', directory, int(cyclic) + 1)
                expected = loader[0]['images'].shape[2] + int(cyclic)
                self.assertEqual(result['panels'], expected)
                self.assertTrue(model.training)
                self.assertFalse(model.grad_enabled)
                self.assertEqual(random.getstate(), python_state)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                torch.testing.assert_close(torch.get_rng_state(), torch_state)
                self.assertEqual(list(Path(directory).glob('*.png')), [Path(result['path'])])
                self.assertGreater(Path(result['path']).stat().st_size, 100)


if __name__ == '__main__':
    unittest.main()
