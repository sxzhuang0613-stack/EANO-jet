"""合成验证：几何方向、跨窗口重建、双向梯度与单向物理约束。"""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from EANO.components import CODALayer, DWT3d, EANOX, WaveletConv3d
from EANO.losses import EANOXLoss
from EANO.main import make_checkpoint, make_optimizer, restore, run_epoch, synthetic_batches, save_latest_checkpoint
from EANO.ulits import (SequenceWindow, accumulate, atomic_save, deformation,
                        load_config, seed_everything, warp)


def small_config():
    config = load_config(Path(__file__).with_name('config.yaml'))
    config['model'].update(input_size=[16, 16], width=4, lifting_hidden_channels=8,
                           projection_hidden_channels=8, weight_mode='shared')
    return config


class EANOXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        seed_everything(17)

    def test_forward_mapping_and_composition(self):
        # 目标图像向右移动一像素，正向位移读取目标后应还原源图像。
        source = torch.zeros(1, 1, 10, 10)
        source[:, :, 3:6, 3:6] = 1
        target = torch.roll(source, 1, dims=-1)
        offset = torch.zeros(1, 2, 10, 10)
        offset[:, 0] = 1
        torch.testing.assert_close(warp(target, offset), source)
        increments = offset.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        cumulative = accumulate(increments)
        torch.testing.assert_close(cumulative[:, 0, :, 2:7, 2:7],
                                   torch.arange(1, 4).float().view(1, 3, 1, 1).expand(1, 3, 5, 5))
        inverse = accumulate(-increments, backward=True)
        torch.testing.assert_close(inverse[:, 0, :, 3:7, 3:7],
                                   -torch.arange(1, 4).float().view(1, 3, 1, 1).expand(1, 3, 4, 4))

    def test_window_identity_and_batch_separation(self):
        for cyclic in (True, False):
            for frames in (2, 3, 8, 10, 17):
                with self.subTest(cyclic=cyclic, frames=frames):
                    window = SequenceWindow(8, 4, cyclic)
                    images = torch.randn(2, 3, frames, 4, 4, requires_grad=True)
                    indices, valid = window.indices(frames, images.device)
                    merged = window.merge(window.split(images, indices), indices, valid, 2, frames)
                    torch.testing.assert_close(merged, images)
                    merged.sum().backward()
                    torch.testing.assert_close(images.grad, torch.ones_like(images))

    def test_dwt_reconstruction_and_weight_modes(self):
        transform = DWT3d()
        field = torch.randn(1, 2, 7, 16, 16)
        reconstructed = transform.inverse_transform(transform.forward_transform(field), 7)
        torch.testing.assert_close(reconstructed, field, atol=2e-6, rtol=2e-5)
        for mode in ('shared', 'official', 'tucker'):
            with self.subTest(weight_mode=mode):
                conv = WaveletConv3d(2, 2, [8, 8, 8], weight_mode=mode)
                value = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
                output = conv(value)
                self.assertEqual(output.shape, value.shape)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_coda_token_equivariance(self):
        coda = CODALayer(4, [8, 8, 8], weight_mode='shared')
        a, b = torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8)
        aa, bb, attention = coda(a, b)
        swapped_b, swapped_a, swapped_attention = coda(b, a)
        torch.testing.assert_close(aa, swapped_a)
        torch.testing.assert_close(bb, swapped_b)
        torch.testing.assert_close(attention, swapped_attention.flip((1, 2)))
        torch.testing.assert_close(attention.sum(-1), torch.ones(1, 2))

    def test_complex_tucker_and_odd_spatial_size(self):
        for kind, size in (('dwt', [8, 15, 17]), ('dtcwt_complex', [8, 16, 16])):
            with self.subTest(transform=kind):
                conv = WaveletConv3d(2, 2, size, weight_mode='tucker', transform_type=kind)
                value = torch.randn(1, 2, *size, requires_grad=True)
                output = conv(value)
                self.assertEqual(output.shape, value.shape)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_uncertainty_weights_have_gradients(self):
        config = small_config()
        config['losses']['weighting'] = 'uncertainty'
        model = EANOX(config['model'], config['sequence'])
        objective = EANOXLoss(config['losses'])
        images = synthetic_batches(config, 1)[0]['images']
        objective(images, model(images))['total'].backward()
        for parameter in objective.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_both_groups_reach_both_motion_modules_and_stress(self):
        for group in ('wno_base', 'eano'):
            for grid, cyclic in (('linear', False), ('periodic', True)):
                with self.subTest(group=group, grid=grid):
                    config = small_config()
                    config['losses']['group'] = group
                    config['sequence'].update(grid_type=grid, cyclic=cyclic)
                    model = EANOX(config['model'], config['sequence'])
                    objective = EANOXLoss(config['losses'], cyclic)
                    images = torch.randn(2, 1, 10, 16, 16)
                    prediction = model(images)
                    expected = 10 if cyclic else 9
                    self.assertEqual(prediction['inf_flow'].shape, (2, 2, expected, 16, 16))
                    result = objective(images, prediction)
                    self.assertTrue(torch.isfinite(result['total']))
                    result['total'].backward()
                    for module in (model.VecInt, model.VecInt_backward, model.projection_B):
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_physics_does_not_use_backward_motion(self):
        for group in ('wno_base', 'eano'):
            with self.subTest(group=group):
                config = small_config()
                config['losses']['group'] = group
                images = torch.rand(1, 1, 4, 8, 8)
                forward = (torch.randn(1, 2, 4, 8, 8) * 0.01).requires_grad_()
                backward = (torch.randn_like(forward) * 0.01).requires_grad_()
                stress = (torch.randn(1, 3, 4, 8, 8) * 0.01).requires_grad_()
                prediction = {'inf_flow': forward, 'neg_flow': backward,
                              'inf_lag_flow': accumulate(forward),
                              'neg_lag_flow': accumulate(backward, backward=True), 'stress': stress}
                objective = EANOXLoss(config['losses'])
                terms = objective(images, prediction)['terms']
                physical = (terms['hyperelastic_loss'] + terms['smoothness_loss'] if group == 'wno_base'
                            else terms['equilibrium'] + terms['constitutive'])
                physical.backward()
                self.assertIsNone(backward.grad)
                self.assertIsNotNone(forward.grad)
                self.assertIsNotNone(stress.grad)

    def test_identity_has_zero_losses(self):
        images = torch.ones(1, 1, 4, 8, 8)
        zero = torch.zeros(1, 2, 4, 8, 8)
        prediction = {'inf_flow': zero, 'neg_flow': zero, 'inf_lag_flow': zero,
                      'neg_lag_flow': zero, 'stress': torch.zeros(1, 3, 4, 8, 8)}
        for group in ('wno_base', 'eano'):
            config = small_config()
            config['losses']['group'] = group
            value = EANOXLoss(config['losses'])(images, prediction)['total']
            self.assertLess(abs(float(value)), 1e-7)

    def test_affine_deformation(self):
        y, x = torch.meshgrid(torch.arange(8.), torch.arange(10.), indexing='ij')
        flow = torch.stack((0.1 * x, -0.2 * y)).unsqueeze(0)
        _, jac = deformation(flow)
        torch.testing.assert_close(jac, torch.full_like(jac, 1.1 * 0.8))

    def test_checkpoint_and_partial_accumulation(self):
        config = small_config()
        config['training'].update(accumulation_steps=3, epochs=2)
        model = EANOX(config['model'], config['sequence'])
        objective = EANOXLoss(config['losses'])
        optimizer, scheduler = make_optimizer(model, objective, config)
        before = model.projection_A.layers[0].weight.detach().clone()
        stats = run_epoch(model, objective, synthetic_batches(config), config, 'cpu', optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(stats['total'])))
        self.assertFalse(torch.equal(before, model.projection_A.layers[0].weight))
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            atomic_save(make_checkpoint(model, objective, optimizer, scheduler, config, 1, 1.0), path)
            restored = EANOX(config['model'], config['sequence'])
            restored_loss = EANOXLoss(config['losses'])
            restored_optim, restored_scheduler = make_optimizer(restored, restored_loss, config)
            restore(path, restored, restored_loss, config, restored_optim, restored_scheduler)
            value = synthetic_batches(config, count=1)[0]['images']
            with torch.no_grad():
                torch.testing.assert_close(model(value)['inf_flow'], restored(value)['inf_flow'])
            self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
            incompatible = copy.deepcopy(config)
            incompatible['sequence']['cyclic'] = False
            with self.assertRaises(ValueError):
                restore(path, restored, restored_loss, incompatible)

    def test_only_latest_training_checkpoint_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('best.pt', 'epoch_0001.pt', 'epoch_0002.pt', 'prediction.pt'):
                atomic_save({'epoch': 0}, output / name)
            save_latest_checkpoint({'epoch': 3}, output)
            save_latest_checkpoint({'epoch': 4}, output)
            self.assertEqual({path.name for path in output.glob('*.pt')}, {'last.pt', 'prediction.pt'})
            self.assertEqual(torch.load(output / 'last.pt', weights_only=True)['epoch'], 4)


if __name__ == '__main__':
    unittest.main()
