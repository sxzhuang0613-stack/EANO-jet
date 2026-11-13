"""两组可整体切换的损失：双向运动、单向应力与正向力学，无标签依赖。"""
import math

import torch
from torch import nn

from ulits import (FourierGrad, deformation, flatten_time, normalized_xy,
                    spatial_derivatives, warp_sequence)


WNO_TERMS = ('forward_recon_loss', 'backward_recon_loss',
             'forward_lag_recon_loss', 'backward_lag_recon_loss',
             'optical_flow_loss', 'hyperelastic_loss', 'smoothness_loss')
EANO_TERMS = ('photometric', 'equilibrium', 'constitutive',
              'alignment_global', 'alignment_adjacent')


class EANOXLoss(nn.Module):
    """损失内部计算不修改预测场，所有参与项均保留对相应分支的梯度。"""
    def __init__(self, config, cyclic=True):
        super().__init__()
        self.config = config
        self.group = config['group']
        self.cyclic = cyclic
        self.grad = FourierGrad()
        if self.group not in ('wno_base', 'eano'):
            raise ValueError("losses.group must be wno_base or eano")
        self.terms = WNO_TERMS if self.group == 'wno_base' else EANO_TERMS
        self.weights = dict(config[self.group]['weights'])
        if set(self.weights) != set(self.terms):
            raise ValueError("Loss weights must match exactly the selected group's terms")
        if any(not math.isfinite(float(w)) or w < 0 for w in self.weights.values()) or not any(self.weights.values()):
            raise ValueError("Weights must be finite and nonnegative, with at least one active term")
        self.weighting = config['weighting']
        if self.weighting not in ('fixed', 'uncertainty'):
            raise ValueError("weighting must be fixed or uncertainty")
        if self.weighting == 'uncertainty':
            if self.group != 'wno_base':
                raise ValueError("Uncertainty weighting is available only for wno_base")
            # 与基础训练入口相同：0.5*exp(-s)*L+0.5*s。
            self.log_vars = nn.ParameterDict({key: nn.Parameter(torch.tensor(-math.log(2 * value)))
                                             for key, value in self.weights.items() if value > 0})
        self.epsilon = config['jacobian_epsilon']
        if self.epsilon <= 0:
            raise ValueError("jacobian_epsilon must be positive")

    def _pairs(self, images):
        if self.cyclic:
            source, target = images, images.roll(-1, dims=2)
        else:
            source, target = images[:, :, :-1], images[:, :, 1:]
        reference = images[:, :, :1].expand_as(target)
        return source, target, reference

    def _gn_loss(self, image, flow, ix, iy, it):
        parameters = self.config['wno_base']
        a, am = parameters['gn_a'], parameters['gn_am']
        if a <= 0 or am <= 0:
            raise ValueError("GN coefficients must be positive")
        ux, _, _, _ = self.grad(flow[:, 0:1])
        _, vy, _, _ = self.grad(flow[:, 1:2])
        residual = it + ix * flow[:, 0:1] + iy * flow[:, 1:2] - image * (ux + vy)
        return (((ix / a + iy / a - image / am) * residual).square()).mean()

    def _linear_mechanics(self, flow, stress):
        parameters = self.config['wno_base']
        u, v = flow[:, 0:1], flow[:, 1:2]
        ux, uy, _, _ = self.grad(u)
        vx, vy, _, _ = self.grad(v)
        exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
        sxx, syy, sxy = stress.split(1, dim=1)
        sxx_x, _, _, _ = self.grad(sxx)
        _, syy_y, _, _ = self.grad(syy)
        sxy_x, sxy_y, _, _ = self.grad(sxy)
        balance = ((sxx_x + sxy_y).square() + (syy_y + sxy_x).square()).mean()
        lam, mu = parameters['lambda'], parameters['mu']
        constitutive = (((lam + 2 * mu) * exx + lam * eyy - sxx).square()
                        + ((lam + 2 * mu) * eyy + lam * exx - syy).square()
                        + (2 * mu * exy - sxy).square()).mean()
        # 沿用基础版能量成本；它不是平方残差，允许出现负值。
        elastic = (0.5 * (exx * sxx + eyy * syy) + exy * sxy).mean()
        weights = parameters['mechanical_weights']
        total = weights['balance'] * balance + weights['constitutive'] * constitutive + weights['elastic'] * elastic
        return total, {'balance': balance, 'linear_constitutive': constitutive, 'elastic_cost': elastic}

    def _deformation_regularization(self, flow):
        parameters = self.config['wno_base']
        ux, uy, ut, _ = self.grad(flow[:, 0:1])
        vx, vy, vt, _ = self.grad(flow[:, 1:2])
        # 保留基础版无区域权重分支，包含原来的时间导数项。
        trace = ux.square() + uy.square() + vx.square() + vy.square() + 2 * ux + 2 * vy + 2 * ut + 2 * vt
        jac = 1 + ux + vy + ux * vy - uy * vx
        low, high = parameters['jacobian_clamp']
        jac = jac.clamp(low, high)
        volume = (parameters['volume_weight'] * (jac - 1).square()).clamp(0, 1)
        return (2 * trace - 2 * jac.log() + volume).abs().mean()

    def _wno(self, images, prediction):
        source, target, reference = self._pairs(images)
        f, b = prediction['inf_flow'], prediction['neg_flow']
        lf, lb = prediction['inf_lag_flow'], prediction['neg_lag_flow']
        # 正向映射读取目标图像；反向映射读取源图像，两组采用一致几何定义。
        terms = {
            'forward_recon_loss': (warp_sequence(target, f) - source).square().mean(),
            'backward_recon_loss': (warp_sequence(source, b) - target).square().mean(),
            'forward_lag_recon_loss': (warp_sequence(target, lf) - reference).square().mean(),
            'backward_lag_recon_loss': (warp_sequence(reference, lb) - target).square().mean(),
        }
        ix, iy, it, _ = self.grad(images)
        if self.cyclic:
            forward_grad = (ix, iy, it)
            backward_grad = (ix.roll(-1, 2), iy.roll(-1, 2), -it.roll(-1, 2))
        else:
            forward_grad = (ix[:, :, :-1], iy[:, :, :-1], it[:, :, :-1])
            backward_grad = (ix[:, :, 1:], iy[:, :, 1:], -it[:, :, 1:])
        optical_forward = self._gn_loss(source, f, *forward_grad)
        optical_backward = self._gn_loss(target, b, *backward_grad)
        terms['optical_flow_loss'] = optical_forward + optical_backward
        terms['hyperelastic_loss'], details = self._linear_mechanics(f, prediction['stress'])
        terms['smoothness_loss'] = self._deformation_regularization(f) + self._deformation_regularization(lf)
        details.update(optical_forward=optical_forward, optical_backward=optical_backward)
        return terms, details

    @staticmethod
    def _transport(source, target, flow):
        # 使用归一化 XY 速度和匹配的空间步长；一步时间间隔定义为一帧。
        velocity = normalized_xy(flow)
        ix, iy = spatial_derivatives(source, normalized=True)
        ux, uy = spatial_derivatives(velocity, normalized=True)
        residual = (target - source + velocity[:, 0:1] * ix + velocity[:, 1:2] * iy
                    + source * (ux[:, 0:1] + uy[:, 1:2]))
        return torch.log1p(residual.square()).mean()

    def _eano(self, images, prediction):
        source, target, reference = self._pairs(images)
        f, b = prediction['inf_flow'], prediction['neg_flow']
        lf, lb = prediction['inf_lag_flow'], prediction['neg_lag_flow']
        photo_forward = self._transport(source, target, f)
        photo_backward = self._transport(target, source, b)
        sxx, syy, sxy = prediction['stress'].split(1, dim=1)
        stress = torch.stack((torch.cat((sxx, sxy), dim=1), torch.cat((sxy, syy), dim=1)), dim=1)
        # 应力散度在当前构形求导，本构关系在参考构形计算。
        sx, sy = spatial_derivatives(stress, normalized=True)
        divergence = sx[:, :, 0] + sy[:, :, 1]
        equilibrium = torch.log1p(divergence.square().sum(dim=1)).mean()
        matrix, jacobian = deformation(flatten_time(normalized_xy(lf)), normalized=True)
        left = torch.einsum('nikhw,njkhw->nijhw', matrix, matrix)
        eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)[None, :, :, None, None]
        jac = jacobian[:, None, None]
        constitutive = left - eye + jac * (jac - 1) * eye
        flat_stress = stress.flatten(1, 2)
        pulled = flatten_time(warp_sequence(flat_stress, lf)).reshape_as(constitutive)
        predicted = jac * pulled
        # 与来源代码一致，缩放分子分母避免大应力直接平方溢出；不截断或分离梯度。
        scale = torch.maximum(predicted.abs().amax(dim=(1, 2)), constitutive.abs().amax(dim=(1, 2))).clamp_min(1)
        aa = predicted / scale[:, None, None]
        bb = constitutive / scale[:, None, None]
        numerator = (aa - bb).square().sum(dim=(1, 2))
        denominator = scale.reciprocal().square() + aa.square().sum(dim=(1, 2)) + bb.square().sum(dim=(1, 2))
        constitutive_loss = 0.5 * (numerator / denominator).mean()
        adjacent_forward = (warp_sequence(target, f) - source).abs().mean(dim=(1, 3, 4)).sum()
        adjacent_backward = (warp_sequence(source, b) - target).abs().mean(dim=(1, 3, 4)).sum()
        global_forward = (warp_sequence(target, lf) - reference).abs().mean(dim=(1, 3, 4)).sum()
        global_backward = (warp_sequence(reference, lb) - target).abs().mean(dim=(1, 3, 4)).sum()
        terms = {'photometric': photo_forward + photo_backward, 'equilibrium': equilibrium,
                 'constitutive': constitutive_loss,
                 'alignment_global': global_forward + global_backward,
                 'alignment_adjacent': adjacent_forward + adjacent_backward}
        details = dict(photometric_forward=photo_forward, photometric_backward=photo_backward,
                       alignment_adjacent_forward=adjacent_forward,
                       alignment_adjacent_backward=adjacent_backward,
                       alignment_global_forward=global_forward,
                       alignment_global_backward=global_backward)
        return terms, details

    def forward(self, images, prediction):
        expected_pairs = images.shape[2] if self.cyclic else images.shape[2] - 1
        for key, channels in (('inf_flow', 2), ('neg_flow', 2), ('inf_lag_flow', 2),
                              ('neg_lag_flow', 2), ('stress', 3)):
            if prediction[key].shape != (images.shape[0], channels, expected_pairs, *images.shape[-2:]):
                raise ValueError("Prediction shape mismatch: " + key)
        terms, details = self._wno(images, prediction) if self.group == 'wno_base' else self._eano(images, prediction)
        contributions = {}
        for key in self.terms:
            if self.weights[key] == 0:
                continue
            if self.weighting == 'uncertainty':
                s = self.log_vars[key]
                contributions[key] = 0.5 * torch.exp(-s) * terms[key] + 0.5 * s
            else:
                contributions[key] = self.weights[key] * terms[key]
        total = sum(contributions.values())
        with torch.no_grad():
            _, jacobian = deformation(flatten_time(prediction['inf_lag_flow']))
            diagnostics = {'jacobian_min': jacobian.min(), 'jacobian_mean': jacobian.mean(),
                           'negative_jacobian_fraction': (jacobian < 0).float().mean(),
                           'near_zero_jacobian_fraction': (jacobian.abs() < self.epsilon).float().mean(),
                           'flow_magnitude': prediction['inf_flow'].square().sum(1).sqrt().mean(),
                           'stress_magnitude': prediction['stress'].abs().mean()}
        return {'total': total, 'terms': terms, 'details': details,
                'contributions': contributions, 'diagnostics': diagnostics}
