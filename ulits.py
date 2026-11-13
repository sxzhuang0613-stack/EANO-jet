"""几何、谱导数、窗口及配置工具。

公开向量场使用像素 XY 分量和 [B,C,T,H,W] 排列；前向位移将前一帧
的点映射到后一帧。速度定义为每帧位移，不进行指数积分。
"""
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F


def load_config(path):
    """保留既有工具入口，委托给独立的 EANOX 配置解析器。"""
    from config_utils import load_eanox_config
    return load_eanox_config(path)


def seed_everything(seed):
    import numpy as np
    random.seed(seed)
    # NumPy 只接受 32 位种子，兼容配置中用于 PyTorch 的较大整数。
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FourierGrad(nn.Module):
    """基础版谱导数：空间镜像填充，时间采用周期边界。"""
    def forward(self, field, dx=1.0, dy=1.0, dt=1.0):
        _, _, time, height, width = field.shape
        px, py = width // 2, height // 2
        padded = F.pad(field, (px, px, py, py, 0, 0), mode="reflect")
        spec = torch.fft.fftn(padded, dim=(-3, -2, -1))
        frequencies = [torch.fft.fftfreq(n, d=step, device=field.device,
                                        dtype=field.dtype) * (2 * torch.pi)
                       for n, step in [(width + 2 * px, dx),
                                       (height + 2 * py, dy), (time, dt)]]
        shapes = [(1, 1, 1, 1, -1), (1, 1, 1, -1, 1), (1, 1, -1, 1, 1)]
        result = []
        for freq, shape in zip(frequencies, shapes):
            derivative = torch.fft.ifftn(1j * freq.reshape(shape) * spec,
                                        dim=(-3, -2, -1)).real
            result.append(derivative[..., py:py + height, px:px + width])
        return (*result, field)


def make_grid(images, grid_type="periodic"):
    batch, _, frames, height, width = images.shape
    options = dict(device=images.device, dtype=images.dtype)
    y = torch.linspace(0, 1, height, **options).view(1, height, 1).expand(frames, -1, width)
    x = torch.linspace(0, 1, width, **options).view(1, 1, width).expand(frames, height, -1)
    if grid_type == "linear":
        t = torch.linspace(0, 1, frames, **options).view(frames, 1, 1).expand(-1, height, width)
        fields = [t, y, x]
    elif grid_type == "periodic":
        phase = torch.arange(frames, **options) * (2 * torch.pi / frames)
        fields = [v.view(frames, 1, 1).expand(-1, height, width)
                  for v in (phase.sin(), phase.cos())] + [y, x]
    else:
        raise ValueError("grid_type must be linear or periodic")
    return torch.stack(fields).unsqueeze(0).expand(batch, -1, -1, -1, -1)


def warp(source, displacement, mode="bilinear", padding_mode="zeros"):
    """按恒等坐标加像素 XY 位移采样，输入排列为 [N,C,H,W]。"""
    n, _, height, width = source.shape
    y = torch.arange(height, device=source.device, dtype=source.dtype)
    x = torch.arange(width, device=source.device, dtype=source.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gx = 2 * (xx + displacement[:, 0]) / (width - 1) - 1
    gy = 2 * (yy + displacement[:, 1]) / (height - 1) - 1
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(source, grid, mode=mode, padding_mode=padding_mode,
                         align_corners=True)


def flatten_time(field):
    return field.permute(0, 2, 1, 3, 4).flatten(0, 1)


def warp_sequence(source, flow, mode="bilinear"):
    batch, channels, frames, height, width = source.shape
    sampled = warp(flatten_time(source), flatten_time(flow), mode=mode)
    return sampled.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def accumulate(flow, backward=False, interpolation="nearest"):
    """保留基础版最近邻复合，明确正反向映射顺序。

    正向为 p+q(Id+p)，反向为 q+p(Id+q)。结果与各帧对终点对齐，
    不额外插入参考帧零位移；周期序列最后一项对应闭合回参考帧。
    """
    current = flow[:, :, 0]
    result = [current]
    for increment in flow.unbind(2)[1:]:
        if backward:
            current = increment + warp(current, increment, mode=interpolation)
        else:
            current = current + warp(increment, current, mode=interpolation)
        result.append(current)
    return torch.stack(result, dim=2)


def normalized_xy(flow):
    height, width = flow.shape[-2:]
    scales = flow.new_tensor([width - 1, height - 1]).view(1, 2, 1, 1, 1)
    return flow / scales


def spatial_derivatives(field, normalized=False):
    height, width = field.shape[-2:]
    steps = (1 / (height - 1), 1 / (width - 1)) if normalized else (1.0, 1.0)
    dy, dx = torch.gradient(field, spacing=steps, dim=(-2, -1), edge_order=1)
    return dx, dy


def deformation(flow, normalized=False):
    """返回 [N,2,2,H,W] 变形梯度与未截断的行列式。"""
    dx, dy = spatial_derivatives(flow, normalized=normalized)
    matrix = torch.stack((torch.stack((1 + dx[:, 0], dy[:, 0]), dim=1),
                          torch.stack((dx[:, 1], 1 + dy[:, 1]), dim=1)), dim=1)
    jacobian = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
    return matrix, jacobian


class SequenceWindow:
    """保留基础版周期窗口及汉明加权重建，并修正多样本批量重组。"""
    def __init__(self, size=8, overlap=4, cyclic=True):
        if size < 2 or not 0 <= overlap < size:
            raise ValueError("Require window size>=2 and 0<=overlap<size")
        self.size, self.overlap, self.cyclic = size, overlap, cyclic

    def indices(self, frames, device):
        stride = self.size - self.overlap
        starts = list(range(0, frames, stride))
        if frames > self.size and (frames - self.size) % stride:
            starts.append(frames - self.size)
        raw = torch.tensor(starts, device=device)[:, None] + torch.arange(self.size, device=device)
        valid = torch.ones_like(raw, dtype=torch.bool) if self.cyclic else raw < frames
        index = raw.remainder(frames) if self.cyclic else raw.clamp_max(frames - 1)
        return index, valid

    def split(self, field, indices):
        # 将窗口并入批次维，保持每个样本内部的窗口顺序。
        b, c, _, h, w = field.shape
        return field[:, :, indices].permute(0, 2, 1, 3, 4, 5).reshape(-1, c, self.size, h, w)

    def merge(self, field, indices, valid, batch, frames):
        n, length = indices.shape
        c, h, w = field.shape[1], field.shape[-2], field.shape[-1]
        values = field.reshape(batch, n, c, length, h, w).permute(0, 2, 1, 3, 4, 5)
        window = torch.hamming_window(length, periodic=False, device=field.device, dtype=field.dtype)
        weights = window[None] * valid
        values = values * weights.view(1, 1, n, length, 1, 1)
        merged = field.new_zeros(batch, c, frames, h, w)
        merged = merged.index_add(2, indices.flatten(), values.flatten(2, 3))
        counts = field.new_zeros(frames).index_add(0, indices.flatten(), weights.flatten())
        return merged / counts.clamp_min(1e-8).view(1, 1, frames, 1, 1)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def flow_to_hsv(flow, maximum, white_threshold_pixels=0.05):
    """像素 XY 位移转 RGB：方向为色相，幅度为饱和度，零位移为纯白。"""
    import numpy as np
    from matplotlib.colors import hsv_to_rgb
    flow = np.asarray(flow)
    if flow.shape[0] != 2 or not np.isfinite(flow).all():
        raise ValueError('位移必须是有限的 [2,H,W] 数组')
    magnitude = np.hypot(flow[0], flow[1])
    hue = np.mod(np.arctan2(flow[1], flow[0]) / (2 * np.pi), 1.0)
    saturation = np.clip(magnitude / max(float(maximum), 1e-12), 0, 1)
    saturation = np.where(magnitude <= white_threshold_pixels, 0, saturation)
    return hsv_to_rgb(np.stack((hue, saturation, np.ones_like(hue)), axis=-1))


def save_epoch_visualization(model, loader, config, device, output_dir, epoch):
    """随机选择训练序列，绘制参考帧至各帧累计位移，并覆盖上一轮结果。"""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .data_adapter import get_images

    options = config.get('visualization', {})
    if not options.get('enabled', True):
        return None
    columns = options.get('columns', 6)
    dpi = options.get('dpi', 120)
    threshold = options.get('white_threshold_pixels', 0.05)
    if not isinstance(columns, int) or columns < 1 or dpi <= 0 or threshold < 0:
        raise ValueError('可视化列数、分辨率必须为正数，白底阈值必须非负')

    # 独立随机源保证新增可视化不会改变下一轮训练样本顺序或随机状态。
    rng = random.Random(config['runtime']['seed'] + epoch)
    dataset = getattr(loader, 'dataset', loader)
    if not hasattr(dataset, '__getitem__') or len(dataset) == 0:
        raise ValueError('每轮可视化需要可索引且非空的训练数据集')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            selected = dataset[rng.randrange(len(dataset))]
            if isinstance(selected, list):
                selected = selected[rng.randrange(len(selected))]
            sequences = get_images(selected, config, 'cpu')
            slice_index = rng.randrange(len(sequences))
            sequence = sequences[slice_index:slice_index + 1].to(device)
            model.eval()
            prediction = model(sequence)
            flow = prediction['inf_lag_flow'][0].detach().cpu().numpy()
            frames = sequence.shape[2]
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    cyclic = config['sequence']['cyclic']
    expected = frames if cyclic else frames - 1
    if flow.shape[0] != 2 or flow.shape[1] != expected or not np.isfinite(flow).all():
        raise ValueError('累计位移的形状或数值不符合序列定义')
    # 第一幅显式加入参考帧恒等位移；周期模式最后一幅保留闭合残差。
    fields = np.concatenate((np.zeros_like(flow[:, :1]), flow), axis=1)
    labels = ['0 -> 0 (reference)'] + [
        f'0 -> {(index + 1) % frames}' + (' (cycle)' if cyclic and index == frames - 1 else '')
        for index in range(expected)]
    maximum = float(np.hypot(fields[0], fields[1]).max())
    info = selected.get('info', {})
    filename = info.get('filename', str(selected.get('case_id', 'sequence')))
    actual_slice = info.get('slice_index', -1)
    if actual_slice == -1:
        actual_slice = slice_index
    count = fields.shape[1]
    columns = min(columns, count)
    rows = (count + columns - 1) // columns
    figure = Figure(figsize=(2.5 * columns, 2.7 * rows), facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    try:
        for index, axis in enumerate(axes.flat):
            axis.set_facecolor('white')
            axis.set_axis_off()
            if index < count:
                axis.imshow(flow_to_hsv(fields[:, index], maximum, threshold), interpolation='nearest')
                axis.set_title(labels[index], fontsize=9)
        figure.suptitle(f'Epoch {epoch} | {filename} | slice {actual_slice}\n'
                        f'Reference -> frame | shared scale: {maximum:.3g} px | '
                        'hue=direction, saturation=magnitude, white=zero', fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0, w_pad=0.8)
        path = Path(output_dir) / 'displacement_hsv.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.png')
        figure.savefig(temporary, dpi=dpi, facecolor='white')
        temporary.replace(path)
    finally:
        figure.clear()
    return {'path': str(path), 'epoch': epoch, 'filename': filename,
            'slice_index': int(actual_slice), 'panels': count,
            'maximum_magnitude_pixels': maximum}
