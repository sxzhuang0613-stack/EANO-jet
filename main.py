"""eano 入口：训练、验证、预测及不依赖真实数据的合成链路检查。"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import torch
from tqdm.auto import tqdm

# 同时支持直接运行本文件和以包方式运行，不导入旧工程代码。
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from components import eano
    from losses import EANOXLoss
    from data_adapter import build_dataloader, data_protocol, get_images
    from ulits import atomic_save, load_config, seed_everything, save_epoch_visualization
else:
    from .components import eano
    from .losses import EANOXLoss
    from .data_adapter import build_dataloader, data_protocol, get_images
    from .ulits import atomic_save, load_config, seed_everything, save_epoch_visualization


def scalar_record(result):
    record = {'total': float(result['total'].detach())}
    for section in ('terms', 'details', 'contributions', 'diagnostics'):
        record.update({section + '/' + key: float(value.detach())
                       for key, value in result[section].items()})
    return record


def print_epoch_losses(epoch, config, stats, learning_rate, seconds, visualization=None):
    """按列输出本轮训练均值，同时区分原始损失和实际加权贡献。"""
    labels = {
        'forward_recon_loss': '正向相邻配准', 'backward_recon_loss': '反向相邻配准',
        'forward_lag_recon_loss': '正向累计配准', 'backward_lag_recon_loss': '反向累计配准',
        'optical_flow_loss': '双向光流约束', 'hyperelastic_loss': '正向力学约束',
        'smoothness_loss': '正向形变正则', 'photometric': '双向光度约束',
        'equilibrium': '应力平衡', 'constitutive': '本构一致性',
        'alignment_global': '双向累计配准', 'alignment_adjacent': '双向相邻配准',
    }
    line = '─' * 64
    print(f'\n{line}\nEpoch {epoch:03d}/{config["training"]["epochs"]}  |  '
          f'{config["losses"]["group"]}  |  lr={learning_rate:.3e}  |  {seconds:.1f}s')
    print(f'训练总损失：{stats["total"]:.6f}\n{line}')
    print('损失项                        原始值          加权贡献')
    for key, value in stats.items():
        if not key.startswith('terms/'):
            continue
        name = key.split('/', 1)[1]
        label = labels.get(name, name)
        # 中文字符按双列宽补齐，避免中英文混排导致表格错位。
        width = sum(2 if ord(char) > 127 else 1 for char in label)
        padded = label + ' ' * max(1, 26 - width)
        contribution = stats.get('contributions/' + name, 0.0)
        print(f'{padded}{value:>14.6g}  {contribution:>14.6g}')
    print(line)
    if visualization is not None:
        print(f'累计位移图：{visualization["path"]}')
    print('最新权重：last.pt（覆盖保存）\n', flush=True)


def save_latest_checkpoint(payload, output):
    """成功覆盖最新检查点后，清理本程序旧策略产生的历史权重。"""
    import re
    output = Path(output)
    atomic_save(payload, output / 'last.pt')
    for path in output.iterdir():
        if path.is_file() and (path.name == 'best.pt' or re.fullmatch(r'epoch_\d+\.pt', path.name)):
            path.unlink()


def run_epoch(model, objective, loader, config, device, optimizer=None, epoch=None):
    training = optimizer is not None
    model.train(training)
    objective.train(training)
    accumulation = config['training']['accumulation_steps']
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive")
    parameters = list(model.parameters()) + list(objective.parameters())
    sums, samples, pending = {}, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def update(count):
        # 包括最后不足整组的批次，按实际累积次数平均，不丢弃尾部梯度。
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(count)
                if not torch.isfinite(parameter.grad).all():
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite gradient; optimizer update cancelled")
        clip = config['training']['gradient_clip_norm']
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    stage = '训练' if training else '验证'
    description = stage if epoch is None else f'Epoch {epoch}/{config["training"]["epochs"]} {stage}'
    with torch.set_grad_enabled(training), tqdm(
            loader, desc=description, disable=not config['training'].get('progress_bar', True),
            dynamic_ncols=True, unit='批次') as progress:
        for batch in progress:
            # 列表由真实 loader 返回，病例时间长度和标签相位数可以不同。
            records = batch if isinstance(batch, list) else [batch]
            if not records:
                continue
            for record in records:
                sequences = get_images(record, config, device)
                slices = len(sequences)
                unit_values = {}
                # 2D 模式每项只有一个切片；3D 模式完整病例逐切片反传并取均值。
                for sequence in sequences.split(1):
                    result = objective(sequence, model(sequence))
                    if not torch.isfinite(result['total']):
                        raise FloatingPointError("Non-finite total loss")
                    if training:
                        (result['total'] / (slices * len(records))).backward()
                    for key, value in scalar_record(result).items():
                        unit_values[key] = unit_values.get(key, 0.0) + value / slices
                for key, value in unit_values.items():
                    sums[key] = sums.get(key, 0.0) + value
                samples += 1
            progress.set_postfix(loss=f'{sums["total"] / samples:.5g}')
            if training:
                pending += 1
                if pending == accumulation:
                    update(pending)
                    pending = 0
    if training and pending:
        update(pending)
    if samples == 0:
        raise ValueError("The selected data split is empty")
    return {key: value / samples for key, value in sums.items()}


def make_optimizer(model, objective, config):
    options = config['training']
    parameters = list(model.parameters()) + list(objective.parameters())
    if options['optimizer'] not in ('adam', 'adamw'):
        raise ValueError("optimizer must be adam or adamw")
    cls = torch.optim.AdamW if options['optimizer'] == 'adamw' else torch.optim.Adam
    optimizer = cls(parameters, lr=options['learning_rate'], weight_decay=options['weight_decay'])
    name = options['scheduler']
    if name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=options['epochs'], eta_min=options['eta_min'])
    elif name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=options['step_size'], gamma=options['gamma'])
    elif name == 'none':
        scheduler = None
    else:
        raise ValueError("scheduler must be cosine, step or none")
    return optimizer, scheduler


def make_checkpoint(model, objective, optimizer, scheduler, config, epoch, best=None):
    """保留旧 best 调用参数以兼容已有调用，但不再存储最佳验证结果。"""
    return {'format_version': 1, 'model': model.state_dict(),
            'objective': objective.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'config': copy.deepcopy(config), 'epoch': epoch,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore(path, model, objective, config, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError("Not an eano checkpoint; legacy model weights are incompatible")
    # 切换损失或几何定义必须作为新实验，不能悄悄恢复旧优化器状态。
    sections = ('model', 'sequence', 'losses', 'data', 'dataset_name',
                'dataset_paths', 'train_fraction', 'normalization_epsilon', 'data_protocol')
    for section in sections:
        previous, current = checkpoint['config'].get(section), config.get(section)
        # 兼容旧版记录验证集的检查点；恢复训练只核对实际使用的训练集合。
        if section == 'data_protocol' and optimizer is not None:
            previous = previous.get('train') if previous else None
            current = current.get('train') if current else None
        if previous != current:
            raise ValueError("Checkpoint configuration mismatch: " + section)
    model.load_state_dict(checkpoint['model'])
    objective.load_state_dict(checkpoint['objective'])
    if optimizer is not None:
        for key in ('optimizer', 'scheduler', 'learning_rate', 'weight_decay',
                    'accumulation_steps', 'gradient_clip_norm', 'eta_min', 'step_size', 'gamma'):
            if checkpoint['config']['training'][key] != config['training'][key]:
                raise ValueError("Checkpoint training configuration mismatch: " + key)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return checkpoint


def synthetic_batches(config, count=2):
    """只用于执行检查，不代替 dataloader 中的真实数据接口。"""
    height, width = config['model']['input_size']
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    frames = config['sequence']['window_size'] + 2
    phases = torch.arange(frames) * (2 * torch.pi / frames)
    sequence = torch.stack([torch.exp(-8 * ((x - 0.1 * phase.sin()).square() + y.square()))
                            for phase in phases]).unsqueeze(0).unsqueeze(0)
    return [{config['data']['image_key']: sequence.clone(), 'case_id': ['synthetic']}
            for _ in range(count)]


def main():
    parser = argparse.ArgumentParser(description='EANOX：无标签双向运动与正向应力学习')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--mode', choices=['train', 'validate', 'predict'], default='train')
    parser.add_argument('--loss-group', choices=['wno_base', 'eano'])
    parser.add_argument('--device')
    parser.add_argument('--output-dir')
    parser.add_argument('--checkpoint')
    parser.add_argument('--progress-bar', action=argparse.BooleanOptionalAction, default=None,
                        help='是否显示每轮训练和验证进度条，默认开启；关闭使用 --no-progress-bar')
    parser.add_argument('--input', help='预测输入文件：图像张量或包含 images 的字典')
    parser.add_argument('--smoke', action='store_true', help='缩小模型，用合成数据检查训练和保存链路')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.loss_group:
        config['losses']['group'] = args.loss_group
    if args.device:
        config['runtime']['device'] = args.device
    if args.output_dir:
        config['runtime']['output_dir'] = args.output_dir
    if args.progress_bar is not None:
        config['training']['progress_bar'] = args.progress_bar
    if args.smoke:
        # 只在显式检查模式缩小配置；有效配置完整写入输出目录。
        config['model'].update(input_size=[16, 16], width=4,
                               lifting_hidden_channels=8, projection_hidden_channels=8,
                               weight_mode='shared')
        config['training'].update(epochs=1, accumulation_steps=3)
    if config['training']['epochs'] < 1:
        raise ValueError("epochs must be positive")
    seed_everything(config['runtime']['seed'])
    # 在构建大模型前检查真实数据接口，同时记录病例划分。
    train_loader = None
    if args.mode == 'train' and not args.smoke:
        train_loader = build_dataloader(config, 'train')
        config['data_protocol'] = {'train': data_protocol(config, train_loader)}
    requested_device = config['runtime']['device']
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = eano(config['model'], config['sequence']).to(device)
    objective = EANOXLoss(config['losses'], config['sequence']['cyclic']).to(device)
    output = Path(config['runtime']['output_dir']).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint or config['runtime']['resume']

    if args.mode != 'train':
        if not path:
            raise ValueError("validate/predict requires --checkpoint")
        # 独立验证不依赖当前训练 loader；使用检查点记录的训练数据协议核对配置。
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if 'data_protocol' in saved.get('config', {}):
            config['data_protocol'] = saved['config']['data_protocol']
        restore(path, model, objective, config)
        model.eval()
        objective.eval()
        if args.mode == 'validate':
            loader = synthetic_batches(config) if args.smoke else build_dataloader(config, 'val')
            expected = config.get('data_protocol', {}).get('val')
            if not args.smoke and expected is not None and expected != data_protocol(config, loader):
                raise ValueError('当前验证病例集合与检查点不一致')
            stats = run_epoch(model, objective, loader, config, device)
            (output / 'validation.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(stats, ensure_ascii=False))
        else:
            if not args.input:
                raise ValueError("predict requires --input")
            batch = torch.load(args.input, map_location='cpu', weights_only=False)
            if isinstance(batch, torch.Tensor):
                batch = {config['data']['image_key']: batch}
            with torch.no_grad():
                # 完整病例分切片预测，避免一次保留整例的中间特征。
                outputs = [model(sequence) for sequence in get_images(batch, config, device).split(1)]
                prediction = {key: torch.cat([item[key] for item in outputs], dim=0)
                              for key in outputs[0]}
            atomic_save({key: value.cpu() for key, value in prediction.items()}, output / 'prediction.pt')
            if isinstance(batch.get('info'), dict):
                atomic_save(batch['info'], output / 'prediction_info.pt')
            print(str(output / 'prediction.pt'))
        return

    if not path and any((output / name).exists() for name in ('last.pt', 'history.jsonl')):
        raise FileExistsError("Output directory already contains training results; use resume or a new directory")
    if args.smoke:
        train_loader = synthetic_batches(config)
    optimizer, scheduler = make_optimizer(model, objective, config)
    start = 0
    if path:
        checkpoint = restore(path, model, objective, config, optimizer, scheduler)
        start = checkpoint['epoch']
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    for epoch in range(start, config['training']['epochs']):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        train_stats = run_epoch(model, objective, train_loader, config, device, optimizer, epoch=epoch + 1)
        if scheduler is not None:
            scheduler.step()
        # 使用本轮更新后的模型另做一次无梯度预测，每轮覆盖同一张图。
        visualization = save_epoch_visualization(model, train_loader, config, device, output, epoch + 1)
        record = {'epoch': epoch + 1, 'train': train_stats}
        if visualization is not None:
            record['visualization'] = visualization
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        checkpoint = make_checkpoint(model, objective, optimizer, scheduler, config, epoch + 1)
        save_latest_checkpoint(checkpoint, output)
        print_epoch_losses(epoch + 1, config, train_stats, learning_rate,
                           time.perf_counter() - started, visualization)


if __name__ == '__main__':
    main()
