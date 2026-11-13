"""EANOX 的小波算子、基础版注意力与双分支模型。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Literal, Optional

from pytorch_wavelets import DWT, IDWT
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class ChannelMLP(nn.Module):
    """逐位置通道映射，沿用基础版的线性层与中间激活。"""

    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 n_layers=2, n_dim=3, non_linearity=F.gelu, dropout=0.0, **kwargs):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        sizes = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.layers = nn.ModuleList([nn.Conv1d(a, b, 1) for a, b in zip(sizes, sizes[1:])])
        self.activation = non_linearity
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(self.activation(x))
        return x.reshape(shape[0], x.shape[1], *shape[2:])

class SoftGating(nn.Module):
    """基础版的可学习逐通道残差门控。"""
    def __init__(self, channels, n_dim=3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, *([1] * n_dim)))

    def forward(self, x):
        return x * self.weight

class DWT3d(nn.Module):
    """移植基础版的空间二维与时间一维可分离小波变换。"""

    DWT_SUBBANDS = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
    DTCWT_SUBBANDS = ['low'] + [f'd{i}_{t}' for i in range(6) for t in ['a', 'd']]
    DTCWT_COMPLEX_SUBBANDS = DTCWT_SUBBANDS
    
    def __init__(self, wavelet: str = 'db4', level: int = 1, transform_type: str = 'dwt') -> None:
        super().__init__()

        if level != 1:
            raise ValueError(f"The baseline transform supports exactly one level, got {level}")

        if transform_type not in ['dwt', 'dtcwt', 'dtcwt_complex']:
            raise ValueError(f"transform_type must be one of ['dwt', 'dtcwt', 'dtcwt_complex'], got {transform_type}")

        self.wavelet = wavelet
        self.level = level
        self.transform_type = transform_type

        self.dwt_1d = DWT1DForward(J=level, wave='db4', mode='periodization')
        self.idwt_1d = DWT1DInverse(wave='db4', mode='periodization')

        if transform_type in ['dtcwt', 'dtcwt_complex']:
            
            self.dtcwt_2d = DTCWTForward(J=level, biort='near_sym_a', qshift='qshift_a')
            self.idtcwt_2d = DTCWTInverse(biort='near_sym_a', qshift='qshift_a')
        else:
            
            self.dwt_2d = DWT(J=level, mode='periodization', wave=wavelet)
            self.idwt_2d = IDWT(mode='periodization', wave=wavelet)
    
    def get_subband_names(self) -> list[str]:

        if self.transform_type == 'dtcwt_complex':
            return self.DTCWT_COMPLEX_SUBBANDS
        elif self.transform_type == 'dtcwt':
            return self.DTCWT_SUBBANDS
        else:
            return self.DWT_SUBBANDS
    
    def forward_transform(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        if self.transform_type == 'dtcwt_complex':
            return self.dtcwt3d_complex(x)
        elif self.transform_type == 'dtcwt':
            return self.dtcwt3d(x)
        else:
            return self.dwt3d(x)

    def inverse_transform(self, subbands: Dict[str, torch.Tensor], T_orig: int) -> torch.Tensor:

        if self.transform_type == 'dtcwt_complex':
            return self.idtcwt3d_complex(subbands, T_orig)
        elif self.transform_type == 'dtcwt':
            return self.idtcwt3d(subbands, T_orig)
        else:
            return self.idwt3d(subbands, T_orig)
    
    def dwt3d(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        B, C, T, H, W = x.shape

        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        low_2d, high_2d = self.dwt_2d(x_2d)

        hl_2d, lh_2d, hh_2d = high_2d[0][:,:,0], high_2d[0][:,:,1], high_2d[0][:,:,2]
        _, _, H2, W2 = low_2d.shape

        subbands = {}
        spatial_mapping = {'LL': 'aa', 'LH': 'ad', 'HL': 'da', 'HH': 'dd'}
        
        for name_2d, coeff_2d in [('LL', low_2d), ('LH', lh_2d), ('HL', hl_2d), ('HH', hh_2d)]:
            coeff_5d = rearrange(coeff_2d, '(b t) c h w -> b t c h w', b=B, t=T)
            coeff_1d_input = rearrange(coeff_5d, 'b t c h w -> (b h w) c t')
            
            low_1d, high_1d = self.dwt_1d(coeff_1d_input)
            
            low_5d = rearrange(low_1d, '(b h w) c t -> b c t h w', b=B, h=H2, w=W2)
            high_5d = rearrange(high_1d[0], '(b h w) c t -> b c t h w', b=B, h=H2, w=W2) if len(high_1d) > 0 else torch.zeros_like(low_5d)
            
            spatial_name = spatial_mapping[name_2d]
            subbands['a' + spatial_name] = low_5d
            subbands['d' + spatial_name] = high_5d
        
        return subbands
    
    def idwt3d(self, subbands: Dict[str, torch.Tensor], T_orig: int) -> torch.Tensor:

        B, C, T_sub, H2, W2 = subbands['aaa'].shape
        
        spatial_mapping = {'aa': 'LL', 'ad': 'LH', 'da': 'HL', 'dd': 'HH'}
        reconstructed_2d = {}
        
        for spatial_name, name_2d in spatial_mapping.items():
            low_5d = subbands['a' + spatial_name]
            high_5d = subbands['d' + spatial_name]
            
            low_1d = rearrange(low_5d, 'b c t h w -> (b h w) c t')
            high_1d = rearrange(high_5d, 'b c t h w -> (b h w) c t')
            
            rec_1d = self.idwt_1d((low_1d, [high_1d]))[..., :T_orig]
            rec_5d = rearrange(rec_1d, '(b h w) c t -> b c t h w', b=B, h=H2, w=W2)
            reconstructed_2d[name_2d] = rearrange(rec_5d, 'b c t h w -> (b t) c h w')
        
        low_2d = reconstructed_2d['LL']
        high_2d = torch.stack([reconstructed_2d['HL'], reconstructed_2d['LH'], reconstructed_2d['HH']], dim=2)
        
        x_2d = self.idwt_2d((low_2d, [high_2d]))
        return rearrange(x_2d, '(b t) c h w -> b c t h w', b=B, t=T_orig)
    
    def dtcwt3d(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        B, C, T, H, W = x.shape

        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        Yl, Yh = self.dtcwt_2d(x_2d)  

        H_high, W_high = Yh[0].shape[3], Yh[0].shape[4]  
        
        subbands = {}

        Yl_down = F.avg_pool2d(Yl, kernel_size=2, stride=2)  
        low_5d = rearrange(Yl_down, '(b t) c h w -> b t c h w', b=B, t=T)
        low_1d_input = rearrange(low_5d, 'b t c h w -> (b h w) c t')
        low_1d, high_1d = self.dwt_1d(low_1d_input)
        
        low_5d_a = rearrange(low_1d, '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
        subbands['low'] = low_5d_a  

        Yh_5d = rearrange(Yh[0], '(b t) c d h w ri -> b t c d h w ri', b=B, t=T)

        for i in range(6):
            
            dir_coeffs = Yh_5d[:, :, :, i, :, :, :]  
            dir_real = dir_coeffs[..., 0]  
            dir_imag = dir_coeffs[..., 1]
            
            dir_mag = torch.sqrt(dir_real**2 + dir_imag**2 + 1e-8)

            dir_1d_input = rearrange(dir_mag, 'b t c h w -> (b h w) c t')
            low_1d, high_1d = self.dwt_1d(dir_1d_input)
            
            low_5d = rearrange(low_1d, '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
            high_5d = rearrange(high_1d[0], '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
            
            subbands[f'd{i}_a'] = low_5d   
            subbands[f'd{i}_d'] = high_5d  
        
        return subbands
    
    def idtcwt3d(self, subbands: Dict[str, torch.Tensor], T_orig: int) -> torch.Tensor:

        B, C, T_sub, H2, W2 = subbands['low'].shape

        low_5d = subbands['low']
        low_1d = rearrange(low_5d, 'b c t h w -> (b h w) c t')
        
        high_1d = torch.zeros_like(low_1d)
        rec_low = self.idwt_1d((low_1d, [high_1d]))[..., :T_orig]
        rec_low_5d = rearrange(rec_low, '(b h w) c t -> b c t h w', b=B, h=H2, w=W2)
        
        rec_low_5d_up = F.interpolate(
            rearrange(rec_low_5d, 'b c t h w -> (b t) c h w'),
            scale_factor=2, mode='bilinear', align_corners=False
        )
        Yl = rec_low_5d_up  

        Yh_list = []
        for i in range(6):
            low_5d = subbands[f'd{i}_a']
            high_5d = subbands[f'd{i}_d']
            
            low_1d = rearrange(low_5d, 'b c t h w -> (b h w) c t')
            high_1d = rearrange(high_5d, 'b c t h w -> (b h w) c t')
            
            rec_1d = self.idwt_1d((low_1d, [high_1d]))[..., :T_orig]
            rec_5d = rearrange(rec_1d, '(b h w) c t -> b t c h w', b=B, h=H2, w=W2)
            rec_2d = rearrange(rec_5d, 'b t c h w -> (b t) c h w')

            rec_complex = torch.stack([rec_2d, torch.zeros_like(rec_2d)], dim=-1)
            Yh_list.append(rec_complex)

        Yh = [torch.stack(Yh_list, dim=2)]
        
        x_2d = self.idtcwt_2d((Yl, Yh))
        return rearrange(x_2d, '(b t) c h w -> b c t h w', b=B, t=T_orig)
    
    def dtcwt3d_complex(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        B, C, T, H, W = x.shape

        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        Yl, Yh = self.dtcwt_2d(x_2d)  
        
        H_high, W_high = Yh[0].shape[3], Yh[0].shape[4]
        subbands = {}

        Yl_down = F.avg_pool2d(Yl, kernel_size=2, stride=2)
        low_5d = rearrange(Yl_down, '(b t) c h w -> b t c h w', b=B, t=T)
        low_1d_input = rearrange(low_5d, 'b t c h w -> (b h w) c t')
        low_1d, _ = self.dwt_1d(low_1d_input)
        low_5d_a = rearrange(low_1d, '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
        subbands['low'] = low_5d_a

        Yh_complex = torch.view_as_complex(Yh[0].contiguous())  
        Yh_5d = rearrange(Yh_complex, '(b t) c d h w -> b t c d h w', b=B, t=T)
        
        for i in range(6):
            
            dir_complex = Yh_5d[:, :, :, i, :, :]  

            dir_real = dir_complex.real
            dir_imag = dir_complex.imag

            real_1d_input = rearrange(dir_real, 'b t c h w -> (b h w) c t')
            real_low, real_high = self.dwt_1d(real_1d_input)
            real_low_5d = rearrange(real_low, '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
            real_high_5d = rearrange(real_high[0], '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)

            imag_1d_input = rearrange(dir_imag, 'b t c h w -> (b h w) c t')
            imag_low, imag_high = self.dwt_1d(imag_1d_input)
            imag_low_5d = rearrange(imag_low, '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)
            imag_high_5d = rearrange(imag_high[0], '(b h w) c t -> b c t h w', b=B, h=H_high, w=W_high)

            low_complex = torch.complex(real_low_5d, imag_low_5d)   
            high_complex = torch.complex(real_high_5d, imag_high_5d)
            
            subbands[f'd{i}_a'] = low_complex   
            subbands[f'd{i}_d'] = high_complex  
        
        return subbands
    
    def idtcwt3d_complex(self, subbands: Dict[str, torch.Tensor], T_orig: int) -> torch.Tensor:

        B, C, T_sub, H2, W2 = subbands['low'].shape

        low_5d = subbands['low']
        low_1d = rearrange(low_5d, 'b c t h w -> (b h w) c t')
        high_1d = torch.zeros_like(low_1d)
        rec_low = self.idwt_1d((low_1d, [high_1d]))[..., :T_orig]
        rec_low_5d = rearrange(rec_low, '(b h w) c t -> b c t h w', b=B, h=H2, w=W2)
        rec_low_5d_up = F.interpolate(
            rearrange(rec_low_5d, 'b c t h w -> (b t) c h w'),
            scale_factor=2, mode='bilinear', align_corners=False
        )
        Yl = rec_low_5d_up

        Yh_list = []
        for i in range(6):
            
            low_complex = subbands[f'd{i}_a']   
            high_complex = subbands[f'd{i}_d']  

            real_low = low_complex.real
            imag_low = low_complex.imag
            real_high = high_complex.real
            imag_high = high_complex.imag

            real_low_1d = rearrange(real_low, 'b c t h w -> (b h w) c t')
            real_high_1d = rearrange(real_high, 'b c t h w -> (b h w) c t')
            rec_real = self.idwt_1d((real_low_1d, [real_high_1d]))[..., :T_orig]
            rec_real_5d = rearrange(rec_real, '(b h w) c t -> b t c h w', b=B, h=H2, w=W2)
            rec_real_2d = rearrange(rec_real_5d, 'b t c h w -> (b t) c h w')

            imag_low_1d = rearrange(imag_low, 'b c t h w -> (b h w) c t')
            imag_high_1d = rearrange(imag_high, 'b c t h w -> (b h w) c t')
            rec_imag = self.idwt_1d((imag_low_1d, [imag_high_1d]))[..., :T_orig]
            rec_imag_5d = rearrange(rec_imag, '(b h w) c t -> b t c h w', b=B, h=H2, w=W2)
            rec_imag_2d = rearrange(rec_imag_5d, 'b t c h w -> (b t) c h w')

            rec_complex = torch.stack([rec_real_2d, rec_imag_2d], dim=-1)
            Yh_list.append(rec_complex)
        
        Yh = [torch.stack(Yh_list, dim=2)]
        x_2d = self.idtcwt_2d((Yl, Yh))
        return rearrange(x_2d, '(b t) c h w -> b c t h w', b=B, t=T_orig)

class WNOBlock(nn.Module):
    """保留基础版的前归一化、局部路径、门控残差与通道映射。"""

    def __init__(self, width: int, size: list, n_layers: int = 2, wavelet: str = 'db4',
                 weight_mode: str = 'tucker', use_channel_mlp: bool = True,
                 channel_mlp_expansion: float = 0.5, transform_type: str = 'dwt', rank: float = 0.5) -> None:
        super().__init__()

        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")

        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise ValueError(f"size must be a list/tuple of 3 elements [T, H, W], got {size}")

        if any(s <= 0 for s in size):
            raise ValueError(f"All size dimensions must be positive, got {size}")

        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        if not 0 < channel_mlp_expansion <= 1.0:
            raise ValueError(f"channel_mlp_expansion must be in (0, 1], got {channel_mlp_expansion}")

        self.n_layers = n_layers
        self.use_channel_mlp = use_channel_mlp

        self.wavelet_convs = nn.ModuleList([
            WaveletConv3d(width, width, size=size, wavelet=wavelet, weight_mode=weight_mode, transform_type=transform_type, rank=rank)
            for _ in range(n_layers)
        ])
        self.local_convs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.InstanceNorm3d(width, affine=True) for _ in range(n_layers)])

        self.wno_skips = nn.ModuleList([SoftGating(width, n_dim=3) for _ in range(n_layers)])

        self.post_wno_norms = nn.ModuleList([nn.InstanceNorm3d(width, affine=True) for _ in range(n_layers)])

        if use_channel_mlp:
            hidden_channels = int(width * channel_mlp_expansion)
            self.channel_mlps = nn.ModuleList([
                ChannelMLP(in_channels=width, out_channels=width,
                          hidden_channels=hidden_channels, n_layers=2, n_dim=3)
                for _ in range(n_layers)
            ])
            self.mlp_skips = nn.ModuleList([SoftGating(width, n_dim=3) for _ in range(n_layers)])
            self.mlp_norms = nn.ModuleList([nn.InstanceNorm3d(width, affine=True) for _ in range(n_layers)])
            self.post_mlp_norms = nn.ModuleList([nn.InstanceNorm3d(width, affine=True) for _ in range(n_layers)])
        
    def forward(self, x: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:

        if index is not None:
            return self._forward_single(x, index)

        for i in range(self.n_layers):
            x = self._forward_single(x, i)
        return x

    def _forward_single(self, x: torch.Tensor, i: int) -> torch.Tensor:

        x_norm = self.norms[i](x)

        x_wno = self.wavelet_convs[i](x_norm) + self.local_convs[i](x_norm)
        x_skip_wno = self.wno_skips[i](x)
        x = x_wno + x_skip_wno  
        x = self.post_wno_norms[i](x)  

        if i < self.n_layers - 1 or self.use_channel_mlp:
            x = F.gelu(x)

        if self.use_channel_mlp:
            x_mlp = self.channel_mlps[i](self.mlp_norms[i](x))
            x_skip_mlp = self.mlp_skips[i](x)
            x = x_mlp + x_skip_mlp  
            x = self.post_mlp_norms[i](x)  
            
            if i < self.n_layers - 1:
                x = F.gelu(x)
        
        return x

class WNOBlocks(nn.Module):
    """用于替换运动后处理算子的堆叠小波组件。"""

    def __init__(self, in_channels: int, out_channels: int, size: Optional[list] = None,
                 n_layers: int = 1, wavelet: str = 'db4', weight_mode: str = 'tucker',
                 use_channel_mlp: bool = True, channel_mlp_expansion: float = 0.5,
                 transform_type: str = 'dwt', rank: float = 0.5, **kwargs) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"Channels must be positive, got in_channels={in_channels}, out_channels={out_channels}")

        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        if not 0 < channel_mlp_expansion <= 1.0:
            raise ValueError(f"channel_mlp_expansion must be in (0, 1], got {channel_mlp_expansion}")

        self.n_layers = n_layers
        self.use_channel_mlp = use_channel_mlp
        self.transform_type = transform_type

        if size is None:
            size = [8, 40, 40]
        elif not isinstance(size, (list, tuple)) or len(size) != 3 or any(s <= 0 for s in size):
            raise ValueError(f"size must be a list/tuple of 3 positive elements [T, H, W], got {size}")

        self.in_proj = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.wavelet_convs = nn.ModuleList([
            WaveletConv3d(out_channels, out_channels, size=size, wavelet=wavelet,
                         weight_mode=weight_mode, transform_type=transform_type, rank=rank)
            for _ in range(n_layers)
        ])
        self.local_convs = nn.ModuleList([nn.Conv3d(out_channels, out_channels, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.InstanceNorm3d(out_channels, affine=True) for _ in range(n_layers)])

        self.wno_skips = nn.ModuleList([SoftGating(out_channels, n_dim=3) for _ in range(n_layers)])

        self.post_wno_norms = nn.ModuleList([nn.InstanceNorm3d(out_channels, affine=True) for _ in range(n_layers)])

        if use_channel_mlp:
            hidden_channels = int(out_channels * channel_mlp_expansion)
            self.channel_mlps = nn.ModuleList([
                ChannelMLP(in_channels=out_channels, out_channels=out_channels,
                          hidden_channels=hidden_channels, n_layers=2, n_dim=3)
                for _ in range(n_layers)
            ])
            self.mlp_skips = nn.ModuleList([SoftGating(out_channels, n_dim=3) for _ in range(n_layers)])
            self.mlp_norms = nn.ModuleList([nn.InstanceNorm3d(out_channels, affine=True) for _ in range(n_layers)])
            self.post_mlp_norms = nn.ModuleList([nn.InstanceNorm3d(out_channels, affine=True) for _ in range(n_layers)])
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        x = self.in_proj(x)

        for i in range(self.n_layers):
            
            x_norm = self.norms[i](x)

            x_wno = self.wavelet_convs[i](x_norm) + self.local_convs[i](x_norm)
            x_skip_wno = self.wno_skips[i](x)
            x = x_wno + x_skip_wno
            x = self.post_wno_norms[i](x)  
            
            if i < self.n_layers - 1 or self.use_channel_mlp:
                x = F.gelu(x)

            if self.use_channel_mlp:
                x_mlp = self.channel_mlps[i](self.mlp_norms[i](x))
                x_skip_mlp = self.mlp_skips[i](x)
                x = x_mlp + x_skip_mlp
                x = self.post_mlp_norms[i](x)  
                
                if i < self.n_layers - 1:
                    x = F.gelu(x)
        
        return x

class WaveletConv3d(nn.Module):
    """各子带独立处理，支持稠密、张量分解及位置共享权重。"""

    def __init__(self, in_channels: int, out_channels: int, size: list, wavelet: str = 'db4',
                 weight_mode: Literal['official', 'tucker', 'shared'] = 'official',
                 rank: float = 0.5, transform_type: str = 'dwt', share_t: bool = False) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"Channels must be positive, got in_channels={in_channels}, out_channels={out_channels}")

        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise ValueError(f"size must be a list/tuple of 3 elements [T, H, W], got {size}")

        if any(s <= 0 for s in size):
            raise ValueError(f"All size dimensions must be positive, got {size}")

        if weight_mode not in ['official', 'tucker', 'shared']:
            raise ValueError(f"weight_mode must be one of ['official', 'tucker', 'shared'], got {weight_mode}")

        if transform_type not in ['dwt', 'dtcwt', 'dtcwt_complex']:
            raise ValueError(f"transform_type must be one of ['dwt', 'dtcwt', 'dtcwt_complex'], got {transform_type}")

        if not 0 < rank <= 1.0:
            raise ValueError(f"rank must be in (0, 1], got {rank}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.wavelet = wavelet
        self.weight_mode = weight_mode
        self.transform_type = transform_type
        self.share_t = share_t  

        self.dwt_module = DWT3d(wavelet=wavelet, level=1, transform_type=transform_type)

        subband_names = self.dwt_module.get_subband_names()

        dummy = torch.randn(1, 1, *size)
        dummy_subbands = self.dwt_module.forward_transform(dummy)
        first_subband_key = subband_names[0]
        _, _, modes_t, modes_h, modes_w = dummy_subbands[first_subband_key].shape
        self.modes = (modes_t, modes_h, modes_w)

        if self.share_t:
            weight_shape = (in_channels, out_channels, modes_h, modes_w)  
        else:
            weight_shape = (in_channels, out_channels, modes_t, modes_h, modes_w)  
        
        scale = (2 / (in_channels + out_channels)) ** 0.5

        self.complex_subbands = set()
        if transform_type == 'dtcwt_complex':
            for name in subband_names:
                if name != 'low':  
                    self.complex_subbands.add(name)

        if weight_mode == 'official':
            self.weights = nn.ParameterDict()
            for name in subband_names:
                if name in self.complex_subbands:
                    
                    weight_real = scale * torch.randn(*weight_shape)
                    weight_imag = scale * torch.randn(*weight_shape)
                    weight_complex = torch.complex(weight_real, weight_imag)
                    self.weights[name] = nn.Parameter(weight_complex)
                else:
                    self.weights[name] = nn.Parameter(scale * torch.randn(*weight_shape))

        elif weight_mode == 'tucker':
            self.weights = nn.ModuleDict({
                name: _TLTorchWrapper(weight_shape, rank=rank, factorization='tucker')
                for name in subband_names
            })
            
            if transform_type == 'dtcwt_complex':
                import warnings
                warnings.warn("复小波的张量分解模式沿用基础版实值权重近似，不学习独立虚部。")

        else:  
            self.weights = nn.ModuleDict()
            for name in subband_names:
                if name in self.complex_subbands:
                    
                    self.weights[name + '_re'] = nn.Conv3d(in_channels, out_channels, 1, bias=False)
                    self.weights[name + '_im'] = nn.Conv3d(in_channels, out_channels, 1, bias=False)
                else:
                    self.weights[name] = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, T, H, W = x.shape

        subbands = self.dwt_module.forward_transform(x)

        transformed = {}
        for name, sb in subbands.items():
            is_complex = name in self.complex_subbands

            if self.weight_mode == 'shared':
                if is_complex:
                    
                    sb_real = sb.real
                    sb_imag = sb.imag
                    W_re = self.weights[name + '_re']
                    W_im = self.weights[name + '_im']
                    out_real = W_re(sb_real) - W_im(sb_imag)
                    out_imag = W_re(sb_imag) + W_im(sb_real)
                    transformed[name] = torch.complex(out_real, out_imag)
                else:
                    transformed[name] = self.weights[name](sb)
            else:
                W = self.weights[name]
                if hasattr(W, 'to_tensor'):
                    W = W.to_tensor()
                # 保留实值分解权重的含义，只匹配复系数运算所需的数据类型。
                if sb.is_complex() and not W.is_complex():
                    W = W.to(dtype=sb.dtype)

                if is_complex and self.weight_mode == 'official':
                    
                    _, _, t, h, w = sb.shape
                    wt, wh, ww = self.modes
                    
                    if t == wt and h == wh and w == ww:
                        
                        transformed[name] = torch.einsum("bixyz,ioxyz->boxyz", sb, W)
                    else:
                        
                        raise RuntimeError(
                            f"WaveletConv3d size mismatch in complex mode! "
                            f"subband '{name}' shape (t,h,w)=({t},{h},{w}) vs weight modes ({wt},{wh},{ww}). "
                            f"Please ensure model 'size' parameter matches input spatial dimensions."
                        )
                else:
                    _, _, t, h, w = sb.shape
                    wt, wh, ww = self.modes

                    if self.share_t:
                        
                        _, _, wh, ww = W.shape
                        if h == wh and w == ww:

                            transformed[name] = torch.einsum("bityz,ioyz->botyz", sb, W)
                        else:
                            
                            raise RuntimeError(
                                f"WaveletConv3d size mismatch in share_t mode! "
                                f"subband '{name}' shape (h,w)=({h},{w}) vs weight shape ({wh},{ww}). "
                                f"Please ensure model 'size' parameter matches input spatial dimensions."
                            )
                    else:
                        
                        if t == wt and h == wh and w == ww:
                            transformed[name] = torch.einsum("bixyz,ioxyz->boxyz", sb, W)
                        else:
                            
                            raise RuntimeError(
                                f"WaveletConv3d size mismatch! "
                                f"subband '{name}' shape (t,h,w)=({t},{h},{w}) vs weight modes ({wt},{wh},{ww}). "
                                f"Please ensure model 'size' parameter matches input spatial dimensions."
                            )

        # 奇数空间尺寸可能在逆变换后多出一个采样点，裁回输入域。
        return self.dwt_module.inverse_transform(transformed, T)[..., :x.shape[-2], :x.shape[-1]]
    
    def num_params(self) -> int:

        return sum(p.numel() for p in self.parameters())

class _TLTorchWrapper(nn.Module):
    """按基础版初始化张量分解权重，仅在需要时导入依赖。"""

    def __init__(self, shape: tuple, rank: float = 0.5, factorization: str = 'tucker') -> None:
        super().__init__()
        from tltorch import FactorizedTensor
        self.tensor = FactorizedTensor.new(shape, rank=rank, factorization=factorization)
        self.tensor.normal_(0, (2 / (shape[0] + shape[1])) ** 0.5)

    def to_tensor(self) -> torch.Tensor:

        return self.tensor.to_tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.einsum("bixyz,ioxyz->boxyz", x, self.to_tensor())

__all__ = ['DWT3d', 'WaveletConv3d', 'WNOBlock', 'WNOBlocks']

class CODALayer(nn.Module):
    """保留基础版的函数令牌、单头注意力、归一化及残差，仅替换交互算子。"""
    def __init__(self, width, size, wavelet='db4', weight_mode='tucker',
                 transform_type='dwt', temperature=0.8, rank=0.7):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.width = width
        self.temperature = temperature
        # 基础版默认单头，三个投影各是一层线性 ChannelMLP。
        self.Key = ChannelMLP(width, width, n_layers=1, non_linearity=nn.Identity())
        self.Query = ChannelMLP(width, width, n_layers=1, non_linearity=nn.Identity())
        self.Value = ChannelMLP(width, width, n_layers=1, non_linearity=nn.Identity())
        self.norm1 = nn.InstanceNorm3d(width, affine=True)
        self.attention_normalizer = nn.InstanceNorm3d(width, affine=True)
        self.mixer_in_normalizer = nn.InstanceNorm3d(width, affine=True)
        self.mixer_out_normalizer = nn.InstanceNorm3d(width, affine=True)
        # 两层交互算子逐令牌共享参数，沿用基础版的排列等变分支。
        self.mixer = WNOBlock(width, size, n_layers=2, wavelet=wavelet,
                              weight_mode=weight_mode, transform_type=transform_type,
                              rank=rank, channel_mlp_expansion=1.0)

    def forward(self, first, second):
        batch, channels, time, height, width = first.shape
        tokens = torch.stack((first, second), dim=1).reshape(batch * 2, channels, time, height, width)
        normalized = self.norm1(tokens)
        q, k, v = [op(normalized).reshape(batch, 2, -1)
                   for op in (self.Query, self.Key, self.Value)]
        weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) /
                                (q.shape[-1] ** 0.5 * self.temperature), dim=-1)
        attention = torch.bmm(weights, v).reshape_as(tokens)
        attention = self.attention_normalizer(attention + tokens)
        output = self.mixer(self.mixer_in_normalizer(attention))
        output = self.mixer_out_normalizer(output) + attention
        output = output.reshape(batch, 2, channels, time, height, width)
        return output[:, 0], output[:, 1], weights

class eano(nn.Module):
    """无标签双分支时空模型；速度就是像素 XY 相邻位移，不进行指数积分。"""
    def __init__(self, model_config, sequence_config):
        super().__init__()
        from ulits import FourierGrad, SequenceWindow
        self.model_config = dict(model_config)
        self.cyclic = sequence_config['cyclic']
        self.grid_type = sequence_config['grid_type']
        self.input_size = tuple(model_config['input_size'])
        self.composition_interpolation = sequence_config['composition_interpolation']
        self.window = SequenceWindow(sequence_config['window_size'],
                                     sequence_config['overlap'], self.cyclic)
        self.grad = FourierGrad()
        width = model_config['width']
        if width < 2 or len(self.input_size) != 2 or min(self.input_size) < 4:
            raise ValueError("Require width>=2 and a spatial size>=4")
        if self.grid_type not in ('linear', 'periodic'):
            raise ValueError("grid_type must be linear or periodic")
        if model_config['levels'] != 1:
            raise ValueError("The baseline wavelet implementation supports levels=1 only")
        grid_channels = 3 if self.grid_type == 'linear' else 4
        hidden = model_config['lifting_hidden_channels']
        self.lifting_A = ChannelMLP(4 + grid_channels, width, hidden, n_layers=2)
        self.lifting_B = ChannelMLP(1 + grid_channels, width, hidden, n_layers=2)
        size = [self.window.size, *self.input_size]
        common = dict(size=size, wavelet=model_config['wavelet'],
                      weight_mode=model_config['weight_mode'],
                      transform_type=model_config['transform_type'], rank=model_config['rank'])
        depths = model_config['stage_depths']
        if len(depths) != 3 or any(depth < 1 for depth in depths):
            raise ValueError("stage_depths must contain three positive integers")
        self.flow_stages = nn.ModuleList([
            WNOBlock(width, n_layers=depth, channel_mlp_expansion=model_config['channel_mlp_ratio'],
                     **common) for depth in depths])
        self.stress_stages = nn.ModuleList([
            WNOBlock(width, n_layers=depth, channel_mlp_expansion=model_config['channel_mlp_ratio'],
                     **common) for depth in depths])
        coda = model_config['coda']
        # 基础版固定两次交互，不改变令牌数量或交互位置。
        self.coda = nn.ModuleList([
            CODALayer(width, size, wavelet=model_config['wavelet'],
                      weight_mode=model_config['weight_mode'],
                      transform_type=model_config['transform_type'],
                      temperature=coda['temperature'], rank=coda['rank']) for _ in range(2)])
        hidden = model_config['projection_hidden_channels']
        self.projection_A = ChannelMLP(width, 2, hidden, n_layers=2)
        self.projection_B = ChannelMLP(width, 3, hidden, n_layers=2)
        # 尺寸传原始窗口大小，子带大小由实际小波变换推导，避免重复减半。
        post = dict(common, rank=model_config['postprocess_rank'])
        self.VecInt = WNOBlocks(2, 2, n_layers=model_config['postprocess_depth'], **post)
        self.VecInt_backward = WNOBlocks(2, 2, n_layers=model_config['postprocess_depth'], **post)
        self.scale = model_config.get('output_scale_pixels')
        if self.scale is None:
            self.scale = (self.input_size[0] - 1) * 0.5

    def _predict_window(self, feature_a, feature_b):
        a, b = self.lifting_A(feature_a), self.lifting_B(feature_b)
        weights = []
        for stage in range(3):
            a, b = self.flow_stages[stage](a), self.stress_stages[stage](b)
            if stage < 2:
                a, b, attention = self.coda[stage](a, b)
                weights.append(attention)
        raw = self.projection_A(a) * self.scale
        # 保持基础版两套独立后处理及正负输入关系；对外统一命名前向位移。
        return {'flow_z': raw, 'inf_flow': self.VecInt_backward(-raw),
                'neg_flow': self.VecInt(raw), 'stress': self.projection_B(b)}, torch.stack(weights, dim=1)

    def forward(self, images):
        from ulits import make_grid, accumulate
        if images.ndim != 5 or images.shape[1] != 1 or images.shape[2] < 2:
            raise ValueError("images must have shape [B,1,T,H,W], T>=2")
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError("Image spatial size must match model.input_size")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("images must be finite floating point tensors")
        batch, _, frames, _, _ = images.shape
        # 与基础版一致，在窗口内计算导数；坐标在完整序列生成后再切窗。
        grid = make_grid(images, self.grid_type)
        indices, valid = self.window.indices(frames, images.device)
        image_chunks = self.window.split(images, indices)
        grid_chunks = self.window.split(grid, indices)
        dx, dy, dt, image = self.grad(image_chunks)
        a = torch.cat((image, dx, dy, dt, grid_chunks), dim=1)
        b = torch.cat((image, grid_chunks), dim=1)
        prediction, attention = self._predict_window(a, b)
        prediction = {key: self.window.merge(value, indices, valid, batch, frames)
                      for key, value in prediction.items()}
        pairs = frames if self.cyclic else frames - 1
        prediction = {key: value[:, :, :pairs] for key, value in prediction.items()}
        prediction['inf_lag_flow'] = accumulate(prediction['inf_flow'], backward=False,
                                               interpolation=self.composition_interpolation)
        prediction['neg_lag_flow'] = accumulate(prediction['neg_flow'], backward=True,
                                               interpolation=self.composition_interpolation)
        prediction['sigma_xx'], prediction['sigma_yy'], prediction['sigma_xy'] = prediction['stress'].split(1, dim=1)
        prediction['attention'] = attention.reshape(batch, indices.shape[0], 2, 2, 2)
        return prediction

__all__ += ['EANOX', 'CODALayer']

