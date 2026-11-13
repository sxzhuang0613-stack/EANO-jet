# EANOX

基于本地 WNO 基础版整理的独立实现：时空双分支、三阶段、两次 CODA、
双向运动与单向应力。删除全部标签训练依赖；原交互与运动后处理中的
傅里叶神经算子全部改为小波神经算子。原工程文件未修改。

## 文件

| 文件 | 用途 |
|---|---|
| `main.py` | 训练、验证、预测、断点恢复及合成检查入口 |
| `components.py` | 基础版小波组件、通道映射、CODA、EANOX |
| `losses.py` | 两组损失及诊断输出 |
| `config.yaml` | 外部参数 |
| `ulits.py` | 坐标、谱导数、采样、累计复合、汉明窗口重建 |
| `dataloader.py` | 用户提供的带元数据 NPZ 病例加载器，保留原读取和预处理逻辑 |
| `data_adapter.py` | 病例/切片布局转换、变长批次组织与数据协议记录 |
| `config_utils.py` | 保留原渲染配置入口，新增独立 EANOX 配置解析 |
| `test_eanox.py` | 合成几何、梯度、重建及恢复验证 |

## 运行

训练和验证进度条默认开启，可在配置中设置 `training.progress_bar: false`，
或运行时添加 `--no-progress-bar` 关闭，`--progress-bar` 显式开启。

训练循环只运行训练，不构造或运行验证/测试集。每轮结束后在终端按表格输出
总损失、各项原始损失和加权贡献，同时显示学习率与耗时；完整历史仍写入
`history.jsonl`。检查点只保留覆盖更新的 `last.pt`，不再保存最佳或逐轮权重。
在同一输出目录恢复训练后，成功保存最新检查点时会清理旧策略产生的
`best.pt` 与 `epoch_数字.pt`；预测结果等其他文件不受影响。
独立的 `--mode validate` 仍可在需要时手动运行。

每轮结束后使用更新后的模型，从训练集随机选择一个切片时间序列，绘制第一帧
至各帧的累计前向位移。全部帧拼在同一张 `输出目录/displacement_hsv.png` 中，
新一轮覆盖旧图。第一幅为白色恒等参考帧；周期模式最后一幅保留完整周期闭合
位移。全序列共用幅度标尺：方向映射为色相，幅度映射为饱和度，零位移显示白色。
绘图不读取分割标签；默认小于 0.05 像素的位移显示纯白。
`visualization` 下可设置 `enabled`、`columns`、`dpi`、`white_threshold_pixels`。
可视化随机抽样与前向不改变训练随机状态，不生成额外的逐轮图片文件。

需要 Python 3.10 或更新版本，依赖见 `requirements.txt`。无需安装原工程，
也不依赖其任何模型或工具。当前机器可使用：

```bash
/home/zhuangshuxin/miniconda3/envs/3dgs/bin/python EANO/main.py --smoke --device cpu --output-dir /tmp/eanox-check-wno
/home/zhuangshuxin/miniconda3/envs/3dgs/bin/python EANO/main.py --smoke --loss-group eano --device cpu --output-dir /tmp/eanox-check-eano
/home/zhuangshuxin/miniconda3/envs/3dgs/bin/python -m unittest EANO.test_eanox EANO.test_data_adapter -v
```

以上命令从 `src` 目录运行。也可从任意位置用绝对路径执行 `main.py`。
`--smoke` 显式将空间尺寸改为 16×16、宽度改为 4、使用位置共享权重，
训练一轮合成序列；完整有效配置写入输出目录。它不代表默认模型或真实数据精度。
重复检查应使用新目录，或显式恢复。

真实训练已接入用户的 NPZ loader，先在 `config.yaml` 设置 `dataset_name`、
`dataset_paths`、`train_fraction` 和 `normalization_epsilon`：

```bash
python EANO/main.py --config EANO/config.yaml --output-dir outputs/run01
python EANO/main.py --config EANO/config.yaml --mode validate --checkpoint outputs/run01/last.pt
python EANO/main.py --config EANO/config.yaml --mode predict --checkpoint outputs/run01/last.pt --input sequence.pt
```

预测输入文件可以是图像张量，也可以是包含配置指定图像键的字典。
也支持保存的原生 loader 字典，包含 `cmr` 和 `info.cmr_layout`。
预测结果保存在 `prediction.pt`，原生病例的元数据另存为 `prediction_info.pt`。
恢复训练通过 `--checkpoint` 或
`runtime.resume` 指定本实现保存的检查点。旧工程权重不兼容。

## 输入输出与几何

真实数据加载器读取 `cmr`、`labels`、`info` 三项组成的 NPZ。其 JSON 元数据
必须含轴序、相位索引和原始间距等字段；标签相位可以为空，但结构字段仍需存在。
训练仅消费 `cmr`，不会使用标签损失或按标签覆盖率筛选病例。

默认 `data.dim: 2D`：病例文件名排序后按 `train_fraction` 划分，再展开切片；
每个切片保留完整时间轴，原生 `[T,H,W]` 转为 `[1,1,T,H,W]`。
这沿用 loader 的比例划分，不改为 ACDC 官方训练/测试划分。
可选 `data.dim: 3D`：原生 `[S,T,H,W]` 逐切片运行，完整病例全部切片平均梯度
和日志，使病例等权。切片模式下则是切片等权，两种训练单位不应混作同一实验。

批次采用原生样本列表，支持不同病例时间长度、切片数和标签相位数，无需补时间帧。
`batch_size` 表示切片数或病例数；每批按记录数平均，再按实际梯度累积批数平均。
loader 已完成完整病例归一化和固定 128×128 缩放，适配层不再重复处理。
真实数据要求 `model.input_size: [128,128]`；合成检查仍可使用更小尺寸。
当前 loader 的增强分支未执行实际空间变换，适配器要求关闭增强。

EANOX 使用 `config_utils.load_eanox_config`，不触发原渲染项目的配置检查。
数据目录的相对路径以配置文件目录为基准。检查点记录数据集、路径、
训练患者列表及预处理参数，恢复训练时核对，避免悄悄改变实验划分。

- 输入图像：`[B,1,T,H,W]`，已预处理的有限浮点数，T 至少为 2；模型不二次归一化。
- 空间尺寸固定为 `model.input_size`；同一批内时间长度一致。
- 自动生成网格：线性编码为 3 通道，周期编码为 4 通道；两分支输入通道自动匹配。
- 输出向量统一为像素 XY 分量；相较基础版的 YX 存储顺序已显式统一。
- `inf_flow`、`neg_flow`：正反向相邻位移，即速度，不使用独立时间步长或指数积分。
- `inf_lag_flow`、`neg_lag_flow`：按映射方向复合的累计位移，不是逐项相加。
- 前向图像对齐读取 `目标图像(恒等坐标+inf_flow)`，与源图像比较；反向对齐相反。
- 应力 `stress` 为 `[B,3,K,H,W]`，依次为 xx、yy、xy；额外导出对应单通道字段。
- 运动场均为 `[B,2,K,H,W]`；循环序列 K=T，非循环序列 K=T−1。
- 累计场按帧对终点排列，不插入零参考帧。循环模式最后一项对应完整周期闭合，目标为首帧。
- `flow_z` 仅为中间场，不直接参与任何运动损失。
- `attention` 为 `[B,窗口数,2次交互,2个查询分支,2个键分支]`。

保留基础版周期训练、8 帧窗口、4 帧重叠、汉明加权重建、累计最近邻采样、
图像双线性采样和零边界。短序列循环取索引；非循环模式边界填充不参与窗口权重。
多样本窗口重组已修正，不混合病例。损失在重建后的完整序列计算一次。

## 组件约定

保留基础版独立 lifting、双分支深度 `[2,2,1]`、门控残差与归一化。
CODA 的 Q/K/V 仍为独立的一层线性 ChannelMLP；单头注意力对整张时空场
的两个分支令牌计算。保持温度、令牌归一化、注意力残差、两层共享交互
算子及输出残差，不改成交互结果只进入局部网络的另一种架构。

小波变换保留基础版的空间二维加时间一维实现，默认空间 db4、时间 db4、
周期延拓、单层分解。基础代码的变换重建只消费一层子带，因此 `levels`
仅允许 1。支持 `shared`、`official`、`tucker` 三类权重，张量分解依赖按需导入。
CODA 与运动后处理使用同一类小波组件，分别保留自己的层数和秩参数。

`FourierGrad` 是基础版的谱求导工具，不是神经算子网络层，继续保留。
基础组使用谱导数；EANO 组的物理空间差分沿用其来源实现，并匹配归一化坐标步长。

## 损失切换

只需修改 `losses.group`，或使用 `--loss-group`：

| 分组 | 运动项 | 正向物理项 |
|---|---|---|
| `wno_base` | 双向相邻及累计 MSE、双向 GN 光流 | 平衡、线弹性本构、能量成本、正向单步及累计形变正则 |
| `eano` | 双向相邻及累计 L1、双向保守输运光度 | 平衡、累计形变下的相对 Kirchhoff 本构 |

两组都不使用分割标签，不预测材料参数，不生成反向应力；反向运动不进入物理项。
两组使用相同网络和映射定义。基础组保留原公式，但修正 GN 运算的时间/通道
维度混用，并将运动约束统一到后处理后的位移。正向线弹性系数在全图固定，
不再按区域调制。原能量成本可能为负，作为单独监测值保留。

EANO 组沿用当前来源实现的逐点 `log1p` 光度/平衡残差及相对 Kirchhoff 本构，
将预测应力拉回参考构形后乘 Jacobian。空间与时间平均方式保留来源约定：
光度、平衡、本构取均值；L1 对齐先平均空间，再对批次及帧对求和。因此不同
序列长度的总损失不能直接比较。双向运动项按两方向求和，不额外取平均。

基础组支持固定权重或来源训练入口的不确定性权重；EANO 组只使用固定权重。
每项权重为 0 时不接入总损失计算图，仍记录该项原始值。

记录包含 `terms`、`details`、`contributions`、`diagnostics`：原始损失、内部子项、
实际加权贡献、原始 Jacobian 与场幅度。Jacobian 诊断使用未截断累计位移，
不把负 Jacobian 改为正值。基础组能量内部仍保留来源截断公式。

## 验证范围

自动验证覆盖两组双向运动和应力梯度、反向位移不进入物理项、恒等形变零损失、
平移与仿射几何、短序列/长序列/多批次窗口重建、CODA 令牌交换等变性、
默认离散小波重建、三类小波权重、尾批梯度累积及检查点恢复。
合成验证不代表心脏影像训练效果或无折叠保证。

复小波配合张量分解时沿用基础版的实值核近似，不学习独立虚部；已补齐实数
权重与复数系数的类型适配。默认离散小波不涉及这一近似。
