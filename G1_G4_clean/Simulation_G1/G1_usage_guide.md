# G1 Simulation Usage Guide

## 配方训练集：2026-10-23

入口为 `python -m Simulation_G1.training_recipes`。只生成配置、空间支撑和索引，
不预先生成完整源／EEG 数组。当前使用一个共享网格／leadfield；记录内容哈希，
后续多种前向模型的组合与资源加载另行扩展。

### 数据量与难度

推荐先用2,500个支撑验证训练流程，再扩大到默认25,000个支撑。按支撑做80/10/10划分，
每个训练支撑32个变体，每个验证／测试支撑16个变体：

| 阶段 | 训练样本 | 验证样本 | 测试样本 | 总配方数 |
|---|---:|---:|---:|---:|
| 小规模训练：2,500支撑 | 64,000 | 4,000 | 4,000 | 72,000 |
| 标准训练：25,000支撑 | 640,000 | 40,000 | 40,000 | 720,000 |
| 学习曲线仍受益时扩至50,000支撑 | 1,280,000 | 80,000 | 80,000 | 1,440,000 |

数量是初始训练设计，不是充分样本量证明。优先增加独立空间支撑，避免仅增加同一支撑的噪声重复。
四级难度按支撑约20/35/35/10分配；各级参数范围有意重叠，避免把一个波形类型等同一种难度。

| 难度 | 源数 | 各源面积cm² | 请求sensor SNR dB | 有源背景时source SNR dB |
|---|---|---|---|---|
| easy | 1–2，其中单源抽样概率60% | 0.5–4 | 15–30 | 25–40 |
| medium | 2–4 | 0.5–8 | 5–20 | 15–30 |
| hard | 3–6 | 0.25–12 | −5–10 | 8–20 |
| extreme | 4–8 | 0.25–20 | −10–5 | 0–12 |

预计约88%为多源；实际比例见`dataset.json`。面积逐源按对数均匀分布抽样，可同场出现大小源。
难度越高通常源越多、SNR越低；面积大小本身不是严格单调的定位难度指标。
所有前景patch均完整、连接且不重叠。困难场景也不会以放宽重叠约束来凑样本。

覆盖14个已有波形族，同族不同参数、混合波形、同步波形，以及独立／相关／离散因果耦合。
均匀权重、中心衰减、扩散、收缩、扩散后回缩五类空间形式按五个变体一组安排；
组内共享时间波形配置和相关随机流。传感器噪声／SNR可变化，因此不是纯粹的单因素对照实验。
单源不应用多源耦合。验证／测试较少的变体数也覆盖因果场景，不只在训练中出现。

### SNR与短事件

配方默认使用平均参考后的整窗`global_rms`标定sensor SNR；不以短暂高能窗口的功率来放大整段噪声。
`sensor_noise_peak_ratio=1`进一步限制全通道全时段的最大噪声绝对值不超过干净传感器信号的最大绝对值。
它只对整个噪声乘同一个缩小系数，不截断波形，不单独缩放各通道；这不保证每个通道／每个时刻噪声都小于信号。

限制触发时，实际SNR会高于请求值。记录`noise_peak_limit_applied`、`noise_peak_limit_scale`、
`realized_sensor_snr_db`，并额外输出`event_sensor_snr_db`：参考后干净EEG的GFP达到其峰值20%的时点上计算。
事件窗口SNR只是诊断，不用于标定。评估和学习难度应依据实际SNR，不把请求的负SNR范围当作已实现的覆盖。
要关闭峰值保护，可设CLI `--sensor-noise-peak-ratio 0`或Python参数`None`。

原G1配置默认不启用峰值限制，旧数值行为保留。新增`iid_white`表示真正的传感器白噪声（平均参考前）；
原`white`仍为带限且有共享成分的噪声。当前配方还可采样pink、powerlaw、specparam_resting和既有mixed。
真实噪声本轮不启用，原`real_sensor_noise`数组入口保留；自动加载、重采样、导联对齐及真实／白噪声按功率比例混合待后续实现。

### 生成和调用

以下命令会创建新目录，遇到已有目录拒绝覆盖。原路径使用工作区Python环境；
合集从其根目录运行，并用`--cortex`、`--leadfield`提供外部资产。

```powershell
# 初步训练：64,000训练配方，不产生64,000份EEG数组。
.\.venv\Scripts\python.exe -m Simulation_G1.training_recipes --output-dir generated_simulation_dataset/G1_recipes/train_stage1 --n-supports 2500

# 标准规模：640,000训练配方＋80,000验证/测试配方。
.\.venv\Scripts\python.exe -m Simulation_G1.training_recipes --output-dir generated_simulation_dataset/G1_recipes/train_standard --n-supports 25000 --variants-per-support 32 --eval-variants-per-support 16
```

默认2秒、250Hz，可用`--duration-s`和`--srate`更改；本波形采样方案要求至少1秒、100Hz。
较低频静息研究需要更长窗时应单独设计时长，不把短窗当作充分的低频谱估计。

```python
from Simulation_G1.training_recipes import RecipeDataset

# simulator为已加载相同mesh/leadfield的G1SourceSimulator，可常驻复用。
train = RecipeDataset("generated_simulation_dataset/G1_recipes/train_stage1", simulator, split="train")
sensor, clean, noise, source, metadata = train[0]
config, recipe = train.recipe(0)  # 只读取配方，不生成信号
validation = RecipeDataset("generated_simulation_dataset/G1_recipes/train_stage1", simulator, split="val")
```

默认只读取训练划分；`split=None`仅用于全库审阅。返回G1五项数据，不自动把不等长metadata拼成tensor；
训练时分别整理固定形状的sensor/source数组和metadata。此版提供随机访问与单实例资源复用，
尚未实现GPU生成或多worker批量预取，也未完成稀疏空间基底投影优化。

### 存储、复现与划分边界

`dataset.json`保存共享参数、资源指纹、代码／Python／NumPy／SciPy版本、分布统计；
`supports.jsonl`每个空间支撑只存一次；`recipes-*.jsonl`分块保存参数变体，配套offset索引；
`train/val/test.indices.npy`提供按划分随机访问，数组索引不是模拟数据。
`generator_snapshot/Simulation_G1`保留三个生成代码文件，以便归档旧版本；加载器不会自动执行这些快照。

配方固定实际顶点集合、中心、增益、完整波形参数和spatial/waveform/gain/background/sensor五个随机流。
逐源波形再由waveform种子和源序号派生，不依赖batch大小、访问顺序或worker编号。
重建时校验共享资源和生成代码／环境版本，拒绝静默用新版本解释旧配方；使用归档版本需显式恢复对应代码与环境。
同一环境下已验证JSON往返与重复索引读取逐元素一致；跨硬件逐位相同仍不作保证。

同一支撑的全部衍生变体归入同一划分；支撑ID对patch排列顺序不敏感，避免同一空间支撑换顺序后跨集合。
这属于具体场景支撑留出，不等于脑区严格留出；不同支撑仍可位于相邻／相同脑区。
严格空间域外泛化和跨头模型泛化需要另建解剖区域或forward级留出协议。

本轮实际生成128支撑／3,680配方的小集合，未生成标准规模72万配方，也未启动模型训练。


2026-10-20: Disjoint foreground patches and shared spatial dynamics are available. See [spatial scenarios](../SIMULATION_GUIDELINE.md#g1g2-spatial-scenarios-2026-10-20).

本文档是 `Simulation_G1` 的快速使用说明。本集合不附带历史设计计划，这里只保留实际生成数据时最常用的配置、推荐组合和注意事项。

## 1. 核心入口

推荐从稳定入口导入：

```python
from Simulation_G1.g1_simulation import (
    G1SimulationConfig,
    G1SourceSimulator,
    G1ScenarioSampler,
    G1DifficultyScenarioSampler,
)
```

常用生成接口：

```python
sensor, clean, noise, source, manifest = simulator.generate_sensor(config)
```

返回值含义：

```text
sensor: noisy scalp EEG/MEG, n_sensors x n_times
clean:  leadfield @ source, n_sensors x n_times
noise:  sensor-level noise, n_sensors x n_times
source: cortical source truth, n_sources x n_times
manifest: JSON-safe metadata
```

FS20k示例脚本：

```powershell
.\.venv\Scripts\python.exe Simulation_G1\generate_fs20k_g1_demo.py
```

示例输出：

```text
Simulation_G1/generated_fs20k_demo/
```

## 2. 必填配置

最小配置通常包括：

```python
config = G1SimulationConfig(
    sample_id="G1_example",
    template_id="fsaverage5",
    source_space_id="Cortex_FS_20k",
    leadfield_id="Gain_20k_FS_fixed_orientation",
    modality="EEG",
    srate=250,
    time_window=(-0.2, 0.8),
    n_patch=1,
    patch_area_mm2=180.0,
    decay_mm=12.0,
    waveform_params=[
        ("erp", {
            "components": [
                {"type": "gaussian", "latency_s": 0.12, "width_s": 0.045, "amplitude_rel": 1.0}
            ],
            "amplitude": 1.0,
        })
    ],
    sensor_noise_type="pink",
    sensor_snr_db=12.0,
    sensor_snr_mode="robust_global_rms",
    random_seed=12345,
)
```

## 3. Source Time-Series Types

### Event-Like

适合定位、ERP/spike重建、低维可解释benchmark。

```text
erp
spike
gaussian_pulse
```

推荐用途：

- `erp`: 事件相关成分，适合验证定位偏差和时间重建。
- `spike`: 短暂高能量活动，适合测试稀疏源成像。
- `gaussian_pulse`: 简单单峰控制样本，适合debug。

### Oscillatory

适合频段活动、ERSP、burst和节律重建。

```text
ersp
resting
oscillation
burst_oscillation
nonsinusoidal_oscillation
ar
```

推荐用途：

- `ersp`: 事件锁定节律增强/抑制。
- `resting`: 连续band-limited随机节律。
- `oscillation`: 连续振荡，可加平滑包络和频率漂移。
- `burst_oscillation`: 间歇性节律活动。
- `nonsinusoidal_oscillation`: 含谐波的非正弦alpha/theta。
- `ar`: AR(2)式窄带振荡。

### Aperiodic / Resting-Like

适合模拟真实背景源活动、1/f非周期信号和resting-state风格信号。

```text
powerlaw
knee_powerlaw
specparam
synaptic_noise
eegsourcesim_resting_noise
resting_pinknoise
pink
white
```

推荐用途：

- `powerlaw`: Timmer-Konig式1/f非周期活动。
- `knee_powerlaw`: 带低频knee的1/f活动。
- `specparam`: aperiodic背景 + Gaussian periodic peak。
- `synaptic_noise`: Poisson突触事件卷积，非正弦、宽频。
- `eegsourcesim_resting_noise`: resting-like 1/f + alpha局部峰。

## 4. Source Background

`source_background_type` 是加在源空间的低秩背景活动，和sensor noise不同。

可选：

```text
None
pink
powerlaw
knee_powerlaw
specparam
synaptic
eegsourcesim_resting
alpha
beta
theta
delta
gamma
white
```

使用建议：

- 干净验证：`source_background_type=None`
- 训练数据：`powerlaw`、`specparam`、`eegsourcesim_resting`
- 难例：`synaptic`、`eegsourcesim_resting`、`alpha`

如果设置了 `source_background_type`，通常同时设置：

```python
source_snr_db = 25.0  # 常用范围 20-35 dB
```

## 5. Sensor Noise

`sensor_noise_type` 是头皮传感器层噪声。

可选：

```text
white
pink
powerlaw
specparam_resting
mixed
eegsourcesim_resting
real
```

使用建议：

- `white`: 最干净的基线噪声。
- `pink`: 温和1/f-like噪声，适合默认benchmark。
- `powerlaw`: 更明确的1/f随机噪声。
- `specparam_resting`: 1/f + moderate alpha peak，适合resting-like EEG。
- `mixed`: 混合非周期、宽带和resting成分，适合训练。
- `eegsourcesim_resting`: 更接近resting EEG风格，但不要过度用于所有样本。
- `real`: 注入真实噪声片段，需要传入 `real_sensor_noise`。

真实噪声注入：

```python
config.sensor_noise_type = "real"
sensor, clean, noise, source, manifest = simulator.generate_sensor(
    config,
    real_sensor_noise=real_noise,  # shape: channels x times, or times
)
```

## 6. SNR Settings

推荐默认：

```python
sensor_snr_mode = "robust_global_rms"
sensor_snr_clip_quantile = 0.95
```

不同信号推荐：

```text
continuous/resting/oscillation: robust_global_rms
erp/spike/burst:              robust_global_rms or active_rms
strict legacy comparison:     global_rms
```

推荐SNR范围：

```text
easy:    15-25 dB
medium:   8-18 dB
hard:     0-12 dB
extreme: -5-8 dB
```

manifest会记录：

```text
realized_sensor_snr_db
realized_sensor_snr_reference_db
source_psd_summary
clean_sensor_psd_summary
noise_sensor_psd_summary
```

## 7. Recommended Generation Modes

### Mode A: Clean Localization Sanity Check

用于确认leadfield、patch、inverse算法基本正确。

```text
waveform: erp / gaussian_pulse
n_patch: 1
patch_area_mm2: 200-500
source_background_type: None
sensor_noise_type: white or pink
sensor_snr_db: 20-25
connectivity_mode: independent
```

### Mode B: Standard Training Samples

用于常规训练集，覆盖事件、节律和非周期活动。

```text
waveform: erp / ersp / powerlaw / specparam / resting
n_patch: 1-3
patch_area_mm2: 100-350
source_background_type: None / powerlaw / specparam
sensor_noise_type: pink / powerlaw / specparam_resting
sensor_snr_db: 8-18
connectivity_mode: independent or correlated
```

### Mode C: Resting-State-Like Samples

用于模拟静息态或连续脑活动。

```text
waveform: specparam / eegsourcesim_resting_noise / resting / powerlaw
n_patch: 1-5
source_background_type: powerlaw / specparam / eegsourcesim_resting
sensor_noise_type: specparam_resting / eegsourcesim_resting / mixed
sensor_snr_db: 5-15
connectivity_mode: correlated
```

### Mode D: Sparse Transient Challenge

用于测试spike、短时ERP、burst下的算法鲁棒性。

```text
waveform: spike / erp / burst_oscillation
n_patch: 1-3
patch_area_mm2: 40-200
source_background_type: powerlaw / synaptic
sensor_noise_type: powerlaw / mixed
sensor_snr_mode: active_rms or robust_global_rms
sensor_snr_db: 0-12
```

### Mode E: Multi-Source Interaction

用于测试多源混叠、相关源和伪因果结构。

```text
waveform: mixed list per patch, e.g. erp + ersp + ar + specparam
n_patch: 3-8
connectivity_mode: correlated or causal
coupling: 0.4-0.9 for correlated; n_patch x n_patch for causal
source_background_type: powerlaw / eegsourcesim_resting
sensor_noise_type: mixed / specparam_resting
sensor_snr_db: 0-12
```

### Mode F: Domain Randomization

用于深度学习训练时提高泛化。

```python
sampler = G1DifficultyScenarioSampler(difficulty="medium", random_seed=1)
config = sampler.sample(
    sample_id="G1_random_000001",
    template_id="fsaverage5",
    source_space_id="Cortex_FS_20k",
    leadfield_id="Gain_20k_FS_fixed_orientation",
)
```

难度推荐：

```text
easy:    调试、单源定位、单元测试
medium:  默认训练/验证
hard:    鲁棒性评估
extreme: 压力测试，不建议作为主要训练分布
```

## 8. Suggested Dataset Mix

一个平衡的G1训练集可以按如下比例采样：

```text
20% clean event-like:
  erp / gaussian_pulse, no source background, white/pink noise

25% standard event/burst:
  erp / spike / burst_oscillation, powerlaw or mixed noise

25% oscillatory:
  ersp / resting / oscillation / nonsinusoidal_oscillation

20% resting/aperiodic:
  powerlaw / knee_powerlaw / specparam / eegsourcesim_resting_noise

10% multi-source hard cases:
  correlated/causal, 3-8 patches, mixed/specparam_resting noise
```

验证集建议更结构化：

```text
single-source clean:      20%
single-source noisy:      20%
multi-source independent: 20%
multi-source correlated:  20%
resting/aperiodic:        20%
```

## 9. Practical Defaults

推荐默认参数：

```text
srate: 250 Hz
time_window: (-0.2, 0.8) for event samples
time_window: (0.0, 2.0) or longer for resting/aperiodic samples
patch_area_mm2: 100-350
decay_mm: 8-18
min_seed_distance_mm: 20-30
sensor_snr_mode: robust_global_rms
average_reference: True for EEG
normalize_source: True
normalize_sensor: False
```

## 10. Quick QC Checklist

每批生成后至少检查：

- `manifest_index.json` 是否包含所有样本；
- `sensor`, `clean_sensor`, `sensor_noise`, `source` 是否全为有限值；
- `realized_sensor_snr_reference_db` 是否接近目标SNR；
- `source_psd_summary` 是否符合目标波形；
- `noise_sensor_psd_summary` 是否有合理1/f趋势和不过强的窄带振荡；
- `*_summary.png` 中 source waveform、sensor traces、noise traces、PSD 是否合理。

生成总览图：

```text
Simulation_G1/generated_fs20k_demo/G1_fs20k_overview.png
```
