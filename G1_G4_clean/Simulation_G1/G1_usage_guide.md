# G1 Simulation Usage Guide

## 皮层几何与网络约束生成：2026-10-28

新增 `G1SimulationConfig.source_model="lbo"` / `"lbo_network"`，原默认 `"patch"` 不变。
`spectral` 字典定义模式、时间频谱和事件参数；支持 `rest`、`erp`、`ersp`。
几何模块直接使用当前 leadfield 对应的 Brainstorm 三角网格，cotangent FEM 刚度矩阵及 lumped
面积质量矩阵求解广义特征问题，省略各连通分量的常数模态。`n_modes` 指双半球总模态数。
当前 fsaverage5 256 模态文件对应每半球 128 模态；完整原始网格未另行切除 medial wall。
基底及网络投影在 simulator 内缓存最近一组，文件或网格改变时重新验证，不缓存随机样本。

时间过程是随机 Gaussian FFT 的 1/f 背景加 Gaussian alpha 峰，可选 beta 峰；
网络图以面积加权方式投影到模态空间。此模型提供统计源信号，不等同于 NMM 生理参数模型。
网络图必须是与当前网格逐顶点对齐的 `signed_maxabs` NPZ；符号不解释为兴奋/抑制。
未接入尚未配准的完整 MACM 共激活矩阵。几何方法的参考实现见
[BrainEigenmodes](https://github.com/NSBLab/BrainEigenmodes)，EEG/MEG 几何基参考见
[GBF 研究](https://pubmed.ncbi.nlm.nih.gov/42045673/)；本实现不是其数值方法的逐项复现。

### 调用

```powershell
python scripts/prepare_g1_lbo.py --modes 256
python simulate_source_sensor.py --engine g1 --source-type erp --g1-source-model lbo --g1-spectral-config Simulation_G1/spectral_rest.example.json --duration-s 5 --srate-hz 256 --sensor-noise-type iid_white --sensor-snr-db 20 --sensor-snr-mode global_rms --sample-id lbo_rest_demo
```

第二条命令的 `--source-type` 是原接口字段，新模式的实际场景由 JSON 中 `scenario` 决定。
也可直接用 Python，保留既有五返回值接口：

```python
config = G1SimulationConfig(
    source_model="lbo", srate=256, time_window=(0, 5), random_seed=42,
    sensor_noise_type="iid_white", sensor_snr_db=20., sensor_snr_mode="global_rms",
    spectral={"basis_path": "Anatomy/G1_LBO/fsaverage5_lbo256.npz",
              "n_modes": 256, "scenario": "rest", "temporal_exponent": 1.5,
              "alpha_hz": 10.5, "alpha_width_hz": 2., "alpha_power_ratio": .5,
              "network_fraction": 0.},
)
sensor, clean, noise, source, metadata = simulator.generate_sensor(config)
```

需要网络图时选择 `lbo_network`，提供 `network_path`、`network_fraction` 和可选 `network_indices`。
建议冻结 `basis_sha256`、`network_sha256`，输出 metadata 同样记录实际哈希。
`spectral_pilot_profiles.json` 保存五个探索性候选设置；选取其中一个字典传入 `spectral`，
不要将整份 profile 表直接当作单个设置。纯 rest 的恢复目标是整个静息场，独立 noise baseline
不含该目标；ERP/ERSP 的恢复目标分别是诱发项/调制增量，baseline 含未调制的静息背景。
整个静息场作为目标时的无目标 baseline 是理想化仿真定义，不能当作真实静息 EEG 中自动可辨的无信号时段。
默认 baseline 仍是 2 秒，默认协方差只从它估计，oracle 仅供诊断。
五返回值中的 `source` 仍为总源信号；快速 `generate_coefficients` 接口另行提供 target/background。
新模式使用 `erp_amplitude` / `ersp_depth` 控制目标相对背景，不接受 legacy `source_snr_db`。
原自动训练配方生成器仍采样 patch；spectral 模式通过直接配置接口使用。

### 参数及解释

| 参数 | 含义 |
|---|---|
| `spatial_length_mm`, `spatial_exponent` | 模态能量随空间频率衰减的尺度和强度 |
| `temporal_exponent` | 非周期过程的输入 PSD 指数；不保证传感器拟合斜率严格相等 |
| `alpha_hz`, `alpha_width_hz`, `alpha_power_ratio` | alpha 中心、Gaussian PSD 标准差、相对非周期功率；ratio=0 可关闭 |
| `beta_hz`, `beta_width_hz`, `beta_power_ratio` | 可选 beta；默认 ratio=0 |
| `network_fraction`, `network_indices` | 网络过程在模态能量中的混合比例及地图列索引 |
| `erp_latency_s`, `erp_width_s`, `erp_amplitude`, `erp_latency_jitter_s` | ERP 的相对潜伏期、宽度、任意模型尺度幅度、试次抖动 |
| `event_onset_s`, `event_duration_s` | 事件位置与 ERSP 包络长度，均相对于 time_window 的时间坐标 |
| `ersp_depth`, `ersp_beta_depth` | alpha/beta 振幅调制深度，负值为抑制、正值为增强，最小 -1 |

ERP 使用带平滑起始边界的因果 Gaussian，事件前目标严格为零；ERSP 保留随机相位，
使用 Hann 形幅度包络调制 alpha/beta，避免把诱发均值与诱导功率变化混为一谈。

### 已运行的小规模真实对照

运行 `scripts/prepare_g1_validation_eeg.py` 准备官方 PhysioNet S001–S008 的 R01/R02/R04，
再使用 `scripts/evaluate_g1_spectral.py` 和 `scripts/plot_g1_spectral_validation.py`。
评估还要求已准备的 EEGMAT/arithmetic 原生导出和匹配的解剖资产。
脚本采用共有通道和 CAR、256 Hz、相同 1–40 Hz 滤波；比较频谱形状，不宣称绝对电流/电压幅度已校准。

- EEGMAT rest：8 名 train、4 名 val、4 名 test；324 组配置，各 8 个随机种子。
  train 前十候选由 val 选择，test 不用于选择，最终另用 8 个独立种子报告。
- PhysioNet EO/EC：各 8 名被试，S001–S004 train、S005–S008 test；54 个状态候选。
- PhysioNet cued MI：每人 15 次，共 120 次；真实事件前完整 2 秒 T0，事件后 3 秒任务。
  32 个事件候选；ERP 比较每人 15 次平均的 GFP/RMS，ERSP 在 C3/Cz/C4 分析。
- PSD 指标为描述性 Theil–Sen log-log 拟合（2–35 Hz 排除 7–14 Hz），不是 specparam。
  Alpha 指标是候选峰；没有显著峰时峰位置不应作生理解释。ERSP 使用 bandpass/Hilbert
  pooled power 的 baseline 比值，同时报告去 ERP 后的 induced 指标；TFR 图另用 STFT 作诊断。

| 场景 | train/val 选择的候选 | 独立检验结论 |
|---|---|---|
| EEGMAT rest | 256 模态，指数1.5，alpha10.5Hz、SD2Hz、功率比0.5，无额外网络混合 | alpha峰10.84 vs真实11.02Hz，alpha占比0.484 vs0.468；指数1.50 vs1.23，仍有误差 |
| PhysioNet EO | 指数1.5，额外alpha功率比0 | 频谱/时序距离较小；不可据此声称真实睁眼完全没有alpha |
| PhysioNet EC | train偏好指数2、alpha功率比0.5 | 对test强度明显过估，固定强alpha配置不泛化 |
| Cue ERP | 幅度0.5、潜伏期300ms、宽60ms、抖动40ms | 合成GFP峰约246ms，test中位约307ms；仅部分描述性匹配，不是P300验证 |
| Motor ERSP | 运动网络混合0.5、alpha调制深度-0.1 | train较接近，test alpha/beta仍失配，需保留受试者/状态变化 |

当前没有证据证明网络混合普遍优于纯 LBO，也未证明全脑 FC、源成像准确率或临床真实性。
不建议只生成一个固定“最佳”设置；先使用验证过的范围构建状态与个体变体，并保留跨生成模型测试集。
原路径结果位于 `generated_simulation_dataset/G1_spectral_validation/`，含 JSON/CSV、协议/哈希和 PNG/SVG。
合集不包含真实 EEG、42MB 左右的基底缓存或生成结果；下载数据须遵守 PhysioNet ODC Attribution 许可。

实测本机两线程：256 模态计算约16.7秒；输出全20484顶点的1秒样本约0.64–0.92秒
（含2秒baseline，少量样本计时，非训练吞吐基准）。仅需 sensor 特征时，预计算 `L @ Phi`
并使用模态接口可避免反复展开全顶点数组。

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
