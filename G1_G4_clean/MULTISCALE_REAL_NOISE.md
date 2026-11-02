# 多尺度源活动与注册真实噪声

> 当前接口以 [PRODUCTION_SAMPLE_INTERFACE.md](PRODUCTION_SAMPLE_INTERFACE.md) 为准。
> 下文保留前一阶段的验证记录；当前所有谱场均应用 mask，新配方为 G1 v3 / NMM v5，且训练采用命名数组。

本次增加两个可选功能：G1 的粗尺度 LBO 场加精细空间残差，以及 G1 / 统一 G2–G3 的真实 sensor 噪声资源调用。默认仍使用原来的合成噪声；不改写已有数据或模型参数。

## G1 多尺度活动

```python
from Simulation_G1.g1_simulation_core import G1SimulationConfig

config = G1SimulationConfig(
    source_model='lbo',
    spectral=dict(basis_path='path/to/basis.npz', scenario='erp', n_modes=128),
    multiscale=dict(fraction=.1, metric='vertex_mean', n_components=8, length_mm=10.),
    srate=256, time_window=(0., 1.), baseline_duration_s=2., random_seed=42,
)
components = {}
sensor, clean, noise, source, metadata = simulator.generate_sensor(
    config, source_components=components)
```

`simulator` 是已加载 leadfield 与皮层网格的 `G1SourceSimulator`。
`scenario` 支持 `rest`、`erp`、`ersp`；`lbo_network` 同样支持残差。
patch 前景的静息背景使用 `source_background_type='spectral_rest'`、
`source_background_spectral={...}`、`source_snr_db=...`，并通过
`source_background_multiscale={...}` 设置精细背景。

| 参数 | 含义 |
|---|---|
| `fraction` | 精细成分功率 /（粗成分功率 + 精细成分功率），范围 `[0,1)`；默认 0 |
| `metric` | `vertex_mean` 或 `area_weighted`；后者使用 FEM 顶点质量 |
| `n_components` | 精细空间模式数，默认 8；不能超过剩余空间维数 |
| `length_mm` | FEM 平滑长度，默认 10 mm；它不是皮层 patch 半径 |
| `seed` | 可选独立种子；省略时从样本种子派生，不改变原有随机流 |

空间残差通过 `(M + length_mm² K)^-1 M^0.5 ε` 生成，再去除保留的 LBO
子空间和每个连通皮层分量的常数项。投影在实际 cortical mask 上进行，mask 外为零。
这是有限元 Gaussian resolvent 近似，不宣称为精确的连续 Matérn 模型。
粗、精成分在面积内积下正交，顶点平均内积下可能存在交叉项；metadata 单独记录交叉功率。
实现诊断保存在 `multiscale_metadata`，原始 `multiscale` 配置保持不变以便重放。
因此 `fraction=.1` 不必等于精细成分功率 / 最终混合信号功率。

ERP 精细成分沿用同一事件潜伏期；ERSP 沿用同一调制区间，并调制自身的静息背景。
背景及其单独的 baseline 使用同一个增益。输出沿用原有的逐顶点时间去均值约定，
因此 ERP 输出的事件前部分可能有常数偏移；独立 baseline 不含 ERP 前景。
`fraction=0` 保持旧数值输出与随机流，亦保留旧 spectral masking 约定；启用残差时粗、精成分均应用 mask。

可选 `source_components` 输出 `foreground_coarse/residual`，以及存在背景时的
`background_coarse/residual`、`baseline_background_coarse/residual`。这些数组跟随源归一化、
source SNR 和 physical amplitude 增益；微小 float32 舍入误差纳入分解以保持重建一致。
`save_sample(..., save_source=True)` 在启用残差时同时保存分解；
`save_components=True` 可只保存分解。大数组进入 NPZ，不进入 JSON。

高层 CLI 新增 `--g1-multiscale-config residual.json` 和
`--background-multiscale-config residual.json`。
配方生成器也支持 `source_background_multiscale`；新语义使用 `g1-recipes-v2`，默认配方仍为 v1。

## 准备与注册真实 sensor 噪声

```powershell
.\.venv\Scripts\python.exe scripts/prepare_real_noise.py recording.edf noise.npz --record-id S001-rest --recording-id S001-R01 --subject-id S001 --noise-role resting_background --split train --target-fs 256 --reference average
```

支持本地 EDF、FIF，以及显式声明数据键、通道名、采样率、通道类型和 SI 单位的二维 NPZ。
不下载数据，不把三维 epochs 展平，不插值补通道，不把任务数据自动认定为噪声。
重采样使用带抗混叠滤波的 polyphase 方法；输出 NPZ 和 `.record.json`，拒绝覆盖。
通道名仅统一大小写、空格和末尾点号；不做解剖位置别名替换。
`--channel-names-json` 可显式选取与 forward 一致的通道，选取后重新做 EEG 平均参考。
FIF 标记的坏通道默认排除；混合 EEG/MEG 不允许统一平均参考。

```python
import json
from Simulation_G1.real_noise import create_noise_registry

record = json.load(open('noise.record.json', encoding='utf-8'))
create_noise_registry([record], 'noise_registry.json')
```

registry 只保存本地资源位置、配置、来源和校验和，不复制信号。
同一采集、受试者或已知相同内容的记录不得跨 train/val/test 分组。
`recording_id` 应保持真实采集身份；哈希无法识别被独立裁剪且错误改名的所有重叠记录。
跨数据驱动模型训练与噪声资源的分组，也需要使用一致的采集身份，不能仅凭波形推断。

```python
simulator = G1SourceSimulator.from_brainstorm_cortex(
    leadfield, cortex, channel_names=names_in_leadfield_order)
config = G1SimulationConfig(
    srate=256, time_window=(0., 1.), baseline_duration_s=2., random_seed=42,
    sensor_noise_type='real', sensor_snr_db=10., sensor_snr_mode='active_rms',
    real_noise_registry='noise_registry.json', real_noise_record_id='S001-rest',
    real_noise_split='train',
)
sample = simulator.generate_sensor(config)
```

加载时严格检查采样率、完整通道集合、单位、参考、split 和文件哈希；通道仅按名称重排。
截取连续且互不重叠的 baseline/event，长度不足直接报错，不循环拼接。
两段共用 noise 增益；协方差从导出的 baseline 估计，默认 baseline 长度仍为 **2 秒**。
`active_rms` 用活动窗定义 SNR；它与全事件 `global_rms` 的数值不可直接等同。
同单位的 relative 模式按 SNR 缩放；混合单位必须先统一或使用显式 physical unit contract。
physical 模式需声明正的 `sensor_noise_std`；未指定或全部为零时拒绝注册噪声，避免加载了真实数据却静默输出零噪声。

`resting_background` 包含真实脑活动，不能视作纯仪器噪声。与额外 source 背景一起使用时，
须设 `allow_combined_real_source_background=True`；纯 `empty_room` 不需要该选项。
该选项只表示有意叠加，不保证两个背景过程统计独立。
旧 bare-array API 保留，输出会标记未验证的来源/单位/参考信息。

G1 高层 CLI 对应 `--sensor-noise-type real --real-noise-registry ...
--real-noise-record-id ... --real-noise-split train --channel-names ...`。
G1 训练配方的 `build_recipe_dataset(..., real_noise_registry=...)` 根据空间支撑所属 split
选择噪声记录；每次回放核对固定的 registry SHA。

## G2–G3 与数据驱动 NMM

统一 `NMMSimulationConfig` 使用同名 `real_noise_registry`、`real_noise_record_id`、
`real_noise_split`、`allow_combined_real_source_background` 字段。
`NMMSourceSimulator(backend, channel_names=names)` 可调用自定义匹配导联的 backend。
保存的 provenance 保留原记录截取范围、单位转换和全部父记录分组。

`build_period_index(..., sensor_noise_types=('real',), real_noise_registry=...)`
生成注册噪声的 `nmm-periods-v4`，并冻结 registry SHA；
`NMMRecipeDataset(..., channel_names=names)` 回放时同时校验源与噪声的分组身份。
合成噪声仍为 v3，reader 兼容 v1–v4。
G4 的独立参数训练流程本次未重训；数据驱动 NMM 导出到统一 bank 后可使用同一后端噪声机制。

## 验证与边界

最终原目录测试 **531 项：529 通过、2 跳过**；合集隔离测试 **523 项：512 通过、11 跳过**，
均无失败或错误。合集测试未从原目录导入代码；跳过项为外部 anatomy/AAL 数据或可选集成测试。
19 个本次 Python 文件通过 AST 语法检查，162 个登记文件的源/合集校验和一致。
环境未安装 ruff/mypy，因此没有将语法检查等同于完整 lint/typecheck。
日志仅出现 TVB 缺少可选 geodesic 扩展的既有警告；本次残差与 mesh-edge 路径不依赖该扩展。

本地 PhysioNet S001R01 静息记录已用于端到端验证，采样率 160 → 256 Hz。
与 Brainstorm 显式匹配 **62 通道**：省略 Brainstorm P9/P10 和 PhysioNet T9/T10，
没有把它们当成同一位置。四个 G1 patch、1 秒事件、2 秒 baseline 的结果可精确重放；
全事件目标 10 dB 实测 **9.999999967 dB**，协方差与导出 baseline 重算的相对误差为
**9.39×10⁻¹⁷**。结果位于原工作区 `generated_simulation_dataset/real_noise_validation/g1_smoke/`。
合集不携带这些真实记录或大型 anatomy 资源。

真实 fs5 的 18 个配置（3 种时序 × 3 个残差比例 × 2 种功率指标）也通过检查：
比例最大误差 7.82×10⁻¹¹、分解重构相对误差 2.05×10⁻¹⁷；
2 个 BLAS 线程下，缓存后的完整 sensor 生成约 0.68–1.68 秒/样本
（256 Hz、1 秒事件、2 秒 baseline、128 粗模式与 8 个精细模式；此范围包含关闭残差的对照）。
`fraction=0` 在三种时序下均保持 source、sensor、noise、baseline 和 covariance 逐元素一致。

这些检查证明数值、时间分割和接口的一致性，不证明新残差参数已达到最优生理真实性。
尚需多受试者、多个任务条件下的频谱、ERSP、空间相关及源成像恢复比较。
未增加依赖，未重建大规模训练集，未推送 remote。
