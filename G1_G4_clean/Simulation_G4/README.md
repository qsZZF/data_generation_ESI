# G4: data-guided NMM simulation

G4 learns an inverse from synthetic EEG to neural-mass parameters, applies it to
measured EEG, and generates fresh stochastic source signals from those estimates.
Measured EEG supplies the observation domain and reconstruction target; it does
not supply known physiological parameter labels. This guide describes the
implemented workflow, not a claim of successful parameter recovery or completed
training. Read each run's reports for its actual status and metrics.

This code-only copy excludes pilot results and trained checkpoints.
See [external assets](../EXTERNAL_ASSETS.md) and [copy validation](../VALIDATION.md).

## Model, anatomy and reference

The reference is [NeuroDyn-EEG, arXiv:2609.36773](https://arxiv.org/abs/2609.36773)
and its [released implementation](https://github.com/Gnosis-Neurodynamics/NeuroDyn-EEG/tree/c9dd9074818e80bfe8e8ba7b2787cf5d8a2be38c),
pinned at `c9dd9074818e80bfe8e8ba7b2787cf5d8a2be38c`. `inverse.py` adapts its
`TrainWorks/src/model.py` under Apache-2.0; preserve [NEURODYN_LICENSE](NEURODYN_LICENSE)
and [NEURODYN_NOTICE](NEURODYN_NOTICE). No upstream trained weights are included.

The executable G4 simulator is the paper's **dual-timescale Jansen–Rit model**,
implemented with a compiled Heun integrator. It is distinct from G3's TVB
single-branch JR model. G4 parameter arrays therefore must not be passed directly
to TVB's JR class. The approach extends G3's source-to-sensor workflow while
following the reference model for inversion.

The initial complete observation pipeline uses **78 cortical AAL parcels** on
the repository's 20,484-vertex fsaverage5 mesh. Bilateral hippocampus, amygdala,
caudate, putamen, pallidum and thalamus are excluded from AAL90. The supplied
64-channel leadfield is projected through area-weighted, unit-sum parcel columns;
unassigned vertices remain zero. Atlas and SC orders are matched by original
AAL90 numbers on both matrix axes.

The mesh matches fsaverage5 exactly in vertex order and topology. Atlas labels
are derived with categorical sampling through official neuromaps
registration-fusion coordinates. **AAL's Colin27 coordinates are treated as
nominal MNI coordinates for an MNI152 mapping; no specific Colin27-to-MNI152
nonlinear deformation is applied.** This remains a template-boundary
approximation, recorded in every run's atlas manifest. See the
[external atlas requirements](../EXTERNAL_ASSETS.md) for coverage, exclusions,
resource hashes and upstream licensing, including neuromaps' CC-BY-NC-SA terms.

The original 994-parcel G3 resources remain available. The G4 simulator and
inverse accept arbitrary region counts, but the current G4 CLI loads cortical
AAL connectivity. A 994-region tensor/architecture check is not a completed
994-region atlas/SC training or reconstruction experiment.

## Forward and inverse conventions

- Each simulation retains neural state through a default one-second warm-up,
  discards that prefix, and exports a new clock starting at zero. The default
  bank stores ten seconds at 256 Hz. Requested integration steps are at most
  0.5 ms; the actual step divides a fixed 256 Hz drive period and is recorded.
- Stable sigmoids, physical 0–10 ms sliding delays with fractional interpolation,
  source-indexed cumulative firing statistics and zero initial state correct or
  explicitly replace corresponding release conventions. Shared Gaussian drive
  has mean 220 and standard deviation 22 spikes/s, held on the 256 Hz grid.
  Timestep refinement preserves this drive convention.
- Connectivity is the mean of the published 88 healthy AAL90 matrices,
  transposed to `(target, source)` and subset to cortical parcels. Self-edges are
  removed. The paper's `sqrt(2*w - w*w)` transformation is applied without hidden
  row normalization. The downloaded SC mirror's SHA256 matches its published
  OSF digest; see `Anatomy/G4_resources/manifest.json`.
- Source outputs are model postsynaptic potentials in **mV**, not measured
  cortical currents. Multiplying these by the leadfield supplies a spatial
  observation pattern; it does not establish an absolute physical EEG voltage
  scale. Reconstruction fits one observation gain from fitting windows and
  saves the resulting sensor predictions in volts.
- Measured files carry verified volts, channel names, native sampling rate,
  subject, condition and available trial/block metadata. Recorded benchmark
  scaling is reversed. Channels are matched by name/standard aliases, with at
  least eight matching channels required; unmatched channels are recorded.
  Matching channel names does not establish identical electrode coordinates:
  these runs use the existing template head/leadfield, not individual MRI or
  digitized electrode geometry from each dataset.
- Synthetic and measured EEG receive the same within-window fourth-order
  1–40 Hz filter, temporal demeaning and common-average reference. The inverse
  then uses one global standard deviation per window, preserving relative
  channel power. Absolute amplitude is consequently not a recovery target.

The inverse retains the released scalp-temporal, leadfield-inverse and spectral
branches, regional self-attention and cross-attention. `--profile paper` uses
the released model widths; `compact` reduces widths/depth for limited GPU memory.
Both use physical-unit targets, prior-consistent bounded heads, a relative-ridge
leadfield inverse, frequency bins selected using the actual sampling rate, and
span-normalized MAE with the paper's parameter-family weights. Training uses
Adam, learning rate 0.001, weight decay 0.00001, and patience 20; the lowest
validation-loss state is restored. These adaptations prevent this experiment
from being described as an exact reproduction of the paper.

## Parameters and validation

Each draw contains eleven regional parameter arrays and one global delay:

| Parameters | Bounds | Units |
|---|---|---|
| `tau_e1_ms`, `tau_i1_ms` | 10–35 | ms |
| `tau_e2_ms` | 3.9–8.4 | ms |
| `tau_i2_ms` | 7.3–16.8 | ms |
| `theta` | 5.4–6.6 | mV |
| `beta` | 0.5–0.62 | 1/mV |
| `rmax` | 2.5–7.5 | spikes/s |
| `C1` | 0.5–1.5 | dimensionless |
| `C2` | 0.4–1.2 | dimensionless |
| `C3`, `C4` | 0.125–0.375 | dimensionless |
| `delay_ms` | 0–10 | ms, global |

Hierarchical truncated-normal draws share a whole-brain mean while allowing
regional variation. Inhibitory time constants are sampled conditionally between
the excitatory constant and 2.4 times that value, intersected with their absolute
bounds. Heads enforce the same joint restrictions. At 78 regions there are 859
outputs; a 2,048-draw bank is a **bounded pilot**, not the paper's large-scale
training regime or proof that these parameters are identifiable.

Whole-brain draw identities split once into 70% training, 10% validation, 10%
calibration and 10% final test. Every montage view inherits that split. The bank
records parameter/noise seeds and hashes of the atlas, SC and simulator; mismatched
caches are rejected. Each dataset gets its own montage/window-specific inverse,
while reusing the same independent source draws. Dataset subject counts are
unrelated to the number of synthetic supervised training examples.

Reports compare final-test parameter recovery with a **training-prior-mean
predictor**, not just training loss. Calibration supplies pooled synthetic
family-wise absolute-error radii and held-out coverage; these are not posterior
distributions, real-EEG confidence intervals or calibrated uncertainty for
averaged subject estimates.

All measured windows receive estimates. Subject/condition reconstruction averages
parameters over fitting windows, then generates three independent stochastic
realizations. Forward checks use held-out trials when available, otherwise
separate blocks with a two-window guard. Groups with too little data retain
parameter estimates but are marked as lacking an independent forward check.
Log-PSD error/correlation are compared with a prior-mean simulation baseline;
both fit observation gain using fitting windows only. This checks distributional
agreement, not phase-aligned waveform or true physiological parameter recovery.

## Data and commands

Run from the repository root using its Python environment. The initial scope is
`arithmetic_zyma2019`, `physionet` and `faced`, spanning arithmetic/rest, motor
imagery and video-elicited emotion. The benchmark's PhysioNet subset contains
four imagery classes; do not infer an execution-versus-imagery contrast from the
original parent dataset's broader protocol. The registry pins their
[OpenEEGBench](https://github.com/braindecode/OpenEEGBench) revisions and subject
partitions. Other registry entries are metadata, not completed dataset pipelines.

The loader reads native Zarr arrays without executing remote code, verifies
Git-blob or LFS SHA256 content against the pinned repository manifest, and saves
download/voltage/preprocessing provenance. `--endpoint` may select an explicit
public mirror; the same revision and content hashes remain required. Default
Hub access is `https://huggingface.co`; the commands below use the mirror used in
this environment. No `--max-windows` means every window in each requested
recording. Omitting `--recordings` downloads only recording 0, not a full dataset.

```powershell
# Rebuild the atlas from cached resources and verify/fetch the SC resource.
.\.venv\Scripts\python.exe -m Simulation_G4.atlas
.\.venv\Scripts\python.exe -m Simulation_G4.resources

# All recordings in the three pinned dataset revisions.
.\.venv\Scripts\python.exe -m Simulation_G4.data hub --datasets arithmetic_zyma2019 --recordings (0..71) --endpoint https://hf-mirror.com
.\.venv\Scripts\python.exe -m Simulation_G4.data hub --datasets physionet --recordings (0..653) --endpoint https://hf-mirror.com
.\.venv\Scripts\python.exe -m Simulation_G4.data hub --datasets faced --recordings (0..122) --endpoint https://hf-mirror.com

# Resumable 2,048-draw AAL78 pilot: ten seconds, 256 Hz, one-second warm-up.
.\.venv\Scripts\python.exe -m Simulation_G4.bank --draws 2048 --workers 2
```

The manifest marks `full_dataset` only when every pinned recording is present.
Downloads are byte-capped per recording and fail explicitly for missing chunks.
An optional `data eegmat` command retrieves original PhysioNet arithmetic EDFs;
it is a separately labelled distribution, not the benchmark-preprocessed data.

Train and reconstruct each prepared dataset:

```powershell
$g4Registry = Get-Content Simulation_G4\dataset_registry.json -Raw | ConvertFrom-Json
foreach ($g4Name in @('arithmetic_zyma2019', 'physionet', 'faced')) {
    $g4Revision = $g4Registry.datasets.$g4Name.revision
    $g4Data = "generated_simulation_dataset/G4/data_access/$g4Name/$g4Revision"
    .\.venv\Scripts\python.exe -m Simulation_G4.pipeline --dataset $g4Name --data-dir $g4Data --profile paper --epochs 150 --batch-size 8 --device auto
}
```

Add `--train-only` to train/check synthetic recovery while matching-montage EEG
downloads continue. Later pass `--reconstruct-only` with the same dataset and
complete prepared directory to apply the saved checkpoint without retraining.
That mode verifies the checkpoint's dataset, bank, montage, window length and
leadfield. New observations with a different montage or window definition
require a matching model. `--max-reconstruction-groups N` limits forward checks,
so a limited run must not be presented as full-dataset reconstruction.

Use `--reconstruction-workers 4` to parallelize independent subject-condition
forward checks. Verified per-group caches resume completed work; changed data,
parameters, numerical settings or reconstruction code invalidate those caches.
Downloads support `--record-workers 3 --workers 2` for bounded record/file
concurrency, retaining Retry-After handling and per-record completion manifests.

Outputs live under `generated_simulation_dataset/G4/`:

- `data_access/<dataset>/<revision>/`: verified recording cache, prepared NPZ
  EEG windows and download manifest.
- `aal78_bank_256hz_v1/`: per-draw source/parameter files, `config.json`, `splits.json`.
- `trained_256hz/<dataset>/`: `run_manifest.json`, `training_progress.jsonl`,
  `best_checkpoint.pt`, `training_prior.npz`, `training_report.json`,
  `estimated_parameters.npz`, `reconstruction_report.json`, and
  `reconstructed_sources/group_*.npz` with source PSP, sensor predictions,
  fitted gain, prior baseline and fitting/evaluation window identities.

`run_manifest.json` preserves the measured records that established the training
montage and window format. After `--reconstruct-only` applies that checkpoint to
additional subjects, `reconstruction_report.json` records the complete current
measured cohort. The synthetic training count remains 1,433 independent draws.

Inspect recovery against the prior baseline and held-out forward checks before
using inferred parameters as scenario priors. A completed training job alone
does not show that the inverse learned useful regional parameters.

## 从独立 baseline 估计噪声协方差（2026-10-27）

投影后的每个训练 draw 生成默认 1 秒的 baseline，并在 `training_noise_covariance.npz`
中保存 `baseline_sensor`、由它估计的 float64 `noise_covariance`，以及仅用于诊断的
`oracle_noise_covariance` / `sensor_noise_covariance`。各矩阵形状为 `[draw, channel, channel]`，
记录 draw 索引、baseline 随机种子及通道顺序。baseline 使用独立噪声随机流，沿用活动段
的噪声增益，不另行归一化。可通过 `--baseline-duration-s` 调整训练 baseline 时长。
baseline 和活动段使用相同的 1–40 Hz 滤波及平均参考，处于逆模型全局 z-score 之前；
若随后将信号除以尺度 s，协方差须除以 s²。
源轨迹 bank 本身尚无指定 sensor montage，因此在投影阶段生成这些矩阵。

当前重建导出的 `clean_sensor_V` 没有额外加入噪声，各 draw 保存 1 秒零 baseline 和零协方差，
明确标记无噪声、baseline 不含估计信息；它不是实测 EEG 的噪声估计。白化方法、秩处理和统一接口详见
[NMM 使用指南](../NMM_USAGE.md#per-sample-baseline-covariance-for-esi-whitening)。

## 256 Hz 默认采样率（2026-10-25）

新建 bank 的 Python API 和 CLI 默认输出 256 Hz，与 G2–G3 对齐。训练与重建
读取 bank 配置中的实际采样率；积分步长、warm-up 和 1–40 Hz 分析频带保持原设定。
实测 EEG 原始采样率须至少为 100 Hz；160/200/250 Hz 等数据可以通过多相重采样
对齐到 256 Hz 网格，但不增加原始频带信息。来源 metadata 记录原始及分析采样率、
是否上采样与重采样方法；epoch 起止样本索引仍属于原始记录网格。

默认新目录为 `aal78_bank_256hz_v1/` 和 `trained_256hz/`。旧 `aal78_bank_v1/`
及 `trained/` 保留；复用旧模型时显式传入对应的 `--bank-dir` 和 `--output-dir`。
100 Hz bank 不能改写采样率 metadata 后当作 256 Hz 使用；应重新生成 bank 并训练
匹配的逆模型。更高采样率本身不代表模型已学会 40 Hz 以上的实测特征。

## PLANNED: G4 接入统一训练 bank 与实测背景（2026-10-24）

本节是下一阶段方案，不代表以下适配器、PCA 噪声或 residual 噪声已经实现。
现有 G4 可以训练逆模型、估计参数并生成新的随机轨迹；其 `bank.py` 仍使用
`config.json` / `splits.json` 和逐 draw NPZ，新建 bank 默认 256 Hz。G2 的 schema 2
区域 bank、period 配方与 G4 原始 bank 不是同一种文件协议，不能直接改目录名复用。

### 统一位置：共享数据协议，保留不同的生成模型

建议流程为：

```text
实测记录的固定划分
  → 训练侧参数推断 / 状态先验与独立的背景估计
  → 参数配置 + 新的随机驱动 → G4 连续 NMM 轨迹
  → 丢弃 warm-up → 256 Hz 区域 bank
  → 共同的 period 索引、空间配方、投影、SNR 和训练读取
```

G4 使用双时间尺度 JR，G3 使用单分支 TVB JR；共享的是区域时序、空间映射和
读取协议，不是参数数组。适配器按 `model_id` 分派参数校验及生成器，禁止直接把
G4 的十一组区域参数传给 G3。新轨迹可以在积分后输出 256 Hz；现有 100 Hz bank
不能通过上采样恢复缺失频带，应保持原带宽标记或用保存的参数重新生成。
已有逆模型训练于 1–40 Hz 的观测，生成较高频输出并不意味着高频特征已被实测数据约束。

每条记录需要以下契约，未得到的量明确写 `null` 或 `not_estimated`：

| 字段组 | 必须保留的信息 |
|---|---|
| 模型身份 | `model_id`、实现版本/代码哈希、参数名称/单位/边界、积分与 warm-up 设置 |
| 解剖与观测 | atlas ID、区域名称及原始编号顺序、SC 哈希、区域到顶点映射及 leadfield 哈希、通道顺序、参考/滤波/采样率 |
| 数值尺度 | `source_unit=mV_PSP`、归一化与缩放链、仅由 fitting 数据估计的 `observation_gain` 及其单位约定 |
| 推断来源 | checkpoint 哈希、原始 dataset/revision、subject/session/run/condition、fit fold、窗口身份及拟合方法 |
| 参数不确定性 | 点估计、跨窗变异；若另行实现后验/ensemble，则保存方法、抽样身份及校准范围 |
| 复现与分组 | 参数配置 ID、独立模拟种子、噪声种子、`simulation_group_id`、全部原始数据父身份、固定 split |
| 存储与目标 | 时钟单位、输出带宽、codec、保护区、前景定义、近似误差及解码后训练标签定义 |

当前 G4 的 synthetic calibration error radii 不能改名为实测参数后验。区域 PSP
也不是皮层电流；不能仅靠 `mV_PSP` 字段就声称投影后具备绝对物理 EEG 电压。
写入公共 bank 时必须显式验证单位与模型身份；当前 G2 写入器根据 G3 的
`manifest.model.class` 推断单位，G4 导出适配需要修正这一边界，不能伪装成 G3 的模型类。
独立静息背景的资格校验也要按 `model_id` 扩展；当前 period 接口只识别原 G3 JR
静息记录，不能把 G4 背景仅改名为 `resting_jr` 来绕过模型检查。

静息 G4 可能全脑都有活动，不能自动假定只有一个激活区。全网络场景优先保存
dense/int16 作为参考；需要 ESI 前景定位任务时，另外声明选区/参数干预及前景真值。
把全网络轨迹事后选区重映射应标记为合成增强，不等同于原始拟合网络的生理重建。
固定 k 背景可用于训练近似，但传感器和源标签必须来自同一解码结果；拟合验证保留
完整参考，不通过压缩来改善拟合指标。

### 实测数据如何成为背景：先片段，再 PCA，最后 residual

1. **独立静息片段作为首版。** 使用同一 split 内、与目标拟合窗口独立的真实记录，
   完成单位、坏段、通道与参考检查后，保存完整 float32 或受控量化数组。
   sensor 维度较小，默认不必压低秩。真实静息 EEG 含脑活动，应称为观测域背景，
   不能称为纯测量噪声；同时加入 NMM 静息背景时须检验背景重复计入的问题。
   真实背景的皮层来源未知，监督标签只覆盖合成前景及已知模拟背景；不能为实测
   背景伪造全脑源真值，也不能把包含它的总 EEG 解释为仅由已保存模拟源投影而来。
2. **PCA 作为可选统计增强。** 只在训练数据上拟合均值、空间基、秩和系数模型，
   记录原始通道/参考。直接保留连续时间系数片段，或按共同时间块重采样全部系数，
   能比逐点 IID 重采样更好地保留包络、非平稳性和跨通道时间关系。
   若生成新系数，需要另行建模自谱/交叉谱或多变量时序；PCA 本身只分解瞬时协方差。
   大方差 PC 既可能是脑节律也可能是伪迹，不能自动当作噪声或直接删除。
   截断项的时序结构与空间协方差要评估，必要时保留完整残差，不能默认用白噪声替代。
3. **Residual 留作单独验证分支。** 当前 G4 用独立随机实现作频谱检查，没有与实测
   相位对齐。直接计算 `measured - independently_simulated` 会把两份脑活动都留进
   差值，不能解释为噪声。仅在确定性/相位对齐的预测或任务平均预测中，才评估逐点
   residual；拟合 observation gain、滤波、参考和通道必须一致，并通过交叉拟合在
   未参与参数估计的窗口取 residual。它仍包含未建模脑活动、模型失配与伪迹。

另一条研究分支是按频率估计背景交叉谱：在独立数据上比较实测与预测的谱矩阵，
对差矩阵作 Hermitian 对称化和半正定约束后才构造随机过程。两者直接相减未必
半正定，而且存在相关信号/噪声时加性分解并不成立；应记录被修正的谱量、频率平滑
和独立留出检验，称为“拟合残差背景模型”，不声称发现了纯噪声真值。

### 参考、SNR 与真实/白噪声混合

统一沿用 G2–G3 的两级定义：source SNR 是时间去均值后、皮层面积加权的
前景/背景功率比；sensor SNR 是参考后完整 clean sensor / measurement noise
功率比，同时报告 foreground / (projected background + measurement noise)
的有效目标 SNR。实测背景即使在 sensor 域相加，也需单独标注其生理来源。

真实片段与白噪声的混合比例定义为**名义功率比例**。先分别执行同一滤波、参考、
去均值及单位功率标准化，再按 `sqrt(alpha)` 和 `sqrt(1-alpha)` 混合。
有限窗中两者可能相关，混合总功率含交叉项；应测量实际总功率和交叉项，再整体
缩放到目标 SNR，不能将振幅比例误当功率比例。若启用峰值保护或活动窗口 SNR，
应作为显式配置并记录实现值；不能静默改变既有 global RMS 的含义。

外部噪声与 leadfield 必须匹配通道名称、顺序、坐标/布局、单位、参考与采样率。
缺失导联不得补零冒充观测；明确取公共导联或使用经过验证的映射，并同步调整
leadfield。不同导联布局的 PCA 基也不能仅凭相同矩阵尺寸复用。EEG 背景不能直接
作为 MEG 传感器噪声；MEG 需要相应传感器类型和测量单位的数据。

### 分组、优先数据与分阶段验收

先固定原始被试/会话/记录层面的 train/validation/test，再做任何参数拟合、PCA、
秩选择、归一化或 observation gain 估计。所有窗口、参数估计、新随机轨迹、压缩
视图及噪声衍生物继承父 split；训练样本不得借用测试被试背景。多个父记录的混合
仅在同一 split 内完成。未见被试的适配实验可在该测试被试的指定 fitting 段估计
参数，但其 adaptation/evaluation 段必须明确隔离，且不得反馈修改训练模型。
当前 G2 period 构建器默认随机分配父组件；G4 适配层应按既定 fold 分别构建索引，
或新增显式 fold 映射入口并检查冲突，不能对已有实测数据划分重新随机分配。
测试配方、噪声片段及种子固定。共享原始父数据的不同模拟轨迹需要更高层分组，
不能只凭各自不同的 `simulation_group_id` 判断独立性。

把全局训练得到的checkpoint/PCA变换作为拟合产物单独审计，与直接复用的记录、
个体参数拟合数据等样本父身份区分。个体参数的随机模拟后代继承该个体划分；
应用训练侧学到的固定变换不应被误记成重新使用训练记录的时间序列。
测试使用独立被试背景，或明确标注为固定PCA分布的新随机实现测试，两者不能混称。

| 阶段 | 交付与数据 | 验收条件 |
|---|---|---|
| A：格式适配 | G4 source/parameter/gain 导出公共 bank；保留既有三数据集作为回归基线 | 单位/模型/atlas 顺序不被改写；warm-up 时钟与连续切窗一致；codec 前景保护；失败输入明确拒绝 |
| B：真实背景基线 | PhysioNet R01/R02 小试点，再接 LEMON EO/EC；完整噪声片段优先 | 通道和参考一致；固定种子/配方重复读取一致；实测 SNR 与元数据吻合；原始父身份无跨 split 泄漏 |
| C：统计增强 | 训练侧 PCA 与连续系数块；COG-BCI 多状态/重复会话扩展 | 独立留出 PSD、alpha 峰、协方差/交叉谱、包络分布对照；新增增强对 ESI 的影响单独报告 |
| D：残差研究 | 交叉拟合的对齐残差或半正定谱残差模型 | 与真实片段/白噪声/PCA 基线比较；量化任务信号残留与模型失配；未通过则不进入默认训练集 |

PhysioNet R01/R02 与已有 MI 是同一队列，必须联动被试划分；LEMON 和 COG-BCI
目前是优先接入计划，不能从注册表条目推断已有完整适配。详细获取范围见
dataset priority CSV (original workspace: reviews/neuralbench/dataset_priority_access.csv; not bundled)。最终同时报告
合成参数恢复、未参与拟合的实测统计重建、以及独立 ESI 测试；其中任何一项均不能
单独证明实测 NMM 参数或皮层源真值已被恢复。本方案不要求本轮下载新 EEG 或训练新模型。
