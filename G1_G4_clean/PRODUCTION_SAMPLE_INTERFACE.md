# 当前生产接口：G1 与统一 G2–G3

> 当前 G1 生产配方已升级为 v4，并默认使用明确的 IID measurement noise 与有资源约束的谱背景。
> 在线/完整 QC 和真实噪声增强对照见 [QUALITY_AND_TRAINING_POLICY.md](QUALITY_AND_TRAINING_POLICY.md)。

本轮落实 mask、样本真值、事务保存、SNR 与训练读取五项改造。
旧数据不会被改写；G4 exporter、G4 新 bank/逆模型训练、trial 模式和网络资产重建不属于本轮。

## 1. 训练直接读取命名数组

```python
sample = simulator.generate_sample(config, compute_diagnostics=False)  # G1
y = sample['sensor']
x = sample['source_foreground']
C = sample['noise_covariance']
baseline = sample['baseline_sensor']
metadata = sample['manifest']
```

统一 NMM 同样调用 `generate_sample(config)`，返回下列公共字段。
G1 `RecipeDataset[i]` 已改为相同字典形式，不再返回五元素 tuple。
NMM `NMMRecipeDataset[i]` 延续命名数组接口。

| 字段 | 含义 |
|---|---|
| `source_foreground` | 当前前景定义下的监督真值；对于整场静息任务，前景可以是整个静息场 |
| `source_background` | 添加的 source nuisance；不存在时为零 |
| `source_total` | 前景加背景 |
| `source` | `source_total` 的内存别名；不能默认解释为仅前景 |
| `foreground_sensor` / `background_sensor` | 两类源经投影、去均值和参考后的信号 |
| `clean_sensor` | 前景投影加背景投影，尚未添加测量噪声 |
| `sensor_noise` / `sensor` | 测量噪声 / 最终观测 |
| `baseline_sensor` | 默认 2 秒的独立窗口，不含前景目标 |
| `noise_covariance` | 从 baseline 估计的默认白化协方差，float64 |
| `oracle_noise_covariance` | 活动段实际总干扰的协方差，仅供诊断 |
| `manifest` | JSON-safe 配置、来源、单位、SNR 和诊断；不含 baseline 或 covariance 数组 |

训练通道不再把大数组转成 Python 列表。`generate_sensor()` 仅作为旧五返回值调用的薄适配层，
仍会在该边界转换 whitening 数组；新训练代码应使用 `generate_sample()`。
输出转换到 float32 后，分量相加可能出现舍入量级差异，不能要求每次浮点加法逐位一致。

普通 patch 的 `source_components={}` 与 `save_components=True` 现在也提供完整分解；
没有精细成分时 `foreground_residual` 为零。多尺度分解沿用
`foreground_coarse/residual`、`background_coarse/residual`、`baseline_background_coarse/residual`。

## 2. Mask 与 ERP 时间含义

谱场在 residual fraction 为 0 或非零时均应用相同 cortical mask；
不再为了旧数值兼容而让纯粗尺度场落到被排除顶点。
这项修正会改变旧版 fraction=0 的谱结果，旧结果只能用其冻结代码复现。

`coefficient_erp_support` 描述原始因果 ERP 系数。
`exported_foreground_centering` 与 `source_temporal_representation` 描述输出的 AC 表示：
整事件时间去均值会在刺激前产生常量偏移，不代表事件提前，也不能标记为刺激前逐点全零。
单独的 baseline 不含该前景。

## 3. SNR 统一采用显式模式

新 G1、统一 NMM、G1 CLI 和新训练配方默认 `sensor_snr_mode='active_rms'`。
`global_rms` 和 `robust_global_rms` 仍可显式选择；physical 模式仍固定噪声幅度，SNR 是结果。

`active_rms` 使用参考后的 clean-sensor GFP，取不低于
`sensor_snr_clip_quantile` 分位数的时点；默认 `.95`，即最高约 5% 的时点（含并列值）。
噪声增益用这些时点的 clean 功率除以**整个事件段**的 measurement-noise 功率校准。
baseline 使用同一个增益，不参与目标功率计算。

- `realized_sensor_snr_reference_db`：所选定义下的实际 SNR，应与请求值比较；峰值限制可能提高它。
- `realized_sensor_snr_db`：全时窗 clean / measurement-noise 功率比，可与前者不同。
- `effective_target_snr_db`：全时窗 foreground / (source background 投影 + measurement noise)。
- `effective_target_active_snr_db`：前景活动窗功率 / 全事件总干扰功率。
- `effective_target_active_window_snr_db`：前景与总干扰都限制在前景活动窗内。

前景活动窗单独由 foreground GFP 的同一分位数确定。G1 旧 `event_sensor_snr_db`
仍是明确标注的 20%-peak-GFP 诊断，不用于新模式校准。混合物理单位不报告没有单位不变性的原始 pooled SNR。

新 G1 配方只写 **g1-recipes-v3**；新 NMM period 只写 **nmm-periods-v5**。
旧记录读取时遵循其显式字段；旧 NMM 缺少模式字段时仍解释为 global RMS。
这不等于允许旧生成器身份的配方在新代码中静默重放。

## 4. 保存、覆盖保护和 resume

```python
manifest = simulator.save_sample(
    'output', config, save_sensor=True, save_source=True, save_components=True)  # G1
# 相同请求验证通过后直接复用，G1 不重新生成。
manifest = simulator.save_sample(
    'output', config, save_sensor=True, save_source=True, save_components=True, resume=True)
```

目录统一为：

```text
output/<sample_id>/
  <sample_id>_data.npz
  <sample_id>_manifest.json
  commit.json
```

G1 不再将同一样本拆成多个平铺文件。NPZ 始终保存 baseline 与 covariance，其他数组按保存选项选择。
manifest 中的文件名相对于样本子目录。只有带有效 `commit.json` 的目录视为完整样本。
保存先在同一父目录写临时数据，再独占发布；同名输出默认拒绝，故障不会覆盖原结果。
旧文件、旧目录和不完整目录不会被自动删除或修复。

G1 的 `resume=True` 在生成前核对 seed/config、导联/网格、生成代码、谱资源、
注册噪声 metadata **及实际信号文件**，再核对已提交文件哈希。无显式种子不能提前 resume。
统一 NMM 和通用高层 CLI 的 resume 采用生成/回放后验证，**不承诺跳过 online 仿真**。
Windows 已验证；Linux 使用 `renameat2(RENAME_NOREPLACE)`，本机未执行 Linux 分支；不支持的发布方式明确报错。

## 5. 按窗噪声读取与空间缓存

```python
from Simulation_G1.real_noise import NoiseWindowReader
reader = NoiseWindowReader(max_cache_bytes=256 * 1024**2)
sample = g1_simulator.generate_sample(config, noise_reader=reader)
# NMM: NMMSourceSimulator(backend, channel_names=names, noise_reader=reader)
```

两个 recipe dataset 默认维护各自的 per-process reader。
NPZ 首次读取完整验证后缓存，后续仅截取所需窗口；准备脚本还支持 `.npy` 输出，供只读 mmap 按窗访问。
NPY 配置和 provenance 放在 sidecar，登记资源仍通过 `create_noise_registry()`。

缓存预算限制解压后 NPZ 数组；NPY 映射另有 16 项上限，驻留页由操作系统管理，不计入这个堆内存预算。
进程变化或 pickle 后不携带已缓存记录。返回窗口是可写副本，不会污染缓存。
快速模式依赖训练资源不可变：stat 变化重新验证；需要每次确认内容时使用 `strict=True`。
直接 `load_noise_window()` 仍逐次严格校验。registry metadata 每次均检查。

同一 patch 支撑的测地距离使用有界缓存，默认 16 MiB；图连接、边长或 mask 改变会失效。
本地缓存指纹用于检查资源变化，不替代文件 provenance 的 SHA256。

## 验证记录

- 原目录：568 项测试，566 通过、2 跳过；合集隔离运行：560 项，549 通过、11 跳过；无失败或错误。
- 本轮 28 个 Python 文件通过 AST 解析；169 个登记文件的源与合集哈希一致。未安装 ruff/mypy，未把语法检查当作完整静态类型检查。
- 实际 fs5 的 rest/ERP/ERSP × fraction 0/0.1/0.3 共 9 组配置均无 mask 外活动，baseline 为 2 秒；协方差重算误差约 1e-16。
- 实际 PhysioNet 62 通道案例中，缓存与严格直接读取逐元素一致；活动参考 SNR 为 10 dB，全时窗 SNR 约 0.21 dB。
- 局部性能探针：64×20,000 float32 NPZ 的预热缓存约 4 ms/窗，严格直接读取约 88 ms/窗；首次验证不计入缓存时延。实际 fs5 六 patch 的重复变体中，距离缓存中位数约 0.121 s/样本，直接计算约 0.132 s/样本；不代表所有配置有相同比例加速。

这些验证覆盖接口、数值、缓存和保存一致性，不证明真实脑活动恢复准确率或最优生理参数。

本轮未增加依赖，未修改旧 bank，未重训 G4，也未推送 remote。
