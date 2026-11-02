# G4 数据驱动源：统一 bank 导出与新版训练

> G4 现已支持按四个训练阶段隔离的真实噪声 registry。该功能保持可选，
> 取舍和实际对照结果见 [QUALITY_AND_TRAINING_POLICY.md](QUALITY_AND_TRAINING_POLICY.md) 与 [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)。

G4 保留双时间尺度 Jansen–Rit 方程，G3 保留 TVB 单分支 JR 方程。
两者共享区域信号存储、源到 sensor 投影、噪声与 period 接口，不互换模型参数。

## 从新 G4 重建结果导出

```powershell
.\.venv\Scripts\python.exe -m Simulation_G4.export_nmm --run-dir path/to/trained/physionet --atlas-dir Anatomy/G4_AAL_cortical_membership --output-dir path/to/new_g4_nmm_bank
```

导出需要完整的 `run_manifest.json`、`reconstruction_report.json` 和
`reconstructed_sources/group_*.npz` 及校验 sidecar。旧文件缺少来源分组时会报错，
不会通过改名或猜测补出来源。目标目录必须是新目录，全部输入校验后才发布完整 bank。

每个随机重复分别存为一个二维 `region_source`，时钟转换为从零开始的毫秒。
bank 为 schema 2，可由现有 `BankWindowReader`、`NMMRecipeDataset`、统一 NMM 接口读取。
它显式声明：

- `source_unit='mV_PSP'`，`normalize=False`；不把 PSP 当作已校准偶极电流。
- `model.class='DualTimescaleJansenRit'`、`signal_type='data_guided_jr'`；不伪装为 TVB JR。
- `active_regions=[]`；拟合参数并不提供离散激活区域真值，整场网络活动可作为前景。
- atlas、区域顺序、projection 和 SC 身份匹配；保存拟合参数、随机种子及原记录校验和。

```python
from nmm_simulation import NMMSimulationConfig, NMMSourceSimulator

config = NMMSimulationConfig(
    atlas='aal78_membership', provider='bank', bank_dir='path/to/new_g4_nmm_bank',
    nmm_sample_id='g4_0000_00', source_mode='network', source_snr_db=None,
    duration_s=1., srate=256., sensor_noise_type='iid_white', sensor_snr_db=10.,
    sensor_snr_mode='active_rms', random_seed=42,
)
sample = NMMSourceSimulator().generate_sample(config)
```

`sample['source_foreground']` 此时是该参数拟合条件下的模型网络活动。
bank 保存原始区域 PSP，不自动套用 G4 前向检查中的拟合 `observation_gain` 或 1–40 Hz 过滤。
要复现报告中的 sensor 预测，应读取原重建产物并应用其观测流程；统一 ESI 样本按自身幅度和 SNR 配置生成。
使用 `source_mode='selected'` 时必须显式指定区域；那是对已生成网络的空间筛选，不能称作人体激活真值。
本次没有为 G4 增加伪造的“默认静息背景”标签；混合背景需独立、有正确语义的记录。

## 来源和划分

新重建产物保存 `parent_data_group_ids`、拟合与评估的 recording/window IDs。
同一参数拟合的全部重复共享 simulation group；同一受试者的不同条件共享 subject parent，
因此 period 划分必须把这些记录放入同一 fold。统一 NMM sample 现在也保留继承的实测祖先 ID。

身份按 `dataset:subject:recording` 命名，各组成部分做 URL 转义以避免分隔符冲突。
若将同一采集登记为真实噪声，应使用产物中的 `qualified_subject_id`、`fitting_recording_ids`
作为 registry 的 subject/recording ID，以共享同一父分组。旧的未限定 S001 等名称不自动匹配。
只有拟合记录/窗口进入生成来源；评估窗口单独记录，不能误记为训练输入。

## 当前训练默认与报告

G4 bank 和 pipeline CLI 的新默认 atlas 是 cortical AAL membership；原 area-unit-sum 资产保留，
需要论文约定对照时显式指定。新 bank 默认 256 Hz、独立区域驱动，旧 bank 不自动升级。
改变驱动、采样率或投影后的模型必须重新验证并训练相应逆模型。

训练报告新增 `heldout_recovery_detail`，分别给出参数整体、全脑均值和去除均值后的区域差异恢复，
并与**仅由训练划分计算**的先验均值比较。`heldout_parameter_recovery.npz`
保存测试 draw ID、预测、真值和先验，便于独立复核。

重建的三次随机轨迹用于分布性前向检查，不期望与真实 EEG 逐点相位一致。
干净重建输出的零噪声协方差仅表示未添加噪声，不能用作真实 EEG 白化矩阵。
基于真实 baseline、PCA 或 residual 的 G4 训练噪声适配仍是后续独立任务。

新版实际训练的范围和结果见 `G4_REFRESH_RESULTS.md`；该报告会区分合成监督训练与实测前向检查，
不将训练完成或 PSD 改善解释为人体生理参数被唯一恢复。
