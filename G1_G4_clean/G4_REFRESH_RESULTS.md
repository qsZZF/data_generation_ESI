# G4 新版训练、统一导出与验证结果

本轮已完成三个数据集的新模型训练和 G4 → 统一 NMM bank → ESI period/sample 调用。生成结果可作为模型条件化模拟源；**不能解释为已可靠恢复逐脑区生理参数**。

## 运行配置

- AAL 78 个皮层分区，fsaverage5 membership 投影。
- 2,048 个独立参数 draw；256 Hz；每条保留 10 秒，丢弃 1 秒 warm-up；独立区域驱动。
- 合成划分：1,433 train / 205 validation / 205 calibration / 205 test；三种 montage 使用同一划分。
- 每个数据集重新训练 compact 逆模型，batch 16，最多 60 epochs，验证集 patience 20，恢复最佳权重。
- 训练采用滤波前全时窗 10 dB IID sensor 噪声，2 秒独立 baseline、OAS 协方差；未使用真实噪声/PCA/residual 增强。
- 每个数据集选择两个受试者进行实测参数估计及留出窗口前向检查。仅使用固定 revision，排除历史重复副本。
- 实测数据原生频率为 200 Hz，上采样至 256 Hz 仅匹配网格；共同分析频带为 1–40 Hz。
- 三种模型分别使用与实测窗口一致的 3 秒（PhysioNet）、5 秒（arithmetic）、10 秒（FACED）输入；共享 bank 存储 10 秒轨迹。

## 覆盖与前向结果

PSD 误差为 1–40 Hz 通道平均的绝对 log10 功率误差，先对每组窗口平均谱，再对条件组等权平均；越小越好。拟合模型和训练先验均值模型分别仅用 fitting windows 校准 observation gain，evaluation windows 不参与该校准。

| 数据集 | 受试者 | 窗口 / 通道 | 最佳轮 / 运行轮 | 条件组 | PSD：拟合 / 先验 | 改善组 | 导出记录 |
|---|---|---:|---:|---:|---:|---:|---:|
| physionet | 1, 10 | 174 / 62 | 7 / 27 | 8 | 1.656 / 1.847 | 8/8 | 24 |
| arithmetic_zyma2019 | 00, 01 | 96 / 19 | 10 / 30 | 4 | 1.660 / 1.803 | 2/4 | 12 |
| faced | 000, 001 | 312 / 26 | 6 / 26 | 16 | 1.310 / 1.602 | 16/16 | 48 |

没有删掉不改善的组，也没有按结果重新选择受试者。各组详细结果、拟合/评估窗口及源数据校验和保存在本地运行目录。

## 参数恢复的限制

| 数据集 | 全脑均值 rmax R² | 各区域参数族的 regional-residual R² 范围 |
|---|---:|---:|
| physionet | 0.592 | -0.006 ～ 0.016 |
| arithmetic_zyma2019 | 0.600 | -0.032 ～ 0.012 |
| faced | 0.504 | -0.003 ～ 0.007 |

regional residual 是每个 draw 内减去全脑均值后的区域差异；R² 接近零或负数说明该部分未被可靠恢复。
PSD 比先验改善只支持有限的分布性前向拟合，不证明估计参数唯一、源位置正确或参数具有临床意义。
实测检查仅覆盖每个数据集两个受试者，不是全库重建，也不是受试者泛化实验。
本轮与历史 pilot 的驱动、投影、采样率、噪声和模型宽度均不同，因此不能把差异直接归因于某一项升级。

## 统一调用和来源隔离

- 已验证 6 个 dataset-qualified subject group 没有跨 fold。
- 同一拟合的三个随机重复共享 simulation group；同一受试者的不同条件共享 subject parent。
- 区域源始终显式标记为 mV_PSP；模型标识为 DualTimescaleJansenRit，未伪装成 G3 TVB JR。
- 统一 sample 提供 source 前景/背景/总活动、sensor 和 2 秒 baseline covariance；活动参考 SNR 校验通过。
- ESI period 划分根据所有祖先关系进行，并验证 train/val/test 各自的读取。
- 84 条记录切成 612 个 1 秒 period，每个 period 两个变体，共 1,224 个配方；train/val/test 为 684/480/60。这里受试者隔离优先于样本比例，少量受试者及不同窗口长度导致比例不均，不能作为正式大规模训练配比。

## 代码验证

原目录 579 项测试：577 通过、2 跳过；干净合集 571 项：560 通过、11 跳过；均无失败或错误。
跳过项为缺少外部资源或未启用的可选集成测试。实际新版模型训练、原 bank 到统一接口和三种 fold 的读取另行完成验证。
三个磁盘 checkpoint 均已重新加载，并从保存的 seed、bank 和噪声设置重建一个测试 draw；预测与训练时保存的值一致到 float32 舍入量级，最大绝对差约 3.8×10⁻⁶。

## 文件与复现

- 调用方法：[G4_UNIFIED_EXPORT.md](G4_UNIFIED_EXPORT.md)。
- 新训练 bank：`generated_simulation_dataset/G4/aal78_membership_256hz_independent_2048/`。
- 本次运行：`generated_simulation_dataset/G4/refresh_membership_256hz_20261101/`。
- `refresh_plan.json`：准确 source files、数据 revision、受试者、seed、代码身份与训练设置。
- 各数据集目录：checkpoint、training/reconstruction reports、测试预测/真值/先验、估计参数和重建源。
- `unified_nmm/<dataset>/`：可由 G2–G3 接口读取的导出 bank；`esi_periods/` 为联合训练配方。
- `results_summary.json` 和 `unified_replay_validation.json`：完整数值结果与调用验证。

合集仅同步代码与报告，不携带原始 EEG、大型 bank 或 checkpoint。本轮未推送 remote。

### arithmetic_zyma2019 未改善的条件组

- 受试者 00，mentalArithmetic：先验误差减拟合误差 = -0.0841。
- 受试者 00，rest：先验误差减拟合误差 = -0.1003。
