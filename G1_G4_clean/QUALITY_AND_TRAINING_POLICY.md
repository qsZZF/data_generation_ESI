# 当前训练数据策略与质量检查

本轮聚焦数据质量、默认配置和一个真实噪声增强对照。没有扩展参数后验、多试次框架、
network/coactive 重建，也没有重新生成大规模 NMM source bank。

## 默认配置

- G1、G2 配置与高层 CLI 默认使用明确的 `iid_white` measurement noise。
- 无空间资源的随机 scenario sampler 不再自动添加随机空间混合的 source 背景。
- G1 生产配方提供经身份校验的 LBO 资源时，使用 `spectral_rest` 背景；否则明确不添加背景。
- 旧 `white`、随机混合及其他噪声族仅保留为显式消融选项，不再随机进入生产默认配方。
- 所有背景仍遵循 cortical mask；显式旧随机背景也不能在 mask 外产生活动。
- 真实静息 EEG 标记为脑背景观测，不当作纯仪器噪声；默认不自动开启真实噪声。

## 两种检查范围

G1 `generate_sample`、统一 NMM `generate_sample` 和 NMM period reader 均附带
`sample['manifest']['quality']`。状态是 `passed`、`warning` 或 `failed`，并逐项列出检查与跳过原因。

在线采用 `online_contract`：检查数组形状、sensor 有限值与成分相加关系、支撑索引/重叠、
参考方式、baseline 长度、协方差基本约束和实际 selected SNR。
为避免降低训练吞吐量，完整 source 扫描/数值闭合和协方差重算留给离线审计；
这些项目会明确标记 `skipped`，不能把在线 passed 解读为已经完成全部科学验证。

```python
from Simulation_G1.quality import assess_sample

quality = assess_sample(
    sample, cortex_mask=simulator.cortex_mask,
    full_source=True, full_covariance=True)
```

完整审计检查 source 有限值、分量闭合、mask、source SNR，
并按记录的 empirical/OAS 估计方式从 baseline 重算协方差。物理小幅值使用相对误差，
不以固定的绝对阈值掩盖 MEG 或 A·m 信号中的错误。
低 SNR、零目标、不可观测目标与特殊 spike 形态不会被简单当作坏样本删除；
明确数值错误或数据契约违背才触发失败。

## 正式配方与试生成

```powershell
python -m Simulation_G1.training_recipes --config Simulation_G1/training_recipe.production.json
```

生产配置为 256 Hz、2 秒事件、2 秒 baseline、active-RMS SNR、OAS 协方差。
空间相关背景采用现有 256-mode LBO，精细残差比例 0.1、8 个分量、10 mm。
25,000 个空间支撑，训练支撑各 32 个变体，验证/测试支撑各 16 个，目标为 **720,000 个配方**。
这是配置目标，本轮没有生成全部 720,000 个样本。

新版仅写 `g1-recipes-v4`，绑定生成代码和资源身份。
`--config` 中 `path_root` 相对于 JSON 文件；当前配置的 `..` 对应项目根目录。显式 CLI 参数覆盖 JSON。
同一空间支撑只属于一个 split；这不等于跨解剖区域或跨头模型泛化。

本轮试生成命令：

```powershell
python -m Simulation_G1.training_recipes --config Simulation_G1/training_recipe.production.json --output-dir generated_simulation_dataset/G1_production_pilot_20261102_v2 --n-supports 60 --variants-per-support 20 --eval-variants-per-support 10
python scripts/audit_training_recipes.py generated_simulation_dataset/G1_production_pilot_20261102_v2 --output-dir generated_simulation_dataset/G1_production_pilot_audit_20261102_v2 --samples 240 --seed 42
```

先生成 60 个支撑、1,080 条 metadata-only 配方，再固定分层抽取 240 条实际生成并完整审计。
选择覆盖 split、难度、源数量、uniform/decay/diffuse/contract/pulse、波形组合等状态。
失败记录不会被替换；逐条结果在 `samples.jsonl`，总体及选中样本分布在 `summary.json`。
只保存配方与审计结果，未保存整批稠密 source/sensor 数据。

## G4 真实噪声可选增强

```python
run_dataset(
    ..., real_noise_registry='noise/registry.json',
    noise_records_by_split={
        'train': ['train_record'],
        'validation': ['validation_record'],
        'calibration': ['calibration_record'],
        'test': ['test_record'],
    })
```

CLI 对应 `--real-noise-registry` 和 `--noise-records-config`。
registry 中 validation/calibration 都使用 `val` 标签，但 G4 要求它们采用互不共享祖先关系的 donor。
同一采集、受试者或已知相同内容不能跨四个阶段使用。通道、采样率、参考和单位必须匹配，
不隐式重采样、补通道、重参考或循环拼接。

窗口包含互不重叠的 baseline 和 event，二者使用同一缩放。增益按过滤前全事件 SNR 校准；
过滤前后 SNR 都记录，防止把不同频谱经过同一过滤后的 SNR 当成相同。
`project_bank(..., draw_indices=[...])` 可生成指定 draw，保留全局种子、fold 与截取身份，
用于让不同模型在同一个测试输入上比较。

本轮对照使用本地 PhysioNet S003/S004/S005/S006 静息记录，分别用于 train/validation/calibration/test，
避开实测前向检查的受试者 1 和 10。IID 控制模型沿用前一轮结果，并验证原/新 IID 投影、目标、
noise 和 baseline 在 10 个实际 draw 上逐元素一致；真实噪声模型使用相同 source bank、训练设置和 seed。
随后将两个模型分别测试在相同的 IID 与真实噪声测试输入上。

这仍是单个训练噪声 donor、两个实测受试者的小对照，不是全面域泛化验证。
实际数值结果与默认开关的取舍见 [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)。
