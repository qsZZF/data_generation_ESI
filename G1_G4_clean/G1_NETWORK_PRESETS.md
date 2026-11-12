# G1静息与分布式ERP预设

以下预设显式启用，普通G1调用的默认参数保持不变。

| 预设 | 用途 | 证据范围 |
|---|---|---|
| `rest_calibrated` | LBO＋network静息；空间长度40 mm、1/f指数1.6、alpha10.5 Hz/宽度1.5 Hz/相对功率0.2 | arithmetic有限受试者上选参及分开确认；综合误差改善约24.2%，不代表所有静息状态 |
| `rest_lbo` | 纯LBO静息，空间长度20 mm、同一频谱参数 | 有竞争力的家族候选，保留作比较 |
| `erp_network` | 共享network＋35 mm测地子区域、多个ERP时间成分、独立静息背景 | 机制验证：有效秩及拓扑变化改善；尚未完成标准ERP实测验证 |

默认256 Hz、2秒事件、2秒baseline、独立白measurement noise、active-RMS SNR20 dB、OAS covariance。`alpha_power_ratio=0.2`是相对非周期成分的生成参数，不是最终sensor中20%功率的保证。PhysioNet闭眼PSD在当前静息配置下反而偏差增大，应对强alpha状态另行标定。

## Python

```python
from Simulation_G1.presets import make_g1_config

config = make_g1_config(
    'rest_calibrated',
    resource_root=r'E:\SourceDataGeneration',
    random_seed=42,
    spectral={'alpha_power_ratio': 0.1},  # 显式覆盖，其余谱设置保留
)
sample = simulator.generate_sample(config)
```

`simulator`使用已有G1SourceSimulator，必须装载与basis/network顶点顺序一致的皮层和leadfield。资源默认相对当前代码项目根目录；代码合集单独使用时，通过`resource_root`指向外部资源所在项目根目录。构造配置不自动下载资源，生成时检查文件、网格与资源身份。

预设资源：`generated_simulation_dataset/G1_spectral_validation/fsaverage5_lbo256.npz`和`projected_source_priors/fsaverage5_20k/neuro_motifs/neuro_motifs_fsaverage5_20k_source_priors_signed_maxabs.npz`。可通过`spectral`显式替换路径及`network_indices`；替换后的组合不自动继承原验证结论。

## 统一CLI

```powershell
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --g1-preset rest_calibrated --resource-root E:\SourceDataGeneration --random-seed 42 --sample-id rest_example
.\.venv\Scripts\python.exe simulate_source_sensor.py --engine g1 --g1-preset erp_network --resource-root E:\SourceDataGeneration --random-seed 43 --sample-id erp_example
```

显式`--srate-hz`、`--duration-s`、`--sensor-snr-db`、`--covariance-estimator`等参数优先于预设。`--g1-spectral-config`提供局部谱JSON覆盖。使用预设时不再同时传`--source-type`，两者冲突会报错；`--resource-root`仅供预设资源使用，普通调用仍使用`--anatomy-dir`等参数。

## 配方与监督目标

`python -m Simulation_G1.network_recipes`生成配置型数据集，可用已有`RecipeDataset`读取；生成组在train/val/test间隔离，同一组的预设和时间/噪声变体不跨组。这不代表解剖支撑或network map互不重叠，尤其静息目标覆盖同一皮层。具体可用参数见模块`--help`。

```powershell
python -m Simulation_G1.network_recipes --resource-root E:\SourceDataGeneration --output generated_simulation_dataset/network_recipes --n-groups 30 --variants-per-group 3
```

新配方冻结资源hash和生成器代码快照；读取旧配方时不要绕过版本校验，应使用其原代码快照。这个配方入口目前支持相对幅度和合成sensor噪声；物理幅度或注册真实噪声配置会明确报错。

`erp_network`以`erp_roi_power_ratio`控制主ROI中前景/背景功率比；不同时使用`erp_amplitude`。当前不支持与额外multiscale残差叠加，避免未经验证的混合被静默启用。

静息预设以整个静息场作为`source_foreground`目标，其baseline不含这一目标，只含measurement noise。ERP主、次级响应均属于前景，baseline含背景而无ERP。新模式的source空间ROI功率比、基线与中心化规则写入manifest；sensor参考和时间中心化可能带来事件前常量偏移，不能据此判定提前激活。

这些预设不更改G2–G4。真实单位仍需显式物理幅度标定；默认是相对活动。所有实测适配结论均限于报告中的图谱、head model、通道与预处理范围。

## 接入验证（2026-11-12）

原路径与`G1_G4_clean`均完成fsaverage5真实网格检查：三个预设各生成64×512 sensor、64×512 baseline及64×64 covariance，完整样本QC通过，相同种子精确复现。两份代码输出逐数组hash一致。另生成18条配置型配方并重放三类预设，确认生成组不跨train/val/test。

原路径G1测试105项、配方测试38项通过；合集G1测试101项、配方测试38项通过（合集不包含4项研究指标/选模测试）。共享CLI回归16项和事务保存7项通过。

接入前相关源码已归档到`reports/G1_WX_preintegration_sources_20261112`，原实验结果未覆盖。新模式生成器身份已更新；这不是自动迁移旧配方。
