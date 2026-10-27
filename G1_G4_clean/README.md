# G1–G4 干净代码集合

此目录是独立复制的代码快照。原项目中的代码和数据未被移动或修改。
保留了算法、统一入口、示例配置、测试、必要的小型区域对照表和版权说明；
不包含 EEG、NMM 信号库、解剖二进制资产、训练权重、生成结果或运行缓存。

## 目录与入口

G1 现支持皮层 LBO / LBO＋网络图的 rest、ERP、ERSP 生成，以及小规模真实 EEG 特征对照。
调用和候选配置的适用范围见 [G1 使用指南](Simulation_G1/G1_usage_guide.md)。

每个新生成的 sensor 样本附带独立 baseline 和由它估计的默认噪声协方差；活动段成分真值
只作为 `oracle_noise_covariance` 诊断字段。详见 [baseline 与白化说明](NMM_USAGE.md#per-sample-baseline-covariance-for-esi-whitening)。

| 路径 | 用途 |
|---|---|
| `Simulation_G1/` | 数学波形、源斑块和传感器模拟 |
| `Simulation_G2/` | NMM库读取、投影、源背景和传感器噪声混合 |
| `Simulation_G3/` | TVB NMM生成、稀疏JR、AAL资产导出及波形评估 |
| `Simulation_G4/` | 数据引导的双时间尺度JR模拟、参数反演和验证 |
| `nmm_simulation.py` | G2–G3统一接口；bank/cache/online及两级SNR |
| `simulate_source_sensor.py` | 原G1/G2/G3统一命令行入口 |
| `scripts/` | G4实际依赖的网格读取工具 |
| `tests/` | 原有回归测试；缺失真实资产的检查明确跳过 |

使用说明：[统一NMM API](NMM_USAGE.md)、[通用命令行](SIMULATION_GUIDELINE.md)、
[G1](Simulation_G1/G1_usage_guide.md)、[G2信号库](Simulation_G2/NMM_bank_usage.md)、
[G3](Simulation_G3/README.md)、[G4](Simulation_G4/README.md)。

## 环境

直接依赖列于 `requirements.txt`，本机验证版本见 `tested_environment.json`。
该清单不是跨平台锁文件；PyTorch应选择适合目标机器的CPU/CUDA版本。
本次整理未安装或修改任何软件包。原工作区的`.venv`没有复制。

当前机器可在此目录下使用原来的解释器，例如：

```powershell
cd E:\SourceDataGeneration\G1_G4_clean
..\.venv\Scripts\python.exe -m nmm_simulation --help
..\.venv\Scripts\python.exe simulate_source_sensor.py --help
..\.venv\Scripts\python.exe -m Simulation_G4.pipeline --help
```

独立新环境可根据依赖清单安装所需部分。G1/G2主要需要NumPy、SciPy、h5py、
matplotlib；TVB运行需要tvb-library；G4还需要Numba／PyTorch，数据与atlas工具
使用MNE、pandas、zarr、requests、nibabel和nilearn。

## 运行仿真前准备资产

真实仿真还需要外部前向模型和分区资产，完整路径见
[EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md)。复制版不会偷偷回退到原项目的数据目录。
将需要的资产按该清单放置后，可运行：

```powershell
python -m nmm_simulation --config Simulation_G3/nmm_unified_example.json --output-dir generated_simulation_dataset/NMM_examples
python -m Simulation_G3.generate_nmm_bank --plan Simulation_G3/aal78_pilot_plan.json --output-dir generated_simulation_dataset/AAL_bank
```

这些命令会生成新结果，不会自动附带或恢复原来的实验记录。G4训练还需要
合适的合成bank、测量数据和配置；本目录没有训练checkpoint。

## 验证与复制记录

```powershell
python -B -m unittest discover -s tests -q
```

测试使用临时夹具验证大多数计算与接口。缺少外部AAL资产及未开启的真实数据
smoke检查会明确跳过。缺少Python依赖时，应先安装对应依赖，不能把导入失败视为通过。
本次实测结果见 [VALIDATION.md](VALIDATION.md)。

`COPY_MANIFEST.json`记录每个复制文件的原路径、原始／目标SHA256以及全部
必要调整。主要调整仅为：独立打包AAL区域表、修正`Anatomy`大小写、示例缓存
目录与缺失外部资产时的测试跳过条件。没有更改动力学或SNR计算方法。

原bank/checkpoint若移入本集合，应先核对其atlas、投影、代码身份和路径；
不要绕过已有provenance校验。更改配置或代码后可生成新的bank。

G4中的外部代码及区域表来源保留在`Simulation_G4/NEURODYN_NOTICE`，对应
Apache-2.0文本为`Simulation_G4/NEURODYN_LICENSE`；不据此替整个项目另行指定许可证。
