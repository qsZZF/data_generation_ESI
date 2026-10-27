# G3 使用入口

本模块负责产生区域NMM轨迹。对外通常使用根目录的
[`nmm_simulation.py`](../nmm_simulation.py)，由统一接口处理缓存、投影和两级SNR。

```powershell
python -m Simulation_G3.g3_tvb_core --help
python -m Simulation_G3.generate_nmm_bank --help
python -m Simulation_G3.prepare_aal_tvb --help
```

准备好[外部资产](../EXTERNAL_ASSETS.md)后，使用`--atlas aal78`选择78个皮层分区；
不指定时保留原998→994方案。低层接口支持自定义连接文件和输出映射。

| 文件 | 用途 |
|---|---|
| `g3_tvb_core.py` | TVB轨迹生成、预热丢弃与采样 |
| `tvb_jr.py` | 稳定JR方程与保留延迟的稀疏耦合 |
| `atlas_profiles.py` | 命名分区、顺序和资产校验 |
| `prepare_aal_tvb.py` | 从已有AAL/SC资产导出TVB连接包 |
| `generate_nmm_bank.py` | 根据显式JSON计划生成并校验bank |
| `nmm_sources.py` | 统一接口的bank/cache/online provider |
| `compare_spike_bank.py` | 与旧G2库对比，需要另行提供旧库及区域选择JSON |
| `evaluate_aal_pilot.py` | 评估固定AAL三记录试验，需要先按计划生成bank |

`nmm_bank_plan.example.json`为通用计划；`aal78_pilot_plan.json`为AAL示例；
`spike_comparison_plan.json`保留先前固定六区对照参数；`nmm_unified_example.json`
由根目录统一入口读取。预设名不是对任意输出波形的检测结果。

历史结果图和生成数据不随本代码集合附带。重复运行实验需生成新数据或显式
指定外部bank。更改代码或参数后，原缓存若身份不匹配，应建立新bank。
