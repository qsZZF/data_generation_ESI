# 外部资产清单

以下路径均相对于本代码集合根目录。这里只列出要求，没有复制这些二进制数据。
原工作区中相同相对路径可作为本机资产来源；在另一台机器上需自行准备合法且
对应版本的资源。数据和代码分别保存，避免把缓存、训练集或模型权重混入代码快照。

| 用途 | 需要的文件或目录 |
|---|---|
| 通用fsaverage5投影／EEG | `Anatomy/Cortex_FS_20k.mat`、`Anatomy/Gain_20k_FS.mat`、`Anatomy/Chanlocs.mat` |
| 998→994分区 | `Anatomy/FS_994nmm_to_20k.mat`、`Simulation_G3/anatomy/connectivity_998.zip` |
| AAL78投影 | `Anatomy/G4_AAL_cortical/atlas.npz`、`projection.npz`、`parcels.csv`、`manifest.json` |
| AAL78 TVB连接 | `Anatomy/G3_AAL78_TVB/connectivity_78.zip`及其`manifest.json` |
| G4结构连接／重建AAL连接包 | `Anatomy/G4_resources/SCmatrices88healthy.mat`及来源说明 |
| 从原始模板重建AAL | `Anatomy/G4_AAL_cortical/resources/`中的AAL体积、原ROI表、registration-fusion坐标；还需fsaverage5表面与公共前向资产 |
| 旧G2尖波库／历史波形比较 | `nmm_spikes/`；新G3生成bank不依赖这套旧库 |
| G4实测数据、合成bank及模型 | 单独的EEG导出目录、合成bank、训练checkpoint；通过G4命令行配置路径 |

本集合另带的 `Simulation_G4/reference/aal_regions.csv` 是G3导出器依赖的
小型名称／编号对照表，不是完整atlas、leadfield、结构连接或EEG数据。

## 复用本机资产的约定

最简单的方式是把所需资产复制到上表位置。部分原入口允许显式传入
`--anatomy-dir`、`--projection-path`、`--connectivity-path`、`--atlas-dir`、
`--connectivity-dir`等路径；以各入口的`--help`为准。命名atlas和统一NMM入口
会校验配套资产身份，不能只用相同维数的矩阵替换。

也可由用户在本机建立目录映射来共享现有数据，但本次没有建立任何隐藏链接。
共享目录适用于读取；若要运行atlas重建等写入流程，应使用独立资产副本，避免
修改原工作区使用的资产。代码快照本身不依赖原项目的绝对路径。

## 已知方法边界

G1 LBO 模式另外需要基底缓存：在已配置当前 Brainstorm cortex 后，运行
`python scripts/prepare_g1_lbo.py --modes 256`，写入 `Anatomy/G1_LBO/`。
网络模式还需当前网格对齐的 `projected_source_priors/fsaverage5_20k/` 中的 signed_maxabs NPZ。
这些二进制资产不随代码上传，不能直接用 Nilearn 顶点顺序替换。
真实信号验证脚本可下载一个小型官方 PhysioNet 子集，保存原始 EDF 的 URL、许可和 SHA256；
完整的 EEGMAT 对照还需要原 G4 流程已准备的 arithmetic 导出。

- AAL78是已有fsaverage5派生皮层atlas；其边界含名义MNI映射近似。
- AAL TVB传导长度来自区域质心距离代理，不是实测纤维长度。
- G3是TVB单分支JR，G4是双时间尺度JR；不能直接混用参数文件。
- 真实EEG、区域参数、仿真源和训练权重是不同资产，不能由复制代码自动补齐。
