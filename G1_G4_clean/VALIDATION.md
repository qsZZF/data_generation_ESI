# 复制版验证

## 2026-10-28：默认 baseline 统一为 2 秒

G1、legacy G2、统一 G2–G3、period 索引/读取及 G4 的默认 baseline 改为 2 秒，
包括对应 CLI 和 G4 无噪声重建的零 baseline。256 Hz 下为 512 点，200 Hz 下为 400 点。
显式给出的 baseline 时长仍优先；已有配置中明确保存的 1 秒值不自动改写。
新测试检查默认 NMM baseline 长度和分区背景非重叠窗口；保留显式短窗覆盖与来源划分测试。
原目录共 295 项测试：293 项通过、2 项跳过；合集共 288 项：279 项通过、9 项跳过，均无失败或错误。
源码同步、语法及清单校验通过。未重建已保存的样本或训练模型。

以下保留历史记录，当前默认时长和测试总数以本节为准。

## 2026-10-27：baseline 估计替代默认 oracle 协方差

原目录测试共 293 项：291 项通过、2 项跳过；合集隔离测试共 286 项：277 项通过、9 项跳过，均无失败或错误。
每个 sensor 样本生成默认 1 秒、无定位目标的 `baseline_sensor`；默认 `noise_covariance`
从保存的 baseline 估计，活动段已知成分协方差仅保留为 `oracle_noise_covariance` 诊断字段。
baseline 与活动段共享噪声设置、空间混合、标定增益、参考及归一化尺度，不单独调整 baseline 功率。
G1 和合成 sensor 噪声使用联合流中的非重叠区间；NMM source 背景来自同一 parent 的非重叠窗口，
记录边界及 split，长度不足报错。真实噪声输入需要完整 baseline 加活动段，不循环拼接。
G4 训练投影使用独立可复现随机流产生 baseline，按活动段的增益缩放后同样滤波；clean 重建
保留无噪声零 baseline，并标注其不含协方差估计信息。
公开五返回值接口、legacy G4 双返回接口保持兼容；新增 baseline、oracle 字段及 CLI 时长配置。
修复 legacy G2 无噪声时 SNR 除法溢出；标准导出检查 baseline/协方差维度与有限值。

真实 fsaverage5 验证：G1 的 64×512 baseline 与 AAL78 NMM 的 64×256 baseline 均检测到秩 63；
保存的协方差与 baseline 直接估计一致。白化 baseline 的单位阵误差分别约 9.6×10⁻¹⁵、1.8×10⁻¹⁵。
两例 baseline 协方差与活动段 oracle 的相对 Frobenius 差约 0.369 和 0.570；这是有限样本估计差异，
不是源恢复准确率证据。NMM replay 的 baseline 重复调用逐元素一致，并验证了背景非重叠窗口来源。
保留原有源背景、SNR、空间采样、period、训练和重建回归覆盖；源码与合集字节核对、语法、JSON 检查通过。
无新增依赖，未批量重写旧数据或重训真实数据模型；短窗 baseline 的统计估计仍有不确定性。
旧 G1 配方需要使用其冻结代码快照重现；新代码生成的配方记录更新后的代码身份。

以下是历史验证记录，当前行为与测试总数以本节为准。

## 2026-10-25：逐样本噪声协方差与白化支持

原目录测试共 275 项：273 项通过、2 项跳过；合集隔离测试共 268 项：259 项通过、9 项跳过，均无失败或错误。
G1、统一 G2–G3 及其 period/batch 接口提供总噪声和 sensor 噪声协方差；
legacy 接口保留五返回值，并在 metadata 中提供矩阵，保存时使用 float64 压缩 NPZ。
G4 训练投影按 draw 保存滤波后、输入标准化前的协方差；clean 重建明确保存无噪声零矩阵。
源轨迹 bank 尚未指定 sensor montage，不提前伪造传感器协方差。
新增 17 项测试覆盖归一化/SNR/峰值限制后的尺度、背景交叉项、平均参考降秩、零噪声、
batch 默认字段、保存接口、滤波及白化；G1 原信号数组与修改前检查的 16 组数组哈希一致。
原路径实际 fsaverage5 验证使用 20,484 顶点、64 通道、256 Hz、1 秒信号，
配置 source 背景与 mixed sensor 噪声，两个 64×64 协方差秩均为 63。
总噪声白化 W C Wᵀ 与单位阵最大绝对差约 7.4×10⁻¹⁵，sensor 噪声白化后约 6.8×10⁻¹⁵。
协方差是当前模拟噪声实现的经验估计，并非独立基线估计；短窗估计精度仍有限。
无新增依赖；没有批量重写旧样本或重训真实数据集，未自动修改输出信号为白化信号。
语法、JSON、合集来源校验和与独立导入检查均通过；未运行环境中不存在的 ruff/mypy。

以下保留此前验证记录，当前测试总数以本节为准。

## 2026-10-25：G4 256 Hz 适配

原目录测试共 258 项：256 项通过、2 项跳过；合集隔离测试共 251 项：242 项通过、9 项跳过，均无失败或错误。
新建 G4 bank 默认输出 256 Hz；训练及重建按 bank 配置使用对应采样率。
新增测试覆盖旧 100 Hz bank 的配置保护、160/200/250/256/500 Hz 实测网格转换、
时长与 10 Hz 测试节律保持、原始样本索引保留，以及 256 Hz 单 epoch 训练、checkpoint 重载和重建衔接。
训练衔接测试采用合成夹具和 mock 模拟器，不作为实测参数恢复证据。
原目录另用真实 AAL78 连接矩阵运行双时间尺度 JR：丢弃 1 秒 warm-up 后，
输出 1 秒、78×256 点的有限值轨迹，时间步长 1/256 秒。
未重新生成正式 bank 或重训真实数据集；1–40 Hz 分析频带保持原设定。
本轮修改了 bank.py、pipeline.py、两个测试文件和 G4 README；代码及测试在两个目录字节一致。
沿用现有重采样、滤波和 bank 配置传递流程，没有新增依赖或接口层。

以下保留此前各轮验证记录，测试总数以本节为准。

- 更新日期：2026-10-24（NMM period 数据集、父身份划分、缓存与批量读取；G4方案）。
- Python源码：68个文件，全部语法解析通过；JSON 文件解析通过。
- 合集测试：共247项，238项通过，9项明确跳过；0失败、0错误。
- 原路径测试：共254项，252项通过，2项明确跳过；0失败、0错误。
- 测试在复制目录作为工作目录和首选导入路径的条件下运行；原项目根路径从导入搜索路径移除。
- 11个主要模块核对了实际`__file__`，全部来自本目录。
- 各轮代码同步详情见`COPY_MANIFEST.json`；本轮period功能12个代码／测试文件已核对原路径与合集字节一致。已更新来源及复制文件SHA256，文档保留合集专用说明。
- 没有在本目录生成模型、数据或字节码缓存；测试临时数据和编译缓存位于集合之外。

## 跳过项目

- `test_aal_cli.AtlasCLITests.test_aal_profile_selects_matching_connectivity_and_projection`：External AAL assets not supplied。
- `test_aal_cli.AtlasCLITests.test_custom_paths_remain_explicit_overrides`：External AAL assets not supplied。
- `test_aal_cli.AtlasCLITests.test_named_g2_atlas_rejects_custom_bank_identity_and_projection`：External AAL assets not supplied。
- `test_aal_cli.AtlasCLITests.test_same_size_reordered_projection_rejected_before_tvb`：External AAL assets not supplied。
- `test_aal_cli.AtlasCLITests.test_spike_preset_matches_core_defaults_and_accepts_overrides`：External AAL assets not supplied。
- `test_aal_tvb_assets.AALTVBTests.test_real_bundle_order_provenance_roundtrip_and_reproducibility`：Local AAL resources absent。
- `test_g3_generation.G3GenerationTests.test_real_tvb_seed_and_sampling`：Optional installed-TVB smoke test。
- `test_g4_atlas.AtlasTests.test_actual_bundle_preserves_sc_order_forward_projection_and_cortical_mask`：Local atlas not built。
- `test_simulation_pipeline.PipelineTests.test_real_high_level_engines`：Optional local anatomy/NMM/TVB integration。

## 验证范围

Period新增验证：整数采样点固定／滑动窗口、丢尾或报错、父身份连通分组、历史转换嵌套来源恢复、独立背景同fold且不同seed、选区／同轨迹分解／全网络模式、缓存字节预算、窗口局部解码、文件变更与校验和拒绝、pickle后worker缓存重置、线程batch和预取复现、区域PSP偏置不产生虚假动态空间活动。

实际AAL78低秩bank建立81条network配方、72条training-only partition配方及5条training-only独立背景配方。按10个相邻1秒窗解码再拼接，与完整10秒记录逐元素一致；抽样回放与batch检查通过，两级SNR吻合。缓存只限制保留的编码数组，不能视为总进程内存上限；NPZ冷读取仍解压整条编码载荷，未完成磁盘分块I/O或GPU吞吐优化。

G4仅新增明确标记PLANNED的适配与实测背景方案，未运行新的G4拟合、PCA或residual噪声模型，也未下载新EEG。

NMM存储新增验证：旧schema1兼容、新schema2的dense/int16/fixed-k解码、配置激活区及继承保护区完整保留、全记录抗混叠降采样、固定k只按矩阵维度截断、量化误差界、校验和和异常维度拒绝、原bank不可覆盖、派生视图保持父轨迹分组。

原路径用6条已有记录生成36个编码视图（AAL78、legacy994、旧G2单源尖波均覆盖），受保护区域均逐元素等于对应降采样float32参考；已验证旧G2压缩bank回放。当前64导EEG模型下计算了背景误差、PSD／协方差、可观测秩及source SNR条件误差。没有重新生成长NMM轨迹，也没有以这些误差直接宣称ESI准确率。sensor噪声编码保持原状。

配方新增验证：JSON往返与索引访问顺序无关的精确复现、五个随机流独立、显式空间支撑、同步波形、白噪声、峰值约束、配置生成不调用波形、按支撑划分、按划分随机访问、校验和及资源／版本冲突拒绝。原有配置四类输出数组在修改前后SHA256保持一致。

原路径实际生成128支撑／3,680配方（约6.1 MB，含代码快照）；约87.4%为多源。抽取33条实际生成、全部有限且重复读取一致；2条触发噪声峰值限制，实际SNR独立记录。该次单线程热读取与生成平均约0.46秒/条（2秒、250Hz），不是多worker或GPU训练吞吐测试。未生成默认标准规模72万配方、未启动训练，真实噪声未启用。

新增验证覆盖完整 patch 支撑不重叠、显式冲突与不可满足面积报错、随机种子复现、面积／邻接阶数阈值、扩散／收缩／扩散后回缩端点、配置映射、按指定 clip 选择旧 NMM 文件，以及空间调制后的 source/sensor SNR。

原路径的实际20,484顶点网格检查：3组各6个约12 cm²的G1 patch均无重叠；994区域与AAL78投影均保持设定3 dB源SNR。另实际运行G1六源与G2六源旧尖波回放CLI短样本，均无支撑重叠，设定参考定义下传感器SNR为10 dB。后者数据不打包进合集。

未安装新的lint／typecheck工具；环境未提供ruff或mypy，因此对应检查未运行。已执行语法解析、模块导入验证、完整单元测试及上述原路径实际资产短样本检查。

测试包含使用临时夹具的模拟、反演、bank、投影和SNR检查。
未复制真实解剖资产、EEG/NMM库或checkpoint，因此没有重新执行依赖这些资产的全规模实验。
Python依赖使用原工作区已有环境，没有重新安装；确切版本见`tested_environment.json`。
验证过程中的初始沙箱启动及Windows多进程测试启动器问题已经在集合外的验证脚本中处理；
上面报告的是修正启动器后的成功结果，不涉及修改复制版模型算法。
