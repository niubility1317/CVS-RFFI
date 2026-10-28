# Phase3 独立无标签流与后续查询集

2026-10-29。本数据工作属于用户已授权的独立 Phase3，不恢复暂停的 Phase2 队列。原始 Full eq 已完成验证；本次只创建新的单观测与角色划分，不新增模型训练、目标预测或评分。run 固定为 `20261029-phase3-full-pool-rx24-r01`。

## 固定划分

目标接收机为 RX24-18，与当前 PAMF 的 source RX1-19、18-2、19-2、2-19、3-19 不交，也不在既有四个 compact 子集的接收机集合内。该事实避免从旧 compact query 转训练；不能据此声称跨接收机接收的是统计独立发射事件。

|角色|日期|builder 私有设备集合|每 TX 每景|物理记录|
|---|---|---|---:|---:|
|无标签适应／发现|2021_03_08|旧 6 类＋11-4、3-13|32|768|
|后续独立 query|2021_03_23|上述 8 类＋12-20、5-20|64|1920|

旧 6 类顺序保持 14-10、14-7、20-15、20-19、6-15、8-20。3 个场景为 low suburban、mid urban、low urban，是等量压力分层，不能解释为实际被动监听概率。每个接收记录只绑定一景、一个随机信道 seed、一份 received IQ，不复用物理记录生成多景。无标签流与 query 日期不交；source 也使用了 D3 的其他 RX 地面数据，因此这里只声明后续独立观测，不声明严格未来日期外推。

抽样和场景分配对全部 TX 使用同一公式：每个 RX/date/TX 单元按固定 selection seed 做索引随机排列，取前 `3 × per_scene` 条，按选中次序对 3 景循环分配；通道 seed 只取决于 channel seed、scene 和 canonical 接收记录 ID。真实 known/unknown 角色不进入 scene、seed、质量筛选或处理路径。输出整体混洗，公共 ID 不编码 TX、RX、日期、索引或场景。

## 输入与处理

复用 `local_artifacts/pamf_phase2_full_eq_validation_20261029_r02` 已完成的 158 PKL／8767867 条原始验证，只用成员索引定位 RX24-18 的 D1、D3 两个成员；不重扫 20 ZIP、不增加 hash 或 receipt 链。实际计数中 D1 最小 1006≥96，D3 最小 264≥192，满足本池需求。builder 仍核对本次选中成员、形状、数量、角色及输出有限值。

Full 原数组为 float64 `[N,256,2]`，已有 eq 身份；不重裁 256 点、不重新均衡。明确按匹配 source 的 `iq_record` 进行 FP32 布局转换和每条 unit RMS，然后直接调用 `ResidualChannel`：25 MHz、2462 MHz、`residual/post_sync/noEQ`、reference SNR 30 dB、reflect boundary、输入状态 `WiSig_equalized1_center256_unit_rms`。该 source 输入归一化与星地接收后处理区分；星地输出不二次归一化、不保存 clean 参考。接收机响应按 RX/date/eq session 和固定 receiver seed 生成，各记录有自己的固定随机信道实现。

选择／receiver／channel／shuffle seed 分别为 2026102901／02／03／04。配置在 `configs/pamf_phase3_full_pool_20261029.json`。Full RX index30 表示24-18，属于 Full 命名空间，不能冒充 Main index。

## 公共与评分边界

每个角色的 `public/` 仅有 `manifest.json`、`ids.npy`、`received.npy`。manifest 使用 `phase3_min_v1` 精确白名单，初始已注册类只列合法旧 6 类；公共输入不含真实未知数量、类别配额、逐条 TX／role／scene、source 原始路径或 truth。无标签算法只能从公共流自行估计簇数。

独立 `truth/truth.npz` 含 opaque ID、真实标签、Full RX index 和 scene，builder 私有报告保留 canonical 接收记录及随机分配映射。无标签流的 truth 也不能进入拟合或注册，后续 query 预测完整固定后才允许评分端读取，用发现流确定匿名簇与 TX 的评分映射，并在后续 query 评价，不在 query 上重新匹配以美化注册表现。旧类误注册、错误合并、拆分和重复注册单独报告。

初始冻结 F 对 4 个新 TX 均未见过。开放 D1 无标签流后，11-4、3-13 已在部署期无标签见过，须按“无标签见过但未注册／随后注册”评价；12-20、5-20 始终不进入该流，用于后续严格未知拒识。适应后的这两种 unknown 不能合并冒充全部严格未见；拒识、旧类适应、发现类未来识别和后续未知分别报告。

builder 一次核对 source/target、两角色和跨场景物理互斥、每记录单观测、公开无标签及实际形状，然后才写 `VALIDATED_ONCE` 完成标记。未变化的池跨方法复用。准备数据不开放 query 训练；正式方法及阈值仍须先在 source 冻结，不从目标结果挑选分支或参数。

## 当前状态

**VERIFIED／VALIDATED_ONCE**。真实构建已完成，独立读回两个public目录、完成标记和32行完整日志通过。U768（每景256）／Q1920（每景640），全局2688个canonical接收记录唯一且跨角色／场景互斥；只读取2个ZIP成员，没有重新全量验证。未加载模型、预测或训练，零GPU，不停止其他owner。原生PID3368及实际命令已核实，完成后进程退出。

实际code/config commit为`e4ebaa2682d8c2e87182a8b8ce36ae5ea5eec4d9`。11项针对性测试及本候选一次独立P0/P1审查PASS，包含真实ResidualChannel三景合成IQ。新池位于`local_artifacts/pamf_phase3_full_pool_20261029_r01/pool/`；独立读回、完整构建日志、元数据分析见同目录上层的build_post_state.json、build.log、analysis_summary.json。原始数据、IQ及truth不提交Git。

|角色|capsule_id|split_id|
|---|---|---|
|discovery|p3full-636b5bdd4dac13a1608e63a7|332811f1ff75c05e18c794e7|
|evaluation|p3full-35f7907e8044d907f22701e4|ff2715f5818c999bcc80fc2c|

这只是数据入口就绪，不是Phase3正式拒识、适应、分类或注册成功。正式query尚未模型推理或评分，后续方法仍先通过源开发门槛，再用同一未变数据池确认。
