# Phase1 CLIC后冻结v2预注册与运行报告

## 1. 状态与目标

- 实验ID：`phase1_clic_postfreeze_20260812_v2`
- 当前状态：`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 目标：从已完成、不可覆盖的训练run`phase1_clic12_20260812_v5`导出12个C／G source clean／source-V／fixed400 proxy工件，随后继续source-L三场景LEO weak、同fold pair、bundle及目标域registered／unknown确认闭环。
- 假设：v1的统一故障仅由导出器错误要求checkpoint的`id_feature_key=z_id`造成；将checkpoint构造键固定为真实`feat_joint`、同时保持正式导出特征键为`z_id`，可恢复工件生成且不改变模型、阈值或科学路线。

## 2. v1故障与v2唯一修复

- v1状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；12／12行在任何NPZ前以同一`CLICSplitExportError: checkpoint arg id_feature_key drifted`退出，NPZ=`0/12`。
- v2只分离两个原本被错误共用的合同：checkpoint模型构造键=`feat_joint`；CLIC后冻结几何输入／导出键=`z_id`。
- 未改变训练checkpoint、terminal、source TX角色、split、ManySig字节、fixed400 proxy、C／G operator、GPU映射、loss、阈值或任何target规则。
- v1目录、日志和报告保持不可覆盖；v2使用全新的run／log／release路径，不恢复、不重试v1。

## 3. 科学与数据边界

- G只从同一份`received_i`提取固定lag=`{1,2,4,8}`的多尺度三点复曲率token；C为参数匹配control。
- 本次第一波只读取ManySig、v5 final checkpoint及terminal receipt；target／query／truth／role访问均为0。
- 每fold后续source-L LEO缓存只生成一次，C／G共享同一received-IQ字节；三scene物理样本互斥，不择优、不融合、不重采样。
- 所有正式target指标最终必须覆盖`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，同时报告registered-known域泛化和registered／unknown拒识；本波不读取或产生target性能。
- ADV3B02只要求训练数据配置与known-test配置信号等价，不要求同capsule、同物理样本或同received-IQ字节。

## 4. 第一波冻结矩阵与资源

|阶段|行数|输入|输出|GPU|
|---|---:|---|---|---|
|source clean／proxy export|12|F1—F6×C/G checkpoint、terminal、ManySig|每臂`source_clean_proxy.npz`|`0,0,1,1,2,2,3,3,4,5,6,7`|

- launcher：`code/scripts/launch_phase1_clic_postfreeze_source12_20260812.sh`
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；工作目录为新release的`code`。
- 训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic12_20260812_v5`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_postfreeze_20260812_v2`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_postfreeze_20260812_v2`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。

## 5. 本地验证与发布门

- TDD RED：将checkpoint fixture改为v5真实`id_feature_key=feat_joint`后，旧导出器精确失败于`checkpoint arg id_feature_key drifted`，与N607 v1的12／12指纹一致。
- GREEN：分离checkpoint键与导出键后，聚焦测试通过；完整`code/tests/test_phase1_clic_postfreeze.py`通过102个参数化用例，仅3条既有AMP弃用warning。
- `py_compile`通过；`git diff --check`通过。
- launcher本地dry-run逐行解析为12行、C6／G6、12行全部绑定v2、target／query／truth／role入口为0。
- 独立Terra审查结论：`P0=0/P1=0/ALLOW`；另以C／G两种operator逐元素确认identity-only导出的`z_id`和`tx_logits`与完整`return_aux`路径一致。
- 对v5真实payload的只读库存核对确认其不含`split_info`、顶层`candidate_id／run_id`副本或冗余`phase1_clic_enabled`；新增回归要求这些副本缺失时由args、目录、operator、terminal checkpoint路径／字节SHA／arm及source／class／physical count+SHA闭合，若旧副本存在但漂移仍拒绝。
- 第二次独立Terra窄审同样为`P0=0/P1=0/ALLOW`，确认`WiSigSubsetDataset.selected`与训练terminal的physical-order哈希正对应重建的labeled索引序列。
- 最终导出器SHA256=`7A9FFC614C56F8D1879967FB5BAB93B49F7B36AF639FE8DDB3F97A6F3EF9C952`；launcher SHA256=`936319E8289E3FC1E8B509A927D03B4569D91C5CD82432812AABA0DC5ED53F0A`。
- 发布前剩余门：本地Git commit、N607只读preflight、远端release hash／compile／help／launcher dry-run及一个真实v5 checkpoint重开smoke。

## 6. 运行与停止合同

- 正式launch恰1次；retry=`NO`。启动后核对outer／12个PID、确切CWD／命令、GPU映射、日志增长及12个输出路径。
- 预期成功工件：`12/12 source_clean_proxy.npz`；12个工件完整前不读取或解释accuracy、loss、AUROC、`u_gap`等性能值。
- 若错误checkout／hash、输出覆盖、协议字段漂移，或至少2个独立行在输出前出现同一确定性异常，立即停止该run并封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 不得因任何性能数值停止、选臂、调参、换scene、换seed或重试。

## 7. 待运行回填

- Git commit与文件SHA：待提交后回填。
- N607 archive／release／SCP次数：待唯一runner回填。
- outer／PID／GPU／日志／工件计数：待运行后回填。
- 最终状态：尚无性能结果。
