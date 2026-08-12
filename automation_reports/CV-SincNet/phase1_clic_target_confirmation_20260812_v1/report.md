# P1-CLIC目标域LEO盲态确认v1

## 当前状态

- 实验ID：`phase1_clic_target_confirmation_20260812_v1`
- 日期：2026-08-12
- 操作者：Codex主控；N607唯一runner待委派
- 状态：`LOCAL_VERIFICATION_IN_PROGRESS / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`
- 目标：一次性构建共同的`p2_min_v1`目标域LEO weak registered-known／unknown缓存，随后让F1—F6的C／G共用同一IQ-only包完成零适配推理；隔离scorer报告域泛化、真实unknown拒识和配置匹配ADV3B02比较。

## 固定数据与矩阵

|字段|固定值|
|---|---|
|receiver|`20-1`|
|registered-known union|`14-10,14-7,20-15,20-19,6-15,8-20`|
|unknown|`1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6`|
|days|`0,1,2`|
|物理样本|每TX 120条，三scene各40条；每physical ID只产生一份received IQ|
|scene|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|seed|dataset=`713101`；scene=`7131010/7131011/7131012`|
|目标预测|F1—F6×C／G，共12份；同一target package；每样本恰一次forward|
|更新权限|target端训练／适配／阈值拟合／温度校准／选择／重试均为0|

共同union6只定义共享缓存。每fold的正式registered-known计分宇宙严格取对应source-policy local4；inactive2显式排除且不得转为unknown。unknown20与registered union6互斥。

## 指标与门

每个fold×arm×scene同row保存：registered-known overall、macro、min-class、min-receiver、min-day、accepted accuracy、false reject、defer；unknown保存AUROC、AUPR-out、FPR95、显式拒识率、误接收率、安全处置率及TX／receiver／day最差切片。全体unknown及每个scene的`decision=unknown`比例均须至少70%，`defer`单列且不进入分子。

ADV3B02无需与CLIC共享capsule、物理样本、received-IQ或seed；只要求训练数据配置与known测试数据配置相同。baseline与candidate均按相同fold local4、receiver、day和scene语义cell比较；禁止union6边际替代local4切片。

## 本地实现与验证

- cache spec：`code/configs/phase1_clic_target_confirmation_20260812_v1.json`
- cache launcher：`code/scripts/launch_phase1_clic_target_cache_20260812.sh`
- source clean v3 launcher：`code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh`
- target evaluator：`code/evaluate_phase1_clic_target_leo.py`
- C predictor sealer/runtime：`code/cvsrffi/phase1_clic_target_leo.py`
- G bundle：`code/export_phase1_clic_deployment_bundle.py`

已完成的窄验证：新cache scope／精确角色／三scene物理互斥／NPZ TOCTOU为5项GREEN；真实`CLEAN.export→bundle`闭环GREEN；target相关34项GREEN；C／G real forward与cache builder禁用`Tensor.numpy`／`torch.from_numpy`后4项GREEN；69 unknown+31 defer按207／300写入失败门而不误计defer。完整回归和独立终审仍在执行，未达到发布门。

## N607预注册

- 普通账号直连preflight；8卡可见后才落地。
- 所有代码先形成干净Git commit和不可覆盖release；SCP后逐文件SHA闭合。
- cache阶段正式入口：`bash <release>/code/scripts/launch_phase1_clic_target_cache_20260812.sh`；GPU0；日志`logs/phase1_clic_target_confirmation_20260812_v1/target_cache.out`。
- clean v3入口：`bash <release>/code/scripts/launch_phase1_clic_postfreeze_source12_v3_20260812.sh`；12进程按GPU表固定分配。
- 预期cache工件：3个scene NPZ、`cache_set.json`、PID表和日志；总行数26×120=3120，每scene1040，registered 6×40=240，unknown 20×40=800，三scenephysical ID两两不交。
- 技术停止：错误checkout/hash、覆盖风险、协议越权或至少两个独立row在prediction前出现同一确定性异常指纹时停止精确run-owned进程；不因任何性能值停止或选择。
- retry：`NO`。技术故障使用新run ID修复，不覆盖本run。

## N607唯一运行席检查点（2026-08-12）

- 冻结提交：`fe086aa81a19a48590ad9b24e83dbac47717b235`；主工作树仍含并行agent dirty/untracked改动，未纳入发布包。
- 本地commit归档：`C:\Users\lh594\AppData\Local\Temp\phase1_clic_target_confirmation_20260812_v1\clean_commit.tar`，SHA256=`7D052A0A34E99AEF0E33521A18B859178ADEE947F43118D94EB60C7637055424`，bytes=`267120640`；未在Windows解包，避免长路径改变归档验证边界。
- 冻结文件SHA：spec=`0A35055A4D1CF1537E3F1D5137C38C376798C0F579EB26A7C7D99906E99D4510`；launcher=`F76C2B6D6A2DC8D6EC0FEAEEC33E097DCFBAED6EA6AC8C44BC3C479E95F2C0DB`；builder=`D58335F7608A99616E91CC3C7578A79ABB205EC073E050406A95248DB849CE6D`；loader=`82A5BA41DF8017CFEC2F02154D928DF20473BA7C627A13D088F485C89955C19E`。
- N607直连preflight：通过；普通账号`szu2070436088`，项目根可见，GPU0为`0%/1MiB`，目标release/run/log路径均ABSENT；形式化启动计数仍为0。
- 当前阶段：`LANDED / SCP=1 / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。`SCP`仅一次成功；首次错误路径探针未落地。

## 风险与待闭合项

- N607的Torch2.1+NumPy2旧桥曾导致原生崩溃；当前直接cache、C forward、G reload路径均已有禁桥回归，但仍需真实F1烟测。
- 当前N607无已证实匹配的新scope cache，也无满足新crossed local4 RX/day合同的ADV3B02 immutable reference；必须新建cache，并为正式非劣评分补齐匹配reference。缺reference不得阻塞IQ-only预测封存，但不得发布非劣结论。
- 本报告当前不含性能数值，不作晋级结论。

- 远端release：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_target_confirmation_20260812_v1_fe086aa8；远端归档SHA=7d052a0a34e99aef0e33521a18b859178adee947f43118d94eb60c7637055424，四文件静态门和dry-run均通过。

## 运行席最终封存（2026-08-12）

- 最终状态：`LAUNCH_ENTRY_TECHNICAL_FAILURE / FORMAL_BUILDER_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 唯一formal launcher invocation计数为`1`，命令为`bash <release>/code/scripts/launch_phase1_clic_target_cache_20260812.sh`；退出码为`3`，输出为`refusing to overwrite target cache run/log root`。未重试。
- builder实际PID=`0`；`pids_target_cache.tsv`、`target_cache.out`、3个NPZ、`cache_set.json`及其他本run cache工件均不存在，故`artifacts=0`，不能进入性能分析。
- 根因：formal入口外层已预创建受保护的`runs/phase1_clic_target_confirmation_20260812_v1`和`logs/phase1_clic_target_confirmation_20260812_v1`目录，冻结launcher按防覆盖规则在builder启动前拒绝退出；未执行删除、改名或远端修复。残留目录及`launcher.out`均由普通账号`szu2070436088`创建，目录mtime约为20:18:33，`launcher.out`全文仅含上述拒绝信息。
- 只读收尾：目标run目录为空；目标log目录仅有48B`launcher.out`；无匹配的旧launcher/builder进程；8卡均为`0%`GPU利用率、约`1MiB`显存占用；本地`ssh/scp`进程及到N607:22连接均为0。
- 远端release静态内容也未形成本次冻结包闭合：实测spec SHA=`2d7f2a93d26b86b037040d4d1d7d90dcd46c053650ea8d887d08bf69efccd280`、launcher SHA=`f76c2b6d6a2dc8d6ec0feaeec33e097dcfbaed6ea6ac8c44bc3c479e95f2c0db`、builder SHA=`bcd6b0d1dd784ae518d26c1889645d2dd4b22bbcdee4158fea3dc606404370f2`，冻结loader文件缺失；上述证据只用于解释本run为何不能晋级，不构成性能结果。
- 后续边界：若继续，必须创建全新的`v2`run/release ID并重新走完整落地门；`v2`是新run，不是对`v1`的retry。`v1`保持不可覆盖、不可重启、不可作性能比较。
