# Phase1单读出local4控制bundle v1真实构建报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.基本信息

|字段|冻结值|
|---|---|
|实验／任务ID|`phase1_single_control_bundle_v1_build_20260809_v1`|
|操作主体|主控`/root`负责方法、版本和结论；唯一N607 runner负责落地、启动、监控和回收|
|目标|从冻结F1C local4控制checkpoint和ManySig source-only分片构建一个不可变的10文件本地证据bundle，并完成真实IQ六字段parity、状态零更新、资源门和CARE N=1技术闭环|
|性质|一次性技术构建；不是训练、性能评测、Phase1晋级、六类deployment、真实unknown拒识或多星协同实验|
|比较基准|本地fixture只证明接口；本轮检验真实F1C字节能否通过同一严格builder／loader合同|
|实现commit|`bb5b7f05dbb3c94e3925fd8754a60321250fb94b`|
|本地Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`，分支`codex/phase3-responsibility-20260807`|
|根报告|本文件位于非Git根`E:\type10-7\automation_reports\CV-SincNet\phase1_single_control_bundle_v1_build_20260809_v1\report.md`；同步镜像进入上述Git工作树|

## 2.假设与证据边界

技术假设：冻结F1C闭集身份头可以在不读取proxy／held／target／query、不更新模型状态、不引入G候选的条件下，导出local4控制bundle；其中`z_dom`只是固定五维IQ统计descriptor，`q`只表示model reliability，不能称为学习域表征、物理链路质量或真实unknown后验。

成功只表示：

1.输入checkpoint、三份训练receipt、ManySig字节SHA、split、类序和场景registry全部闭合；
2.L只进入类几何，U的view seed和descriptor严格标签盲，V只形成技术stress-tail；
3.9个payload加`manifest.json`的allowlist、member SHA、外部content root和strict loader闭合；
4.真实IQ的`z_id、z_dom、q、d_class、e_unknown、p_local`与决策字段通过eager↔runtime parity，前后state digest一致；
5.CARE N=1只验证规范化接口恒等，不构成协同收益；
6.资源门在N607实际环境通过。

本轮不读取或解释任何accuracy、AUROC、FAR、safe rejection、LEO性能或unknown性能。G1仍因ManySig缺少truth-blind跨RX同事件绑定而`BLOCKED_DATA`，本构建不会绕过该阻塞。

## 3.冻结输入

|输入|远端路径|冻结SHA256／合同|
|---|---|---|
|F1C最终checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth`|`0b1e1d24621f5c044b0a77f30915ec1f67342e6132fba8df28f21b43ad6b2ab8`|
|training completion|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json`|`c31edd31f1ec322615b4d0647cfcb9ece4e8ef5c3940d54aaa89c85c60f4431c`|
|terminal status|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json`|`0575ed6ee778e5b7b94e1e5b842e9ff24bf32496b05d36f82f658117a791c3a2`|
|CP terminal|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json`|`5a9677d6eab883f221ceb5c544f8e0bf6bcdb26479bba326766494bb7ce482e0`|
|ManySig|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；receipt内空dataset SHA无权覆盖|
|source split|由receipt机械重建|seed`7281105`；mode`tx_rx_day_1_6_3`；L／U／V=`0.07／0.63／0.30`，数量`3920／35280／16800`|
|local4 TX order|固定|`20-15、20-19、6-15、8-20`；head IDs`0,1,2,3`|
|source／target|固定|source days`0,1`、RX`0..6`；target days`2,3`、RX`10,11,7,8,9`；equalized`1`|
|场景|固定|`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`；每physical独立label-blind seed；无TTA|

三份receipt所在目录前缀均为：

`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12`

## 4.本地变更与验证

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/phase1_single_control_bundle_v1.py`|严格bundle、状态、runtime、local evidence、真实builder|`D44D907E1D346FFF9DB3409B03A7570133B9AA0BF969BF8D36ED862AAA12012B`|
|`code/scripts/build_phase1_single_control_bundle_v1.py`|fixture／real build／external-root verify CLI|`E116788F0B13884C3E3C2AE0F2376CA242BD9D8731C95DE677D6E1C7D11D162E`|
|`code/tests/test_phase1_single_control_bundle_v1.py`|协议、标签盲、CUDA、资源、bundle和负测|`800AB1B0E18D8DCEE8C136F9F2B9FF10F7E530257D27990DA7A346C137DA592D`|
|`analysis/phase1_single_control_bundle_v1_design_20260809.md`|Revision9设计和追溯|`45BC33DDC435260DB3FBC0090F2DF6A258F1A054531C0D9AF673265A572A65BA`|
|`analysis/phase3_final_goal_traceability_20260809.md`|总目标状态和G1数据阻塞|`529F3E438DA3898D2277B68EF902BC5E7BE041C7843FD468E18DC8B800BA761B`|

本地`ssr-gpu`验证：

- `python -m py_compile`：PASS；
- `python -m pytest -q code/tests/test_phase1_single_control_bundle_v1.py`：`16 passed`，实际CUDA路径执行；
- CLI fixture build＋外部content-root verify：PASS；
- 300行测试与独立1000行reference的全量median／MAD严格相等；
- `git diff --check`：PASS，仅Windows换行提示；
- 独立实际diff复审：`P0=0、P1=0、ALLOW`。

## 5.N607落地和唯一命令

冻结release：

`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v1_bb5b7f05`

冻结输出根，启动前必须完全不存在：

`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v1`

冻结日志根：

`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_single_control_bundle_v1_build_20260809_v1`

冻结Python：

`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

冻结物理GPU：`GPU7`。预检若该卡会超过每卡2个训练进程、环境或输入不满足合同，则停止并返回阻塞；不得换GPU、调参数或自行重试。

在release根执行的唯一build命令为：

```bash
CUDA_VISIBLE_DEVICES=7 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python   code/scripts/build_phase1_single_control_bundle_v1.py   --real-build   --project-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_single_control_bundle_v1_build_20260809_v1_bb5b7f05   --checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/final_ssdg.pth   --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl   --completion-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_training_completion_receipt.json   --terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_terminal_status.json   --cp-terminal-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cp_sfce12_20260809_v2/F1C_CP_SFCE12/phase1_cp_sfce_terminal_receipt.json   --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_single_control_bundle_v1_build_20260809_v1   --device cuda
```

runner必须以`nohup`／detached方式仅启动一次，把stdout＋stderr写入`build.out`并记录准确PID、CWD、cmdline和GPU映射；SSH caller超时后只能只读核验是否landed，禁止重复launch。

## 6.预检、健康和停止规则

启动前依次确认：

1.按项目规则先执行direct N607 read-only preflight；若仅TCP／SSH路径不可用，最多使用一次verified lab bridge；身份或target歧义立即停止；
2.release、run、log、staging sibling和outer路径均不存在；
3.固定commit archive无prefix落地，代码结构无`code/code`嵌套；
4.5个授权文件及设计卡的archive member SHA闭合，远端`py_compile`、CLI`--help`通过；
5.冻结checkpoint、三receipt和ManySig均存在且SHA逐项匹配；
6.GPU7资源满足冻结上限，且无同run ID进程；
7.使用exact command唯一启动一次。

停止／失败规则仅看技术事实：

- 错误commit、路径或SHA；
- output／staging覆盖风险；
- receipt、split、class order、scenario或strict load不闭合；
- Traceback、CUDA／OOM、nonfinite、parity、state mutation、resource gate或loader失败；
- 进程非零退出、无manifest或10成员不完整。

任一项触发即标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，保留日志和partial，不重试、不调门、不读取性能。若SSH caller超时但远端进程存在且日志增长，只继续短连接监控，不能视为失败或再次启动。

## 7.成功门和预期工件

预期输出根严格包含9个payload和1个manifest：

1.`runtime/local_evidence.ts`；
2.`state/class_geometry.npz`；
3.`state/domain_descriptor_stats.npz`；
4.`state/rank_tail_summary.npz`；
5.`locks/checkpoint_binding.json`；
6.`locks/class_binding.json`；
7.`locks/source_partition_receipt.json`；
8.`locks/runtime_parity_receipt.json`；
9.`locks/resource_receipt.json`；
10.`manifest.json`，不进入自身members。

技术成功门：

|门|要求|
|---|---|
|成员／root|严格10文件，无missing／extra／symlink；外部expected content root复验PASS|
|bundle状态|`TECHNICAL_LOCAL4_CONTROL_BUNDLE`、`performance_promoted=false`|
|输入绑定|checkpoint、dataset、三receipt、resolved config、class order、scenario和split全部闭合|
|状态边界|runtime／三state payload测量前后digest一致，query零update|
|本地证据|六字段和决策parity PASS；单条canonical JSON`<=64KiB`|
|资源|bundle`<=32MiB`；CPU RSS增量`<=512MiB`；CPU batch1 p99`<=250ms`；CUDA可用时VRAM`<=256MiB`|
|CARE N=1|validate后`p_local`、decision、label、reason和evidence hash规范化恒等|
|声明|`NO_PERFORMANCE_RESULT`；不得声称unknown FAR、协同收益、Stage2-C、注册或在轨部署性能|

成功后runner以manifest的`content_root`执行一次同release CLI`--verify-bundle --expected-content-root ... --expected-status TECHNICAL_LOCAL4_CONTROL_BUNDLE`。只回收日志、manifest和JSON小receipt；不得下载ManySig、源checkpoint或包含冻结权重的`runtime/local_evidence.ts`，NPZ是否回收以小工件安全边界为准。远端完整bundle原样保留。

## 8.已知风险与完成后检查

- 真实构建需遍历L／U／V并生成三种LEO view，耗时可能明显长于本地fixture；只以PID、GPU、日志增长和最终artifact判断健康，不设拍脑袋超时。
- U不读TX标签，但builder为partition审计可在隔离局部变量中枚举physical key；任何标签进入U seed／descriptor排序即失败。
- F1C训练终态为预期`NON_PROMOTABLE_P0_DISABLED／exit8`，本bundle只将它作为字典序首折C臂技术控制，不能改写为Phase1晋级。
- `source_exchangeable_calibration=false`、`finite_sample_exact_conformal=false`必须保留；129点tail只是固定技术rank aggregate。
- G1 real proxy multi-receiver仍缺truth-blind grouping manifest；本轮成功不会解除该数据阻塞。
- 结束后检查：build exit、10成员、member SHA、content root、strict verify、runtime parity、resource receipt、state digest、日志错误指纹、GPU／进程／SSH清理，并把结果写回本报告及Git镜像。
