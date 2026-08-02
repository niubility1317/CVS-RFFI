# d122_g1_sourceheld_d5a1892a_20260803_r1实验报告骨架

##登记信息

|字段|内容|
|---|---|
|Run ID|d122_g1_sourceheld_d5a1892a_20260803_r1|
|时间戳|2026-08-03（Asia/Hong_Kong，预登记）|
|操作员|Codex（D122唯一N607 runner）|
|状态|STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT|
|目标|验证冻结RDCE域适应与D112静态ground head在同一坐标和同一source-held矩阵中的独立效应与交互。|
|假设|RDCE与ground head各自已有正向历史证据；Jacobian方差输运后的M_JOINT可能同时保留M_DA的域适应收益和M_HEAD的K1/floor收益。该假设必须由完整同row结果裁决。|
|比较目标|四臂使用同一矩阵、同一输入包和同一运行边界；仅比较同一行证据。|

##冻结版本与候选臂

|项目|登记值|
|---|---|
|代码commit|d5a1892a|
|设计版本|e8b84afa|
|独立review|MERGE P0=0 P1=0|
|聚焦测试|14 passed|

|候选臂|登记类别|结果状态|
|---|---|---|
|M0|基线臂|TBD|
|M_DA|域适配臂|TBD|
|M_HEAD|分类头臂|TBD|
|M_JOINT|联合臂|TBD|

##本地smoke与矩阵登记

|字段|内容|
|---|---|
|local smoke predictions|E:\type10-7\automation_reports\CV-SincNet\d122_g1_local_smoke_d5a1892a_20260803_r2\predictions|
|manifest SHA|605bba274c67f77ff07913c6c39ab6c1ed8ab23bd461e78c245da187a9ce685a|
|矩阵规模|63 rows/252 units|
|split/seed|d104_source_seed104713_v2/104713|
|查询状态|query zero|
|活动状态|active old6/fallback0|
|rho范围|0.044852635602166874..0.5544932699603999|

##远端输入包与执行登记

|字段|内容|
|---|---|
|输入包|/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages|
|RDCE wire SHA256|20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795|
|tap SHA256|48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f|
|receipt SHA256|24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665|
|checkpoint SHA256|2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98|
|远端项目|/home/szu2070436088/2510044040/CV-SincNet|
|Conda/Python环境|/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python|
|运行根目录|/home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_d5a1892a_20260803_r1|
|GPU|GPU0|
|精确服务器命令|见启动前runner登记；先predict完整封存，再score打开truth。|
|日志路径|/home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_d5a1892a_20260803_r1/runner.log|
|主PID|95089（已退出；exit=1）|
|同步目的地|code/cvsrffi/stage2_d122_rdce_ground_head.py、code/scripts/run_d122_g1_sourceheld_one_shot.py；tap/receipt至新run root/input。|

##预期产物

|产物|预期值|状态|
|---|---|---|
|prediction_manifest|存在且覆盖63 rows|TBD|
|rows|63 rows|TBD|
|truth_open_event|存在|TBD|
|held_scores|存在|TBD|

##本地变更与验证

|字段|内容|
|---|---|
|本次变更|新增D122核心、G1runner和两份定向测试；冻结设计与agent角色路由另行版本化。|
|验证范围|`py_compile`通过；两份定向测试`14 passed`；真实21包无truth smoke覆盖63行/252单元，六旧类全激活、零fallback、新类逐bit边界全通过。|
|Git状态/提交|设计`e8b84afa`；角色路由`13891438`；实现`d5a1892a`；报告提交待补。|
|远端落地|TBD；按本任务明确要求不执行SSH/SCP。|

##健康与停止规则

停止仅P0或两行同fingerprint无prediction。P0包括协议、安全、错误checkout/hash、输出覆盖或查询泄漏等系统性故障。禁止看性能停；不得依据accuracy、H、BA、floor或其他性能数值停止运行。停止后应保留已产生的日志、行退出、prediction计数和其他部分产物，并标记NO_PERFORMANCE_RESULT。

## r1技术闭合

|字段|r1事实|
|---|---|
|最终技术状态|STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT|
|启动与退出|wrapper PID95089；exit=1；进程已退出。|
|唯一异常|predict导入阶段缺少scripts.run_d106_g1_sourceheld_one_shot；D122 runner在零prediction前退出。|
|预测与评分|0个prediction单元；无prediction_manifest、rows、truth_open_event或held_scores；truth未打开。|
|停止决定|技术性导入闭合失败；未查看任何性能数值，未调参、未重启。|
|资源清理|N607 GPU compute-app列表为空；本地ssh.exe=0，到N607:22的ESTABLISHED=0。|
|回收路径|E:\type10-7\automation_reports\CV-SincNet\d122_g1_sourceheld_d5a1892a_20260803_r1\artifacts\n607|
|runner.log SHA256|a479c937e70b11ddcaf10801911e830ef06791e13c33fdcbc268fe969f2805a4|
|main.pid SHA256|67230371c2c86327d1b347dabd2a49231f5131a0bddeb3b1c433eeafa46fbcc6|
|exit_code SHA256|4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865|
|cleanup receipt SHA256|c2cf7188e31d1626436f4a929eae86120cf8673bad8a88f67fcce30d6e3272e4|

##结果表（TBD）

|候选臂|机制/类别|receiver/TX split|K-shot|seed|old|seen_new|unknown|coverage/rollback/defer|loss/adapter|最终判定|
|---|---|---|---|---|---|---|---|---|---|---|
|M0|基线|TBD|1/5/10|104713|TBD|TBD|N/A|N/A|无训练/identity qKNN|TBD|
|M_DA|RDCE域适应|TBD|1/5/10|104713|TBD|TBD|N/A|N/A|无训练/rank-3 RDCE|TBD|
|M_HEAD|静态ground head|TBD|1/5/10|104713|TBD|TBD|N/A|N/A|无训练/单位质量anchor|TBD|
|M_JOINT|RDCE＋同坐标静态ground head|TBD|1/5/10|104713|TBD|TBD|N/A|N/A|无训练/Jacobian方差输运|TBD|

##最终结论

TBD。待完整日志、prediction_manifest、truth_open_event、held_scores和同一矩阵结果返回后填写；当前登记不构成性能结论。

##字段缺口（骨架阶段）

启动与完成时间、主/子PID、实际运行状态、每行退出与prediction计数、完整同一行指标、异常记录、checkpoint/最佳epoch、最终结论和远端同步回执均待补。

## 启动前runner登记

输入：packages=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages；truth=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth.json；truth seal=/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth_input_seal.json；RDCE wire SHA256=20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795；新run root在只读检查时不存在。

同步：stage2_d122_rdce_ground_head.py SHA256=1d244d2ef89fc4bbd9d87c02a83f008548fd65e62bc05a579c77ad5897764197；run_d122_g1_sourceheld_one_shot.py SHA256=07e7902d5c3fddf723af9401e3f500e3029590a6db02b6bc90110e0048581429；tap SHA256=48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f；receipt SHA256=24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665。

固定目录=/home/szu2070436088/2510044040/CV-SincNet；Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python；CUDA_VISIBLE_DEVICES=0：

~~~bash
python code/scripts/run_d122_g1_sourceheld_one_shot.py predict --package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages --rdce-asset-wire /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --d106-tap-archive <run_root>/input/d106_ls_strict_tap.npz --d106-tap-receipt <run_root>/input/d106_ls_strict_tap.receipt.json --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d122_g1_sourceheld_d5a1892a_20260803_r1 --output-dir <run_root>/predictions
python code/scripts/run_d122_g1_sourceheld_one_shot.py score --prediction-root <run_root>/predictions --truth-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth.json --truth-input-seal-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth_input_seal.json --truth-open-event-json <run_root>/truth_open_event.json --output-json <run_root>/held_scores.json
~~~

停止仅P0或两行同一确定性exception fingerprint且零prediction；不得根据性能数值停止、调参或重启。
