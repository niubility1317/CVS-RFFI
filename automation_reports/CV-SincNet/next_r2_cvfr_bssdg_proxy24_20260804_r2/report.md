# NEXT-R2 CVFR-BSSDG Proxy24 r2实验报告

## 登记

- 实验ID：`next_r2_cvfr_bssdg_proxy24_20260804_r2`
- 日期：`2026-08-04`
- 操作者：主agent负责协议、候选、数据/结果分析和最终决策；唯一Luna/max runner负责N607落地、运行和artifact回收
- 当前状态：`LOCAL_VERIFIED / RELEASE_REVIEW_GO / READY_FOR_N607_HANDOFF`
- 目标：修复r1封存器错误后，以同一最小source-held矩阵判断`CVFR-BSSDG/r1`是否产生非恒等且同row联合正收益
- 性能边界：本实验不是Target、125或正式Stage2-C结论；r1为`NO_PERFORMANCE_RESULT`，不得合并或补评分

## r1失败与本轮唯一修复

r1已生成96份JSON、96份NPZ、96份BSSDG wire和48份CVFR wire，但manifest错误地要求每个文件内容SHA全局唯一。合法的相同状态可产生相同内容，因此r1在封存阶段失败，未产生manifest/completion，truth未打开，score未运行。

commit`4aea94c7788aa8406d0ccf486a911e8b7ab0de0b`只把封存条件改为：SHA格式有效；JSON/NPZ路径为run-root相对POSIX路径；后缀正确；路径在JSON/NPZ之间全局唯一；24/96覆盖和顺序精确。内容SHA允许重复。方法、损失、参数、receiver、held class、K、seed、输入和评分规则均不改变。这是第二次也是最后一次发布工程修复。

## 冻结矩阵和四状态指标

|项目|冻结值|
|---|---|
|候选|`CVFR-BSSDG/r1`|
|receiver|从7个source-only receiver按固定SHA256规则选2个，不读性能|
|held class|6个LOCO类|
|K|`1、5`|
|outer key|`2×6×2=24`|
|状态|`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`|
|prediction|`24×4=96`|
|明确排除|Target、125、K10、Cn20、调参、换seed、性能驱动重跑|

|状态|统一含义|允许指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|`A_retained、F_retained、total_correct`；`N/H=NA`|
|`DA1_REG0`|域适应后/新类注册前|`A_retained、F_retained、total_correct`；`N/H=NA`|
|`DA0_REG1`|域适应前/新类注册后|`A_retained、N_seen_new、H_retained_new、F_retained、total_correct`|
|`DA1_REG1`|域适应后/新类注册后|`A_retained、N_seen_new、H_retained_new、F_retained、total_correct`|

报告四个差分：域适应注册前=`DA1_REG0-DA0_REG0`；域适应注册后=`DA1_REG1-DA0_REG1`；无域适应注册效应=`DA0_REG1-DA0_REG0`；有域适应注册效应=`DA1_REG1-DA1_REG0`。DiD只用于四态共有指标。

## 本地版本和验证

- Git工作树：`E:\type10-7\code\snapshots\d92_lite125_20260804_wt`
- 科学候选与初始manifest修复基线commit：`4aea94c7788aa8406d0ccf486a911e8b7ab0de0b`
- 严格路径封存与r1结案commit：`92fc9ad8`
- 修复文件：`code/cvsrffi/stage2_next_r2_runtime.py`，SHA256=`f1d8d678afaa2df8138bf6c23d5721c56d468fa124ce0dee83660184ea04288c`
- 聚焦测试文件：`tests/test_stage2_next_r2_runtime.py`，SHA256=`65a172fe1f0ace0208a9845ed59c0e9db24f1cd8c2686106a7ef5e3e4ac56c28`
- 本地验证：六个NEXT-R2测试文件共`83 passed`；`py_compile`和`git diff --check`通过
- 真实checkpoint无query smoke：r1已在相同checkpoint、科学方法和特征路径上通过，receipt SHA=`e41a1a5809ff2f27360cd032d9b63258c710edbb961c4eac05c92aa893fed24f`；本次只改manifest最终封存，不重复smoke
- 独立修复复核：`P0=0、P1=0、P2=0、GO`；确认8个路径反例拒绝、重复内容SHA允许、无效SHA/错误后缀/重复路径/跨类型碰撞/24-96覆盖顺序继续失败关闭，且方法、矩阵、协议未变

## 固定输入

|输入|N607路径|SHA256|
|---|---|---|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|selected IQ|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`|`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`|
|selected receipt|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.receipt.json`|`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`|
|L_s label join|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d104_split/L_s/features.npz`|`dd315295bc65069f174529137ab0e5089c1e648b5cd902c476d79fbc18dd813d`|

`capsule_id=d106_ls588_proxy_dd315295`，`split_id=d104_source_seed104713_v2`。

## N607冻结执行

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 只读依赖包路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/d137_next_r1_fabr_tsl_proxy84_20260804_r1/source/code/cvsrffi`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r2_cvfr_bssdg_proxy24_20260804_r2`
- GPU：物理GPU0；若预检发现GPU0不满足每GPU最多2个训练进程或存在冲突，runner不得擅自换卡，返回主agent重新登记
- 同步：仅同步修复后的`code/cvsrffi/stage2_next_r2_runtime.py`到正确包路径，校验SHA和`py_compile`；其余7个r1已核验文件只读复核SHA，不重新复制
- output、log、capsule和score路径必须位于新r2 run root且启动前不存在；不得读取、续跑、封存或评分r1 partial output
- builder、predict、score分别为独立进程；只有builder和独立score可读L_s label join，predict参数不得出现label join

冻结CLI参数与r1完全相同，只把`run-id`和`run-root`改为r2，并为r2重新构建capsule。由于服务器现有项目包缺少NEXT-R1依赖，三个进程使用r1已核验的inline wrapper：先从当前`code/`导入`cvsrffi`，只读追加上述d137依赖目录到`cvsrffi.__path__`，设置各CLI的冻结`sys.argv`后用`runpy.run_path(..., run_name='__main__')`执行。runner必须在启动前把完整实际命令、GPU和PID回填本报告。

## 健康、完成和晋级规则

- 启动后核验主PID、精确CWD/cmdline/run-root绑定、GPU映射和日志增长；首个worker wave核验launched/completed/succeeded/failed、prediction/score计数及异常指纹。
- 只因P0协议/安全违规、错误hash/checkout、覆盖风险、launcher确定性故障，或两个不同outer key在prediction前出现相同异常指纹而停止；绝不因中间性能停止。
- 完成要求：24/24 outer key、96/96 state、每态JSON/NPZ/BSSDG wire、48份DA1 CVFR wire、manifest、completion和独立score全部闭合。
- 若全部DA1为identity或全部DA1预测与对应DA0相同，完整评分后记`NO_FUNCTION`并关闭候选。
- 否则K5 pooled的域适应注册后效应必须同时满足：`ΔH>0、Δtotal_correct>0、ΔA_retained≥0、ΔN_seen_new≥0、ΔF_retained≥0`。通过只允许进入更完整source-held复核；失败即关闭，不调参、不重复。

## 结果回填

|candidate|状态|receiver/TX|K|A_retained|N_seen_new|H_retained_new|F_retained|total_correct|资源|判定|
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
|`CVFR-BSSDG/r1`|`DA0_REG0`|待capsule冻结|1/5|待评分|NA|NA|待评分|待评分|待回填|待定|
|`CVFR-BSSDG/r1`|`DA1_REG0`|待capsule冻结|1/5|待评分|NA|NA|待评分|待评分|待回填|待定|
|`CVFR-BSSDG/r1`|`DA0_REG1`|待capsule冻结|1/5|待评分|待评分|待评分|待评分|待评分|待回填|待定|
|`CVFR-BSSDG/r1`|`DA1_REG1`|待capsule冻结|1/5|待评分|待评分|待评分|待评分|待评分|待回填|待定|

完成后回填逐key同row表、四差分、共同指标DiD、CVFR identity比例、预测相同性、资源、异常和最终关闭/晋级决定。

## 终态证据追加：技术停止（2026-08-04）

- 终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。未依据任何性能指标停止，未修改方法/矩阵/输入，未进行score重试。
- 首次predict绑定尝试：PID=`1132599`自然退出，日志`predict.log`仅报告`run root must be a new resolved absolute child of an existing directory`；output不存在且0 prediction。该日志保留并标记`COMMAND_BINDING_ATTEMPT0_ZERO_OUTPUT`。
- 经主agent授权，同一冻结r2内只做一次路径绑定纠正：`--run-root=/home/szu2070436088/2510044040/CV-SincNet/runs/next_r2_cvfr_bssdg_proxy24_20260804_r2/output`，其余参数完全不变；新日志为`predict_retry1.log`，不覆盖首日志。
- predict retry PID=`1134441`自然退出，`CUDA_VISIBLE_DEVICES=0`/`--device cuda:0`，GPU0释放；完成JSON显示`outer_keys_completed=24`、`states_completed=96`、`all_states_sealed=true`、`truth_opened=false`、`scoring_performed=false`。远端计数：state JSON=96、NPZ=96、BSSDG wire=96、CVFR wire=48；根JSON=plan/preregistration/manifest/completion四份。
- 独立score PID=`1135496`自然退出，GPU0释放；`score.log`唯一确定性异常为`NextR2ScoreError: state receipt binding drift`。未生成`score.json`或`scoring_completion.json`，因此没有性能结果。
- 远端日志SHA256：`predict.log`=`b3362565c2dd1f0a3ed4114f30f1acc05a5baf5c0671e8ed66b5f952bbc67207`；`predict_retry1.log`=`f2ae150eb1525772cb0f1bb8eae3a12217c820eb9f95b8431411ed7d8824d29b`；`score.log`=`194a6e43e674a9ac12f8c7966d71f244427c5dc09212f49f2109f7282eaf7e4c`；manifest=`903f2422d776589586fa2d91898f8386d41e4bd8dd5933184280988463924d1a`；completion=`1495409761f53ab4b8ef876664b796d588b017b5edf9f6b430dd5de175c48e06`。
- 完整回收artifact绝对路径：`E:\type10-7\automation_reports\CV-SincNet\next_r2_cvfr_bssdg_proxy24_20260804_r2\artifacts\remote_next_r2_20260804`（346文件；包含input、全部logs、96状态三类artifact及四份根JSON）。
- 所有SSH/SCP短连接结束后本机`ssh.exe=none`、N607/bridge TCP22无ESTABLISHED连接；远端GPU0无compute进程。主agent负责后续只读故障定位和性能空结果记录。
