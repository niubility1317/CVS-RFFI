# Phase1 CLIC source-L LEO weak第二波v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_leo_20260812_v2`。
- 当前状态：`LAUNCH_ENTRY_NOT_LANDED / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`（launch attempt=1；入口前检查失败，未创建正式进程）。
- 目标：修复v1真实烟测发现的Torch 2.1／NumPy 2.x数组桥接崩溃，然后为F1—F6各生成一份source-L单观测LEO weak received-IQ缓存，并由同fold C／G复用完全相同的缓存字节导出特征。
- 科学路线、矩阵、数据、场景、seed、GPU映射及停止条件均不变；本次仅修复N607运行时兼容性，不调参、不读取性能。

## v1失败与根因

- v1状态：`SMOKE_TECHNICAL_FAILURE / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`；报告commit=`3b5c10f4`。
- 唯一F1真实cache smoke在`Tensor.numpy()`处native segmentation fault，exit=`139`；未生成任何cache或receipt，正式run/log/outer路径保持不存在。
- N607环境为Python 3.10.19、Torch 2.1.0+cu121、NumPy 2.2.5。项目已有同一环境组合的兼容说明：旧ndarray C-API桥接不应使用`Tensor.numpy()`或`torch.from_numpy()`。
- 本地修复使用确定性list边界将Torch转为float32 NumPy，并用buffer协议加clone将float32 NumPy转回Torch；两处均保留shape、dtype、有限值和连续内存检查。

## 冻结输入与矩阵

- 训练run：`phase1_clic12_20260812_v5`，12／12 checkpoint与terminal技术闭合。
- source clean第一波：`phase1_clic_postfreeze_20260812_v2`，12／12`source_clean_proxy.npz`已完成。
- 每fold source-L严格3920行=`4 TX×7 RX×140`；每cell按稳定物理ID排序后分配三scene为47／47／46。
- 正式scene：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；每物理样本只绑定一个scene、一个seed和一份received-IQ。
- 输出矩阵：6个共享cache构建，随后12个C／G source-LEO export；同fold C／G同GPU且读取同一cache路径。
- 目标输出：6个cache NPZ、6个cache receipt、12个source-LEO NPZ、12个binding。

## 许可边界

- 只读source-L、对应checkpoint／terminal和ManySig；U、V、proxy、target、query、truth和role均不进入缓存forward、fit、阈值或选择。
- 缓存NPZ exact member仅为`received_iq,tx_ids,rx_ids,day_ids,physical_sample_id,sat_scenarios`，不保存clean IQ、feature、logit或模型状态。
- C／G共用同一fold缓存字节，不按arm重采样；无多scene融合、择优、TTA或零时适配。
- ADV3B02比较只要求训练数据配置和测试数据配置等价，不要求与CLIC复用同一目标包或received-IQ。

## 本地版本与验证

- 预计改动：`code/build_phase1_clic_source_leo_iq.py`、`code/tests/test_build_phase1_clic_source_leo_iq.py`、`code/scripts/launch_phase1_clic_source_leo12_20260812.sh`及本报告。
- TDD RED：两项测试分别禁止`Tensor.numpy()`与`torch.from_numpy()`；实现前均精确失败于兼容桥接API缺失。
- GREEN：两项兼容测试通过；source-L builder专项现为15项全过。
- 联合回归：builder 15项+Phase1 CLIC 190项+postfreeze 106项，共311项全过；仅10条既有AMP弃用warning。
- `py_compile`通过；launcher`bash -n`通过；dry-run精确18行=`6 cache+12 export`、C6／G6，target／query／truth／role参数为0。
- 本地文件SHA：builder=`6E9C34456EBFDEAFBE0D119029DEFDBF89938C1006D965611DCFEA8B1EC32B24`；测试=`11C240342703E974BE95AE115C2DBE5FD7BEAB977F5BA989AD4620455177E7FA`；launcher=`3813EEEA125075D47D1B7C9976C78BD2681A4398418B33EFF58F41B9876B265D`。
- 独立Terra终裁：`P0=0/P1=0/P2=0/ALLOW`。审查同时禁用`Tensor.numpy`与`torch.from_numpy`后运行完整3920行合成builder，并验证非连续float64输入、float32输出、finite、连续性及Torch↔NumPy双向断别名；全部通过。

## N607发布与停止合同

- 新run ID、run root、log root均为`phase1_clic_source_leo_20260812_v2`，不得重用或覆盖v1。
- 本地验证与独立审查通过后，由唯一N607 runner执行一次新archive、至多一次SCP和一次正式launch。
- 正式launch前仅允许一个F1真实cache smoke；它必须生成3920行exact-member NPZ与receipt，并由生产consumer重开成功，且进程／GPU／SSH清零。
- 若smoke失败，正式launch保持0并封存；若smoke通过，才执行唯一正式launch。至少两个正式行在工件前出现同一确定性异常时停止本run。
- 不以accuracy、loss、AUROC、`u_gap`、unknown拒识率或任何性能值做停止、调参或重试依据。

## 待回填

- Git commit及最终报告SHA。
- 本地联合测试精确计数、独立审查结论。
- archive／release／SCP／smoke／唯一launch、PID／GPU／日志和工件闭合。

## v2发布与F1真实cache smoke证据（2026-08-12）

- 冻结Git commit：`5c14d7fd46e71556d6ede01c8daf790fa83c32ce`；仅由该commit生成无前缀干净archive，未携带Task7 dirty文件。
- 本地archive：`E:\type10-7\code\runner_tmp_phase1_clic_source_leo_20260812_v2\phase1_clic_source_leo_20260812_v2_5c14d7fd_git_archive.tar`，SHA256=`24E977CAEB331C3CCAFFA76E0B540A1E84ACD2D84D08444DE350153DE9D84BD1`，bytes=`267008000`，members=`5024`。
- SCP恰1次，远端archive SHA／bytes闭合；原子落地release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_leo_20260812_v2_5c14d7fd`。远端核心文件SHA以archive解出LF字节复算：builder=`25F895E42552D16EA71941CED02F08C1E7E20D7927335AAE94F6EF72D6A45B4D`，export clean=`C9CA2B62CE94FEC8255B0ACD493E5B2513782DF8C375D0E43129638508CE846F`，export LEO=`591B60960F40BFE9585DB0D958C9A0E11DC0A75534AC935DFC3CD497E4E533C9`，phase1_clic=`36FFDE23244D80AD15647F72252C70004D1E33549D7BF3D6E510E99E56461335`，launcher=`3813EEEA125075D47D1B7C9976C78BD2681A4398418B33EFF58F41B9876B265D`。
- 远端静态门：4核心`py_compile`、builder/export `--help`、launcher `bash -n`、dry-run精确18行（6 cache+12 export）均通过；目标／query／truth／role参数为0；run/log/outer/release初始路径均不存在且未覆盖。
- F1 smoke独立路径：`/home/szu2070436088/2510044040/CV-SincNet/.smoke_phase1_clic_source_leo_20260812_v2_F1`，使用真实F1C/F1G final checkpoint与terminal receipt、ManySig及GPU0，未写正式run。builder产出`source_l_received_iq.npz`（9,597,696 bytes）和receipt（5,265 bytes）。首次结构校验因误写3920³笛卡尔积而非工件故障；经批准仅终止run-owned验证PID 2515762（SIGTERM后退出），未重建／改写工件。
- 修正结构校验与生产consumer重开通过：NPZ SHA256=`bd9d2813522fcc722957bfdbf90a33462a23c2f4e77a8fdefd88708345842dbd`；exact members=`received_iq,tx_ids,rx_ids,day_ids,physical_sample_id,sat_scenarios`；row_count=`3920`；28个TX×RX cell各140；84个TX×RX×scene cell计数46/47；physical_sample_id全局唯一3920；received_iq finite；receipt schema=`cvs.phase1.clic_source_leo_received_iq.v1`、`minimum_cell_count=46`、fit/held-validation/proxy forward rows均0、`query_access=false`、`target_access=false`；生产`_load_existing_received_iq`重开PASS。
- smoke结束时该路径下无运行进程、GPU compute为空、SSH客户端与N607 TCP22均清零。正式launch计数仍为0；本报告更新后将提交，然后执行唯一正式launcher入口。

## 唯一正式入口结果（2026-08-12）

- `launch_attempt=1`；`formal_process_launch=0`；未重试。实际远端shell先检查`test -x "$REL/code/scripts/launch_phase1_clic_source_leo12_20260812.sh"`，因Git archive解出的launcher为普通文件mode=`664`而非可执行mode，入口在检查阶段退出；未调用launcher主体。
- 首次调用的远端核心文本为：`set -euo pipefail; ROOT=/home/szu2070436088/2510044040/CV-SincNet; REL="$ROOT/releases/phase1_clic_source_leo_20260812_v2_5c14d7fd"; ...; test -x "$REL/code/scripts/launch_phase1_clic_source_leo12_20260812.sh"; ...; cd "$REL"; nohup env RUN_ID=phase1_clic_source_leo_20260812_v2 TRAINING_RUN_ID=phase1_clic12_20260812_v5 PROJECT_ROOT="$ROOT" CODE_ROOT="$REL/code" PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash "$REL/code/scripts/launch_phase1_clic_source_leo12_20260812.sh" > "$ROOT/phase1_clic_source_leo_20260812_v2_outer.out" 2>&1 < /dev/null &`。本地包装首次返回exit=`1`且stdout/stderr为空；随后只读复核确认无落地，不能将其误记为已启动。
- 远端release仍存在且未覆盖；launcher路径存在、bytes=`6079`、SHA256=`3813eeea125075d47d1b7c9976c78bd2681a4398418b33eff58f41b9876b265d`、mode=`664`。run、log、outer均`ABSENT`；无outer PID、cache/export PID、pids表、正式工件、日志或GPU compute进程。SSH/TCP22已清零。
- 结论：`LAUNCH_ENTRY_NOT_LANDED / NO_PERFORMANCE_RESULT`。该run没有任何性能结果，不得按性能解释；release与smoke工件保留。后续若要修正入口权限，必须由主控另行创建新run ID/新发布决策，不能在本run重试。
