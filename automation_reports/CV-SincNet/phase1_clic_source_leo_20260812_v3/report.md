# Phase1 CLIC source-L LEO weak第二波v3预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_leo_20260812_v3`。
- 当前状态：`LOCAL_VERIFIED / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 目标：为F1—F6各构建一份source-L单观测LEO weak received-IQ缓存，并由同fold C／G读取完全相同的缓存字节，完成12个source-LEO特征导出。
- v3不改变方法、数据、场景、seed、矩阵、GPU映射、阈值或停止规则；唯一发布修复是改用新run ID，并要求运行席以`bash <launcher-path>`调用普通Git归档文件，不再用执行位检查阻断入口。

## 前序证据与修复边界

- 训练run=`phase1_clic12_20260812_v5`：12／12 checkpoint、terminal、config receipt已技术闭合，40 epochs、1200 batches，failure=0。
- source clean run=`phase1_clic_postfreeze_20260812_v2`：12／12`source_clean_proxy.npz`已完成，每臂21120行；其中source-L为3920行，4 TX×7 RX每cell恰140，same-fold C／G物理元数据逐字节一致。
- source-LEO v1只在真实smoke中触发Torch 2.1／NumPy 2.x旧数组桥接native crash，正式launch=0；该桥接已在commit`5c14d7fd`修复并经独立审查`P0=0/P1=0/P2=0/ALLOW`。
- source-LEO v2真实F1 cache smoke已PASS：NPZ SHA256=`bd9d2813522fcc722957bfdbf90a33462a23c2f4e77a8fdefd88708345842dbd`，exact6 members、3920行、28个TX×RX cell各140、84个TX×RX×scene cell各46／47、physical ID全局唯一、finite，生产consumer重开PASS。
- v2正式入口未落地：`launch_attempt=1`但`formal_process_launch=0`。原因是运行包装在调用`bash`前错误要求launcher mode为可执行；归档文件mode=`664`，因此run／log／outer／PID／工件均未产生。该事实已封存在commit`1f6aae85`，不是模型、数据或launcher主体故障。

## 冻结矩阵与数据合同

- fold：F1—F6；arm：C、G；训练checkpoint与terminal来自`phase1_clic12_20260812_v5/{F1C...F6G}_CLIC12`。
- 每fold source-L严格3920=`4 TX×7 RX×140`；物理ID稳定排序后，三formal scene分配为47／47／46。
- scene仅为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；每物理样本只生成一份received-IQ，不跨scene融合、择优、TTA或零时适配。
- 同fold C／G必须读取同一cache路径和相同字节；cache NPZ exact members仅为`received_iq,tx_ids,rx_ids,day_ids,physical_sample_id,sat_scenarios`。
- 只读source-L、对应checkpoint／terminal和ManySig；U、V、proxy、target、query、truth与role不进入缓存forward、fit、阈值或选择。
- ADV3B02只要求训练数据配置和测试数据配置等价，不要求与CLIC复用同一目标包或received-IQ。

## 本地版本、验证与发布合同

- 本次预期Git改动仅为launcher中v2→v3的不可覆盖run ID替换及本报告；source builder、exporter、模型和科学合同字节不变。
- 提交前必须通过launcher`bash -n`和dry-run精确18行=`6 cache+12 export`，C6／G6；检查输出不得出现target／query／truth／role训练输入。
- v3使用新run root、log root和outer路径；它们在N607落地前必须全部不存在。
- 由唯一N607运行席执行：直连preflight、资源与路径检查、干净Git archive、至多一次SCP、远端SHA／静态门、唯一正式launcher调用。
- 因同一builder commit的真实F1 cache smoke已在v2完成并由生产consumer重开，v3不重复该耗时烟测；入口必须直接使用`bash "$REL/code/scripts/launch_phase1_clic_source_leo12_20260812.sh"`，只要求文件存在，不要求`-x`。
- 唯一正式launch后立即核验outer、PID表、固定GPU映射、日志增长和首个cache／export工件；至少两个不同fold在工件前出现同一确定性异常时，只停止该run并封存`NO_PERFORMANCE_RESULT`。
- 不读取accuracy、loss、AUROC、`u_gap`或拒识性能来停止、调参、选择或重试。

## 预期输出

- 6个`F{fold}_SHARED/source_l_received_iq.npz`及6个receipt。
- 12个`F{fold}{C|G}_CLIC12/source_leo.npz`及12个binding。
- `logs/phase1_clic_source_leo_20260812_v3/pids_source_leo12.tsv`和18个阶段日志。
- 完成态只表示`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；source proxy、PAIR、bundle、F6和target评分由后续独立阶段完成。

## 待回填

- Git commit、文件SHA、本地验证结果与独立最窄审查。
- archive／release／SCP／唯一正式launch、PID／GPU／日志、cache／NPZ／binding闭合。
