# Phase1 CLIC source-L LEO weak第二波v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_leo_20260812_v2`。
- 当前状态：`LOCAL_FIX_UNDER_REVIEW / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
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
