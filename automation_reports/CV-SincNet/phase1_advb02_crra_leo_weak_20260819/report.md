# ADVB02 CRRA-S LEO弱信道Phase1实验报告

## 最小预登记

|字段|值|
|---|---|
|运行ID|`phase1_advb02_crra_leo_weak_20260819_r1`|
|候选|`ADVB02_CRRA_S_LEO_WEAK_E200`|
|状态|`LOCAL_VERIFIED`；尚未发布到N607|
|实现Git提交|`fc4629beb90ccbc74dc98287925c07f5019d7e45`，远端分支OID已独立核对一致|
|工作目录|`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/advb02-crra-leo-weak-20260819`|
|远端工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|环境|本地`ssr-gpu`；远端`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|启动前按N607占用选择；每GPU最多2个训练任务|
|固定随机种子|`392034`|
|Phase1角色|`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`|
|训练和测试信道|仅`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|禁止项|不使用`mixed_orbit`；不访问target receiver/query/calibration；不启用CRRA-C/target adapter|

## 冻结候选

- 基座与主训练路由：`ADV3B02_CORE90_SOFT_E200`，E200，label/pseudo=`130/70`，AdamW，`lr=2e-4`，`weight_decay=1e-4`，`lite_d`，`branch_ablation=no_dac`，`domain_enhancer=rcn_stats`与EMA。
- LEO日程：E1–40仅`leo_clear_weak,p=0.30`；E41–90为`leo_low_elev_weak,leo_rain_weak,p=0.60`；E91–200为三种LEO场景并集`p=0.80`。
- CRRA-S：仅身份时间路径；逐对I/Q收缩白化、rank=8 FiLM残差、源域多中心支持门、q条件时间/频率/PA可靠度融合；PA不重构，域分支不使用CRRA。
- CRRA损失：有效卫星KL仅`lambda_sat_cons=0.05`；pair=`0.05`、energy=`0.001`、gate L1=`0.001`、nuisance=`0.02`、q TX adversarial=`0.02`、shell=`0`。
- CRRA日程：E1–16恒等；E17–46线性ramp；E47起CRRA参数组学习率为主学习率的`0.25`。
- 训练视图方式：`concat_masked`。clean分支保留完整Core90目标；同一批卫星视图作为掩码辅助监督，不把所有主损失机械扩展为双批。

## 命令、输入输出与停止规则

|字段|值|
|---|---|
|启动器|`code/scripts/launch_phase1_advb02_crra_leo_weak_20260819.sh`|
|训练输入|`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，仅source Phase1数据|
|输出根目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_crra_leo_weak_20260819_r1/ADVB02_CRRA_S_LEO_WEAK_E200`，不可覆盖|
|训练日志|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_crra_leo_weak_20260819_r1/ADVB02_CRRA_S_LEO_WEAK_E200.out`|
|独立测试日志|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_crra_leo_weak_20260819_r1/ADVB02_CRRA_S_LEO_WEAK_E200.final_eval.out`|
|预期产物|`final_ssdg.pth`、训练终态、clean指标、三种LEO逐场景指标、`independent_final_eval/final_eval.json`、`independent_final_eval/final_eval.txt`及CRRA遥测|
|技术停止规则|仅在协议越界、错误运行ID/输出覆盖、无法启动、确定性重复预预测异常、final checkpoint缺失或独立评估无法闭合时停止；中途性能高低不作为停止条件|

训练器即使将历史P0/P1阈值标为诊断性信息，只要`final_ssdg.pth`合法存在，启动器仍必须完成独立测试。最终状态只能在clean和全部三种LEO逐场景结果都保存后标为`ARTIFACTS_COMPLETE`。

## 启动记录

|字段|值|
|---|---|
|启动时间|2026-08-19 23:58（Asia/Hong_Kong）|
|状态|`RUNNING`|
|启动器PID|`3375058`|
|训练子进程PID|`3375076`|
|GPU|GPU0；启动后`nvidia-smi pmon`确认该PID为GPU0唯一计算进程|
|代码绑定|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_crra_leo_weak_20260819/f32f52b8/workspace/code/SSDG/train_ssdg.py`|
|运行根|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_crra_leo_weak_20260819_r1/ADVB02_CRRA_S_LEO_WEAK_E200`|
|启动器日志|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_crra_leo_weak_20260819_r1/launcher.out`|

首次启动后检查确认训练日志已增长，并写入`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`、`channel=leo_weak only`、三段LEO日程、`concat_masked`、唯一`lambda_sat_cons=0.05`和`lambda_crra_sat_kl=0`。外层nohup shell的CWD为用户主目录，但实际Python脚本、`PYTHONPATH`和run根均显式绑定到上述隔离release与唯一输出目录。

## 本地验证记录

- `LOCAL_FOCUSED_SUITE_PASS`：CRRA核心、模型、协议、训练、评估遥测和启动器聚焦测试共46项通过；`py_compile`、启动器`bash -n`和`git diff --check`通过。
- `LOCAL_COMMAND_PARSE_PASS`：启动器完整训练命令由`SSDG.train_ssdg.build_arg_parser()`成功解析；`--eval_sat_on main`与`--sat_eval_max_batches -1`均为已注册参数。
- `P0_REVIEW_FIXED`：审查发现独立checkpoint评估重建模型后默认`crra_epoch=1`，会关闭最终CRRA门控。已增加`restore_crra_eval_epoch()`，独立评估和协作评估都会从checkpoint恢复训练轮次；定点测试通过。
- `N607_DIRECT_PREFLIGHT_VERIFIED`：2026-08-19 23:52（Asia/Hong_Kong）确认N607项目根目录和8张GPU可见；GPU0空闲，GPU1–3已有其他任务，本run不干预它们。
- `RELEASE_ARCHIVE_READY`：实现提交`f32f52b8f8d8fc51873a830888e4e2ea3a6d8d27`的`code/`和独立评估器归档为`advb02_crra_leo_weak_f32f52b8.tar`，本地SHA256=`2c74568313552b26d544bbb0994ef31c4a0fa918b22d4bb02a345e88440757da`。计划上传到`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_crra_leo_weak_20260819/f32f52b8/`，解压到其`workspace/`子目录；训练输出和日志仍写入主项目的唯一run根目录。
- `RELEASE_ARCHIVE_LANDED_VERIFIED`：远端同一归档SHA256回读为`2c74568313552b26d544bbb0994ef31c4a0fa918b22d4bb02a345e88440757da`；仅做本地/远端这一轮归档校验。
- `REMOTE_COMPILE_AND_DRY_RUN_PASS`：隔离`workspace/`内的改动模块均已编译；启动器干跑显示固定seed、四角色、三段LEO日程和`channel=leo_weak only`，并确认输出中不含`mixed_orbit`。
- `REAL_CHECKPOINT_NOQUERY_SMOKE_PASS`：CPU上严格重建并加载历史`ADV3B02_CORE90_SOFT_E200`真实checkpoint，`missing=0,unexpected=0`；该冒烟不读取query、不参与本次候选性能比较。

## 结果与解释

待运行。届时按同一运行行保留clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`结果；不得将不同场景或不同运行的峰值拼接为单一结论。结果仅代表当前source-only Phase1协议下的弱LEO代理信道证据，不宣称真实全链路卫星性能。
