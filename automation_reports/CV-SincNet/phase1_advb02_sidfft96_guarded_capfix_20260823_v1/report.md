# Phase1 ADV3B02 SID-FFT96残差上界修正验证

## 当前状态

- run ID：`phase1_advb02_sidfft96_guarded_capfix_20260823_v1`
- 状态：`RUNNING`
- 目标：修复公共模型构建器漏传`sid_max_residual_ratio`的问题，仅重跑修正后的`S3G_SIDFFT96_GUARDED`，复用上一轮已完成的S0，不扩大矩阵。
- 科学边界：本轮仍是Phase1 source-only代理实验，不引入target/query，不声明Phase2适应、新类注册、Phase3未知拒识或真实在轨性能。

## 最小预登记

|字段|冻结值|
|---|---|
|候选/矩阵|仅`S3G_SIDFFT96_GUARDED`；S0复用`phase1_advb02_sidfft96_guarded_20260822_v1/S0_FROZEN_CORE90`|
|基座|`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|数据协议|`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`R_s∩R_t=∅`，query/target输入计数必须为0|
|seed|`392002`|
|训练|200个epoch；仅`sid_fft96.*`可训练；`sid_max_residual_ratio=0.10`|
|必评场景|clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|本地工作区|`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-sid-capfix-20260823`|
|Git分支|`codex/advb02-sid-capfix-20260823`|
|Git提交|训练构建修复`52fc45852828f2d336fe65849394648bf2ed9265`；评估重建修复`26477fc49d513bc01f19304f629655180187aee5`|
|N607环境/CWD|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`|
|运行目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_guarded_capfix_20260823_v1`|
|日志目录|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_guarded_capfix_20260823_v1`|
|GPU|GPU7；启动前preflight为0个计算进程，GPU0–6各1个既有训练进程|
|启动命令|以release为`ROOT`，显式绑定主目录的`RUNS_ROOT`、`LOG_ROOT`、`WISIG_PKL`、`BASELINE_CKPT`和`SID_MASK_PATH`，设置`RUN_ID=phase1_advb02_sidfft96_guarded_capfix_20260823_v1`、`ONLY=S3G`、`GPU_S3G=7`，执行`bash code/scripts/launch_phase1_advb02_sidfft96_guarded_20260822.sh --only=S3G`|
|预期artifact|`final_ssdg.pth`、选择checkpoint、200行`metrics_epoch.csv/jsonl`、训练stdout、`independent_final_eval/final_eval.json/txt`、独立评估stdout|

## 停止规则

只允许因协议/query泄漏、错误stage/seed/split/scenario、输出碰撞、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误、进程归属不清或预登记的系统技术故障停止。低性能不停止训练；发现性能不足只在完整评估后判定不晋级。任何停止都只能作用于该run ID绑定的精确进程树，并保留已有artifact。

## 晋级门槛

与复用S0同row比较：clean下降不超过0.3pp；三种LEO overall均值至少提升1.0pp；三种LEO Strict UDU均值至少提升1.0pp；LEO overall floor至少提升0.5pp；运行时残差上界必须为0.10且逐样本残差比不超过0.10；非SID参数零漂移。未满足则`SCIENTIFIC_FAILURE_NO_PROMOTION`，不得进入多seed或完整确认矩阵。

## 设计—实现—验证追踪

|ID|要求|实现位置|验证|状态|
|---|---|---|---|---|
|R1|公共构建器显式透传`sid_max_residual_ratio`|`code/post_stage_common.py::build_baseline_model`|新回归测试构建实际模型并检查运行时属性|`VERIFIED_LOCAL`|
|R2|逐样本残差范数不超过原始身份嵌入的10%|`SIDFFT96Residual.forward`既有裁剪逻辑|构建后实际SID前向并检查每个样本比值|`VERIFIED_LOCAL`|
|R3|真实checkpoint无query smoke且仅SID参数可训练|既有smoke与训练配置|N607 release真实checkpoint smoke|`VERIFIED`|
|R4|只重跑修正S3G并闭合clean及三种LEO_WEAK|既有guarded launcher的`--only=S3G`|N607完整artifact与同row评分|`RUNNING`|
|R5|独立最终评估重建保持相同的0.10上界|`code/evaluation/collaborative_inference_eval.py::build_model_from_checkpoint_args`|checkpoint参数驱动的实际评估模型构建测试|`VERIFIED_LOCAL`|

## 工作区隔离说明

原检出已切换到`codex/advb02-hsid-20260823`并存在3个未提交的HSID相关文件改动。本轮从已发布提交`8c5cae87a91e567ef06c680e5f9a372ebee8e166`创建独立capfix worktree，未切换、暂存、提交或清理原分支的任何改动。

## 本地验证记录

- 基线：修复前SID相关20项测试全部通过，证明既有测试没有覆盖公共构建器透传。
- RED：新增`test_build_baseline_model_applies_sid_residual_cap`后得到运行时`0.0 != 0.1`，准确复现报告中的实现偏差。
- GREEN：补充单一参数透传后，新测试通过；SID、频谱审计、launcher及公共构建器相关21项测试全部通过。
- 新测试将SID projector权重放大后执行实际前向，4/4个样本的残差比均不超过`0.100001`。
- `py_compile`、launcher语法检查与`git diff --check`均通过。

据此，R1、R2和R5状态更新为`VERIFIED_LOCAL`；R3等待真实ADV3B02 checkpoint无query smoke，R4等待N607真实性能artifact。

## 独立P0/P1审查与定点修复

- 首次独立审查未发现P0，但发现1个P1：训练公共构建器已得到0.10，launcher调用的独立最终评估构建器仍漏传该参数，导致clean及三种LEO_WEAK可能以0.0上界评估。
- 按TDD定点复现：checkpoint参数为0.10时，修复前评估模型运行时属性为0.0。
- 最小修复：仅在评估构建器增加同一参数透传；对应测试转绿，训练/评估两条构建路径及相关SID、launcher测试共29项通过。
- 唯一一次定点复审结论为`RESOLVED`：评估构建器从checkpoint参数恢复0.10，定点测试通过，未重新扩展审查范围。
- 首版release`phase1_advb02_sidfft96_guarded_capfix_52fc4585`在审查返回前只完成归档传输、SHA核对、远端编译和dry-run，从未启动训练，现标记为`SUPERSEDED_NOT_LAUNCHED`。真实实验只能使用包含评估修复的新release。

## 发布与启动前证据

- 可启动release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`。
- release归档本地与远端SHA-256均为`b33b0e0d6268432a11fe72f2c7b9816a1c2053c1961f6a8d6e56c84e68becc4a`，状态`VERIFIED`。
- 远端`py_compile`和只选择S3G的完整命令dry-run均为`VERIFIED`；dry-run确认训练和独立评估均从修复release加载代码。
- 真实ADV3B02 checkpoint smoke为`VERIFIED`：`batch_role=L_s`、`query_input_count=0`、`target_input_count=0`、raw/SID初始差异为0、非SID可训练参数为0、SID可训练参数为41,280。
- 新run root在preflight时不存在；log root仅包含本轮smoke，不存在S3G训练输出，launcher仍可执行其不可覆盖检查。

## 启动健康检查

- N607服务器时间2026-08-23 02:42（Asia/Hong_Kong）启动；launcher PID为`1252254`，主训练PID为`1252274`。
- launcher CWD和训练代码均绑定`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_guarded_capfix_26477fc4`，cmdline绑定唯一run root、seed 392002、`--only=S3G`和`--sid_max_residual_ratio 0.10`。
- GPU7在启动前计算进程数为0，启动后`nvidia-smi pmon`确认PID`1252274`为GPU7计算进程；没有触碰GPU0–6的既有任务。
- 首个5秒采样恰处于日志静默窗口，训练日志保持12,474字节；随后的只读复核增长至18,167字节并闭合E2/200，因此日志增长状态为`VERIFIED`，不是启动失败。
- E2时source validation TX为98.83%，source satellite mean/floor为87.93%/86.20%；heldout test保持按预登记不在训练期执行。未发现Traceback、OOM、Killed或协议错误。
