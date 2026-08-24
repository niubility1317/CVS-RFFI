# PHASE1_CCOI_PA_V2_S20260824_20260825A实验报告

## 预登记

- 状态：`LOCAL_VERIFIED/PRE_REGISTERED`；尚未启动N607实验。
- 候选：`CCOI-PA-V2`，单seed最小矩阵`C0/C1/C2/C3/C4`。
- 科学对照：冻结`ADV3B02_CORE90_SOFT_E200`；沿用V1的同split、同seed、同训练/评估预算和四场景，C1–C4保持同容量。
- 修复范围：原始`meta.rx_i`接收机导出；算子独立分类与源域`V_cal`尺度对齐凸融合；有界码本有效数/集中度正则。Core90、source roles、场景和目标/query边界不变。
- Git实现提交：`8a959d00da768d1134ce859bd366052f4ea9c109`，分支`codex/phase1-ccoi-pa-v1-20260824`，远端OID已独立核对一致。
- 主要文件：`code/train_phase1_ccoi_pa.py`、`code/score_phase1_ccoi_pa.py`、`code/cvsrffi/ccoi_pa.py`、V2 launcher/config、聚焦测试及V2设计/追踪报告。
- 本地验证：`ssr-gpu`中35项CCOI聚焦测试通过；三个生产Python文件语法编译通过；C0–C4 dry-run通过；一次定点P0/P1检查闭合。
- 本机Git Bash：既有路由证据为`FAILED`，未在错误Bash通道执行launcher；发布后在N607运行`bash -n`。
- 源域协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`rho_label≤0.1`；目标域、query、query role和query truth不进入训练、校准或选择。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别输出。
- seed：训练和卫星扰动均固定为`20260824`，用于与V1同row比较。
- N607环境/CWD：普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；release目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- GPU：`0`；2026-08-25 01:05直连预检显示8张RTX 3090利用率均为0、显存占用1MiB，无compute app和`train_phase1_ccoi_pa.py`进程。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A`；smoke使用同名`_REAL_CKPT_NO_QUERY_SMOKE`独立不可覆盖根。两者预检均不存在。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A.out`；launcher监督日志使用同run ID独立文件。
- release归档：`E:\type10-7\release_archives\phase1_ccoi_pa_v2_8a959d00.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00.tar.gz`；本地SHA256为`976bfe2919f4632e5b5b277b915ec418c7754866c9d2a058859439429eab5628`，远端SHA待同步后核对一次。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_8a959d00 && ROOT=$PWD CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_20260825 RUN_ID=PHASE1_CCOI_PA_V2_S20260824_20260825A GPU=0 bash code/scripts/launch_phase1_ccoi_pa_v2_20260825.sh`。
- 预期artifact：`protocol_and_smoke.json`、挑战预训练历史、每row校准、sidecar、challenge audit、`prediction.jsonl`、后置`truth.jsonl`、`metrics.json`、matrix manifest和完整日志。
- 直接技术停止规则：仅在协议/query泄漏、错误checkout/CWD/run-root/GPU、输出碰撞、无法启动、prediction无法闭合，或至少两个row出现相同确定性预prediction异常时停止；不因低准确率停止。只处理该run绑定的进程树并保留全部局部artifact。
- 科学门槛：C2或更高row相对C1的LEO均值和receiver-floor分别至少提升0.30个百分点；clean下降不超过0.50个百分点；C4 holdout NMSE相对C1下降至少5%且R²大于0。未过线记负结果，不中止健康运行。
- 新run授权：本报告仅授权唯一run ID `PHASE1_CCOI_PA_V2_S20260824_20260825A`；不得重复启动或覆盖旧run。

## 运行更新

待N607发布、smoke和正式矩阵后追加。

