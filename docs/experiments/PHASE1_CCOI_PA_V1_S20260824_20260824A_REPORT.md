# PHASE1_CCOI_PA_V1_S20260824_20260824A实验报告

## 预登记

- 状态：`LOCAL_VERIFIED/PRELAUNCH`；没有性能结果。
- 候选：`CCOI-PA-V1`，单seed最小矩阵`C0/C1/C2/C3/C4`。
- 科学对照：冻结`ADV3B02_CORE90_SOFT_E200`；C1–C4同容量、同初始化、同split、同seed、同训练和评估预算。
- Git实现提交：`f7f2ab4a8431091d1674439fb99f2e414010ae6e`，分支`codex/phase1-ccoi-pa-v1-20260824`，远端OID已独立核对一致。
- 本地变更：`code/model.py`仅增加无参数`pa_token_map`；新增`ccoi_pa.py`、`ccoi_losses.py`、runner、独立scorer、launcher、配置、设计/追踪报告及聚焦测试。
- 本地验证：`ssr-gpu`环境；5个Python文件`py_compile`通过；55项聚焦测试通过；一次P0/P1只读复审完成并定点修复C1真实q旁路、缺少卫星激励、holdout感受野循环和truth写出顺序。
- 本机Git Bash：`FAILED`，指定Git for Windows后仍被路由为`/bin/bash`且`MSYSTEM`为空；未执行launcher。远端发布前必须运行`bash -n`。
- 源域协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`rho_label≤0.1`；目标域、query、query role和query truth不进入训练/校准/选择。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别输出。
- 命令：`RUN_ID=PHASE1_CCOI_PA_V1_S20260824_20260824A GPU=<preflight-selected> bash code/scripts/launch_phase1_ccoi_pa_v1_20260824.sh`。
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v1_20260824/PHASE1_CCOI_PA_V1_S20260824_20260824A`；smoke使用独立不可覆盖后缀根。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v1_20260824/PHASE1_CCOI_PA_V1_S20260824_20260824A.out`。
- GPU：5。预检时GPU0–4各有1个既有训练进程，GPU5–7无训练进程；本run只在GPU5启动1个训练进程，不触碰既有任务。
- 技术停止规则：仅在协议/数据越权、错误checkout/seed/split、输出已存在、真实checkpoint严格加载失败、非有限loss、prediction闭合失败、独立scorer连接失败或同一确定性预prediction异常重复时停止；不得因中间性能低而停止。
- 预期artifact：`protocol_and_smoke.json`、挑战预训练历史、每row sidecar、`prediction.jsonl`、后置`truth.jsonl`、`metrics.json`、challenge audit、matrix manifest和完整日志。
- N607预检：直连普通账户、项目根、checkpoint、ManySig数据和Python环境均`VERIFIED`；目标run与smoke根均不存在。
- release归档：`E:\type10-7\release_archives\phase1_ccoi_pa_v1_3ed07d9b.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v1_3ed07d9b.tar.gz`；本地/远端SHA256均为`15b33bc9ce88ca8ab0bd2bff8bfe23e9bf7a33f570bdea60d9e6a2e977fb13fd`。
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v1_3ed07d9b`；远端5个Python文件编译通过且`.pyc`已读回，launcher远端`bash -n`通过。
- 启动后PID/CWD/cmdline/GPU/log增长：`PENDING_LAUNCH`。

## 结果

尚未启动。科学增益与C0–C4排序均为`UNKNOWN`。
