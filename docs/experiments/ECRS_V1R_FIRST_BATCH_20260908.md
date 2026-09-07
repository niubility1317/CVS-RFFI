# ECRS V1R首批N607实验

run_id：`phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1`。
状态：`LOCAL_VERIFIED`，启动后追加读回结果。用户于2026-09-08明确授权启动实验。

## 冻结矩阵与版本

沿用[已发布矩阵](ECRS_V1R_V2_MATRIX_RELEASE_20260907.md)，seed=392005、epochs=200，正式行均从头训练，不加载smoke或历史checkpoint。基线提交`7db8af32f1deef8ba8330ecab5bfe4db5c1454cb`；本次部署helper和报告由后续Git提交固定，实际release从该提交归档，不修改模型/损失。

|GPU|行|含义|
|---|---|---|
|0|B0|匹配原训练入口，关闭ECRS|
|1|B2-V1|原V1 R7非融合控制，gate关闭|
|2|B2|V1R接线与独立响应头修复包|

B1零作用工程检查沿用已有定点与真实CLI合成检查证据，不额外排200轮；后续V2/融合/策略行未在本批启动。

ManySig equalized=1，source RX=1,3,4,6,8，day=1,2,3；L/U/V=0.07/0.63/0.30。target不构造评估loader、不参与选模。保持已发布训练、增强、损失与48次source验证安排。该训练入口不声称完整SSDG Core90等价，也不将B2修复包解释成单因素效果。

## 环境、输入输出与命令

- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，当前实测torch=2.1.0+cu121，CUDA可用。
- 数据：项目根下`Dataset_WigSig/ManySig.pkl`，现有只读数据，不修改或重建。
- release/CWD：项目根下`releases/phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1_<commit8>`。
- checkpoint/output：项目根下`runs/phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1/<row>`。
- stdout/诊断/计划：项目根下`logs/phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1`及其`<row>`子目录。
- 唯一launch owner：当前任务主Agent；新helper使用独占log root、dispatch owner及run root防止重复启动。

```text
<python> -u <release>/code/scripts/run_ecrs_v1r_first_batch_20260908.py --project-root <project-root> --run-id phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1 --smoke-checkpoint <project-root>/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2/ADV3B02_ECRS_R7/best.pth --detach
```

helper先严格重建现有真实R7 checkpoint，用合成IQ执行无query有限值前向；不将旧权重传入正式训练。PASS直接继续，不创建smoke许可artifact。三行实际argv和PID写入logs下`launch_plan.json`、`children.jsonl`，退出码写入`exit_status.jsonl`；stdout分别为`B0.stdout.log`、`B2-V1.stdout.log`、`B2.stdout.log`。

2026-09-08 00:08服务器preflight：N607身份/项目路径核实，8张3090计算进程均为0，磁盘可用7.2TB。本次每卡仅一个训练实验，不影响其他任务。启动前检查所用GPU仍有每卡最多两个实验的容量。

## 检查、停止与完成边界

本地36项相关回归及15行真实parser检查已在矩阵发布完成；本次独立P0/P1审查未发现首批E200训练路径阻断，已核对source-only、四场景source prediction、输出保护及torch2.1调用。新helper完成语法/CLI检查，远端执行一次归档SHA比较和一次编译，再由实际checkpoint smoke确认环境运行。

低准确率不停止健康训练。仅既定协议/输入/checkout错误、输出碰撞、无法执行、不能形成合法prediction等技术故障触发处理；只处理本run所属进程并保留所有产物，禁止广泛停止或重启。dispatcher只记录退出，不自动重跑、不干预其他行。

预期每row输出best/latest checkpoint、完整训练指标和ECRS诊断（适用行）、source clean及三LEO逐样本prediction和汇总、选择配置及恢复状态。source行完成记`SOURCE_SCREEN_COMPLETE`，不冒充最终target `ARTIFACTS_COMPLETE`。同row与B0差值、RX/day分层和实际机制激活分别报告；单seed不晋级默认配置，不使用target结果调参。训练总时长待真实进度估计，不用历史耗时作保证。
