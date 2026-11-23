# LoRa residual DURA 扰动门配对验证

## 候选和比较

沿用 `20261123-phase1-lora-dura-residual-practical-m1-r01` 的 30,000 个物理包、15 台未见目标接收机、256 kHz residual practical 观测和种子 2026092702。三路共用对应的冻结 Tri-set。对照为直通和 R69 半量投影。候选为在同一半量投影前加源域扰动门，沿用 0.50 倍阈值余量，门内对六条 chirp 施加原半量修正。比较固定七景，单独报告 clean、六景均值、低强度城区、每台接收机开门数、净纠错和宏 F1。另列既有 full practical 复响应同种子结果，保持各自信道定义。

R69 从父运行的 L/U IQ 训练继承。检测门独立在同一契约的 4,000 个 L/U 源包拟合，在另外 6,000 个 clean 源包校准，无 F、TX 标签或目标采样。模型冻结后依次做真实 R69 源包 smoke、对已冻结盲视图预测及本机独立评分。源域留一支持的改进目标是保留 clean 并恢复高相干受扰包的六景收益。三种子 57.0%、低城区 43.0%、clean 71.6556% 的总目标继续使用原口径，本次单种子配对提供该目标的一行证据。

## 发布与产物

使用本仓库当前提交与匹配上游分支。发布回执中的 `code_commit` 记录实际完整提交 ID，源程序与正式候选同时固定。N607 使用现有 LoRa 普通用户 Python，工作目录为该 run 的独立 release，父运行目录及预测产物只读。每次一个本 run 工作者，卡序为 4、2、5、1、6、0、3，每卡计算进程上限为 4，计入已有任务。

```text
python -m experiments.pamf_lora_main.residual_dura_gate_run publish --repo <repo> --output <gate-release> --context <lora-runtime> --ssh-config <ssh-config> --parent-submit <residual-parent-submit>
python -m experiments.pamf_lora_main.residual_dura_score --disturbance-gate --submit <gate-release>/submit.json --ssh-config <ssh-config> --dataset <original-lora-archive-root> --output <gate-readout> --full-control <complex-response-confirm-readout>
```

远端新根为 `runs/20261123-phase1-lora-dura-residual-gate-m1-r01`，预期记录 `gate.json`、真实源包 `smoke.json`、`activity.json`、315 份预测数组及冻结清单。本机评分侧先检查完整 15 RX × 7 景 × 3 路数组、门控统计和 full 对照，再读取原档案真值，输出 `scores.json`。源对齐、预测或 smoke 出错时停止本 run 并保留产物。父任务失败时终止缺少依赖的流程。处于可核实的父训练等待期时，健康父进程继续运行。

## 发布状态

**VERIFIED**。发布提交为 `c99556d506b583be3db15722c6ad820a719aab50`，本机与上游 OID 一致。N607 传输校验和模块编译通过，经独立检查确认入口、工作目录和 GPU 2 的拟合工作者。独立审核以父源配对已就绪为前提通过，发布前核实 28,000 源包和两套配对已就绪。针对性测试确认缺预测、门开数为零但预测与直通不符时在读真值前退出。模拟齐全的 315 数组可进入独立评分。

正式拟合完成。4,000 个训练包和 6,000 个校准包得到阈值 0.4528636538，F、TX 真值和目标读取均为零。控制器处于 `WAITING_FOR_RESIDUAL_R69`，独立评分已启动并等待预测冻结。当前完成的是门控拟合与发布核查，正式 R69 smoke、目标预测和成绩待出。
