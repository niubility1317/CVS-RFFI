# Phase2 ADV3B02 OVC-CI open-verifier collaborative inference

## 基本信息

- 实验ID：`phase2_adv3b02_ovc_ci_20260704`
- 时间：2026-07-04
- 操作员/agent：Codex
- 目标：在`ADV3B02_CORE90_SOFT_E200`特征基础上，为Stage2-C qknn8协同推理增加可部署的open verifier，优先提升未知类拒识，同时保持旧类准确率不下降。
- 对比目标：COTE-CI、MOE-CI、OPR-CI、SLEV/ENPC既有结果。
- 结论边界：本地结果未达标，不能登记为Stage2-C成功、部署成功或论文主结论；当前结果是算法负证据。

## 算法设计

OVC-CI冻结ADV3B02特征与qknn8原型路径，只训练一个小型开放集验证器。验证器输入为原型相似度、margin、top-k相似度均值、softmax entropy、proxy_unknown centroid相似度、known centroid相似度及二者差值。阈值由source old与target old/seen-new support的known风险分位数确定；`proxy_unknown`只作为源侧开放集负例；`target_unknown`仅用于最终评估。

部署状态量：

|项目|字节|
|---|---:|
|prototype_fp16_bytes|2560|
|verifier_fp16_bytes|44|
|centroid_fp16_bytes|640|
|total_fp16_state_bytes|3244|

## 协议与集合审计

源特征文件：`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz`

|role|rows|TX IDs|receiver IDs|
|---|---:|---|---|
|source|2400|`14-10,14-7,20-15,20-19,6-15,8-20`|`1-1,1-19,14-7,18-2,19-2,2-1`|
|proxy_unknown|1600|`1-1,1-10,1-11,1-12`|`1-1,1-19,14-7,18-2,19-2,2-1`|
|target_old|2400|`14-10,14-7,20-15,20-19,6-15,8-20`|`20-1,3-19,7-14,7-7,8-8`|
|target_new|800|`19-3,3-8`|`20-1,3-19,7-14,7-7,8-8`|
|target_unknown|800|`10-1,10-10`|`20-1,3-19,7-14,7-7,8-8`|

审计结论：

- `target_unknown_training_count=0`，`target_unknown_eval_only=800`。
- 训练角色仅为`source`、`proxy_unknown`、`target_old_support`、`target_new_support`。
- `proxy_unknown`与`target_unknown`的TX集合不重叠；`proxy_unknown`与target receiver集合不重叠。
- 协同接收机为5个target receivers：`20-1,3-19,7-14,7-7,8-8`。
- `collab_counts=all`实际覆盖`1..5`，对应从单接收机到全部target receivers。

## 本地变更与验证

新增文件：

|文件|用途|SHA256|
|---|---|---|
|`code/scripts/phase2_open_verifier_ci_eval.py`|OVC-CI算法与评估入口|`867846C86206AE5D639935A9A74DF18A55FF2555581414518B0EFA80F035B8D8`|
|`code/tests/test_phase2_open_verifier_ci_eval.py`|target_unknown评估专用与collab_count覆盖测试|`DFAF3961DF2EC8C3548D8062B2FC7D35525F502D33045AA7972F8BA70A7A1D53`|

本地验证命令：

```powershell
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_open_verifier_ci_eval.py code\tests\test_phase2_open_verifier_ci_eval.py
conda run -n ssr-gpu python code\tests\test_phase2_open_verifier_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。

本地全量命令：

```powershell
conda run -n ssr-gpu python code\scripts\phase2_open_verifier_ci_eval.py --feature_npz local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz --output_json local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci.json --output_summary_csv local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci_summary.csv --output_evidence_csv local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --query_per_class 20 --qknn_k 8 --device cuda:0 --verifier_epochs 500 --max_event_bytes 1152 --max_event_latency_ms 20
```

本地输出：

- JSON：`E:\type10-7\local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci.json`
- Summary CSV：`E:\type10-7\local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci_summary.csv`
- Evidence CSV：`E:\type10-7\local_artifacts\phase2_adv3b02_ovc_ci_20260704\local_k8\ovc_ci_evidence.csv`
- JSON SHA256：`0191708DE3D325CD68F7523F7D72D707CEAE247AACDD24A9967753967109C77E`

## 本地结果表

目标门槛：`old_acc>=0.99`，`min_old>=0.95`，`seen_new_acc>=0.97`，`min_seen>=0.93`，`unknown_reject>=0.99`。

|profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|bytes/event|latency_ms|target_pass|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|ovc_unknown_guard|4|0.8250|0.5500|0.0500|0.0000|0.3750|0.6250|0.0313|0.0500|192.0|0.25|False|
|ovc_unknown_guard|5|0.8333|0.6500|0.0250|0.0000|0.3500|0.6500|0.1938|0.3500|240.0|0.25|False|
|ovc_old_guard|5|0.9000|0.7500|0.2500|0.2500|0.0000|1.0000|0.0188|0.0750|240.0|0.25|False|
|ovc_balanced|5|0.9000|0.7500|0.2250|0.2000|0.0000|1.0000|0.0188|0.0750|240.0|0.25|False|
|ovc_unknown_guard|2|0.3000|0.1000|0.0000|0.0000|0.6250|0.3750|0.2375|0.2250|96.0|0.25|False|

同row解释：当profile偏向旧类保持时，旧类可到0.90但unknown拒识为0；当profile偏向未知拒识时，unknown最高到0.625但旧类/min_old崩塌。OVC-CI没有解决“未知类拒识提升且旧类准确不下降”的目标。

## N607同步与复测计划

远程工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

同步映射：

|本地文件|远程文件|
|---|---|
|`E:\type10-7\code\scripts\phase2_open_verifier_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_open_verifier_ci_eval.py`|
|`E:\type10-7\code\tests\test_phase2_open_verifier_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_open_verifier_ci_eval.py`|

远程环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

远程复测命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_open_verifier_ci_eval.py --feature_npz /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_proxy_unknown_ci_20260704/features_proxy_unknown.npz --output_json /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_ovc_ci_20260704/k8_available/ovc_ci.json --output_summary_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_ovc_ci_20260704/k8_available/ovc_ci_summary.csv --output_evidence_csv /home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_ovc_ci_20260704/k8_available/ovc_ci_evidence.csv --profiles all --collab_counts all --collab_group_policy available_up_to_k --k_shot 8 --query_per_class 20 --qknn_k 8 --device cuda:0 --verifier_epochs 500 --max_event_bytes 1152 --max_event_latency_ms 20
```

N607预检：

- 命令：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- 结果：直连`N607`通过；远程项目根目录存在；服务器时间`2026年07月04日13:09:31 CST`。
- GPU快照：GPU0-GPU7均为`NVIDIA GeForce RTX 3090`，显存占用均为`10/24576 MiB`，因此选择GPU0。

远程验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_open_verifier_ci_eval.py code/tests/test_phase2_open_verifier_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_open_verifier_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。远程出现一次NumPy内部命名DeprecationWarning，不影响执行。

远程全量测试结果：

- 远程输出目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_ovc_ci_20260704/k8_available`
- 远程日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_ovc_ci_20260704/k8_available.log`
- 本地归档：`E:\type10-7\local_artifacts\phase2_adv3b02_ovc_ci_20260704\remote\k8_available`
- 本地日志归档：`E:\type10-7\local_artifacts\phase2_adv3b02_ovc_ci_20260704\remote\k8_available.log`
- 远端结果JSON归档SHA256：`7E87F4163E229C9145EA8BA0A1089E555AE8EC4034F528E4DD2EC5430CA60AB5`
- 远端日志归档SHA256：`709CEE25E4AE57CC197E3267941FE8DC843AFAAE67AA2F4F4C7E169F1E38105D`
- 远端summary行数：15。
- 远端collab_count覆盖：`1,2,3,4,5`。

远端最佳同row：

|profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|bytes/event|latency_ms|target_pass|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|ovc_unknown_guard|4|0.8250|0.5500|0.0500|0.0000|0.3750|0.6250|0.0313|0.0500|192.0|0.25|False|

SSH/SCP清理：

- 每次SCP/SSH后均执行本地`ssh.exe`进程与`172.31.111.215:22`、`172.31.105.18:22` ESTABLISHED连接检查。
- 检查未发现残留SSH进程或ESTABLISHED连接。

Git镜像提交：

- 仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 分支：`codex/cvs-rffi-release-20260626`
- 提交状态：已提交；以仓库`git log --oneline`为准，避免报告自引用导致amend后hash漂移。

## 下一步建议

本轮结果表明，基于当前ADV3B02特征的后置拒识器无法同时保旧类和拒未知。下一条更合理路线应进入表征训练层：在地面训练阶段加入open-set margin/EVT-aware proxy unknown、class-conditional compactness、receiver-invariant old-class floor约束，生成新的特征权重后再做Stage2-C协同推理。星上侧仍保持轻量原型/EVT/KNN门控，而不是继续在当前特征上调阈值。
