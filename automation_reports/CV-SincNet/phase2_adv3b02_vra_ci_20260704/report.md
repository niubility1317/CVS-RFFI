# Phase2 ADV3B02 VRA-CI virtual-radius adapter collaborative inference

## 基本信息

- 实验ID：`phase2_adv3b02_vra_ci_20260704`
- 时间：2026-07-04
- 操作员/agent：Codex
- 目标：在`ADV3B02_CORE90_SOFT_E200`特征基础上，评估更强的轻量表征适配是否能解决未知类拒识，同时保持旧类与seen-new识别。
- 对比目标：OVC-CI、MOE-CI、OPR-CI、COTE-CI既有负结果。
- 结论边界：本地三组参数均未达标；VRA-CI不能登记为Stage2-C成功、部署成功或论文主结论。

## 算法设计

VRA-CI复用OPR-CI的低秩残差adapter，但训练目标加入虚拟边界未知样本：

1. 旧类/source和target old/seen-new support保持原型分类与known margin。
2. target support向对应原型紧致化。
3. source-side `proxy_unknown`远离所有known原型。
4. 从known prototype和proxy_unknown之间采样virtual boundary negatives，并将其推离known原型。
5. 使用强`old_preserve`和residual约束限制adapter破坏旧类。

部署边界：该方法只需要低秩adapter、原型与已有qknn8后端；不使用`target_unknown`训练或调阈值。

## 协议与集合审计

源特征文件：`E:\type10-7\local_artifacts\phase2_adv3b02_proxy_unknown_ci_20260704\remote\features_proxy_unknown.npz`

|role|rows|TX IDs|receiver IDs|sat_scenarios|
|---|---:|---|---|---|
|source|2400|`14-10,14-7,20-15,20-19,6-15,8-20`|`1-1,1-19,14-7,18-2,19-2,2-1`|``|
|proxy_unknown|1600|`1-1,1-10,1-11,1-12`|`1-1,1-19,14-7,18-2,19-2,2-1`|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|target_old|2400|`14-10,14-7,20-15,20-19,6-15,8-20`|`20-1,3-19,7-14,7-7,8-8`|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|target_new|800|`19-3,3-8`|`20-1,3-19,7-14,7-7,8-8`|`leo_clear_weak,leo_low_elev_weak`|
|target_unknown|800|`10-1,10-10`|`20-1,3-19,7-14,7-7,8-8`|`leo_clear_weak,leo_low_elev_weak`|

审计结论：

- `target_unknown_training_count=0`，`target_unknown_eval_only=800`。
- 训练角色仅为`source`、`proxy_unknown`、`target_old_support`、`target_new_support`、`virtual_boundary_negative`。
- `proxy_unknown`与`target_unknown`的TX集合不重叠；`proxy_unknown`与target receiver集合不重叠。
- 协同接收机为5个target receivers：`20-1,3-19,7-14,7-7,8-8`。
- `collab_counts=all`覆盖`1..5`，对应从单接收机到全部target receivers。

## 本地变更与验证

新增文件：

|文件|用途|SHA256|
|---|---|---|
|`code/scripts/phase2_virtual_radius_adapter_ci_eval.py`|VRA-CI算法与评估入口|`8A8ED0066943235D0EE01F542FFE87C1CD65C056BC654977FB9D5E1FA624C651`|
|`code/tests/test_phase2_virtual_radius_adapter_ci_eval.py`|target_unknown评估专用测试|`8BAD5546E3AAC625C9B492AFC7ADF68D5C2E6CEA42FDBEBC97F4A1E58B39715F`|

本地验证命令：

```powershell
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_virtual_radius_adapter_ci_eval.py code\tests\test_phase2_virtual_radius_adapter_ci_eval.py
conda run -n ssr-gpu python code\tests\test_phase2_virtual_radius_adapter_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。

## 本地全量结果

公共参数：`k_shot=8`，`qknn_k=8`，`query_per_class=20`，`collab_counts=all`，`collab_group_policy=available_up_to_k`，`max_event_bytes=1152`，`max_event_latency_ms=20`。

目标门槛：`old_acc>=0.99`，`min_old>=0.95`，`seen_new_acc>=0.97`，`min_seen>=0.93`，`unknown_reject>=0.99`。

|配置|backend/profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|target_pass|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|default strong|ENPC/enpc_unknown_strict|5|0.3979|0.0263|0.2833|0.2000|0.7167|0.2500|0.0637|0.0333|False|
|default strong|SLEV/slev_energy_strict|5|0.3037|0.0000|0.1333|0.1000|0.9000|0.0833|0.0438|0.0167|False|
|mid|ENPC/enpc_unknown_strict|5|0.4293|0.1316|0.0500|0.0500|0.7833|0.1333|0.0876|0.0833|False|
|mid|SLEV/slev_energy_strict|5|0.3141|0.0000|0.0000|0.0000|0.9500|0.0500|0.0558|0.0000|False|
|guarded|ENPC/enpc_known_anchor|4|0.8377|0.6000|0.2000|0.1000|0.0000|1.0000|0.0000|0.0000|False|
|guarded|SLEV/slev_known_anchor|4|0.8377|0.6000|0.2000|0.1000|0.0000|1.0000|0.0000|0.0000|False|

同row解释：

- 中/强VRA-CI可以把unknown_reject提升到`0.90-0.95`，但旧类准确率和每类下限严重崩塌。
- 保守VRA-CI可以维持`old_acc≈0.84`，但unknown_reject为`0`。
- 当前ADV3B02特征上的轻量表征适配仍无法同时满足旧类、新类与未知类门槛。

## N607同步与复测计划

远程工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

同步映射：

|本地文件|远程文件|
|---|---|
|`E:\type10-7\code\scripts\phase2_virtual_radius_adapter_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_virtual_radius_adapter_ci_eval.py`|
|`E:\type10-7\code\tests\test_phase2_virtual_radius_adapter_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_virtual_radius_adapter_ci_eval.py`|

远程环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

N607预检：

- 命令：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- 结果：直连`N607`通过；远程项目根目录存在；服务器时间`2026年07月04日13:24:57 CST`。
- GPU快照：GPU0-GPU7均为`NVIDIA GeForce RTX 3090`，显存占用均为`10/24576 MiB`，因此选择GPU0。

远程验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_virtual_radius_adapter_ci_eval.py code/tests/test_phase2_virtual_radius_adapter_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_virtual_radius_adapter_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。

远程输出：

|配置|远程run目录|远程日志|本地归档|
|---|---|---|---|
|default strong|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_vra_ci_20260704/k8_default`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_vra_ci_20260704/k8_default.log`|`E:\type10-7\local_artifacts\phase2_adv3b02_vra_ci_20260704\remote\k8_default`|
|mid|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_vra_ci_20260704/k8_mid`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_vra_ci_20260704/k8_mid.log`|`E:\type10-7\local_artifacts\phase2_adv3b02_vra_ci_20260704\remote\k8_mid`|
|guarded|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_vra_ci_20260704/k8_guarded`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_vra_ci_20260704/k8_guarded.log`|`E:\type10-7\local_artifacts\phase2_adv3b02_vra_ci_20260704\remote\k8_guarded`|

远程归档SHA256：

|文件|SHA256|
|---|---|
|`remote\k8_default\vra_ci_summary.json`|`32F26F130B33EF699055B57133C57207836E8D169E1F9149F8D506D660382B5B`|
|`remote\k8_mid\vra_ci_summary.json`|`8A3039694A5E914D9D26CFD620B2CAB289CF56056C132940F7003817C3B25FB1`|
|`remote\k8_guarded\vra_ci_summary.json`|`4958B5D8956AF92514AD261C939EBC2D613F23CC60287DEF04493E3CD3CE50F3`|
|`remote\k8_default.log`|`6E098AD887ED1ADD55F060652B71876364B2D8D7ABC0C87A31322126F8FFCCAA`|
|`remote\k8_mid.log`|`045F63BDFA2D396B810F5CC4B6D6A8DDA3504CC6E360AB2732E37A9ED3F61A48`|
|`remote\k8_guarded.log`|`A52156E93F72199610871DB361B86D1310084603FB3EBAB95FAAABE75E6CA065`|

远程全量结果：

|配置|backend/profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|target_pass|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|default strong|ENPC/enpc_unknown_strict|5|0.4241|0.1579|0.3333|0.2500|0.6667|0.3000|0.0876|0.0333|False|
|default strong|SLEV/slev_energy_strict|5|0.3455|0.1000|0.2000|0.1500|0.8667|0.1333|0.0518|0.0000|False|
|mid|ENPC/enpc_known_anchor|3|0.8010|0.6579|0.2833|0.2500|0.0000|1.0000|0.0000|0.0000|False|
|mid|SLEV/slev_known_anchor|3|0.8010|0.6579|0.2833|0.2500|0.0000|1.0000|0.0000|0.0000|False|
|guarded|ENPC/enpc_known_anchor|5|0.8063|0.6316|0.2333|0.2000|0.0000|1.0000|0.0000|0.0000|False|
|guarded|SLEV/slev_known_anchor|5|0.8063|0.6316|0.2333|0.2000|0.0000|1.0000|0.0000|0.0000|False|

远程summary覆盖：

- default strong：ENPC/SLEV各20行，`collab_count=1..5`。
- mid：ENPC/SLEV各20行，`collab_count=1..5`。
- guarded：ENPC/SLEV各20行，`collab_count=1..5`。

SSH/SCP清理：

- 每次SSH/SCP后均检查本地`ssh.exe`进程与`172.31.111.215:22`、`172.31.105.18:22` ESTABLISHED连接。
- 检查未发现残留SSH进程或ESTABLISHED连接。

Git镜像提交：

- 仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 分支：`codex/cvs-rffi-release-20260626`
- 提交：`284392d Add VRA-CI virtual radius adapter evaluation`

## 下一步建议

VRA-CI进一步确认：仅靠星上轻量adapter在当前特征空间中做unknown边界修复，会出现保旧类与拒未知的强冲突。下一步应进入地面训练阶段，使用open-set source/proxy unknown训练目标重训或微调ADV3B02级别模型，使`z_id`本身具备更强的unknown margin；星上侧再保留轻量qknn8、EVT/KNN和协同融合。
