# Phase2 ADV3B02 SOM-CI source-only open metric collaborative inference

## 基本信息

- 实验ID：`phase2_adv3b02_som_ci_20260704`
- 时间：2026-07-04
- 操作员/agent：Codex
- 目标：验证“地面侧source-only open-set表征修复”的最小近似是否能改善Stage2-C未知类拒识，同时保持旧类和seen-new性能。
- 对比目标：VRA-CI、OVC-CI、MOE-CI、OPR-CI、COTE-CI负结果。
- 结论边界：本地default和guarded两组均未达标；SOM-CI不能登记为Stage2-C成功、部署成功或论文主结论。

## 算法设计

SOM-CI学习一个紧凑的对角metric，只使用地面/source old与source-side `proxy_unknown`，模拟地面训练阶段加入open-set约束。target old/seen-new support不参与metric训练，只在最终Stage2-C qknn8注册与协同推理时使用。`target_unknown`仅用于最终拒识评估。

训练目标：

1. source old保持原型分类、margin与compactness。
2. source-side `proxy_unknown`远离所有source old原型。
3. 从source old原型与proxy_unknown之间采样virtual source negatives并推离已知原型。
4. 使用old-preserve与metric正则限制表征破坏。

部署状态量：

|配置|metric_fp16_state_bytes|source_prototype_fp16_bytes|total_fp16_state_bytes|
|---|---:|---:|---:|
|default/guarded|320|1920|2240|

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

- metric训练角色仅为`source`、`proxy_unknown`、`virtual_source_negative`。
- `target_support=0`用于metric训练；target support只在qknn8后端注册阶段使用。
- `target_unknown_training_count=0`，`target_unknown_eval_only=800`。
- 协同接收机为5个target receivers：`20-1,3-19,7-14,7-7,8-8`。
- `collab_counts=all`覆盖`1..5`，对应从单接收机到全部target receivers。

## 本地变更与验证

新增文件：

|文件|用途|SHA256|
|---|---|---|
|`code/scripts/phase2_source_open_metric_ci_eval.py`|SOM-CI算法与评估入口|`8CCD23AC18133E2345EAFD42E9981EAC87271760785E019A0343F328053979D0`|
|`code/tests/test_phase2_source_open_metric_ci_eval.py`|source-only训练边界测试|`948FA0F041C144564BBBAFB1E6C9186750BC35B9E44DE4EDD42E24427D979864`|

本地验证命令：

```powershell
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_source_open_metric_ci_eval.py code\tests\test_phase2_source_open_metric_ci_eval.py
conda run -n ssr-gpu python code\tests\test_phase2_source_open_metric_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。

## 本地全量结果

公共参数：`k_shot=8`，`qknn_k=8`，`query_per_class=20`，`collab_counts=all`，`collab_group_policy=available_up_to_k`，`max_event_bytes=1152`，`max_event_latency_ms=20`。

|配置|backend/profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|target_pass|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|default|ENPC/enpc_old80_unknown_probe|5|0.4247|0.0500|0.2500|0.1500|0.8500|0.1500|0.0000|0.0000|False|
|default|SLEV/slev_energy_strict|5|0.3441|0.0000|0.1000|0.1000|0.9000|0.0667|0.0528|0.0333|False|
|guarded|ENPC/enpc_balanced|4|0.8011|0.6750|0.3500|0.2500|0.1500|0.8500|0.0163|0.0000|False|
|guarded|SLEV/slev_known_anchor|4|0.8280|0.7000|0.4833|0.4000|0.0000|1.0000|0.0000|0.0000|False|

同row解释：

- default可提升unknown_reject到`0.85-0.90`，但旧类与seen-new严重崩塌。
- guarded可保住`old_acc≈0.83`并提高seen-new到`0.48`，但unknown_reject为`0`。
- source-only open metric未能解决“未知类拒识提升且旧类不下降”的核心冲突。

## N607同步与复测计划

远程工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

同步映射：

|本地文件|远程文件|
|---|---|
|`E:\type10-7\code\scripts\phase2_source_open_metric_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_source_open_metric_ci_eval.py`|
|`E:\type10-7\code\tests\test_phase2_source_open_metric_ci_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_source_open_metric_ci_eval.py`|

远程环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

N607预检：

- 命令：`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- 结果：直连`N607`通过；远程项目根目录存在；服务器时间`2026年07月04日13:43:24 CST`。
- GPU快照：GPU0显存占用`10/24576 MiB`，因此选择GPU0。

远程验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_source_open_metric_ci_eval.py code/tests/test_phase2_source_open_metric_ci_eval.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_source_open_metric_ci_eval.py
```

结果：语法检查通过；单元测试`1 passed`。远程CVS-RFFI环境对NumPy内部模块有DeprecationWarning；脚本已避免使用触发ImportError的NumPy min/std路径，警告不影响执行。

远程输出：

|配置|远程run目录|远程日志|本地归档|
|---|---|---|---|
|default|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_som_ci_20260704/k8_default`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_som_ci_20260704/k8_default.log`|`E:\type10-7\local_artifacts\phase2_adv3b02_som_ci_20260704\remote\k8_default`|
|guarded|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_som_ci_20260704/k8_guarded`|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_som_ci_20260704/k8_guarded.log`|`E:\type10-7\local_artifacts\phase2_adv3b02_som_ci_20260704\remote\k8_guarded`|

远程归档SHA256：

|文件|SHA256|
|---|---|
|`remote\k8_default\som_ci_summary.json`|`F1F732317A23DE5105FDAB29389A2F36C0C97517061AB13C2C6005E6ED450BBE`|
|`remote\k8_guarded\som_ci_summary.json`|`03ADA8BC8244723DA713CE1B3552AFFEFA0F3841CE672E47558206B4EC3A7228`|
|`remote\k8_default.log`|`FFE2188FF05669B792AE7E56F464F03EC1DEB5E6AC378EE3348F3A7E0B9C5873`|
|`remote\k8_guarded.log`|`559DDB26C54420D7BC3C43E5F0F39990F8143AA869F03A12BDA9E3E73F5CFC84`|

远程全量结果：

|配置|backend/profile|collab_count|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_defer|unknown_defer|target_pass|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|default|ENPC/enpc_old80_unknown_probe|5|0.4247|0.0500|0.2500|0.1500|0.8500|0.1500|0.0000|0.0000|False|
|default|SLEV/slev_energy_strict|5|0.3441|0.0000|0.1000|0.1000|0.9000|0.0667|0.0528|0.0333|False|
|guarded|ENPC/enpc_balanced|4|0.8011|0.6750|0.3500|0.2500|0.1500|0.8500|0.0163|0.0000|False|
|guarded|SLEV/slev_known_anchor|4|0.8280|0.7000|0.4833|0.4000|0.0000|1.0000|0.0000|0.0000|False|

远程summary覆盖：

- default：ENPC/SLEV各20行，`collab_count=1..5`。
- guarded：ENPC/SLEV各20行，`collab_count=1..5`。

SSH/SCP清理：

- 每次SSH/SCP后均检查本地`ssh.exe`进程与`172.31.111.215:22`、`172.31.105.18:22` ESTABLISHED连接。
- 检查未发现残留SSH进程或ESTABLISHED连接。

Git镜像提交：

- 仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 分支：`codex/cvs-rffi-release-20260626`
- 提交：`9bba3df Add SOM-CI source open metric evaluation`

## 下一步建议

SOM-CI说明source-only开放集约束本身可以制造拒识边界，但会显著扭曲target old/seen-new目标域邻域。下一步更合理路线不是继续做feature-level后处理，而是训练ADV3B02后续模型时显式加入LEO星地信道source view、多receiver episodic split和open-set proxy unknown，同时把old-class receiver floor作为训练约束。
