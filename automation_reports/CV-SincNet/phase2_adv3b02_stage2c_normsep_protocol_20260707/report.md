# qKNNV42真实Stage2-C NORM_SEP协议修复准备

## 基本信息

|字段|内容|
|---|---|
|experiment ID|`phase2_adv3b02_stage2c_normsep_protocol_20260707`|
|timestamp|2026-07-07|
|operator|Codex|
|objective|在qKNNV42目标下继续推进旧类域适应、新类增多坍塌和最低类过低问题；本轮补齐真实Stage2-C导出协议，使NORM_SEP类表征能同时导出`target_old`、`target_new`和`target_unknown`|
|status|本地代码和launcher准备完成；2026-07-07 01:58、02:01与02:04三次N607直连预检/复查均通过，但GPU持续满载且每卡5条Python计算进程，达到连续三次容量阻塞；未同步远端文件，未启动训练|

## 协议边界

已读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`。本轮不修改项目协议，遵守以下边界：

- `R_t=7-14`，与source receivers`0,1,2,3,4,5,6`不相交。
- `target_old`仍为ManySig旧类`14-10,14-7,20-15,20-19,6-15,8-20`。
- `target_new`使用已有完整Stage2-C包中的真实ManyTx标签：`1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4`。
- `target_unknown`使用互斥真实ManyTx标签：`10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20`。
- proxy unknown训练池显式排除上述`target_new`和`target_unknown`，避免把目标新类或未知类用于source-side proxy训练。
- LEO视图固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，不是clean control。

## 本轮变更

|文件|位置|目的|
|---|---|---|
|`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|Git承载面与`E:\type10-7\code`运行面|新增四段`cell`格式：`name:target_rx:target_new_tx_ids:target_unknown_tx_ids`；保留旧三段格式兼容|
|`code/scripts/launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh`|Git承载面与运行面|新增真实Stage2-C NORM_SEP/HEAD_SEP训练导出launcher|
|`code/tests/test_train_apply_phase1_iq_preadapter_cells.py`|Git承载面与运行面|覆盖三段兼容、四段Stage2-C解析和`target_new/target_unknown`重叠拒绝|

`E:\type10-7\code`不是Git仓库；本轮改动已镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`，后续以Git提交为版本承载面。

## 验证

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m py_compile code\scripts\train_apply_phase1_iq_preadapter_20260703.py code\tests\test_train_apply_phase1_iq_preadapter_cells.py`|PASS|
|`conda run -n ssr-gpu python -m unittest discover -s code\tests -p test_train_apply_phase1_iq_preadapter_cells.py -v`|PASS，3个测试通过|
|`bash -n ./code/scripts/launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh`|PASS|
|`bash -lc '... launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh --dry-run'`|PASS，输出确认`target_new`、`target_unknown`和proxy排除声明|

dry-run关键输出：

```text
[STAGE2C-NORMSEP] target_new=1-10,1-12,1-14,1-16,1-18,1-8,10-11,10-4
[STAGE2C-NORMSEP] target_unknown=10-7,11-1,11-10,11-19,11-4,11-7,12-19,12-20
[STAGE2C-NORMSEP] proxy_pool_excludes_target_new_and_target_unknown=true
```

## N607预检与延期记录

2026-07-07 01:58 CST执行项目规定的只读直连预检：

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

预检结果：PASS。直接`N607`目标、身份文件、服务器时间、项目根目录`/home/szu2070436088/2510044040/CV-SincNet`和GPU可见性均通过。

随后执行只读GPU/进程占用检查。关键证据：

```text
0, 100, 12201, 24576
1, 100, 11863, 24576
2, 100, 11967, 24576
3, 100, 11763, 24576
4, 100, 12239, 24576
5, 100, 12267, 24576
6, 99, 11807, 24576
7, 99, 12561, 24576
```

`nvidia-smi --query-compute-apps`显示每张GPU均有多条`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`计算进程；`ps`显示这些进程来自`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`，当前已有`phase1_dgleo_directmetric16_20260706`队列运行。

决策：不在当前占用状态下追加`phase2_adv3b02_stage2c_normsep_protocol_20260707`。理由是当前8张GPU均接近满载，且每卡已有多条训练计算进程，超过默认“每GPU最多两条训练实验”的启动边界。为避免影响在跑队列，本轮未执行`scp`、未远端覆盖脚本、未启动launcher。

本地SSH清理检查：

|检查|结果|
|---|---|
|`Get-Process ssh -ErrorAction SilentlyContinue`|无残留`ssh.exe`|
|`Get-NetTCPConnection -RemotePort 22 -State Established`|无到22端口的ESTABLISHED连接|

### 2026-07-07 02:01 CST复查

按目标续跑要求再次执行只读直连预检，结果仍为PASS：直接`N607`目标、身份文件、项目根目录和GPU可见性均正常。

复查GPU摘要：

```text
0, 99, 12201, 24576
1, 100, 11863, 24576
2, 100, 11967, 24576
3, 99, 11763, 24576
4, 99, 12239, 24576
5, 99, 12267, 24576
6, 100, 11821, 24576
7, 99, 12563, 24576
```

`nvidia-smi pmon -c 1`复查显示每张GPU仍有5条`python`计算进程：

|GPU|python计算PID|
|---:|---|
|0|`3685671,3686096,3790852,3791255,3791658`|
|1|`3686499,3686922,3792474,3792879,3793282`|
|2|`3687359,3688170,3793685,3794089,3794492`|
|3|`3688607,3689015,3794895,3795299,3796119`|
|4|`3689452,3689858,3796539,3796963,3797400`|
|5|`3690703,3691109,3797820,3798224,3798665`|
|6|`3691546,3691953,3799494,3799934,3800355`|
|7|`3692392,3693211,3800775,3801213,3801636`|

复查决策：继续延期，不执行`scp`、不远端覆盖脚本、不启动`phase2_adv3b02_stage2c_normsep_protocol_20260707`。当前状态已经超过默认每GPU最多两条训练实验的边界，追加训练会干扰正在运行的CVS-RFFI队列。

复查后的本地SSH清理检查仍为通过：

|检查|结果|
|---|---|
|`Get-Process ssh -ErrorAction SilentlyContinue`|无残留`ssh.exe`|
|`Get-NetTCPConnection -RemotePort 22 -State Established`|无到22端口的ESTABLISHED连接|

### 2026-07-07 02:04 CST第三次复查

按目标续跑要求第三次执行只读直连预检，结果仍为PASS：直接`N607`目标、身份文件、项目根目录和GPU可见性均正常。

复查GPU摘要：

```text
0, 99, 12201, 24576
1, 99, 11863, 24576
2, 99, 11967, 24576
3, 99, 11763, 24576
4, 99, 12239, 24576
5, 99, 12267, 24576
6, 99, 11821, 24576
7, 99, 12563, 24576
```

`nvidia-smi pmon -c 1`第三次复查显示每张GPU仍有5条`python`计算进程：

|GPU|python计算PID|
|---:|---|
|0|`3685671,3686096,3790852,3791255,3791658`|
|1|`3686499,3686922,3792474,3792879,3793282`|
|2|`3687359,3688170,3793685,3794089,3794492`|
|3|`3688607,3689015,3794895,3795299,3796119`|
|4|`3689452,3689858,3796539,3796963,3797400`|
|5|`3690703,3691109,3797820,3798224,3798665`|
|6|`3691546,3691953,3799494,3799934,3800355`|
|7|`3692392,3693211,3800775,3801213,3801636`|

阻塞审计：这是同一`phase2_adv3b02_stage2c_normsep_protocol_20260707`启动目标连续第三个目标回合遇到同一N607容量阻塞。当前没有安全的本地替代动作可以证明qKNNV42目标已达成；真实Stage2-C路线仍需N607释放到每GPU不超过默认训练并发边界后，才能执行`scp`、远端验证、启动和启动健康检查。因此本轮仍不执行`scp`、不远端覆盖脚本、不启动launcher。

第三次复查后的本地SSH清理检查仍为通过：

|检查|结果|
|---|---|
|`Get-Process ssh -ErrorAction SilentlyContinue`|无残留`ssh.exe`|
|`Get-NetTCPConnection -RemotePort 22 -State Established`|无到22端口的ESTABLISHED连接|

## 预期N607使用方式

下一步恢复启动前，必须重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

预期远端同步：

|local|remote|
|---|---|
|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\train_apply_phase1_iq_preadapter_20260703.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|
|`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh`|

预期启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup bash code/scripts/launch_phase2_adv3b02_stage2c_normsep_protocol_20260707.sh > logs/phase2_adv3b02_stage2c_normsep_protocol_20260707.driver.out 2>&1 & echo $!
```

预期输出：

|类型|路径|
|---|---|
|runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_normsep_protocol_20260707/`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_normsep_protocol_20260707/`|
|summary|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_normsep_protocol_20260707/stage2c_normsep_protocol_summary.json`|

## 成功判据与风险

|项|判据|
|---|---|
|旧类阶段门槛|K5/K10下`old_acc>=0.80`，并关注`min_old_class_acc`|
|seen-new目标|相对冻结Stage2-C包提高`seen_new_acc`和`min_seen_new_class_acc`，重点观察最低类是否脱离20%-40%坍塌区|
|unknown目标|`unknown_FAR<=0.05`为最终部署安全目标；若未达，只能标为诊断或下一阶段优化输入|
|协议安全|`target_unknown`只评估，不参与阈值拟合、训练或model selection|

风险：本launcher只修复真实Stage2-C导出协议并继承NORM_SEP/HEAD_SEP类表征训练思想；它不保证一次达到最终目标。若N607结果仍显示unknown moat弱或seen-new floor低，应转向class/receiver-protected episodic representation training，而不是继续阈值微调。
