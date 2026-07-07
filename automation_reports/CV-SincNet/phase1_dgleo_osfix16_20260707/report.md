# phase1_dgleo_osfix16_20260707

## 基本信息

|字段|值|
|---|---|
|实验ID|`phase1_dgleo_osfix16_20260707`|
|时间|2026-07-07|
|操作者|Codex Phase1地面训练实验agent|
|阶段边界|Phase1 source-only地面域泛化训练|
|数据边界|`Dataset_WigSig/ManySig.pkl`，禁止ManyTx/target/unknown进入训练|
|核心目标|保护overall/strict UDU/receiver floor/satellite floor，同时直接压低known接收域open-set风险代理指标|
|基线参照|`phase1_dgleo_directmetric16_20260706`和`phase1_dgleo_uopt24_20260707`，尤其`DGLEO_DM_P0D_RADIUS_A`、`DGLEO_UOPT_P0_CORE_C`|
|当前状态|本地实现和验证完成，等待同步并在N607启动|

## 假设

上一轮DirectMetric16/UOPT24已经证明直接指标损失和温和无标签利用能提升p95/p99/proxy_vaccept，但source_overflow、source_episode_overflow、tail/overflow接收和弱receiver/星地floor仍是主要矛盾。本轮不是继续间接调参，而是把Phase1训练中的open-set代理指标写进直接优化项：

- labeled/source侧：继续使用`direct_metric_acceptance_loss`直接约束`zid_p50/p95/p99/tail_cvar`、`source_overflow`、`proxy_vaccept`、`bridge_accept_rate`、`low_density_accept_rate`、`tail_accept_rate`、`overflow_accept_rate`、`radius_to_inter_ratio`。
- unlabeled/source侧：新增`unlabeled_known_acceptance_quarantine_loss`，对未通过伪标签门控的source无标签样本，直接降低其落入known local-component接收门的概率。
- 星地视图：保持`concat_sa`样本拼接形式，不使用CE-only；同时保留domain loss、ADV loss、sat consistency、U_s direct metric和U_s quarantine在星地增强视图上的约束。

## 本地改动

|文件|作用|
|---|---|
|`code/cvsrffi/losses.py`|新增`unlabeled_known_acceptance_quarantine_loss`，用source known anchors构建local gate，直接惩罚untrusted U_s被known接收。|
|`code/SSDG/train_ssdg.py`|接入`lambda_u_quarantine_accept`、valid-domain-only U_s direct metric、U_s p50/tail_cvar日志、quarantine日志和损失权重。|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|新增16候选launcher，每张GPU两实验，Phase1 source-only guard，concat_sa full模式。|
|`code/tests/test_unlabeled_quarantine_acceptance_loss.py`|覆盖新增quarantine损失和训练参数入口。|
|`code/tests/test_phase1_dgleo_osfix16_launcher.py`|覆盖OSFIX16协议声明、每卡两实验分配和ManyTx拒绝。|
|`automation_reports/CV-SincNet/phase1_dgleo_osfix16_20260707/local_dry_run.txt`|完整本地dry-run命令记录。|

## 本地验证

|检查|结果|
|---|---|
|根目录Git状态|`E:\type10-7`不是Git仓库，按规则使用`github_publish/CVS-RFFI-repo`作为Git承载面。|
|语法检查|`python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py`通过。|
|launcher语法|`bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`通过。|
|聚焦测试|`14 passed`：DirectMetric旧测试、UOPT入口测试、新quarantine测试、新OSFIX16 launcher测试均通过。|
|完整dry-run|候选数16；GPU分布`0:2 1:2 2:2 3:2 4:2 5:2 6:2 7:2`。|
|dry-run路径|`automation_reports/CV-SincNet/phase1_dgleo_osfix16_20260707/local_dry_run.txt`|

## 实验矩阵

|candidate|GPU|组别|目标|主要直接优化压力|
|---|---:|---|---|---|
|`DGLEO_OSFIX_CORE_A`|0|P0_CORE|UOPT最优安全基线加quarantine|温和source/proxy/tail+U_s quarantine|
|`DGLEO_OSFIX_CORE_B`|0|P0_CORE|DirectMetric半径安全基线加quarantine|radius/inter和source overflow保守加强|
|`DGLEO_OSFIX_DENSITY_A`|1|P0_DENSITY|source overflow密度门|source_overflow、overflow_accept、radius/inter|
|`DGLEO_OSFIX_DENSITY_B`|1|P0_DENSITY|source overflow严格密度门|更强source_overflow和p99/tail_cvar|
|`DGLEO_OSFIX_PROXY_A`|2|P0_PROXY|proxy_vaccept CVaR安全压制|proxy_vaccept、virtual unknown接收|
|`DGLEO_OSFIX_PROXY_B`|2|P0_PROXY|proxy_vaccept CVaR严格压制|更强proxy_vaccept和tail/overflow|
|`DGLEO_OSFIX_BRIDGE_A`|3|P0_BRIDGE|bridge/low-density/shell负样本|bridge_accept、low_density_accept|
|`DGLEO_OSFIX_BRIDGE_B`|3|P0_BRIDGE|bridge/low-density严格负样本|更强bridge、low-density、radius/inter|
|`DGLEO_OSFIX_TAIL_A`|4|P0_TAIL|tail/overflow/radius clamp|tail_accept、overflow_accept、p99|
|`DGLEO_OSFIX_TAIL_B`|4|P0_TAIL|tail/overflow/radius严格clamp|更强tail、overflow、p95/p99|
|`DGLEO_OSFIX_SATOPEN_A`|5|P0_SATOPEN|星地pair open-set floor保护|sat_pair、U_s sat consistency、sat floor|
|`DGLEO_OSFIX_SATOPEN_B`|5|P0_SATOPEN|星地pair严格open-set保护|更强sat_pair、U_s sat、proxy/tail|
|`DGLEO_OSFIX_UQ_A`|6|P1_UQ|无标签quarantine安全版本|U_s untrusted known-accept压制|
|`DGLEO_OSFIX_UQ_B`|6|P1_UQ|无标签quarantine严格版本|更强U_s quarantine和ADV|
|`DGLEO_OSFIX_JOINT_A`|7|P1_JOINT|联合平衡主候选|source/proxy/bridge/tail/sat/U_s joint|
|`DGLEO_OSFIX_JOINT_B`|7|P1_JOINT|联合强约束上界|强open-set代理压制，观察泛化冲突|

## N607同步计划

|本地文件|远端目标|
|---|---|
|`code/cvsrffi/losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`|
|`code/SSDG/train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|
|新增测试文件|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/`对应路径|

计划远端验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py
bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh --dry-run --only=DGLEO_OSFIX_JOINT_A
```

计划启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=8 bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
```

预期输出：

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dgleo_osfix16_20260707`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_osfix16_20260707`
- 每候选输出：`metrics_epoch.csv/jsonl`、`latest_ssdg.pth`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt/json`

## 关键观察指标

|维度|必须跟踪|
|---|---|
|泛化|`overall_tx`、`strict_udu`、receiver floor、satellite mean/floor、best-final gap、最弱receiver|
|known紧致性|`dm_accept_zid_p50/p95/p99`、`dm_accept_zid_tail_cvar_deg`、`zid_compact_tail_cvar_deg`|
|open-set代理风险|`dm_accept_source_overflow`、`source_episode_overflow_rate`、`dm_accept_proxy_vaccept`、`dm_accept_bridge_accept_rate`、`dm_accept_low_density_accept_rate`、`dm_accept_tail_accept_rate`、`dm_accept_overflow_accept_rate`、`dm_accept_radius_to_inter_ratio`|
|无标签利用|`pseudo_selected`、`u_dm_accept_selected`、`u_dm_accept_zid_p50/p95/p99/tail_cvar`、`u_quarantine_accept_rate`、`u_quarantine_low_density_accept_rate`、`u_quarantine_rate`|
|星地视图|`sat_mean/floor`、`dm_accept_sat_pair_angle_p95_deg`、`u_dm_accept_sat_pair_angle_p95_deg`、`loss_u_sat_cons`|

## 成功标准

主推进候选必须同时满足：

- 泛化不低于当前最优量级：final strict UDU不明显回落，receiver floor和satellite floor至少不下降。
- open-set代理指标至少两个P0风险显著下降：`proxy_vaccept`、`source_overflow`、`p99/tail_cvar`、`tail/overflow_accept`、`radius_to_inter_ratio`。
- 无标签不再只是提高闭集：`u_quarantine_accept_rate`下降，且`u_dm_accept_p95/p99/tail_cvar`没有恶化。
- best-final gap可控，不能只出现best checkpoint单点强。

失败判据：

- strict UDU或satellite floor明显下降，而open-set代理改善不稳定。
- `p95`下降但`p99/tail_cvar/source_overflow`仍高。
- `proxy_vaccept`下降但`bridge/low_density/tail_accept`上升。
- U_s quarantine压制导致pseudo_selected大幅下降且receiver floor退化。
- final相对best明显回落，说明训练后期仍在破坏跨域结构。

## 当前不能声明

这是Phase1 source-only训练。本实验启动和完成后也不能直接声明真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc或H_old_new改善；只能说明闭集DG能力、星地压力鲁棒性、known特征几何、proxy/virtual unknown风险和prototype导出质量。

## N607落地状态

更新时间：2026-07-07 10:53-11:05 CST
本地Git承载面commit：`fc83410 Add Phase1 DG-LEO OSFIX16 training route`
状态：已完成本地实现、镜像提交、N607直连preflight、文件同步、远端语法/编译/dry-run验证，并已启动16个Phase1训练候选。当前是启动健康通过，不是完成结果。

### N607 preflight与占用

|检查项|结果|
|---|---|
|直连目标|`N607`可用|
|服务器时间|`2026年 07月 07日 星期二 10:53:50 CST`|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`可见|
|GPU|8张RTX3090可见|
|启动前训练进程|未发现`train_ssdg.py`或同run launcher|
|启动前GPU占用|仅Xorg背景进程|
|磁盘|`/home`约11T，总用量约27%，可用约7.6T|

### 同步与远端验证

|文件|SHA256|远端验证|
|---|---|---|
|`code/cvsrffi/losses.py`|`94d5d26bcc7fc799658bf67a3949e067e4beedc3c0b04b57cb86b6c5007f6d6b`|matched|
|`code/SSDG/train_ssdg.py`|`fcf3ca59cf3f6c6758c97cfb2cb917bbfd04373152ac2ab2478d51692f83688c`|matched|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|`97a78ace17b91116b3db7107b937d99814da6f4f893238980113e00175ea89f8`|matched|
|`code/tests/test_unlabeled_quarantine_acceptance_loss.py`|`6a26d4e50c0d8c41d5602dc875cce4df79ec9722d8f611843c40436c4ffb7aba`|matched|
|`code/tests/test_phase1_dgleo_osfix16_launcher.py`|`b39287eac22d13245ae805127449dfe435c6652502dadc6e67bdcc3c5a0e72ca`|matched|

远端验证命令均通过：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py
bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh --dry-run --only=DGLEO_OSFIX_JOINT_A
```

dry-run确认关键标志存在：`use_concat_sat_channel_aug=1`、`concat_sat_ce_only=0`、`direct_open_set_metric_loss=1`、`unlabeled_quarantine_accept=1`、`stage2_success_claim=0`。

### 启动命令与健康检查

实际启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=8 bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
```

launcher返回：`[OSFIX16-SUBMIT-COMPLETE]`

启动后健康检查：

|检查项|结果|
|---|---|
|GPU训练进程|16个Python训练进程，严格2个/GPU|
|日志文件|16个`.out`已生成|
|run目录|16个候选目录已生成|
|`metrics_epoch.csv`|16个已生成|
|错误扫描|未发现`Traceback`、`RuntimeError`、`Killed`、`unrecognized`|
|配置marker|`CONFIG-U-DIRECT`、`CONFIG-DM-ACCEPT`、`CONFIG-SAT`均存在|

### 候选落地表

|candidate|GPU|主PID|log|run目录|
|---|---:|---:|---|---|
|`DGLEO_OSFIX_CORE_A`|0|4057265|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_A`|
|`DGLEO_OSFIX_CORE_B`|0|4057673|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_B`|
|`DGLEO_OSFIX_DENSITY_A`|1|4058076|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_A`|
|`DGLEO_OSFIX_DENSITY_B`|1|4058891|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_B`|
|`DGLEO_OSFIX_PROXY_A`|2|4059311|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_A`|
|`DGLEO_OSFIX_PROXY_B`|2|4059718|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_B`|
|`DGLEO_OSFIX_BRIDGE_A`|3|4060156|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_A`|
|`DGLEO_OSFIX_BRIDGE_B`|3|4060562|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_B`|
|`DGLEO_OSFIX_TAIL_A`|4|4061000|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_A`|
|`DGLEO_OSFIX_TAIL_B`|4|4061406|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_B`|
|`DGLEO_OSFIX_SATOPEN_A`|5|4062249|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_A`|
|`DGLEO_OSFIX_SATOPEN_B`|5|4062656|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_B`|
|`DGLEO_OSFIX_UQ_A`|6|4063093|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_A`|
|`DGLEO_OSFIX_UQ_B`|6|4063500|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_B`|
|`DGLEO_OSFIX_JOINT_A`|7|4063937|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_A`|
|`DGLEO_OSFIX_JOINT_B`|7|4064343|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_B`|

注意：`pgrep -af`会显示DataLoader worker继承的长命令行，不能把这些worker都当作主训练PID。上表主PID来自`nvidia-smi pmon`中的GPU计算进程。

### 后续完成判定

本轮只完成启动与早期健康验证。完成后必须按同候选同epoch联合判断：

- 泛化：`overall_tx`、`strict_udu`、receiver floor、satellite mean/floor、best-final gap。
- open-set代理：`proxy_vaccept`、`source_overflow`、`bridge_accept_rate`、`low_density_accept_rate`、`tail/overflow_accept`、`radius_to_inter_ratio`、`zid_p50/p95/p99`、`zid_tail_cvar`。
- 无标签：`u_dm_accept_*`和`u_quarantine_accept_rate/low_density_accept_rate`是否改善，同时不牺牲receiver floor和satellite floor。
- 候选推进：只有同时保护泛化并降低p99/tail/proxy/source overflow风险的候选，才可进入Stage2真实unknown评估；Phase1结果本身不能声明真实unknown_FAR或FPR95改善。
