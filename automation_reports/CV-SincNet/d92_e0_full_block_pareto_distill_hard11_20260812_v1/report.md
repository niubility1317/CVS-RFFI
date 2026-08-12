# D92 E0 FULL/BLOCK Pareto蒸馏Hard11实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT / REJECT_ROUTE`|
|时间|2026-08-12 13:03 HKT|
|操作者|Codex `/root`；科学实现与机械执行闭环分属独立owner；N607由唯一runner负责|
|候选|`E0_FULL_BLOCK_PARETO_DISTILL`；candidate=`d92_e0_full_block_pareto_distill`|
|协议|`p2_min_v1`、复用`VALIDATED_ONCE`数据，不重复验证|
|claim scope|`DEVELOPMENT_ONLY_HARD_SCREEN`|
|正式基线|`E0_FULL_ONLY`完整Target125历史artifact；本轮不重跑E0或D92|

本轮目标是在冻结Hard10同排上同时提高H、old BA、`c_old_acc`、old floor和seen-new，同时降低forgetting、new→old与old→new；query MAC和永久state与E0精确相同，注册计算远低于原D92。

## 2.既有证据与假设

Hard10原始D92 score已从原retry2任务只读取回10/10份，未修改远端。与E0同排复算如下：

|候选|H|old BA / c_old|old floor|seen-new|forgetting|new→old|old→new|
|---|---:|---:|---:|---:|---:|---:|---:|
|E0_FULL_ONLY|73.3472%|74.8611%|44.8333%|72.0333%|12.9167%|15.0417%|15.3333%|
|原D92|74.1673%|76.1389%|48.5000%|72.3750%|11.6389%|15.1000%|14.6944%|
|D92−E0|+0.8201pp|+1.2778pp|+3.6667pp|+0.3417pp|−1.2778pp|**+0.0583pp**|−0.6389pp|

原D92不是八项Pareto解：new→old略差，seen-new只在5/10行不降。因此本轮不压缩复刻D92，而只提取BLOCK协方差的互补方向，并在support上联合约束新旧两组。

受控Hard12证据同时排除了三条路线：D46 LOO提高floor但损害H/new/forgetting；D62 Fisher/Pareto净收益近零且计算翻倍；固定50提高floor但其余核心指标下降。FloorBoost和NewGuard也已分别因新类崩损、八项全零而拒绝。

## 3.冻结方法

设计与追溯：

- `docs/superpowers/specs/2026-08-12-d92-full-block-pareto-distill-design.md`
- `analysis/d92_full_block_pareto_distill_traceability_20260812.md`
- Git设计提交：`b925e820`

K>2时，同一份D81变换support只估计一次`Sigma=0.5 Sigma_old+0.5 Sigma_new`。FULL解使用`Sigma`，BLOCK解只取同一`Sigma`的160/96/32块对角，不重估中心或协方差。经组均衡support-logit RMS对齐后，以`theta(beta)=theta_FULL+G(diag(beta)D)`形成单头候选，`0≤beta_c≤1`且所有类使用同一公式。

一次词典序求解分三层：先最大化七个固定tail约束的共同增益，包括六个旧类各自的lower-Q20 true-vs-all margin均值和一个将所有新类support合并后的lower-Q20 true-new-vs-old margin均值；再最小化old→new与new→old最坏方向hinge；最后最小化归一化方向范数。固定tail集合只由E0部署support确定，优化中不更换，也不扫描权重。

部署阶段分别对E0和唯一候选做一次真实D42预览，复用实际`scale1/scale2`计算support跨组margin量子。候选只有在old→new或new→old至少一项部署变化达到该量子、七项tail和双向hinge均不劣于E0、且部署头非E0 byte-exact时才激活；否则精确回退E0。正式状态只发布一个D42 F0头，并要求最终解码SHA等于候选预览SHA。

禁止PRESS、K折LOO、Fisher、Pareto枚举、旧类统一bias、NewGuard多尺度回缩和query选择。部署头E0-byte-exact、未跨真实量化步或support约束失败时，本地无效或exact E0 fallback，不进入性能发布。

## 4.冻结矩阵

10个performance outer与NewGuard/FloorBoost Hard10完全相同；另含一个K1 liveness：

|序号|outer|角色|
|---:|---|---|
|1|`rx_7_7__seed_713106__k_10__new_5`|performance / K>2 smoke|
|2|`rx_7_7__seed_713104__k_5__new_20`|performance|
|3|`rx_7_7__seed_713103__k_10__new_5`|performance|
|4|`rx_8_8__seed_713103__k_5__new_20`|performance|
|5|`rx_8_8__seed_713103__k_10__new_5`|performance|
|6|`rx_8_8__seed_713106__k_5__new_20`|performance|
|7|`rx_7_14__seed_713104__k_10__new_10`|performance|
|8|`rx_3_19__seed_713102__k_10__new_5`|performance|
|9|`rx_7_7__seed_713105__k_10__new_20`|performance|
|10|`rx_7_7__seed_713104__k_10__new_5`|performance|
|11|`rx_20_1__seed_713106__k_1__new_20`|liveness|

每个outer固定`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，共11 job、33 scene-arm、1 arm、8 shard。K1不进入性能均值。

历史E0 paired baseline：
`E:\type10-7\local_artifacts\d92_e0_full_only_target125_20260812_v1\analysis\paired_rows.csv`，SHA256=`6ebb37fac77d5a218924bcb51ad27424abff4a162a3b8a45a340947fe6d8de6a`。

## 5.性能与资源门

八项均值必须同时严格优于E0。目标幅度为H≥+1.00pp、old BA≥+1.50pp、`c_old_acc`≥+1.00pp、floor≥+4.00pp、seen-new≥+0.50pp、forgetting≤−1.50pp、两向混淆各≤−0.50pp。任一方向平或反向直接`REJECT_ROUTE`。

资源门：

- K>2 two-state fit=4，`DA1_REG1`实际solve=2；FULL/BLOCK共享一套协方差统计；
- query MAC与永久state精确等于E0；
- wall目标≤120ms且≤1.25×E0，硬门≤150ms且≤1.50×；
- peak≤E0+512KiB；
- 注册计算相对原D92至少下降80%。

全方向正确但幅度、稳定性或目标资源不足且硬门未破，只允许`REVISE_ONCE`；全部通过才`ADVANCE_TO_TARGET125_CANDIDATE`，本轮绝不自动跑125。

## 6.本地实现与验证

实现已闭合并进入Git：设计`b925e820`，Hard11机械闭环`6205901a`/`231070cf`，科学实现`51b4605d`，D42跨组真实量子收据与机械门`fa809723`，最终科学量子门`ba866ff8`。

本地`ssr-gpu`一次整合验证覆盖core、共享协方差、probe、slim、query、Hard11 builder、runner和analyzer，共133项通过；9个发布模块`py_compile`通过，配置JSON、两个CLI帮助、`git diff --check`全部通过。独立监督者仅复核此前唯一P1后给出`P0=0 / P1=0 / APPROVE`。

真实D42 RED证据为：实际量子约`0.00788`而跨组部署变化约`0.001`时，旧实现会错误激活；修复后精确回退E0。K>2 active必须满足`quantum>0`、`change>=quantum`、`quantum_pass=true`，K1/REG0三字段必须为`None`。

本地历史回收K10包在进入fit前因旧SOMP-H detached seal schema不兼容而停止，没有性能结果；未据此修改方法。N607当前封存包的真实checkpoint K10 no-query smoke被保留为shard前硬门：只有active、非fallback、fit=4/2、query零访问和最终D42 SHA闭合后才会启动8个shard。P2不阻塞实验。

## 7.N607预注册

|字段|冻结值|
|---|---|
|普通账号|SSH alias `N607`；禁止管理员账号|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_pareto_distill_source_snapshot_20260812_v1`|
|output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|logs root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_block_pareto_distill_hard11_20260812_v1`|
|GPU|GPU0–7各一个shard，`CUDA_VISIBLE_DEVICES=i`内使用`cuda:0`|
|expected|11 job receipt、22正式prediction/COMMIT/fit/resource、11 score、8 summary|
|runtime commit|`ba866ff8f3292faad8ca847e5115f6591a0f71d4`|
|runtime archive|`d92_pareto_distill_runtime_closure_ba866ff8.tar.gz`；5,082,551 bytes；SHA256=`8351fcf9241a73b2ee89865d2a12c6add4aba517dd168716b7e8a8fb88a3dab5`|
|method lock|`stage2_d92_full_block_pareto_distill_hard11_v1.json`；SHA256=`6c3e1a1b41e08ecf7444c30607cbfdf5d59bcea06f9a902eacac91186d8f62c7`|
|launch|`launch.sh`；SHA256=`19e2f3e12281a918144b781270f7ab8eb72631ba1c9947808222861c0f9fd5cc`|
|selection|SHA256=`969fecb2fe723fa04db766cec0390f83771398124494c459e1918d90b91da8df`|

本地到远端只同步三件固定输入：外部报告目录的runtime archive到source root；Git仓库中的method lock到`source_root/configs/`；Git仓库报告目录的`launch.sh`到source root。runtime归档含1,312项，包含`code/cvsrffi/__init__.py`、科学核心和runner入口，且不存在`code/code`。

冻结exact command：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_pareto_distill_source_snapshot_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

启动前source/output/logs及本地取回路径必须不存在；唯一runner只启动一次，`fresh_run_retry=false`。PID、CWD、GPU映射、manifest SHA、smoke receipt和最终artifact树在runner回传后追加。

health stop仅限协议/安全错误、launcher确定性故障、prediction闭包失败或两个不同outer出现同一pre-prediction确定性异常。不得读取性能决定停止。技术失败保留artifact并标记`NO_PERFORMANCE_RESULT`，不得覆盖或在同一run ID重启。

## 8.风险与完成后检查

最大工程风险是共享统计后的BLOCK solve仍超过150ms/1.50×E0；最大科学风险是support margin改善不能外推为query Pareto。前者先由真实checkpoint资源smoke筛除，后者只能由完整10/10 Hard10一次性证伪。

完成后必须回填：Git差异与测试、真实smoke、同步映射与哈希、exact command/PID/GPU、11/11闭合、完整取回、逐outer八指标、receiver/K/scene/六旧类分解、wall/peak/fallback以及唯一裁决。

## 9.N607唯一runner回收记录（2026-08-12）

### 9.1状态与预检

|字段|事实证据|
|---|---|
|runner状态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；fresh-run retry=`false`|
|本地版本|`E:\\type10-7\\code\\snapshots\\d92_125wt`分支`codex/d92-be-hard12-strict-pareto-20260811`，HEAD=`621a5b0af3d318c7b172737cfabbcdf1b99bb346`，启动前工作树干净|
|直连预检|普通账号直连N607 PASS；项目根、Python 3.10.19、8×RTX3090可见|
|启动前冲突检查|source/output/logs及本地回收根均不存在；无匹配run进程；GPU空闲|

### 9.2同步与远端验证

串行同步`archive→config→launch`，每次SCP后均确认本地无残留`ssh.exe`和N607 TCP22连接。远端三件文件哈希与冻结值一致；`bash -n`、临时解包`compileall`和11个冻结模块import closure均PASS。归档成员数为1,312，包含`code/cvsrffi/__init__.py`、Pareto核心与runner，未发现`code/code`。

|远端文件|大小|SHA256|
|---|---:|---|
|`source_root/d92_pareto_distill_runtime_closure_ba866ff8.tar.gz`|5,082,551|`8351fcf9241a73b2ee89865d2a12c6add4aba517dd168716b7e8a8fb88a3dab5`|
|`source_root/configs/stage2_d92_full_block_pareto_distill_hard11_v1.json`|7,154|`6c3e1a1b41e08ecf7444c30607cbfdf5d59bcea06f9a902eacac91186d8f62c7`|
|`source_root/launch.sh`|4,252|`19e2f3e12281a918144b781270f7ab8eb72631ba1c9947808222861c0f9fd5cc`|

### 9.3唯一启动与健康结果

唯一执行冻结命令，SSH返回0。prepare与import closure完成后，K10 smoke在shard启动前失败，未执行任何正式outer。确定性异常指纹为：

```text
D92ParetoDistillHard11RunnerError: fit audit ParetoDistill cross-group quantum receipt drift
```

三种LEO场景的fit audit均记录：`active=false`、`fallback=true`、`fallback_reason=deployment_full_head_byte_exact`、two-state fit=4、实际FULL+BLOCK solve=2、candidate affine SHA=`null`、E0/head-state SHA闭合、`deployed_head_state_closure_pass=true`；跨组change/quantum为`null`且`quantum_pass=false`。query truth/fit/update/selection/role-oracle/class-quota/global-reassignment访问全部为`false`。由于smoke未通过active/非fallback/quantum硬门，launcher未创建8个shard。

### 9.4产物与回收

远端产物完整保留。远端与本地回收树的文件数、总字节和内容树SHA一致：

|树|文件数|总字节|内容树SHA256|
|---|---:|---:|---|
|`source_snapshot`|1,329|70,658,686|`e2972c82d5ad199e3312c9c3d641d474db653f23b9db707ed327f7a791244c34`|
|`logs`|6|1,735|`93354b7353790cb88782ae44d795fd2053db531ccaaabcd9e560f36908039d9b`|
|`output`|13|761,200|`0169bdb759abc0f29aa917330c9436b904dff5c6d3603d911733527bc1f3237b`|

本地回收根为`E:\\type10-7\\local_artifacts\\d92_e0_full_block_pareto_distill_hard11_20260812_v1`。正式矩阵计数为0/11 job receipt、0/22正式prediction/COMMIT/fit/resource、0/11 score、0/8 shard summary。manifest引用的11份历史truth sidecar已回收到`truth_sidecars/jobs/<outer>/offline/scorer/truth_sidecar.json`，11/11 SHA匹配，共5,306,510字节。`matrix_manifest.json` SHA256=`26ad9a51a31feca4e949b17749d8d17813c7d456815538848ef40a4279e5b9e4`。最终远端无匹配run进程、GPU无compute process；本地无SSH/SCP残留。

## 10.主分析与唯一裁决

### 10.1科学解释

runner抛出的“quantum receipt drift”是硬门的外部表现，不是根因。根因是唯一候选在真实K10 checkpoint的clear、low-elev和rain三场景中，连续优化后的头经D42编码/解码后都与E0 FULL完整头byte-exact；因此候选没有任何可部署的非零改变，按冻结规则精确回退E0。此时change和quantum不适用，候选SHA为空，不能将E0 fallback冒充新方法性能。

这条证据直接否证了本路线的可部署性。继续放宽量化门、扫描方向强度、选择其他smoke outer或让fallback进入正式矩阵，都会改变预注册方法或用target结果选择候选，均被禁止。本run不重启，完整Hard10也不补跑。

### 10.2指标与DA/REG状态

由于shard与truth-side scorer均未启动，新候选没有任何合法性能指标。历史E0与原D92只保留为研发背景，不与本run的smoke诊断拼接成伪同排结果。

|方法|H|old BA|`c_old_acc`|old floor|seen-new|forgetting|new→old|old→new|证据状态|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|E0_FULL_ONLY|73.3472%|74.8611%|74.8611%|44.8333%|72.0333%|12.9167%|15.0417%|15.3333%|历史Target125 artifact在冻结Hard10行上的同排均值|
|原D92|74.1673%|76.1389%|76.1389%|48.5000%|72.3750%|11.6389%|15.1000%|14.6944%|历史retry2 artifact在同一Hard10行上的同排均值|
|Pareto Distill|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|真实K10 smoke三场景均D42-byte-exact回退；0个正式score|

|状态|本run可报告结果|
|---|---|
|`DA0_REG0`|未运行；new类指标与H按定义为`N/A`|
|`DA1_REG0`|仅生成truth-free技术artifact；未连接scorer，性能`N/A`|
|`DA0_REG1`|未运行，性能`N/A`|
|`DA1_REG1`|尝试的候选精确回退E0；未连接scorer，性能`N/A`|

因此不存在10 outer×3 scene性能表，也不存在receiver、K、new-count、旧类或双向混淆分解。缺失是由shard前科学无效门造成的预期结果，不是待补数据。

### 10.3资源诊断边界

三种smoke场景的注册wall分别为284.454ms、170.740ms和280.772ms，中位数280.772ms、诊断p90为283.718ms；峰值增量内存最大15,040,512 bytes。fit计数为two-state 4、实际2，相对D92 K10代理`8×(K+1)=88`减少95.45%；query MAC为3,168，发布state为8,583 bytes，但二者对应exact-E0 fallback状态。由于仅一个outer且候选未激活，这些数字只能作为技术诊断，不能替代冻结Hard10资源裁决；它们也没有提供继续该路线的依据。

### 10.4裁决

唯一裁决：`REJECT_ROUTE`。

理由不是八项中某一项下降，而是方法在真实D42部署态没有形成候选：三场景全部回到E0，无法满足“严格优于E0”的必要前提，也不具备进入完整125的资格。下一轮若继续研发，必须更换为直接在D42离散码空间构造非零、support受约束更新的新方法家族；不得对本候选做强度扫描、量化容差放宽或同run重试。
