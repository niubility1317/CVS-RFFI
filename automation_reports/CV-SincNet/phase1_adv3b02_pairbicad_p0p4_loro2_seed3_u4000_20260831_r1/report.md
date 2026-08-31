# PairBiCAD P0–P4正式矩阵报告

## 当前状态

- 状态：`RUNNING`。
- Run ID：`phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1`。
- 方法范围：`P0/P1/P2/P3/P4`；P5–P9不属于本run。
- source数据：`Dataset_WigSig/ManySig.pkl`；训练day1/day2/day3。
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- folds：1、8；seeds：392001、392002、392003；总计30行。
- 训练预算：每行4000 optimizer updates；物理batch48，clean/LEO拼接网络batch96。
- GPU：N607 GPU0–7；每GPU最多2个本run训练进程。
- 目标、Phase2、support、query、truth：禁止访问。
- 每行预期artifact：final checkpoint、runtime/audit、metrics、clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`、资源遥测、`ARTIFACTS_COMPLETE`或技术失败marker。
- 允许技术停止：错误candidate/fold/receiver/day/seed/update、target/Phase2越权、输出覆盖、错误release/CWD、命令无法运行、无合法checkpoint/四场景闭合、同一确定性pre-prediction异常至少2行或进程归属不清。
- 低性能不得停止、重启、热补丁或选择性重跑。
- 兼容性裁决：旧D0–F3保持`concat_sat_ce_only/E80/0.68/0`；新P0–P4预登记`ce_only_plus_pair_selfsup`，卫星TX标签仍只走CE，但P3/P4允许无标签pair/VICReg/delta自监督。该候选级覆盖来自当前报告实现授权，不改变`LEO_WEAK`、source-only或L/U信息权限。

## 报告到实现追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PB-P0-01|第3、15、16节|每update clean/LEO严格成对、单次拼接前向|config、trainer、train_ssdg|verified|单前向测试+真实checkpoint smoke|物理48/网络96|
|PB-P0-02|第5节|有标签clean/LEO TX CE，卫星权重0.5→1.0|trainer、config|verified|loss调用审计|保持concat_sat_ce_only标签边界|
|PB-P1-01|第4、10节|z_dom因素化为z_r/z_d/z_c/z_int|heads、trainer|verified|维度、有限值、隔离与反传测试|z_int=24|
|PB-P1-02|第14节|shared-stem gradient firewall|trainer、gradients|verified|firewall应用计数+smoke|scale=0.05|
|PB-P2-01|第5节|有标签类条件receiver/day/channel DANN|heads、trainer|verified|CAdv调用和梯度测试|不使用U硬标签|
|PB-P2-02|第7节|z_dom主因素反向TX对抗，排除z_int|heads、trainer|verified|TXAdv输入隔离测试|仅L|
|PB-P3-01|第8节|L/U pair identity hinge|pair、trainer|verified|有效数、有限值与反传|epsilon=0.05|
|PB-P3-02|第8节|U clean/LEO预测JS|pair、trainer|verified|U标签不可达测试|不产生伪标签|
|PB-P3-03|第8节|projector VICReg防坍塌|pair、trainer|verified|variance/covariance及batch<2测试|projector=128|
|PB-P4-01|第9节|identity delta信道对抗|pair、trainer|verified|delta adversary审计|L+U|
|PB-P4-02|第9、10节|domain delta解释信道，rx/day稳定|pair、trainer|verified|channel预测与pair损失|L+U|
|PB-P4-03|第9节|delta norm hinge|pair、trainer|verified|半径与梯度测试|delta=0.25|
|PB-GRAD-01|第14节|动态对抗梯度剂量与日志|gradients、trainer|verified|rho和runtime审计|adv0.15–0.25|
|PB-PROTO-01|第2节、项目协议|L/U信息权限和source-only|train_ssdg、tests|verified|协议负测+真实smoke|禁止target/query|
|PB-COMPAT-01|新报告与旧默认冲突|P0–P4候选级扩展，旧候选不变|config、train_ssdg、runtime|verified|旧候选回归+known-default严格恢复|不得把新模式冒充旧默认|
|PB-MATRIX-01|第21、22节|P0–P4×2fold×3seed完整矩阵|launcher、shell|verified|30行dry-run|U4000|
|PB-EVAL-01|指标章节、AGENTS|clean+三种LEO_WEAK逐行闭合|launcher|verified|artifact闭合测试+smoke|不能只报aggregate|
|PB-DEFER-01|最终推荐|P5 soft-U CDAN|无|deferred|N/A|P4多seed source证据后再议|
|PB-DEFER-02|最终推荐|P6 XDC、P7 margin-tail|无|deferred|N/A|不混入P4首轮|
|PB-DEFER-03|第17、18节|P8 hard-LEO、P9 SWAD、Fishr/MixUp|无|deferred|N/A|后续独立消融|

## 版本与发布

- Git分支：`codex/phase1-pairbicad-p4-20260831`。
- 设计：`docs/superpowers/specs/2026-08-31-pairbicad-p4-design.md`。
- 实现计划：`docs/superpowers/plans/2026-08-31-pairbicad-p4.md`。
- 配置注册commit：`f98b00903f11f22f2cb145f9104acefce0c87a57`。
- 因素投影与pair目标commit：`61e4b8117d252602360aeb293a0bc2c40af3670b`。
- 30行矩阵launcher commit：`cadfdee0e7dedb2a88dbb26ce8e812ef17130458`。
- 16L+32U单前向训练commit：`03e328f44001f9a3977da647ad4f2f494f15d78e`。
- 真实checkpoint smoke修复commit：`90e05291d5e6d0ee977583489b6f381a6589ce17`。
- release代码HEAD：`85989ed4d9bc15262b184e1d7b046780916843fa`；归档前远端Git分支OID一致；后续报告提交不改变release代码。
- release归档：本地`E:\type10-7\release_archives\phase1_pairbicad_p0p4_85989ed4.tar.gz`→N607`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_85989ed4.tar.gz`。
- release SHA256：本地/远端均为`044109cd4c0c7dfd417b788b15081ed7874a1894efb8533d6bc3aedfb26d8649`。
- 解压根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_85989ed4`。
- N607账户：`szu2070436088`；未使用管理员账户。
- N607预检：项目根、Python、ManySig存在；`/home`可用7.3TB；GPU0–7无compute process，显存各1MiB；run/release目标路径发布前均不存在。
- 远端编译：`train_ssdg.py`、PairBiCAD模块及launcher全部通过`py_compile`。
- 远端dry-run：30/30行，P0–P4×fold1/8×seed392001/392002/392003，全部U4000、day1/2/3、source-only；GPU队列4/4/4/4/4/4/3/3。
- 正式入口：`code/scripts/launch_phase1_pairbicad_p0p4_n607_20260831.sh`；通过环境变量固定上述release根，`--max-jobs-per-gpu 2`。

## 已落地实现

### P0：严格成对单前向基线

- 每个optimizer update从`L_s`取16条、从`U_s`取32条，共48条物理IQ；对全部48条生成同样本LEO弱信道视图。
- 网络输入按`[48 clean,48 satellite]`拼接为96条，仅调用双骨干模型一次；运行时固定记录`model_forward_count=1`和`extra_forward_count=0`。
- clean有标签行执行TX CE；同一16条对应的satellite行执行TX CE。U_s不读取TX标签，也不生成伪TX标签。
- satellite CE权重按候选配置从0.5调度到1.0；旧D0–F3仍保持E80、0.68、无pair自监督。

### P1：接收机/日期/信道因素分解

- 新增`FactorizedDomainProjector`，把`z_dom`投影为`z_r/z_d/z_c/z_int`；因素维度与交互维度独立，`z_int=24`。
- receiver/day/channel域头读取对应因素；shared stem上的域梯度通过gradient firewall缩放，防止域任务直接覆盖身份主干。

### P2：有标签条件对抗

- 对L_s启用TX条件化receiver/day/channel DANN。
- 对`z_dom`主因素启用TX反向对抗；`z_int`不进入该TX对抗路径。
- U_s不参与需要TX标签的条件对抗。

### P3：无标签pair一致性与防坍塌

- L_s/U_s均可参与identity pair hinge。
- 仅对U_s的clean/LEO预测执行对称JS一致性，不读取其TX真值。
- 128维pair projector执行VICReg invariance/variance/covariance；batch小于2时返回有限、图连接的零项，避免NaN。

### P4：pair-delta信道建模

- 对`delta_id=z_id_sat-z_id_clean`施加信道对抗，使身份变化难以解释信道。
- 对domain/channel delta执行信道预测与channel equivariance，使环境分支吸收信道变化。
- 增加receiver/day pair stability及`delta_radius=0.25`的norm hinge。
- 动态对抗剂量和每组件raw/weighted loss、有效样本数、skip reason、梯度有限性进入`pairbicad_runtime`。

## 本地验证证据

- 配置TDD：111/111通过；旧D0–F3/V1配置语义保持不变。
- 因素/pair TDD：30/30通过；当时完整模块回归279/279通过。
- launcher TDD：8个预期RED后22/22通过；dry-run精确生成30行，GPU总队列分布为4/4/4/4/4/4/3/3，活跃并发由每GPU2个信号量限制。
- 训练接入TDD：7个预期RED后新增9/9通过；主对话随后运行完整模块测试296/296通过。
- smoke修复后完整模块测试297/297通过，`py_compile`和`git diff --check`通过。
- 独立审查：配置阶段发现E80覆盖、U4000/batch48和旧runtime兼容3项P1，均由训练入口/launcher修复；因素/pair提交CLEAN；训练提交未发现直接P0/P1。

### 真实CORE90 checkpoint no-query smoke

- checkpoint：`E:\type10-7\local_artifacts\bicad_xr_real_checkpoint_smoke\best_joint_safe_ssdg.pth`，8,582,116字节。
- r1按旧smoke驱动失败：缺少U_s batch，状态`FAILED/NO_PERFORMANCE_RESULT`，artifact保留。
- 修复后的r3：`PASS`；candidate=P4，16L+32U，physical48，network96，单次前向optimizer step完成。
- 严格重建：`missing_keys=[]`、`unexpected_keys=[]`、`shape_mismatches=[]`。
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`全部输出`[48,6]`有限logits。
- `unlabeled_tx_access=false`，`target_access=false`，`phase2_access=false`，`support_access=false`，`query_access=false`，`truth_access=false`。
- 该smoke只证明技术可运行，不构成P0–P4性能结论。

## 正式矩阵分析规则

- 每行训练与四场景artifact闭合后才计为`ARTIFACTS_COMPLETE`；训练结束但缺任一场景不算完成。
- 先完整核对30/30行的U4000、checkpoint严格重建、source fold/seed/day与运行时计数。
- 只使用source LORO的`V_select`比较P0→P4；同row同时报告clean、三种LEO、LEO mean、LEO floor、clean代价和资源成本。
- 禁止把不同row的单项最大值拼成虚构冠军；禁止使用目标接收机、Phase2或query结果反馈选种、调参、重训或选择性重跑。
- P5–P9继续保持deferred；本run只回答P0–P4逐级机制是否在新数据包Phase1 source-only划分上有效。

## N607正式启动回读

- 启动时间：2026-08-31 13:49 CST后；普通账户`szu2070436088`。
- dispatcher PID：`2807256`，PPID=1；cmdline精确绑定release`phase1_pairbicad_p0p4_85989ed4`、本Run ID、P0–P4、fold1/8、3个seed、U4000和`--max-jobs-per-gpu 2`。
- 第一波worker：16个直属主训练进程；每GPU恰好2个，GPU0–7利用率84%–91%，显存约1.62–1.77GiB/卡。
- 16个worker CWD全部为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_p0p4_85989ed4`；输出分别绑定本run的唯一row目录。
- 第一波覆盖P0全6行、P1全6行和P2前4行；其余14行在dispatcher内排队，不会突破每GPU2个活跃训练进程。
- `plan.json`已存在；16个`train.log`已建立。启动早期日志为0字节，但GPU计算、进程归属和cmdline均持续有效；该trainer在epoch/update汇总前不保证stdout增长，不能据此停止。
- 启动回读未发现`TECHNICAL_FAILURE`或`ARTIFACTS_COMPLETE`marker；当前没有性能结论。
