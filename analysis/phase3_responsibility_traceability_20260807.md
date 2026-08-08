# Phase3阶段职责最终修订追踪表

版本：2026-08-07  
目标分支：`codex/phase3-responsibility-20260807`  
基线提交：`f9050cfa45cb5e8dd3d37181bf4ccf1683084b29`  
目标来源：`E:\codex\home\attachments\c75febfd-60b9-42bb-9825-a0b3b9eda0bb\goal-objective.md`

执行模式：`GOAL_MODE=ACTIVE`

当前状态：`IMPLEMENTING`（Phase1真实checkpoint不可变bundle、truth-free本地证据及预标签物理binding桥已完成本地技术闭环但未性能晋级；Phase3正式性能矩阵等待真实采集receipt）

完成边界：职责文档、代码实现、本地验证、独立`P0/P1`审查、N607非覆盖运行、完整同排artifact与证据受限结论全部闭环后，才允许把目标标为完成。

禁止越级：`GOAL_MODE=ACTIVE`不等于性能目标已达成。Phase1快速实验已经按独立P0/P1审查发布；Phase3在合法事件绑定前不得发布正式性能矩阵，旧R8只保留非部署诊断边界。

## 1.状态口径

只允许使用`pending`、`implemented`、`verified`、`deferred`、`rejected`和`blocked`。代码或文档中出现同名概念不等于实现；只有可达路径、测试和同源artifact共同成立后才能标记为`verified`。

当前阶段在已完成职责、现状和缺口审计的基础上进行并行方法设计与独立交叉审查。未形成冻结候选、实验矩阵、独立P0/P1审查和本地验证前，不发布N607实验。

## 2.基线事实

- 根目录`E:\type10-7`不是Git仓库；科学源文件`项目.md`的修改必须镜像到本分支的`docs/PROJECT_PROTOCOL.md`并留下交接证据。
- 公开仓库原工作树包含与本目标无关的暂存和未跟踪材料；本目标使用独立worktree，避免混入或覆盖用户改动。
- `docs/项目介绍.md`已把unknown观测到Stage2-C注册的生命周期标为Phase3计划，但`docs/PROJECT_PROTOCOL.md`仍把Phase3定义为`unknown rejection备用扩展`。
- `code/evaluation/collaborative_open_set_qknn_eval.py`存在多接收机open-set评估、`strict_same_event_collaboration`和若干融合策略；其是否满足本目标的事件语义、相关性控制、缺失节点、A/B/C/D消融和正式指标仍待逐项验证。
- 历史R8/R9/R10协同评估曾发现严格same-event exact-K与当时数据结构不兼容；旧结果只能作为实现线索和负面证据，不能证明新Phase3目标已完成。

## 3.需求追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| GOV-01 | 一、七、九 | Phase1正式定义为地面开放世界就绪表征学习；Phase3正式定义为部署阶段多接收节点协同推理；Stage2-C只接收授权后的新support | `项目.md`；`docs/source_controls/PROJECT_PROTOCOL.full.md`；`docs/PROJECT_PROTOCOL.md`；`docs/项目介绍.md` | verified | 根源文件与完整镜像规范化内容一致；必需术语、旧语义清除、围栏、链接、排版和`git diff --check`均通过 | 明确当前尚未完成Phase3闭环，不把阶段定义写成性能完成 |
| P1-01 | 一 | Phase1表征同时覆盖TX可分性、跨接收机稳定性、LEO弱信道鲁棒性、紧致/margin、低过置信、可拒识性及后续old适应/new注册能力 | 方法设计；Phase1训练入口；bundle schema | pending | 可达训练路径、导出字段、focused tests | 需区分研发目标与已有性能 |
| P1-02 | 一 | Phase1可训练前端、分支、卷积、`z_id/z_dom`、normalization/projection/fusion及prototype/radius/energy/uncertainty输出 | Phase1配置、模型、loss、exporter | pending | 配置到调用链、checkpoint smoke | 不预先假定全部模块已存在 |
| P1-03 | 一 | Phase1禁止target query真值、确认unknown回流、多星消息、anonymous track、运营身份输出及proxy→真实unknown冒充 | 协议、训练入口、tests | pending | protocol-negative tests | P0边界 |
| P1-04 | 一 | Phase1导出开放世界就绪特征提取器、基础几何、半径/能量/尾部先验、质量/域不确定性和不可变bundle | bundle builder/schema/docs | implemented | 双真实checkpoint runtime parity=0；source-only calibration；content root `97e31efa...`；2400行smoke | bundle技术闭环，不等于真实拒识性能晋级 |
| P1-05 | 二 | 建立TX互斥的`source_known_train_tx`、`source_known_validation_tx`、`source_proxy_unknown_tx`，proxy unknown物理样本完全排除训练 | 数据split/builder、tests | verified | `test_phase1_tx_partition.py`、checkpoint receipt、四臂同TX划分与冻结后审计 | 4/1/1开发screen已闭环；不允许receiver/channel/SNR伪unknown |
| P1-06 | 二 | 支持对比、margin、energy、EVT、held-TX OE、`z_id/z_dom`解耦、半径/协方差、梯度冲突和分层解冻的开放候选空间 | 设计文档、候选配置 | pending | feasibility review、method lock | 不是要求一次候选堆叠全部机制 |
| P1-07 | 二 | Phase1候选五项窄晋级门：不崩溃、known跨接收机不明显退化、floor不严重下降、proxy unknown正信号、真实checkpoint可导出bundle | evaluator、报告模板、release gate | implemented | GeoSat Lite四臂、held-TX审计、双读出与真实bundle同排诊断 | 导出门通过；bundle proxy/held FAR=71.75%/66.75%，无开放世界性能晋级 |
| LOCAL-01 | 三 | 每个节点从同一冻结或合法适配后的Phase1 extractor输出`z_id`、`z_dom`、`q`、`d_class`、`e_unknown`、`p_local` | 本地证据schema/extractor | verified | 真实bundle smoke 2400行；evidence receipt无role/truth；content root绑定 | 当前证据为逐reception proxy，不主张same-event |
| LOCAL-02 | 三 | 本地证据先不可变封存，再进入协同；单节点输出仅为registered/unknown/defer | artifact writer/validator | verified | 精确allowlist；`z_id/z_dom/d_class/e_unknown`必需；raw/cache/member负测；canonical SHA、tamper负测、单节点identity | scorer不得回流 |
| LOCAL-03 | 三 | 单节点不得读取query真值、真实role、真实batch构成、类别配额和独立scorer结果 | predictor API、negative tests | verified | forbidden-field负测；prediction manifest明确`truth_sidecar_opened=false` | 与`p2_min_v1`逐样本边界对齐 |
| BIND-01 | 六；`项目.md`7.3 | 将采集系统在标签可见前生成的物理绑定receipt一对一连接到既有truth-free逐reception证据，形成最终`verified_physical`本地证据；不得由TX、role或truth推导event | `code/cvsrffi/phase3_care_poe.py`；`code/scripts/phase3_bind_physical_evidence.py`；focused tests | implemented | 输入/绑定全覆盖、双ID唯一、event内node唯一、hash tamper、truth/role禁入、输出直入CARE-PoE；39项focused tests通过 | 本地接口闭环；N607库存审计未发现真实receipt，因此尚无正式数据运行 |
| P3-01 | 四 | 报告融合位置；协同输入只能来自已冻结本地证据及合法可见性、时空、频率/波束、轨迹和anonymous-track状态 | Phase3 schema/runner/docs | verified | G0 design、predictor/scorer隔离、N607 artifact | 外部确权证据未进入predictor |
| P3-02 | 四 | 显式处理接收机差异、信道/SNR差异、缺失/延迟、冲突、同一事件、多过境匿名实体和相关证据去重 | fusion core、tests | verified | late/missing/integrity/相关复制/顺序不变性及same-input负测 | 合成技术证据，不是实际在轨性能 |
| P3-03 | 四 | 正式方法必须有真实节点交互；平均、投票、最高置信节点仅作为基线 | fusion core、ablation configs | implemented | CARE-PoE与leader单节点A/B/C/D已可运行 | 真实节点性能等待合法物理事件数据 |
| TASK-01 | 5.1 | 协同unknown拒识以`unknown_false_accept_rate<=5%`和`unknown_safe_rejection_rate>=95%`为目标；registered被reject/defer按身份错误计数 | scorer、report schema | pending | scorer已验证prediction manifest/hash/行数、冻结矩阵完整覆盖及registered reject/defer计错；正式同排性能尚无合法输入 | 不允许全拒绝取巧；技术评分器完成不等于目标达成 |
| TASK-02 | 5.2 | 协同旧类适应比较独立、共享平均、质量加权和完整协同域状态；K10注册后old acc≥92%、最低old acc≥88% | adaptation/fusion matrix、scorer | pending | 同一row四状态指标 | 需沿用`DA0_REG0`等四状态命名 |
| TASK-03 | 5.3 | unknown观测只能关联为`anonymous_entity_id`，不能直接成为新类 | track store/association/tests | verified | anonymous无语义身份且非unknown拒绝生成负测 | 不等于语义身份 |
| TASK-04 | 5.4 | 可信确权输出候选物理身份、证据来源/独立性、冲突、置信度、有效期和`registration_authorized` | authorization credential schema | verified | 缺失/过期/冲突/非独立来源fail-closed测试 | G0使用外部credential fixture，不是现实授权服务 |
| TASK-05 | 5.5 | 授权后重新采集K个独立物理事件；历史unknown query不得转support；再交Stage2-C统一竞争 | Phase3→Stage2-C bridge、tests | verified | fresh-K唯一event/physical ID、历史unknown交叠负测与N607 receipt | 已生成新`split_id`，未执行正式Stage2-C性能矩阵 |
| COUNT-01 | 六 | 代理实验固定评价`N_sat∈{1,2,3,4,5}`，其中1为本地基线、2为最低协同点 | matrix/config/runner | verified | 60行N607预测完整覆盖A-D×N1-N5×3events | 不从结果挑有利子集 |
| COUNT-02 | 六 | 一个`emission_event_id`可对应多个`satellite_reception_id`，跨节点接收仍只计1 shot | data/evidence schema、grouping code | verified | `shot_count_all=1`与same-input reception绑定负测 | 合成fixture证明接口，实际数据仍需采集绑定 |
| COUNT-03 | 六 | 非同步多接收机数据只能声明“多接收节点代理协同”，不得声明真实在轨同步多星验证 | docs/report validator | pending | claim-lint tests | 当前WiSig/ManySig预计属于代理数据 |
| ARCH-01 | 七 | 系统分为共享Phase1本地extractor和部署期Phase3协同推理器，两层接口明确 | architecture docs/schema | verified | 独立Phase3模块和三个职责分离入口的N607 smoke | 未修改Phase2 predictor |
| ARCH-02 | 七 | 完成A原基座+单节点、B新Phase1+单节点、C原基座+协同、D新Phase1+协同的同输入消融 | ablation matrix/scorer | verified | event×node×reception同输入绑定；N1 A=C、B=D | 技术矩阵完整，正式效应值等待合法数据 |
| N607-01 | 八 | Phase1候选八卡并行；本地证据并行提取后缓存；不同`N_sat`与子集复用缓存，不为组合重复运行backbone | launcher/report | deferred | GeoSat Lite与12臂LOTO已并行；bundle runtime/feature缓存后只读build；v3合成矩阵复用15条每bundle evidence生成60行预测 | 尚无合法same-event数据，因此正式`N_sat`性能矩阵延期 |
| N607-02 | 八 | Luna/max仅负责冻结方案的发布、调度、监控和artifact回收；主Agent负责方法与结果决策 | runner handoff/report | verified | 唯一runner按release commit `5501990e`执行v3 CPU闭环，未改方法/矩阵、未重试或解读性能 | 5步exit0；21项artifact回收；连接/GPU清理 |
| CLAIM-01 | 九 | 最终阶段结论必须使用附件指定的职责表述，并同时标明当前实现/证据状态 | `docs/项目介绍.md`；最终报告 | verified | 附件指定段落逐字落入“阶段职责结论”；下一段限定代理数据和未完成状态 | 最终报告仍须复用并结合届时证据状态 |
| EVID-01 | 全文 | 所有实现进入Git，经过focused tests、独立P0/P1审查、非覆盖run ID、完整日志和同排artifact后才可晋级 | Git/tests/reports | verified | v3实现commit `77d9a0e8`、release commit `5501990e`；39 tests、P0=0/P1=0、21项artifact hash一致 | 只晋级技术闭环，不晋级性能 |

## 4.反向审计清单

- [ ] 每个来源要求都有状态。
- [ ] 每个`implemented`项都列出可达目标文件。
- [ ] 每个`verified`项都有实际运行的验证证据。
- [ ] 每个`deferred`、`rejected`或`blocked`项都有原因。
- [ ] 项目科学源文件、Git公开协议和项目介绍不存在Phase1/Phase3职责冲突。
- [ ] 当前能力、设计完成、技术执行和性能达标四种状态未被混写。
- [ ] 若数据不能证明同一`emission_event_id`跨节点绑定，所有结果均保留“多接收节点代理协同”标签。

## 5.验证记录

### 2026-08-07文档职责修订

- 根目录源文件：`E:\type10-7\项目.md`。
- Git完整镜像：`docs/source_controls/PROJECT_PROTOCOL.full.md`，CRLF/LF规范化后与根源文件逐字符一致。
- Git公开摘要：`docs/PROJECT_PROTOCOL.md`。
- 面向读者的路线说明：`docs/项目介绍.md`。
- 必需术语检查通过：`source_proxy_unknown_tx`、`z_id`、`emission_event_id`、`satellite_reception_id`、`anonymous_entity_id`、`registration_authorized`、`D-B-C+A`和“多接收节点代理协同”。
- 旧语义`unknown rejection备用扩展`和`Phase3未知类拒识是独立备用方向`在四份当前文档中均不存在。
- Markdown围栏数分别为24、24、10和28，均为偶数；两个公开文档的相对链接均存在。
- 中文标点后空格扫描无命中；`git diff --check`通过。
- 本阶段没有运行训练、连接N607或生成性能结果。

### 2026-08-08并行设计与独立初审

- 目标工具与本追踪表均显式标为`GOAL_MODE=ACTIVE`；当前生命周期为`DESIGN_DRAFT`。
- Phase1作者提出`P1-OWR-H`与接口回退`P1-OWR-0`；Phase3作者提出相关性感知`CARE-PoE`。
- 独立监督者初审结论为：职责文档`MERGE`，两个机制候选`REVISE`，旧R8评估器作为Phase3预测器及same-event证据`REJECT`。
- 用户进一步要求完整继承既有Phase1 open-world探索经验；主代理已回溯V31、DualGuard16、P0Closed8、CorePath8及P0/P1控制面，禁止重复post-hoc adapter、动态软门、local component并集、reject-all和单纯放大open loss路线。
- 主代理整合修订见`analysis/phase3_openworld_collaboration_design_20260808.md`的`REVISION_2`，其中冻结了proxy不回流、feature-head梯度可达性、`shared invariant core AND local density support`、known hard-core TPR前置门、v2 sibling bundle、双ID、预测/评分隔离、31节点子集、A/B/C/D与四状态的设计门。
- `ssr-gpu`中89项现有loss/协同原语测试通过；真实ADV3B02 checkpoint内存内反向审计证明`L_OW`对8个`id_backbone.cls_head`feature projection tensor产生非零梯度，一步更新改变`z_id`且禁训参数不变。
- 上述结果只关闭梯度可达性疑问；尚未运行候选实现测试、真实checkpoint v2 smoke或N607实验。

### 2026-08-08 Phase1快速实验闭环

- `phase1_geosat_lite_4arm_20260808_v1`以TX级4/1/1互斥划分完成四臂E120/120；独立审查`P0=0/P1=0`，完整metrics/log/checkpoint清单已回收。
- C的clean→LEO一致性相对A把LEO mean/floor提升`9.067pp/9.387pp`；B的known-only角度几何把source proxy FAR从53.50%降到38.25%；D的联合训练把proxy FAR恶化到79.25%，证明两项目标在同一路径上冲突。
- 四臂均未达到5% proxy FAR，故`NO_SINGLE_ARM_PROMOTED`；这些数值全部是source-held开发诊断，不是Phase3真实unknown。
- `phase1_dualreadout_disagree_20260808_v1`完成物理`sig_id`逐行绑定、预测/拒识职责解耦和独立`P0=0/P1=0`复核；两条N607 CPU评估均exit=0。
- 双读出把proxy FAR从B的38.25%降到37.50%、held-known FAR从21.25%降到19.75%，但source full accuracy下降3.50pp，超过预注册2pp门，结论为`REJECTED_KNOWN_GATE`。
- Phase1后续只保留C的类别证据及B/JS连续`e_unknown`研究价值，不再使用跨模型一致性硬拒识。
- Phase3正式性能实验仍被数据语义阻断：旧R8按role/true-label排名构组，缺少采集前生成的独立事件/接收ID；可保留非部署诊断，不可冒充same-emission或真实多星。

### 2026-08-08 Phase1真实bundle闭环

- B/C真实checkpoint先经CPU one-shot导出两个部署子图，batch 1/8/64 eager↔TorchScript parity最大误差均为0；训练期GradReverse和对抗头不进入runtime。
- source-only build使用`(TX,RX,day,sig)`全局作用域physical ID；真实NPZ全体2400行及source 1600行均唯一，裸`sig_id`仅913个唯一值，因此不能直接作为全局ID。
- bundle exact allowlist为两个runtime、`calibration.npz`和receipt；content root=`97e31efaac65cf02ef895b089ffb24ceae2633ffc45aa2ef604e4c3572cb25ff`，无checkpoint、raw IQ、role或truth。
- smoke覆盖2400行且800条proxy/target全部排除fit；emit receipt确认2400条逐reception证据、无role/truth、无same-event主张。
- source-held非部署诊断为known acc=92.5625%、min-class=87.25%、proxy FAR=71.75%、held FAR=66.75%；未达到开放世界目标，结论为`TECHNICAL_BUNDLE_COMPLETE / NOT_PERFORMANCE_PROMOTED`。
- 三轮回顾已拒绝继续做threshold sweep、hard disagreement、receiver/day对齐和同一`feat_joint`角度loss；下一候选只考虑C式LEO一致性加`id_feat_cls`几何，并先补足合法外部proxy TX证据。

## 6.可行性结论（20行内）

1.文档职责修订已封存；Phase1真实checkpoint bundle与2400条truth-free本地证据已完成，Phase3正式性能仍未完成。
2.现有协同评估器支持1到receiver总数的预算、缺失组统计、固定/渐进/自适应融合和unknown拒识。
3.现有scorer会把registered query的reject/defer计为身份错误，不能通过全拒绝抬高known accuracy。
4.R8_SHELL真实回收证据含2200行、5个receiver、1113个代理event。
5.这些event的观测节点数分布为1节点311、2节点541、3节点237、4节点24、5节点0。
6.该artifact使用`receiver_domain_ranked_by_role_tx_scenario`，不是物理same-event对齐，只能作多接收节点代理协同。
7.旧R8的`event_id`仍编码role/true label且融合与scorer共处同一历史函数；新CARE-PoE已用独立预测/评分入口关闭该技术缺口，但不能修复旧数据语义。
8.新`LocalEvidenceV3`已实现双ID、canonical seal、同输入跨bundle reception绑定和不可变本地证据schema；v3预测还必须核对外部binding sidecar/root，旧v2与R8证据行均不满足输入契约。
9.真实Phase1 v2 bundle已从B/C checkpoint导出`z_id/z_dom/q/d_class/e_unknown/p_local`，content root为`97e31efa...`；它只完成技术接口。
10.bundle source-held proxy/held FAR为71.75%/66.75%，未达到5%目标，也未形成相对C的开放世界正信号，因此不得性能晋级。
11.GeoSat Lite入口已用TX级全局互斥替代本轮batch轮换proxy语义；旧proxy loss仍只作历史实现，不进入该run。
12.现有feature exporter具备proxy TX与source/target集合交叠拒绝，可复用为全局split检查基础。
13.现有协同指标含old/new逐类accuracy、floor、unknown FAR/reject和实际receiver直方图。
14.当前协同路径缺少`H_old_new`、`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`、A/B/C/D和difference-in-differences。
15.anonymous entity、外部credential fail-closed和授权后fresh-K桥接均已可达；它们是技术fixture，不是现实运营授权证据。
16.独立Phase3 fixture/predict/score/lifecycle入口已实现，没有继续膨胀带truth的Phase2历史评估函数。
17.CARE-PoE已冻结并通过相关组代表、缺失/延迟、完整性失败、单节点identity和同输入A/B/C/D测试；科学性能尚未读取。
18.R8的truth-ranked分组不满足`proxy_unverified`输入，继续只作历史非部署诊断，不得声称5节点same-event融合。
19.严格同步多节点和正式性能主张需要新的物理event绑定数据；该缺口不阻塞技术闭环，但阻塞真实多星同步结论。
20.Phase3合成G0与真实Phase1 bundle均已完成focused tests、独立`P0=0/P1=0`和N607 artifact闭环；正式性能发布只等待不依赖真值的物理事件绑定输入及后续性能达标候选。

### 2026-08-08 CARE-PoE G0技术闭环

- 冻结实现位于`code/cvsrffi/phase3_care_poe.py`及三个职责分离入口；实现commit为`7c94afac`，release archive commit为`0242d354`。
- 独立初审发现跨bundle reception未逐项绑定、同相关组混合受新增记录影响和scorer重复行/非法role未拒绝；修复后独立复审为`P0=0,P1=0,14 passed`。
- N607 run `phase3_care_poe_g0_synthetic_20260808_v1`四入口各执行一次且exit=0，无retry；60条预测完整覆盖A/B/C/D×`N_sat=1..5`×3events。
- 全部prediction的`shot_count=1`；`N_sat=1`逐event满足A=C、B=D；predictor manifest明确`truth_sidecar_opened=false`。
- anonymous→外部credential→5个fresh independent event已生成`FRESH_K_READY_FOR_STAGE2_C` receipt，历史unknown event未转support。
- 16项小artifact与远端manifest逐项hash匹配；无异常指纹，评估进程、SSH/TCP22均清理，8卡保持空闲。
- 本run状态严格为`ARTIFACTS_COMPLETE_NO_PERFORMANCE_RESULT`；合成metrics不得用于unknown FAR、安全拒绝率或旧类准确率主张。

### 2026-08-08物理binding桥与当前库存

- 独立完成审计发现旧`LocalEvidenceV2`会接受未登记额外字段；新`LocalEvidenceV3`已改为精确allowlist，并把`z_id/z_dom/d_class/e_unknown`设为必需输入。旧v2拒绝加载，必须从源证据重新封存；`raw_iq`、`source_cache`、`member_ids`、truth和role负测均fail closed。
- 新增`cvs.phase3.physical_reception_binding.v1`与`phase3_bind_physical_evidence.py`：binding必须由采集系统在标签可见前生成，包含source reception引用、双ID、node、相关组、delay/deadline和canonical hash。
- 绑定器要求输入reception一对一全覆盖、同一binding receipt和base manifest、node一致、输出`satellite_reception_id`全局唯一且同event内node唯一；不得根据TX、role、truth、rank或score推导event。
- G0 fixture已不再手工伪造verified evidence，而是先生成truth-free proxy evidence及15条封存binding，再通过同一绑定器生成base/new verified evidence，最后运行60行A/B/C/D×N1-N5预测。
- `ssr-gpu`中Phase3与真实bundle相关39项focused tests、`py_compile`和新CLI `--help`通过；独立复审先发现scorer信任manifest自报预算轴的P0，定点修复为强制`[1,2,3,4,5]`并加入自洽截断负测后，复审结论为`P0=0,P1=0,ALLOW_N607_SYNTHETIC_G0_BINDING_V3=YES`。
- 正式predictor要求调用方同时提供binding JSONL与冻结binding root；scorer要求prediction manifest并验证预测hash、行数、`truth_sidecar_opened=false`及A/B/C/D×N1-N5完整覆盖，关闭仅靠行内自报hash或替换预测文件进入评分的旁路。
- N607普通账户只读库存审计未发现任何真实`emission_event_id/satellite_reception_id`资产或采集binding/provenance receipt；现有ManySig/ManyRx/ManyTx/SingleDay和Oracle X/Y文件不能据此形成`verified_physical`输入。
- 远端日志唯一命中只是字段名文本引用，`runtime_cache_audit.json`只有`physical_id_unique`计数；均不是event级资产。库存审计未读IQ/features/labels，未写服务器，结束时GPU、SSH和TCP22全部清零。

### 2026-08-08 LocalEvidenceV3合成技术发布

- 实现commit为`77d9a0e8471603fef60126ad57b822149c09f727`，release commit为`5501990e666cbd42a8eb3e6f89cf8e2bd8d5ab3a`；独立复审关闭为`P0=0,P1=0`。
- N607 run `phase3_care_poe_g0_binding_v3_20260808_v1`的fixture、binding root只读、predict、score和lifecycle五步各执行一次且exit=0，无retry。
- fixture含15条binding、3个event和每bundle 15条reception；binding root为`ca91e1fc2a12547c1935ba378ffd5eeb5c1034e9a1ffd582d9b7b44e8a8c5774`。
- prediction共60行，A/B/C/D各15、N1-N5各12、3个event各20，全部`shot_count=1`；N1的三个event均满足A=C、B=D。
- prediction manifest为`truth_sidecar_opened=false`且binding root匹配；lifecycle到达`FRESH_K_READY_FOR_STAGE2_C`。
- 21项小artifact逐项hash匹配，manifest SHA256为`89fe8230ab976bfab90908cbafd2e5d2a2ad09b72eec334abfeaf0d11e73d132`；无NPZ/checkpoint，错误指纹、残留进程、GPU任务和SSH/TCP22均为0。
- 本run仅证明v3接口、矩阵和生命周期技术闭环；N607仍无真实预标签物理binding，故正式unknown FAR、安全拒绝率和旧类性能保持未评估。
