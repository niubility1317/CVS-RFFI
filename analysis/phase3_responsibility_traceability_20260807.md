# Phase3阶段职责最终修订追踪表

版本：2026-08-07  
目标分支：`codex/phase3-responsibility-20260807`  
基线提交：`f9050cfa45cb5e8dd3d37181bf4ccf1683084b29`  
目标来源：`E:\codex\home\attachments\09a65902-d4a2-49f7-966e-4b03ed53ba09\goal-objective.md`

执行模式：`GOAL_MODE=ACTIVE`

当前状态：`DESIGN_DRAFT`（`FEASIBILITY_REVIEW`已完成，等待独立交叉审查后才能进入`DESIGN_FROZEN`）

完成边界：职责文档、代码实现、本地验证、独立`P0/P1`审查、N607非覆盖运行、完整同排artifact与证据受限结论全部闭环后，才允许把目标标为完成。

禁止越级：`GOAL_MODE=ACTIVE`不等于候选已冻结、实现已验证或性能目标已达成；当前不得发布N607实验。

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
| P1-04 | 一 | Phase1导出开放世界就绪特征提取器、基础几何、半径/能量/尾部先验、质量/域不确定性和不可变bundle | bundle builder/schema/docs | pending | real-checkpoint no-query smoke、schema test | bundle不等于完成真实拒识 |
| P1-05 | 二 | 建立TX互斥的`source_known_train_tx`、`source_known_validation_tx`、`source_proxy_unknown_tx`，proxy unknown物理样本完全排除训练 | 数据split/builder、tests | pending | TX集合/物理ID互斥测试 | 不允许receiver/channel/SNR伪unknown |
| P1-06 | 二 | 支持对比、margin、energy、EVT、held-TX OE、`z_id/z_dom`解耦、半径/协方差、梯度冲突和分层解冻的开放候选空间 | 设计文档、候选配置 | pending | feasibility review、method lock | 不是要求一次候选堆叠全部机制 |
| P1-07 | 二 | Phase1候选五项窄晋级门：不崩溃、known跨接收机不明显退化、floor不严重下降、proxy unknown正信号、真实checkpoint可导出bundle | evaluator、报告模板、release gate | pending | 同排指标、bundle smoke | 门槛中的“明显/严重/明确”由活动方法设计量化 |
| LOCAL-01 | 三 | 每个节点从同一冻结或合法适配后的Phase1 extractor输出`z_id`、`z_dom`、`q`、`d_class`、`e_unknown`、`p_local` | 本地证据schema/extractor | pending | schema和真实checkpoint smoke | 每个字段需来源与形状定义 |
| LOCAL-02 | 三 | 本地证据先不可变封存，再进入协同；单节点输出仅为registered/unknown/defer | artifact writer/validator | pending | hash/immutability和decision enum tests | scorer不得回流 |
| LOCAL-03 | 三 | 单节点不得读取query真值、真实role、真实batch构成、类别配额和独立scorer结果 | predictor API、negative tests | pending | zero-fit/zero-update/zero-selection tests | 与`p2_min_v1`逐样本边界对齐 |
| P3-01 | 四 | 报告融合位置；协同输入只能来自已冻结本地证据及合法可见性、时空、频率/波束、轨迹和anonymous-track状态 | Phase3 schema/runner/docs | pending | manifest验证、forbidden input tests | 外部确权证据不能直接更新Phase2 predictor |
| P3-02 | 四 | 显式处理接收机差异、信道/SNR差异、缺失/延迟、冲突、同一事件、多过境匿名实体和相关证据去重 | fusion core、tests | pending | synthetic invariance/fault tests | 高风险核心项 |
| P3-03 | 四 | 正式方法必须有真实节点交互；平均、投票、最高置信节点仅作为基线 | fusion core、ablation configs | pending | baseline与正式方法结构审计 | 方法尚未冻结 |
| TASK-01 | 5.1 | 协同unknown拒识以`unknown_false_accept_rate<=5%`和`unknown_safe_rejection_rate>=95%`为目标；registered被reject/defer按身份错误计数 | scorer、report schema | pending | hand-calculated fixtures、same-row metrics | 不允许全拒绝取巧 |
| TASK-02 | 5.2 | 协同旧类适应比较独立、共享平均、质量加权和完整协同域状态；K10注册后old acc≥92%、最低old acc≥88% | adaptation/fusion matrix、scorer | pending | 同一row四状态指标 | 需沿用`DA0_REG0`等四状态命名 |
| TASK-03 | 5.3 | unknown观测只能关联为`anonymous_entity_id`，不能直接成为新类 | track store/association/tests | pending | 生命周期状态机测试 | 不等于语义身份 |
| TASK-04 | 5.4 | 可信确权输出候选物理身份、证据来源/独立性、冲突、置信度、有效期和`registration_authorized` | authorization credential schema | pending | fail-closed schema tests | RFFI只是多源证据之一 |
| TASK-05 | 5.5 | 授权后重新采集K个独立物理事件；历史unknown query不得转support；再交Stage2-C统一竞争 | Phase3→Stage2-C bridge、tests | pending | lineage/physical-ID negative tests | 新support将产生新`split_id`并按协议验证 |
| COUNT-01 | 六 | 代理实验固定评价`N_sat∈{1,2,3,4,5}`，其中1为本地基线、2为最低协同点 | matrix/config/runner | pending | matrix coverage test | 不从结果挑有利子集 |
| COUNT-02 | 六 | 一个`emission_event_id`可对应多个`satellite_reception_id`，跨节点接收仍只计1 shot | data/evidence schema、grouping code | pending | exact-K/event-count fixtures | 必须核对现有数据能否绑定同一事件 |
| COUNT-03 | 六 | 非同步多接收机数据只能声明“多接收节点代理协同”，不得声明真实在轨同步多星验证 | docs/report validator | pending | claim-lint tests | 当前WiSig/ManySig预计属于代理数据 |
| ARCH-01 | 七 | 系统分为共享Phase1本地extractor和部署期Phase3协同推理器，两层接口明确 | architecture docs/schema | pending | interface smoke | 不把Phase3塞入Phase2 predictor |
| ARCH-02 | 七 | 完成A原基座+单节点、B新Phase1+单节点、C原基座+协同、D新Phase1+协同的同输入消融 | ablation matrix/scorer | pending | A/B/C/D覆盖和输入parity | 计算`B-A`、`C-A`、`D-B-C+A` |
| N607-01 | 八 | Phase1候选八卡并行；本地证据并行提取后缓存；不同`N_sat`与子集复用缓存，不为组合重复运行backbone | launcher/report | pending | dry-run plan、cache lineage | 未进入发布阶段 |
| N607-02 | 八 | Luna/max仅负责冻结方案的发布、调度、监控和artifact回收；主Agent负责方法与结果决策 | runner handoff/report | pending | handoff完整性审计 | 仅在正式N607 release时执行 |
| CLAIM-01 | 九 | 最终阶段结论必须使用附件指定的职责表述，并同时标明当前实现/证据状态 | `docs/项目介绍.md`；最终报告 | verified | 附件指定段落逐字落入“阶段职责结论”；下一段限定代理数据和未完成状态 | 最终报告仍须复用并结合届时证据状态 |
| EVID-01 | 全文 | 所有实现进入Git，经过focused tests、独立P0/P1审查、非覆盖run ID、完整日志和同排artifact后才可晋级 | Git/tests/reports | pending | commit、test、review、artifact receipts | landing/completion/performance三者分开 |

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

## 6.可行性结论（20行内）

1.文档职责修订已由提交`93e67771`封存；代码和性能仍未完成。
2.现有协同评估器支持1到receiver总数的预算、缺失组统计、固定/渐进/自适应融合和unknown拒识。
3.现有scorer会把registered query的reject/defer计为身份错误，不能通过全拒绝抬高known accuracy。
4.R8_SHELL真实回收证据含2200行、5个receiver、1113个代理event。
5.这些event的观测节点数分布为1节点311、2节点541、3节点237、4节点24、5节点0。
6.该artifact使用`receiver_domain_ranked_by_role_tx_scenario`，不是物理same-event对齐，只能作多接收节点代理协同。
7.当前`event_id`编码role/true label，融合与scorer共处同一函数；这是必须先修复的P0隔离缺口。
8.当前证据行没有独立`emission_event_id`和`satellite_reception_id`，也没有不可变本地证据schema。
9.现有Phase1 runtime只导出normalized `z_id`和old logits，组件另有类/域中心及radius。
10.主模型实际存在`z_dom`，所以扩展bundle v2可行，但`q`、`e_unknown`和本地决策口径仍需冻结。
11.现有Phase1 proxy loss按batch轮换已知label，并不等于TX级全局互斥`source_proxy_unknown_tx`。
12.现有feature exporter具备proxy TX与source/target集合交叠拒绝，可复用为全局split检查基础。
13.现有协同指标含old/new逐类accuracy、floor、unknown FAR/reject和实际receiver直方图。
14.当前协同路径缺少`H_old_new`、`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`、A/B/C/D和difference-in-differences。
15.anonymous entity状态、证据独立性、可信确权凭证和授权后fresh-K桥接尚无可达实现。
16.实施应新建独立Phase3入口，复用已验证融合/计分原语，不继续膨胀带truth的Phase2巨型评估函数。
17.第一组实现只冻结schema、opaque ID、predictor/scorer隔离、单节点/简单基线及negative tests；科学融合候选另行评审。
18.当前数据可运行含缺失节点的`N_sat_deployed=1..5`代理矩阵，但不得声称5节点same-event融合。
19.严格同步多节点主张需要新的物理event绑定数据；该缺口不阻塞代理方法研发，但阻塞真实多星同步结论。
20.在focused tests、真实checkpoint smoke和独立P0/P1审查前，不发布N607实验。
