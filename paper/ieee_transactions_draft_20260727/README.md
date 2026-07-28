# CVS-RFFI IEEE Transactions英文初稿

日期：2026-07-28（Starlink动机、场景、公式与多视角审稿整合）

状态：`EVIDENCE_LOCKED_INITIAL_DRAFT / NOT_SUBMISSION_READY`

本目录给出可继续修改和正式编译的IEEE Transactions双栏英文初稿。正文结构、摘要长度、关键词、编号引用、表格和参考文献样式按IEEE期刊稿组织；当前作者信息保持匿名。

## 文件

- `manuscript.tex`：规范主稿源文件。
- `references.bib`：45条已核验真实文献。
- `manuscript.pdf`：由Tectonic 0.17.0编译的审阅PDF。
- `claim_evidence_matrix.md`：逐项主张、证据和禁用表述。
- `citation_inventory.md`：45份引用PDF的来源、页数、解析结果和SHA-256清单。
- `experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`：覆盖Phase1、Stage2-A/B/C、两阶段交互、资源、连续注册和安全边界的全量消融预注册设计。
- `reviews/00_consolidated_revision_and_experiment_roadmap.md`：综合审稿结论、P0/P1补实验和主张解锁路线。
- `reviews/01_starlink_spaceborne_rffi_literature_audit.md`：Starlink与星上RFFI文献、作用及边界审计。
- `reviews/02_q1_reviewer_novelty_evidence_audit.md`：IEEE一区匿名审稿视角的创新与证据审计。
- `reviews/03_method_experiment_ablation_editorial_audit.md`：方法、公式、LEO模型、实验、消融与编辑审计。

## 2026-07-28修订

- 修正Phase1总损失公式为加权和，补齐所有加号。
- 重写引言与任务建模，明确当前研究空白是“未见目标接收机＋地面标签受限＋部署后K-shot旧类适应与新类注册＋星上存储/更新受限＋全类独立判决”的组合，而不是其中任一单独问题。
- 引入Starlink Gen2和Direct to Cell作为现实系统动机，明确本文方向是“地面发射机→星载接收机”，并限定本文没有Starlink IQ、波形、终端、前端或飞行处理器，不能称为Starlink验证。
- 说明星上RFFI的可辩护作用是辅助物理身份、设备溯源、干扰归因和少样本注册，不能替代密码学、证明恶意或单独触发自动封禁。
- 将Method重构为同一Framework下的两个对等阶段：Phase1和Phase2各自使用一个主小节及五个技术子模块，不再把Phase1压缩成概念性总损失。
- Phase1新增非对称物理多视图编码器、身份—接收机扰动分配、receiver-day门控半监督、尾部风险约束、身份保持源域挑战、200轮调度和不可变deployment bundle的实现一致说明。
- 修正Phase1域表征维度：`z_id=160`、`z_dom=160`；明确LEO一致性项权重为0，正文只保留实际产生梯度的LEO分类项。
- 区分当前统一Phase1协议划分`0.07/0.63/0.30`与ADV3B02历史审计实际使用的`0.10/0.70/0.20`：前者训练池内标签比例为10%，后者为12.5%；历史结果保留为内部方法证据，不能冒充尚未执行的当前协议正式复验。
- 用中性的clear-sky、low-elevation和rainy-link名称呈现三种LEO星地信道场景，论文正文不暴露协议内部场景键。
- 将任务物理链路与实际仿真生成顺序分开建模：当前方法是在已含地面传播与接收机效应的捕获IQ上施加post-capture residual overlay。
- 修正LEO公式中的接收机域、时延、Rician因子、传播场景和slant range符号冲突；公共参数表逐项标记applied、metadata-only和disabled。
- 明确当前capsule使用role-seeded ordered RNG且batch内共享tap delay；可复现权威是sealed capsule bytes，而不是physical ID与role seed的单样本独立重放。
- 新增7份Starlink、上行TT&C RFFI、卫星计算和星载RF指纹相关真实来源，并将PDF加入桌面引用文件夹。
- 组织三个独立子agent从文献、Q1审稿、方法/消融/IEEE编辑角度审查，并形成一份综合补实验路线图。

## 当前可写结论

1. Phase1的`ADV3B02_CORE90_SOFT_E200`属于source-only闭集DG证据：`overall=89.18%`、`strict UDU=84.89%`、`receiver_floor=75.55%`。
2. RTB-IDR（D92）完成125/125 matched诊断；在`K=10,new20`上相对matched control的注册后旧类准确率提高2.622个百分点、最低旧类准确率提高4.600个百分点、`H_old_new`提高0.964个百分点，但新类准确率下降0.653个百分点。
3. RTB-IDR状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。它不能被写成已达到部署目标或已解决Stage2。
4. WiSig/ManySig是地面代理数据，LEO信道属于仿真残余基带模型；不得写成真实在轨实验或完整星地链路预算。
5. Starlink仅用于说明大规模LEO上行接收、移动性和星上计算约束；不得写成本文已处理Starlink信号或已验证其安全性。

## 编译

标准IEEE LaTeX环境可执行：

```text
pdflatex manuscript
bibtex manuscript
pdflatex manuscript
pdflatex manuscript
```

本次审阅PDF使用便携Tectonic编译：

```text
tectonic -X compile manuscript.tex --keep-logs --keep-intermediates
```

正式投稿前应使用目标期刊推荐模板重新编译，并处理正文中的全部`[AUTHOR ACTION: ...]`。

## 投稿前硬缺口

- 补冻结的`p2_min_v1`正式确认矩阵和same-row完整主表。
- 补同capsule、同seed、同新类规模的ProtoNet、单qKNN、adapter-qKNN和最新主候选。
- 补Phase1独立复验、置信区间和模块级消融。
- 补数据manifest和各split物理记录数；采样率与LEO信道参数已进入正文。
- 若继续使用“spaceborne”定位，至少补硬件在环；若主张真实卫星有效性，则必须补真实卫星数据。
- 补目标处理器上的端到端时延、峰值内存、能耗和数值一致性。
- 补连续多session注册、到达顺序、持久状态、rollback和跨接收机切换实验。
- 若投TIFS，补正式威胁模型、unknown/illegal emitter、replay/forgery/support poisoning和FAR/FRR/EER；否则将主张限定为identification与identity assurance。

详细优先级、实验arm、指标、统计设计和claim-unlock关系见`reviews/00_consolidated_revision_and_experiment_roadmap.md`。

45份引用PDF放在桌面交付目录，不纳入Git仓库，以避免把受版权约束或体积较大的论文文件提交到代码版本库。
