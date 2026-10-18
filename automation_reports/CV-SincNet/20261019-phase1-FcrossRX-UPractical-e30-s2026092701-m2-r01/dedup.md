# U 学生 practical 输入：窄去重结论

结论：已经实际训练过“额外 U practical 教学”，不能把 As(U) 称为全新路线；本次有限索引命中及实际代码核对中，未发现当前锁定的精确组合已经实训：P43 强 F＋CROSS_RX 的 ownAs20 student/ownEMA＋10 至 E30、两新行共同去掉学生 strong agreement、固定原 clean/raw teacher confidence＋batch-neighbor temporal 资格、仅候选将既有 U 学生槽位的四分之一从原始 raw U 换成完整 practical(raw U)，总学生／教师 NN 不增加。该未重复结论只覆盖本次命中和代码，不声明全库不存在。

## 最直接的历史：Uurban39

run：20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01。登记和最终报告均为 ANALYZED/VERIFIED；固定 C4 已在读取正式结果前 SOURCE_NEGATIVE 关闭，不扩 seed、不调系数或阈值救跑。

它有 B0／URAW／UAS 三组，各自从零 scratch20＋own10＝E30（13L17U），完整五源 RX、L6300／hidden U56700／V27000。L 为 4L/512 NN 每步，ordinary:As=.75:.25，并非本次 P43 6L/768 NN 的同一个 ownAs20 父。三组都新增一次完整 U 学生前向，E30 每组 963900 个额外 U packet-forward；B0 对新增前向加零损失，URAW 是同成本 raw-U KL，UAS 是同成本 full practical U KL。原 U 学生 strong 和原 mask 完整保留，不是替换现有学生输入。

新 KL 复用原 ownEMA raw-U logits，额外教师 NN=0；q=clamp((max p−.95)/.05,0,1)，batchmean(q·KL)，系数 .16×min(1,(epoch−13)/5)。它另有首次 U 自身 teacher 源资格：precision≥.99、coverage≥.75、每 RX precision≥.98、每 draw mid/low 加权教学机会≥.10。所有 teacher 实际通过，但这不替代源性能判据。

实际源判据要求 UAS−同成本 URAW 在两 draw 各 mid/low 四格均≥+1 pp；四格依次 +.111111／+1.500000／+1.055556／+1.277778 pp，首格失败。fullV clean B0/URAW/UAS=.980555556/.980888889/.980777778；clean 保持条件通过。固定附加 .16 q-KL 配方关闭，不能仅改名或略改门槛重跑。

最少出处：

- E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/report.md（最终状态与完整方法/成本；源关闭段）。
- 同目录 source_development_decision.json：status、decision.checks、decision.urban_deltas_pp、decision.clean、formal_scores_read=false。
- E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_Uurban39/design.py:24-27（4L原配方、三组匹配额外 U、教师资格与损失）。
- 同目录 model.py:18-30（原 base_mask & strong_mask 原样返回）；32-52（额外 U forward 和 KL）；不是 native strong 替换。
- 同目录 criteria.py:3-8（四格源阈值）；quality.py:23-43（首次资格真实无更新读出）；experiments/cvs_U_teacher38/soft_transfer.py:5-12（连续 q 与 batchmean）。

## 相邻但不可混为本次的路线

1. 20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01：ANALYZED，四行 scratch E50/33L17U。SSL_CONF／SSL_WEAK在原 U 学生之外额外做完整 practical U soft-CE（.16）；后者还额外 weak 教师前向。两者不是同 NN 的现有 U 学生替换，且网络不是当前强 F+CROSS_RX。实际 observer.py:78-101 表明原 native_U 保留、再 forward_identity_only(xu)，teacher.py:5-15、23-51 规定 clean-confidence/weak-agreement 与健康校验。其结果不作为本次参数或父模型选择依据。
2. P67：20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01，ANALYZED、源四项 FAIL 关闭。它从对应 P43 own20＋10，仅将 batch_neighbor 改为 epoch_bank；原 strong agreement 保持，U 学生仍 original strong_augment(U,std=.015)，没有 As(U)。实际 experiments/cvs_temporal_bank67/design.py:34-42、57-62；canonical report.md:24-26、70-72。不能同时叠加或把扩大覆盖称为本次 practical 输入因素。
3. P68：20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01，ANALYZED、SOURCE_FAIL/noE50。变的是 L selected64 第二次已有前向中的独立完整 As；U 不变。父为 SAT18 自身 AS_CE20（As-only KL已关），并非 P43 fullKL As20。canonical report.md 的“已有证据与本次关键改变”及“固定矩阵与数据”；experiment.json rows[0].scenario/optimizer/permissions.claim_scope。不能重开独立双 As，也不能与本次 U 输入替换混同。

## 本次检索边界

先通过 E:/type10-7/tools/experiment_registry.py search 检索 urban SSL／urban-ssl、U student satellite／Ustudent、teacher qualification／teacher-qualification、UAS／Uview，以及 P67/P68 精确别名；再仅阅读直接命中的完整报告、experiment.json 和上述实际实现。AsU 一词索引结果过宽，未据此广扫全库。只做历史证据去重，没有运行/注册/修改实验，没有 NN/CP/IQ/SSH/Git 或新的评分；未用正式微差选择本次比例、教师资格或父权重。
