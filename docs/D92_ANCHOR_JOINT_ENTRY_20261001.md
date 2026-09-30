# AJLR support入口与独立摘要（2026-10-01）

状态：入口、独立摘要和合成测试已实现。本worker只执行AST和UTF-8检查。主Agent已验证4项入口测试及修复后的正常完整摘要fixture；剩余摘要拒绝分支正由主Agent串行确认。本文不声明真实拟合、运行或收益。

数学契约以[D92联合设计](D92_JOINT_AFTER_FCR8_DESIGN_20261001.md)为准。核心为`cvsrffi.d92_anchor_joint_local_ridge`，使用`prepare_anchor_joint_training`、`fit_anchor_joint_local_ridge(mode='B'|'C_seq')`、`state.score_with_audit`及`audit_dict`。入口只有R0和R_AJLR_seq；固定Phase1，复用合法未适配target support cache。C只继承当前本行实际B分类函数及原坐标U_B，旧类support和持出物理ID配对。A与B−A为N/A，B0不替代A。

旧类6个，K=1/5/10/20，新增类数0/2/5/10/20，两model seed、两cohort、high/lowurban构成160 parent。非K1采用最多3折物理OOF和全部one-shot anchors；真实K1仍执行B及必要C闭式头，独立持出与proxy为N/A。proxy trainK1也执行真实AJLR闭式头，无adapter监督更新，不强制等于R0。Nnew=0精确复用B，没有额外C preparation/head/优化。全部已注册类别在同一argmax竞争，旧列限制只作事后诊断。

入口保存完整`fit_trace.jsonl`、`compact.jsonl/csv`、`fit_stages.jsonl/csv`、`training_events.jsonl`、`training_events_compact.jsonl/csv`和CVS详细`training.log`。STARTUP包含实际参数、缓存绑定、执行环境、线程和权限；事件保留CE、近端、总目标、梯度、学习率、试探、真实停止、源域验证N/A及耗时。所有U/Z/W/H、梯度、方向、核、q、alpha、Y/M/E和score数组均以有限float64/int64/bool保存在不可覆盖`state_arrays/*.npz`，字符串ID与类别放JSON元数据。完整数组不塞进紧凑日志。`state_manifest.json`和`artifact_manifest.json`保留文件、字节及phase清单，不增加hash或receipt链。

结构数量是160 parent、40 K1、120 OOF、1400 proxy anchors、1800 paths、3240 R0 heads、3240 AJLR preparations和3240 AJLR stages。最多648个可有信息OOF阶段、2592次更新、31104次试探、31752次objective、95256个inner heads、3240个final student heads、864个prior heads；总head/factor上界102600。上界不等于实际值。核心PREPARATION_COUNTERS/STAGE_COUNTERS只按实际累计，共享preparation只算一次，cache展示不收费。R0 head和EDF三角求解分别记录，二者合计；AJLR学生原解、prior原解和CE伴随分别记录。最终outer inference通过`score_with_audit`保存residual与prior的真实距离、核、adapter、字典和时间，reference pair计数是raw work子集，不能再相加。

独立摘要先核对全部4 row的完整marker、source-only checkpoint与既有cache绑定，再打开support-held score和数组。它独立复算旧物理reference下的tau/gamma/s0、q中心K/L、无自由intercept和无二次残差中心化的`alpha=(K+I)^-1(Y-M)`、实际trace正规方程、class-sum、实际B先验、DCT与adapter几何、Z=0原样继承、跨折每类CE均值和RMS、函数近端、纯归一化负梯度、每次12试探的Armijo加非增条件、最后接受状态、完整事件和实际费用。最终outer support-held score由保存的B prior与C residual单样本函数重算，不拟合任何状态，不访问query。

报告保留完整K×新增类数表、receiver/scenario/model/cohort分层和实际资源分层；列出A/B0/B/C旧/C新/H、B−A、B−B0、注册下降、绝对新旧差、旧列诊断及配对正确性转移。H与绝对差先逐parent计算再汇总。新候选只有一个，资源比较需要披露旧run曾包含两个候选。RAM数值字节、最小部署数值状态、训练缓存和NPZ压缩字节分别报告；实际部署包、增量传输、GPU/星载资源未测时为N/A，不能用少参数宣布省算力。

入口依赖既有registration selection/diagnostics、branch support cache loader和branch LocalRidge。核心的FCR/MC/interaction等底层源码仅提供纯数学函数，不读取旧run、历史checkpoint适配状态、目标评分或实验排名。

```text
python tools/evaluate_d92_anchor_joint_probe.py --support-features CACHE --capsule CAPSULE --output NEW_OUTPUT --config CONFIG --expected-capsule-id ID --expected-checkpoint-sha256 SHA --expected-model-seed SEED
python tools/summarize_d92_anchor_joint_probe.py --spec SPEC --run-root COMPLETE_RUN_ROOT --output NEW_SUMMARY
python -m pytest tests/test_evaluate_d92_anchor_joint_probe.py tests/test_summarize_d92_anchor_joint_probe.py -q
```

合成测试覆盖真实K1闭式头、N0复用、proxy实际head、流式JSON/CSV/文本/NPZ、非有限或字符串NPZ拒绝、真实B prior/reference/CE/试探/费用/事件/score/归档篡改、零旧尺度与不完整run提前拒绝。

## 主Agent实际验证与修正

核心22项数值检查全部通过。核心先修复了数值归档对整数label、class index和count的dtype：保留int64，不能统一转成float64后用于索引。该修正保持数学与输入边界不变。

入口和摘要首轮9项检查为8通过、1失败，28.76秒，证据前缀`1790789775879871900`。4项入口检查全部通过。正常摘要fixture先暴露参考pair口径：非identity的参考子集工作量是`m*(m-1)/2+m*(n-m)+h*m`，不能把对称训练矩阵全部行或对角重复收费。摘要已定点修复，并在真实synthetic trial head上要求B的`m=n`和C的`m<n`两种分支均出现且计数正确。

同一fixture第二次核对已通过参考pair，随后暴露JSON原生bool与NumPy比较结果`np.bool_`的类型边界，证据前缀`1790789897088187900`。摘要现在将epsilon转换为原生float，将Armijo与非增比较转换为原生bool；保持与核心完全相同的数值阈值，没有放宽接受条件。

正常完整fixture最终1/1通过，23.74秒，证据前缀`1790790062262768900`。它覆盖actual B prior、固定参考核、全类score、CE、闭式方程、坐标继承、试探、完整事件和真实费用。此前assertRaises分支可能被正常链的早期错误截住，因此主Agent只追加复测其余4项摘要检查，以确认当前正常证书通过后仍正确拒绝篡改；最终结果由主Agent续记。worker没有运行pytest、Conda、SSH、Git、真实数据拟合或评分。
