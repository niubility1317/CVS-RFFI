# P1-CB-SFCE冻结设计与实现追溯卡（2026-08-09）

状态：`LOCAL_VERIFIED`。范围只覆盖source-only的P1-CB-SFCE C/G续训；不构成性能或晋级结论。

## 实现追溯

|ID|来源|冻结要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|CBSFCE-01|主控冻结卡|G唯一新增`lambda=.10,gamma=1`的按当前batch出现TX等权卫星focal CE；C保持原路径|`code/cvsrffi/phase1_cb_sfce.py`,`train_ssdg.py`|verified|数值、C恒等与标签置换测试|不新增head或前向|
|CBSFCE-02|主控冻结卡|只用既有单LEO`tx_logits`和source-known标签；不读取clean z/logit、teacher、RX/domain、proxy、held或LEO-eval|模块、训练接入、测试|verified|参数拒绝和路径测试|共同`lambda_sat_cons=.10`不归因于G|
|CBSFCE-03|监督MERGE|clear/low/rain训练期round-robin等频；local4×3共12格记录rows/loss/finite/nonzero-grad|模块、训练接入、测试|verified|12格覆盖和缺格负测|终态缺格或None/nonfinite失败|
|CBSFCE-04|监督MERGE|G首个有效batch仅一次未缩放新增项/共同base共享encoder-head梯度norm/cos审计|模块、训练接入、测试|verified|raw梯度、None/nonfinite和单次审计测试|符号仅诊断，不改变optimizer|
|CBSFCE-05|主控冻结卡|失败best-effort原子receipt不遮蔽原异常；C为N/A；40E final-only|模块、训练接入、测试|verified|writer失败和terminal负测|不含raw数据|
|CBSFCE-06|主控冻结卡|6fold×C/G共12任务、八卡≤2任务、严格warm-start、新AdamW/AMP、无覆盖|新launcher、测试|verified|`bash -n`和12条dry-run|run默认`phase1_cb_sfce12_20260809_v1`|
|CBSFCE-07|主控冻结卡|postfreeze只读final checkpoint，复用CCPC已验证的12clean＋12LEO＋12proxy＋6pair闭环|新pair evaluator、postfreeze launcher、测试|verified|`py_compile`、focused pytest、`bash -n`、dry-run42|只计算final checkpoint字节SHA，不读取checkpoint权重，避免PAMR native head路径|
|CBSFCE-08|主控冻结卡|clean/LEO/proxy/physical/scenario/TX/RX/ordered metadata及strict checkpoint SHA闭合|新pair evaluator、测试|verified|正负绑定、checkpoint SHA、outer零影响测试|LEO运行时view必须是`single`，manifest profile仍必须为`satellite`|
|CBSFCE-09|主控冻结卡|6折裁决：clean6/6、LEO18格、逐折三场景等权、18格等权与proxy guardrail均不可补偿；prior receipt同matrix/root/fold绑定|新pair evaluator、测试|verified|6折18格通过／拒绝、proxy恶化和prior伪造负测|任何失败为`REJECT_CB_SFCE_PERMANENT`；通过也仅为主控复核候选|

## 冻结方法

令`p^leo_i`为既有单LEO前向的`tx_logits.softmax`，`I_c`为当前batch中local4训练TX`c`的行，`Y_b`为当前出现的TX集合。G唯一新增：

```text
L_G=L_C+0.10·(1/|Y_b|)Σ_c(1/|I_c|)Σ_{i∈I_c}(1-p^leo_{i,y_i})[-log p^leo_{i,y_i}]
```

其中`gamma=1`固定。clear/low/rain由训练期既有单LEO路径round-robin，不使用任何LEO评估物理行；`sat_view_prob=1`、`sat_cons_start_epoch=1`且不允许额外schedule覆盖。共同GeoSat-C`lambda_sat_cons=.10`在C/G均保持不变；它不是CB-SFCE、不得解释G-C。

## 本地验证

- `python -m py_compile code/cvsrffi/phase1_cb_sfce.py code/SSDG/train_ssdg.py code/tests/test_phase1_cb_sfce.py`通过。
- `pytest -q code/tests/test_phase1_ccpc_leo.py code/tests/test_phase1_cb_sfce.py`：35项通过；其中CB-SFCE覆盖数值、类置换、local6→local4绑定、12格终态、None/zero/nonfinite余弦梯度审计、原子失败收据和`lite_d`source-only前反向冒烟。
- `bash -n code/scripts/launch_phase1_cb_sfce12_20260809.sh`通过；`--dry-run`恰输出12条任务。
- 训练CLI以launcher同一禁用集dry-run通过；未访问N607，未产生性能结论或postfreeze结果。

## Postfreeze冻结闭环

`launch_phase1_cb_sfce_postfreeze_20260809.sh`固定执行42步：12个clean开发导出、12个source-only三LEO场景导出、12个source-proxy连续诊断和6个CPU串行C/G pair。导出只读取各候选`final_ssdg.pth`；pair只读取其字节SHA并与clean/LEO manifest的严格load审计和`source_checkpoint_sha256`闭合，不加载模型权重或分类head。

pair evaluator只对source行计算clean和各LEO场景的overall、min-class、min-RX、min-day四项G-C百分点差；target-old和proxy数值均不参与这些分类指标。LEO逐行`channel_views=single`，同时manifest的source profile必须是`satellite`、seed为`7281718`、三场景物理ID两两互斥且各覆盖local4 TX和6个RX。proxy JSON只提供冻结source-calibrated诊断的AUROC/FAR guardrail，不能补偿任何分类门。

第6个pair在同一命令内读取前5个不可覆盖per-fold JSON；每份prior必须与当前pair共享canonical postfreeze根和matrix_id，且严格匹配冻结fold→C/G pair、local4 source-TX顺序、无fit技术策略、`technical_binding=true`和有限[0,1]proxy值。随后输出18格等权四项差与五类非补偿门：clean6/6四floor、LEO18/18四floor、每fold三场景等权overall、18格等权overall、proxy AUROC不降且FAR不升。任一门失败写`REJECT_CB_SFCE_PERMANENT`；全过只写`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`，不产生Phase3 unknown能力声明。

- `python -m py_compile code/scripts/eval_phase1_cb_sfce_pair.py code/tests/test_phase1_cb_sfce_postfreeze.py`通过。
- `pytest -q code/tests/test_phase1_cb_sfce_postfreeze.py`：20项通过；覆盖合法`single`／伪造view、physical/role/scenario/strict-load/class-order/final-SHA负测、outer数值零影响、proxy恶化拒绝、6折18格裁决以及错run/root/pair/source顺序/string technical/NaN或越界proxy prior拒绝。
- `bash -n code/scripts/launch_phase1_cb_sfce_postfreeze_20260809.sh`通过；干跑计数为clean12、LEO12、proxy12、pair6，总计42；`git diff --check`通过。
