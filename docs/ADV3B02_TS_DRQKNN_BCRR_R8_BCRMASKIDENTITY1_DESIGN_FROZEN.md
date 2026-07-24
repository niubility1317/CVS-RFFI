# ADV3B02-TS-DRQKNN-BCRR/r8-bcrmaskidentity1 DESIGN_FROZEN

## 状态与唯一delta

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r8-bcrmaskidentity1`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r7-q3support1`
- 状态：`DESIGN_FROZEN -> IMPLEMENTING -> LOCAL_VERIFIED`
- parent run：`adv3b02_ts_drqknn_bcrr_r7_q3support1_full125_7613bf19_20260724_081350`
- parent终态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 唯一delta：当且仅当support-only BCRR双masked-view LOO触发精确`BCRR masked cross-view degeneracy`时，raw directional路径返回显式`exact_masked_degenerate=true`，raw与dual两条directional LOO均使用确定性零占位，并复用既有`score_normalization_degenerate`语义令该branch的BCRR `omega=0`；qKNN、DA、Q3 support codec、正常BCRR、四臂、prediction、scorer和完整125矩阵均不改变。

## 唯一首源与合法含义

r7首波在0 prediction时9个row均失败，其中两个自然结束row`rx20-1/seed713105,713106/K10/new20`具有同一完整traceback：`append_stage2_c -> _append_bank -> _make_actual_branch -> _cross_view_loo_scores -> SVRNBCRStateError: BCRR masked cross-view degeneracy`。该异常只能在已finite、已单位化的部署support中至少一行masked view范数`<=1e-12`时发生；它不表示全support向量无效，而表示交错mask的一侧没有可归一化的cross-view证据。

共享`stage2_svrn_bcr.make_bcrr_receipt`已经把任一合法BCRR support-side数值退化收缩为`score_normalization_degenerate`、`omega=0`。ADV实际bank路径在提前生成directional SHA时直接调用底层LOO，绕过了这一本来存在的identity fallback。本revision只闭合两条路径的语义一致性，不扩大fallback集合。

真实checkpoint与回收support-only重放进一步定位：`leo_low_elev_weak`的after-bank第119行是完整单位向量（norm=`0.9999999617`），但modulus=`5`、residue=`1`的交错masked view范数精确为0；该行对应class index=`11`、rank=`9`。同一重放没有打开query、truth或生成prediction。clear和rain的同row S_C构建正常，因此这是局部masked-view证据不可辨识，不是完整support无效。

## 冻结回退

对K5/K10的每个qKNN branch：

```text
try:
    directional_qscore, directional_bscore = exact_masked_cross_view_LOO(deployed_Q3_support)
except SVRNBCRStateError as exc where str(exc) == "BCRR masked cross-view degeneracy":
    directional_qscore = zeros([N,C], float64) for both directions
    directional_bscore = zeros([N,C], float64) for both directions
    exact_masked_degenerate = true
    BCRR fallback = existing "score_normalization_degenerate"
    omega = 0
```

`_raw_directional_loo`显式返回该布尔标志；`_directional_dual_loo`只在标志为真时直接返回同shape、同方向、`float64`零qscore，不允许通过“数组恰好全零”反推退化。只有该精确异常类型和完整字符串可进入回退；solve、shape、finite、class、wire、receipt、quantization或其他异常继续fail-closed。K1继续使用既有`K1_identity`。零directional数组只作为不可用LOO的确定性sealed占位；其SHA继续进入actual-bank和BCRR receipt。query阶段因`omega=0`直接返回匹配qKNN logits，不读取BCR权重或更新状态。

## 协议、可辨识性与决策几何

- 判定只读当前row部署support；query、truth、role、quota、跨query状态和optimizer读取均为0。
- K1为既有identity；K5/K10可由support确定masked norm是否退化。
- 正常row的directional logits、omega、BCRR融合和全部预测逐字节不变。
- 退化branch的raw与dual directional证据同时强制不可用，故`M_OTHER=M0`、`M_JOINT=M_DA`；这是无合法互补证据时的强收缩，不伪造分类增益。
- qKNN、DA及邻居几何完全不变；完整125仍按同row检验DA、OTHER和联合协同。若退化覆盖过高导致OTHER或联合无收益，应形成真实性能负结果，不再转化为技术失败。

## 资源、风险与falsifier

state、MAC、时延、显存、optimizer step、trainable parameter和INT8生命周期均无增量；退化branch省去LOO求解且query仅复制qKNN logits。主要风险是异常捕获过宽、显式退化标志未传入dual路径、正常row漂移、identity branch仍持有非零omega或占位SHA不一致。

立即falsifier：

1.除精确masked-view异常外任一错误被吞掉；
2.退化branch的`omega!=0`、fallback不是既有`score_normalization_degenerate`或融合logits不逐字节等于匹配qKNN；
3.正常row的directional SHA、omega或prediction变化；
4.actual-bank与后续`_raw_directional_loo`产生不同占位SHA，或显式退化标志未令dual directional qscore采用同shape零占位；
5.K1、query隔离、Q3旧prefix、state资源或协议负例漂移；
6.真实失败support-only重放仍不能构建S_C state。

## 冻结候选文件

监督裁决为`MERGE / P0=0 / P1=0 / P2=0`，只允许修改：

1.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`：精确异常identity fallback及r8 schema/candidate；
2.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`：精确回退、其他异常fail-closed、正常row不变、K1/K5/K10、真实support无query回归；
3.本文档、r7终态报告和新r8 run报告。

不得修改共享BCRR、runner、scorer、data、authority、method lock、DA、qKNN/BCRR公式、Q3 codec或四臂。专项测试、真实checkpoint无query smoke、独立P0/P1 review和Git提交后立即发布全新完整125；不得复用r7 run。

## 实现闭合

- 完整专项测试文件在`ssr-gpu`退出码0，3项既有平台测试按预期skip；`py_compile`和`git diff --check`通过。
- 真实checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`与回收的三场景enrollment support完成无query的S_B→S_C重放；每场景260行、26类，旧prefix保持不变。
- `leo_low_elev_weak`精确回退后actual、M_OTHER和M_JOINT的双方向SHA一致，`omega=0`；clear/rain未触发新增退化标志。
- 独立review：`MERGE / P0=0 / P1=0 / P2=0`。

`DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`
