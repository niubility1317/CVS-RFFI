# ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1 DESIGN_FROZEN

## 状态与边界

- 状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2`
- 可行性监督：首裁`REVISE`；采纳唯一最小修订后终裁`MERGE / P0=0 / P1=0`
- 协议：继续复用`p2_min_v1`、GEOFF/r8 archive/manifest/coverage，不修改received IQ、物理ID、receiver/TX、场景、K、support/query split或schema，不触发数据重验
- 科学机制：DA、`z_id/z_dom`双qKNN、BCRR、四臂、K、INT8 bank、两级BCR权重codec和完整125均不变

## 触发证据

parent run=`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_full125_posixfix1_ab04f624_20260723_223533`在首波健康检查中形成8份完整prediction集合但均被错误的runtime receipt validator拒绝；另一个row在prediction前因新类support raw`z_id`严格零向量失败。runner已只终止本run，launcher PID=`1214101`、matrix PID=`1214105`、exit=`143`，合法完整row=`0/125`，终态=`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_COMPLETE_PERFORMANCE_RESULT`。

精确失败包只读回收后，同checkpoint复现`leo_low_elev_weak`的class index23、rank5 support raw`z_id`为160维全零；IQ本身有限且非零，同类其余9个peer有效。近当前完整support-only资产750包、67,650个support前向中共发现2个同型零行，均为K10新类单零且各有9个有效peer；K1、K5、整类失效和`z_dom`失效均为0。该扫描不读取query/truth，也不是性能证据。

## 唯一机制delta

冻结规则名：`finite_exact_zero_singleton_class_medoid_v1`。

1. 输入必须是有限FP32`[N,160]`support raw`z_id`、typed class label、唯一typed physical token及`K∈{1,5,10}`；每类必须恰有K行。
2. 零行仅指160个FP32分量逐项严格等于`0.0`。NaN/Inf、范数很小但非零的行不得修复并立即失败关闭。
3. 仅当`K∈{5,10}`、某类恰有1个零行且其余`K-1`行均有限非零时允许修复。K1、同类多零或全零立即失败关闭。
4. 有效peer先按physical token字典序固定顺序，再在FP64单位球面上计算每个实际peer对其余peer的余弦和；取最大者，精确并列按token字典序。
5. 把donor的原始FP32行逐字节复制到零行；其它正常行bitwise不变。不生成均值、原型或跨类替代，不使用`z_dom`。
6. 每个scene/state只生成一次修复输出与不可变receipt；该同一输出供FP32 teacher、affine bank、dual qKNN、BCRR及Stage2-C完整teacher复用。
7. 规则对old/new、class handle、receiver和scene完全同式，不得按诊断中的具体class、rank或场景分支。

## Receipt与验证闭合

repair receipt必须绑定规则、K、输入/输出/单位化输出SHA、零行token root、donor token root、逐类修复数、正常行bitwise保持、修复行数及`query_rows_used_for_fit=0`。runtime state receipt必须暴露实际branch teacher SHA，并与repair receipt的单位化输出SHA一致。

`branch_actual_bank_binding_sha256`在before和after都必须是小写64位hex；after值还必须与同scene append receipt严格一致。缺失、非hex、错SHA、重复/缺少scene-state repair receipt或任一teacher绑定漂移均失败关闭。

## 资源与可辨识性

修复只增加受影响类的`O((K-1)^2×160)`support期计算；query MAC、optimizer step、持久state布局和state bytes不变。K5/K10由4/9个同类有效peer确定实际medoid；K1零行不可辨识，因此不伪造身份并失败关闭。无零行时必须是严格identity。

## 冻结改动范围

- `code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
- `code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`
- `tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`
- 本设计文档、目标文档、活动研发报告和后续新run报告

禁止修改模型、数据、authority、coverage、scorer、DA/qKNN/BCRR公式、四臂、rank、K、fallback或资源上限。

## 冻结测试与立即证伪

- K5/K10单零分别以4/9个有效peer成功修复；K1、同类多零/全零、NaN/Inf、微小非零均失败。
- donor必须是同类实际peer；正常行bitwise不变；support顺序和class/old-new置换后保持等价。
- 每scene/state只生成一次输出/receipt；FP32 teacher SHA与单位化输出SHA一致；before/after binding及after append绑定闭合。
- qKNN/BCRR INT8门、Stage2-C旧前缀、state bytes、query MAC和无query smoke保持原合同。
- 真实失败support-only包必须精确修复1行、其余259行不变并通过完整state门；任一冻结条件失败即停止本revision。
- 独立代码review必须达到`P0=0、P1=0`；随后以新Git commit和全新不可覆盖run ID发布完整125，不复用parent，不发布窄性能子集。

## 本地实现闭合

- 实现严格限制在冻结的method、正式125 runner和专项test三文件；DA、双qKNN、BCRR、四臂、K、数据和资源门未改变。
- `ssr-gpu`下候选与相邻DSSC回归共`103 passed,3 skipped`，3项仅为Windows不执行的POSIX专项；`py_compile`和`git diff --check`通过。
- 真实checkpoint support-only无query smoke覆盖before/after三场景：repair count=`0/1/0`，正常行bitwise不变，teacher/actual-bank/append binding、旧INT8前缀和BCR/qKNN门全部闭合；query/truth读取均为0。
- smoke receipt SHA256=`a2bd0ed6a4c5dc57c906c6a5439fb5b0b118893d00e35f09fb5f33dd8a609cad`；after state bytes=`116764/116765/116764`，未增加query MAC或optimizer step。
- 独立Terra代码review=`MERGE / P0=0 / P1=0 / P2=1`。唯一P2仅涉及非formal调用者省略repair receipt的默认API路径；正式runner显式一次生成并全链复用receipt，不阻塞本次完整125发布。
