# D45冻结outer-B20的head-only LOO可靠度融合探针报告

## 1.身份与目标

- 实验ID：`d45_inner_loo_reliability_fusion_probe_20260718`
- 操作者：Codex`/root`
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问5个confirmation seeds，不生成125结果。

D44固定1:1融合取得当前最高development seen-new/H并改善最低after-old，但rain after-old/forgetting和1个量化翻转失败。D45只改变一个因素：由support内部预测可靠度连续确定full/block全局权重，其他特征、B20、LDA、RMS、量化器、数据和外部门不变。full组件直接复用D42锁定的sklearn解并仅删除类公共仿射项，避免在高维低秩inner折中用另一求解器重构系数；block组件继续使用D43的3-block结构。

## 2.数学机制

定义严格限定为`frozen_outer_b20_head_only_loo`：B20只在outer fit上训练一次，此后冻结；inner折仅重拟合full/block LDA head和各自RMS。因此inner CE只是合法support-derived融合权重，不宣称为全链路nested无泄漏泛化估计；真正无泄漏评价仍由outer-held physical rows承担。

对每个outer fit、每个组件`g∈{full,block}`，在合法fit support内部按support-row rank做leave-one-out。每个inner折只用其余`K−1`个physical support/类拟合组件和RMS，再对held rank的全部类打分。先用稳定log-softmax计算逐fold、逐类inner-held交叉熵`CE_{g,c}`，再令：

`L_g=(1/C)Σ_c CE_{g,c}`，`w_g=exp(−C L_g)/Σ_h exp(−C L_h)`。

最终完整support fit的score为：

`δ_D45=w_full δ_full/s_full+w_block δ_block/s_block`。

具体锁定`log_evidence_g=−C L_g`，再对两个log-evidence做稳定softmax；不clip、不加temperature、不设阈值。乘数`C`来自逐类正确似然的乘积，不是搜索温度。权重对当前fit所有类共享；不读取outer-held、query、old/new角色、handle或场景ID，不扫描weight、threshold、rank、lr、epoch或shrinkage。K1无法形成inner折，但full/block都退化为同一unit-covariance分类器，因此固定1:1是决策等价回退；K2的inner-train为K1，必须由两组件CE证据得到`1:1±1e-12`，否则fail closed。

## 3.资源口径

每阶段包含2个完整组件fit；K>1时再包含每组件K个head-only leave-one-rank-out fit。before/final合计`4+4K`次closed-form LDA；本development outer fit的K=8，因此每条D45行计36次LDA。MAC按`2[L(C_old,K)+L(C_all,K)]+2K[L(C_old,K−1)+L(C_all,K−1)]`逐fit组闭合，不能用单一平均值粗乘。metric B20仍只训练一次、20 epoch/20 optimizer steps；最终只持久化一套融合int8/FP16 state，query MAC仍按一套state。host FP64 covariance峰值继续标记未实测。

## 4.预注册门

相对D42 original固定基准，必须同时满足：协议/lifecycle/source/ground/state/resource闭包；inner train-held互斥且所有support row恰好held一次；类别置换对称；权重有限、严格为正且和为1；K1/K2均为1:1；before在首次new support读取前物化并保持不可变；聚合before/after/new/H与最低before/after/new、joint均不退化，forgetting不增加，至少一个final floor严格改善；clear/low-elev/rain各自before/after/new/H/joint不退化且forgetting不增加；before/final int8-FP32 argmax变化和margin翻转均为0；final old→new/new→old/new-new不超过26/10/18。探针强制identity且禁用full-K10，即使全过也需另行实现正式candidate。

## 5.文件与执行计划

- 探针：`code/scripts/probe_d45_inner_loo_reliability_fusion.py`。
- 单测：`tests/test_probe_d45_inner_loo_reliability_fusion.py`及D42–D44回归。
- 追溯：`analysis/d45_inner_loo_reliability_fusion_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d45_inner_loo_reliability_fusion_probe_20260718\inner_loo_class_likelihood`。
- 环境：本地`ssr-gpu`串行执行；当前不访问N607。

本地预运行验证：

- `python -m py_compile`通过。
- D42–D45定向回归`67 passed`，pytest退出码0。
- pytest退出后出现本机临时目录`pytest-current`的既知`WinError 5`清理噪声，不影响测试结论。
- 初版随机低样本测试暴露D43 full-control在高维低秩折中重复求解的argmax漂移；已改为full组件直接复用D42锁定解，仅删除类公共仿射项，随后回归通过。
- 输出verifier不信任自报布尔字段：它从held索引重算exact-once覆盖，从逐fold CE重算逐类CE、macroCE、log-evidence与权重，并从四个fit组的行数、类数和D42公式重算全部LDA MAC；对应tamper测试通过。

预注册探针已完成；结果为诊断负面，不构成正式candidate、125结果或目标完成。

## 6.执行与证据

- Git提交：`e7b1c983efca6eb8b60a30dd7202708665d7e4fb`。
- 只读worktree：`E:\type10-7\code\snapshots\d45wt`，detached clean。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，device=`auto`。
- 输入：D18 receiver`20-1`/seed`713101`/K10/new5密封capsule；实际outer fit K8；3场景×5折。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d45_inner_loo_reliability_fusion_probe_20260718\inner_loo_class_likelihood`。
- 完成：105/105行，elapsed`78.4283s`，query0，formal/performance claim均为false，N607未访问。
- receipt status：`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；D45 verifier独立通过30条int8/FP32 fit rows。

## 7.同row结果

|Candidate|机制/精度|before-old|after-old|seen-new|H|forgetting|joint|min before|min after|min new|old→new/new→old/new-new|结论|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8|D45 head-only LOO类似然全局融合/int8|92.22%|82.22%|84.00%|83.10%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|负面，不晋级|
|D42-USLDA-FP32-MATCHED|同一D45解/FP32|92.22%|82.22%|84.00%|83.10%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|matched ablation|
|D42-D40-HNBR-INT8-NEGATIVE|old-heavy HNBR/int8|85.56%|85.00%|15.33%|25.98%|0.56pp|0%|66.67%|63.33%|0%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|new-heavy BEC/int8|86.11%|20.56%|78.67%|32.59%|65.56pp|0%|76.67%|0%|36.67%|142/0/32|旧类崩溃|
|B3_SINGLE_IQ_DIAG_FFTRF|单IQ B3比较器|87.78%|75.56%|72.67%|74.08%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|弱比较器|
|D42-PROTOnet-CDA-ZID160|ProtoNet CDA|71.11%|48.33%|52.67%|50.41%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|负面|
|Z0_SUPPORT_ONLY|identity/support-only control|71.11%|48.33%|52.67%|50.41%|22.78pp|0%|33.33%|13.33%|3.33%|0/0/0|control|

固定TX切分为6 old＋5 new，receiver`20-1`、seed`713101`、K10 capsule、3场景、5折；表内每行指标均来自同一candidate的15行，不拼接边际极值。无rollback/defer分支，loss为同一old-only B20共20 epoch/20 optimizer steps，closed-form LDA不增加optimizer step。

|场景|before-old|after-old|seen-new|H|forgetting|joint|min after|min new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|98.33%|90.00%|98.00%|93.83%|8.33pp|40.00%|70.00%|90.00%|4/1/0|
|low-elev|88.33%|80.00%|74.00%|76.88%|8.33pp|20.00%|60.00%|40.00%|7/5/8|
|rain|90.00%|76.67%|80.00%|78.30%|13.33pp|10.00%|30.00%|70.00%|13/2/8|

相对预锁D42 original，D45提高聚合before/after/new/H及最低before/after，最低new和joint持平，混淆由26/10/18降至24/8/16；但aggregate forgetting由8.89pp升至10.00pp，rain after-old由78.33%降至76.67%，rain forgetting由10.00pp升至13.33pp，严格门失败。量化门为before/final/margin`0/0/0`，max score error`0.0016140938`，是D45唯一相对D44新增的闭合正信号。

## 8.可靠度与资源

- before全局`w_full`范围/均值：`0.4258/0.4524/0.4739`，倾向3-block。
- final全局`w_full`范围/均值：`0.4578/0.5144/0.5798`，接近1:1；clear/low-elev/rain均值分别`0.5155/0.5315/0.4964`。
- D44与D45的15个outer prediction SHA全部相同；全局support-LOO权重没有改变任何held预测，不能修复rain旧类遗忘。
- trainable parameters`2016`；20 epoch/20 optimizer steps；persistent int8 state`8583B`；query MAC`6624`；CUDA peak`22,886,912B`。
- LDA fit总数`36`：before main2、final main2、before inner16、final inner16；LDA MAC`1,065,830,400`，metric MAC`4,976,640`，总adaptation MAC`1,070,807,040`。host FP64 covariance peak未测。
- 300条B20 trace全部finite，epoch/step均完整覆盖1–20。

## 9.artifact闭包

|Artifact|Bytes|SHA256|
|---|---:|---|
|training_log.jsonl|4,179,150|`83937b8d7caa68aef99fcc426c7ff2b987b98b4fb4922427277948b6da79d0b9`|
|support_audit.json|313,375|`610f4fb8a54ee6f2361b5c7e519e9377f1386e88486086f2ca3cd27148903784`|
|selection.json|2,990|`fd54e4c0f9684bc4df2c1608d949327b69a7b9771a92289624f71860fe8c416f`|
|RECEIPT.json|4,750|`4b4f180c5c9defe1b0875448269805c0ad531a88e1dd2742e6c280901a423e38`|
|D45_PROBE_METADATA.json|2,041|`cd3443c521d8efae944d6f79de7c33e14cd867b8a0bfb3e3a5ed18ee24127218`|

## 10.D43–D45强制技术复盘

复盘已重读活动goal、`项目.md`、D43/D44/D45报告和四轮完整training logs，并刷新1008条项目conversation index；索引未返回比当前报告更相关的D43–D45历史条目。三轮始终同时评价Stage2-B before/after与Stage2-C seen-new/H/forgetting，保持LEO-weak-only、无clean/source/query truth/role/quota、逐query独立和类无关floor。

- D43 3-block证明分块协方差可提高before/after/H并实现0量化翻转，但牺牲最低new和rain old。
- D44用support RMS固定1:1融合恢复new并改善低尾，形成当前同row最强联合结果，但rain旧类遗忘和1次量化翻转失败。
- D45用合法head-only LOO全局连续权重；权重证据完整且修复量化翻转，但15/15 outer predictions与D44完全一致，说明继续扫描全局权重没有信息价值。
- HNBR/BEC再次证明hard old-protection与hard new-release分别导致新类不可达和旧类崩溃，应拒绝继续走硬门控。

下一轮D46只改变一个机制：由D45每个fit共享一个全局`w_full`，改为每个注册类使用完全相同公式`w_{g,c}=softmax_g(-K×CE_{g,c})`的类级support-LOO似然权重；full/block、B20、RMS、量化、数据与外部门全部不变。该式是每类K个inner-held正确似然的乘积，对类标签置换等变，不使用class ID、old/new角色、场景或query。预期是让不同类真正跨过D44未改变的决策边界，重点观察rain旧类、low-elev/new floor和三类混淆；若15折prediction仍不变，或任一D42严格门退化，则立即拒绝，不扫描温度或权重。

结论：D45为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不正式化、不运行125、不访问N607；下一步进入D46实现与同一development cell最小验证。
