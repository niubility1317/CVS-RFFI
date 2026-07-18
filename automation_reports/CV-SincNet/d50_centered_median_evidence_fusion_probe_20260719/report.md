# D50全局锚定类级中位数证据融合探针

## 1.状态与目标

- 实验ID：`d50_centered_median_evidence_fusion_probe_20260719`。
- 当前状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：综合D45全局LOO融合的稳定性与D46类级LOO对new floor的真实改善，修复D46的类级均值易受少量support held rank拉偏、D47收缩完全退回D45、D48截距残差过强、D49全局cosine nested CE失配。
- development cell固定为receiver`20-1`、seed`713101`、K10/new5、`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`×5 outer折；实际outer fit K8。只复用`VALIDATED_ONCE`的`p2_min_v1`固定received-IQ capsule/split，方法变化不触发数据重验。
- 本轮仅本地开发探针；不访问N607，不运行第二development seed或125。

## 2.唯一方法

继承D45/D46相同的B20、全288d shrinkage-LDA head、3-block shrinkage-LDA head、support RMS、canonical gauge、一次FP32融合和int8/FP16生命周期。对每个匿名注册类`c`及inner held rank`r`定义：

`d_{r,c}=CE_block,r,c-CE_full,r,c`。

D45全局log-odds锚点保持为：

`z0=C×mean_{r,c}(d_{r,c})`。

类级稳健位置与中心偏差为：

`m_c=median_r(d_{r,c})`，`delta_c=K×(m_c-mean_j(m_j))`。

最终：

`z_c=z0+delta_c`，`w_full,c=sigmoid(z_c)`，`w_block,c=1-w_full,c`。

由构造可得`mean_c(z_c)=z0`，因此类级差异不会移动D45的全局log-odds中心。median对偶数K使用两个中间值的算术平均。K1逐位回退D45；K2时full/block均为单位协方差等价head，必须得到精确0.5/0.5，否则fail-close。

该方法只称`D45-anchored centered median rank evidence fusion`，不宣称median是校准posterior、泛化误差估计或场景不变量。

## 3.协议、对称性与禁止项

- `d_{r,c}`只来自合法support标签和严格inner train/held分区；不读取outer-held、query、clean/source、receiver、scene、handle、class ID、old/new角色或任何class quota。
- 每类使用完全相同公式，class-label置换必须等变，rank顺序置换必须不变。
- query继续对全部注册类逐样本独立一次性argmax；truth、role Oracle、true batch class count、quota、global reassignment、query-dependent batch optimization和dense query graph均禁止。
- 不增加temperature、clip、阈值、sign gate、trim比例、权重扫描、第二arm或post-hoc选择；不得根据本轮outer结果切换mean/median。
- sigmoid权重必须有限、严格位于(0,1)且逐类和为1；median、中心、锚点或canonical state任一闭合失败即停止。

## 4.资源预注册

D50不新增B20、LDA fit、optimizer step、query state或sidecar，继承D46的K8 before/final共36次LDA和一个`C×288+C`state。median排序比较单列审计；数值运算保守沿用D47的`O(CK)`、K8两state`1,256`scalar MAC-equivalent上界，因此总适配开销预计不超过D47的`1,077,329,226`MAC-equivalent，query仍为6,624 MAC，参数2,016，state 8,583B，epoch/step为20/20。真实artifact必须报告实测CUDA peak；host FP64 peak未测时继续标未测。

## 5.开发晋级门

D50必须同时满足：

1. 相对D45至少改变1条final outer prediction；否则机制无决策价值。
2. 相对D45的总体和clear/low-elev/rain各自after-old、seen-new、同row H、joint、min-old、min-new均不退化，forgetting不增加。
3. seen-new和min-new至少达到D46的84.67%/73.33%，同时rain after-old/forgetting至少达到D42的78.33%/≤10.00pp；总体forgetting不得高于D42的8.89pp。
4. old→new/new→old/new→new混淆不超过D42的26/10/18。
5. before/final/margin的int8相对FP32翻转为0/0/0；协议、source、ground、state、资源、artifact全部闭包。

任一失败即记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不追加公式变体、第二seed或125。全部通过也仅进入另行formalize，不能直接宣称正式性能。

## 6.完成后详细性能账

实验完成后必须在本报告及根目录镜像写入：7候选总体表、3场景表、11类表、15个outer行、相对D42/D45/D46同条件差值、D50权重/median/均值偏差/锚点分布、20步训练轨迹、逐向混淆、FP32/int8误差与top-tie、资源闭包、全部artifact大小/SHA、缺陷机理和下一轮决策。不得只写失败原因或边际最大值。

## 7.版本与执行占位

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`；工作树有用户无关改动，本轮只stage D50自有文件。
- 根目录`E:\type10-7`的`.git`不可用，根报告仅为运行镜像，不宣称根目录版本化。
- 计划实现：`code/scripts/probe_d50_centered_median_evidence_fusion.py`；测试：`tests/test_probe_d50_centered_median_evidence_fusion.py`；追踪：`analysis/d50_centered_median_evidence_fusion_traceability_20260719.md`。
- 输出、代码提交、clean worktree、输入SHA、exact command、PID/GPU、日志和最终artifact在首次运行前补锁。

## 8.实现与运行前验证

D50通过D46的`reliability_strategy`入口实现，只替换support证据聚合；B20、两个LDA head、RMS、canonical affine融合、int8/FP16编译和runner均未改动。运行后verifier将从每个artifact的逐fold CE重算mean、median、D45锚点、中心偏差、post-log-odds和权重，再把audit临时还原为D46标准权重调用既有D46完整verifier，从而同时验证D50新增公式与D46分区/融合/资源闭包。D50 source closure绑定D47/D46/D45/D44/D43全部helper SHA及D50探针本身。

本地验证：D50定向`8 passed`；D46＋D47＋D50联合`45 passed`；D42–D50全链`152 passed`；`py_compile`通过，所有pytest退出码为0。代码复核未发现P0/P1：K1返回D45等权fallback；K2只允许等价head；rank置换不变、class置换等变；`mean(post_log_odds)=z0`在`1e-12`内fail-close；非有限/分区/CE闭合均拒绝。当前仍未读取outer结果。

## 9.执行锁与exact command

- 实现提交：`003f0babd6791302bbcdcaf03a15e8cedc439c35`；clean detached worktree为`E:\type10-7\code\snapshots\d50wt`，`git status -sb`仅`## HEAD (no branch)`。
- clean worktree探针SHA256：`65e4b31b3add10463744faf6dab9b2e74ef9a2183aba19988cba021dbd5acf53`。主工作树文本因Windows行尾转换具有不同工作树字节SHA，运行只使用上述clean artifact。
- runtime继续只读使用历史锁定`E:\type10-7\code\snapshots\d41wt`；D50 bootstrap对12个runtime模块的内置source closure通过。首次预检在closure通过后因打印了错误诊断属性名退出，未创建输出、未执行fold；改用正确只读字段重跑通过。
- before/after seal、before/after envelope、component manifest、class binding实际SHA依次匹配`53ace286…d9f75`、`c70aedf3…b50ff`、`31a2ad99…ceb0e`、`a2483d6e…be76`、`15b5e144…629c`、`bb89a1db…c901f`。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d50_centered_median_evidence_fusion_probe_20260719\centered_median_evidence_fusion`启动前不存在。本地串行`device=auto`；不访问N607、不生成125。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d50wt\code\scripts\probe_d50_centered_median_evidence_fusion.py' `
  --d50-arm centered_median_evidence_fusion `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d50wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d50_centered_median_evidence_fusion_probe_20260719\centered_median_evidence_fusion' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 10.执行完成状态

- 本地105/105行完成，exit0；wall`80.048s`，receipt elapsed`73.337s`。7候选各15行，receiver`20-1`、seed`713101`、K10/new5、实际fit K8。
- D50末端verifier、source closure、support/query disjoint、ground int8和artifact SHA全部通过；query未打开，未访问clean/source、N607，未运行125。
- runner状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；本报告按预注册门独立复核后定稿为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

## 11.七候选总体性能

H为15个matched row内H的均值；`min-*`先按类跨15行取均值再取类间最小；混淆顺序为`old→new/new→old/new→new`。

|Candidate|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|identity fallback，双端弱|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|与Z0同指标|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|诊断比较器|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|保旧但new注册崩溃|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|偏新导致旧类遗忘|
|D50-INT8|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|本轮候选；精确回到D45决策|
|D50-FP32-MATCHED|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|80.00%|53.33%|70.00%|24/8/16|与int8完全一致|

## 12.分场景性能

|场景|before|after|new|H|forget|joint|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|70.00%|90.00%|4/1/0|new近饱和，但旧类仍未到92%|
|leo_low_elev_weak|88.33%|80.00%|74.00%|75.45%|8.33pp|20.00%|60.00%|40.00%|7/5/8|new和floor明显不足|
|leo_rain_weak|90.00%|76.67%|80.00%|77.45%|13.33pp|10.00%|30.00%|70.00%|13/2/8|旧类最低30%，遗忘13.33pp|
|总体|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|与目标仍有大幅差距|

## 13.逐类性能

O0–O5、N0–N4仅对应opaque handle排序，不参与方法。总体old为before→after；场景old为after；new为注册后准确率。

|角色|类|总体|clear|low-elev|rain|表现|
|---|---|---:|---:|---:|---:|---|
|old|O0/`1f33441e`|90.00→90.00%|100%|80%|90%|稳定|
|old|O1/`33bbd165`|96.67→93.33%|90%|90%|100%|总体最强之一|
|old|O2/`75aa6d50`|96.67→90.00%|90%|90%|90%|各场景一致90%|
|old|O3/`8b02d999`|80.00→53.33%|70%|60%|30%|主要old floor瓶颈|
|old|O4/`a53ca128`|100.00→73.33%|90%|70%|60%|恶劣场景遗忘明显|
|old|O5/`f8dfc2ed`|90.00→93.33%|100%|90%|90%|注册后反而改善|
|new|N0/`09f80039`|70.00%|100%|40%|70%|主要new floor瓶颈|
|new|N1/`1c2ad882`|93.33%|100%|100%|80%|最稳健新类|
|new|N2/`b8fbace5`|76.67%|90%|50%|90%|low-elev失稳|
|new|N3/`d3afb5d1`|90.00%|100%|90%|80%|较稳健|
|new|N4/`f608a348`|90.00%|100%|90%|80%|较稳健|

## 14.十五个outer行

floor顺序为`before/after/new`，混淆为`old→new/new→old/new→new`。

|场景|fold|before|after|new|H|forget|joint|floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|100.00%|90.00%|94.74%|0.00pp|50%|100/100/50%|0/1/0|
|clear|1|100.00%|83.33%|100.00%|90.91%|16.67pp|0%|100/0/100%|1/0/0|
|clear|2|91.67%|83.33%|100.00%|90.91%|8.33pp|50%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|low-elev|0|91.67%|75.00%|80.00%|77.42%|16.67pp|50%|50/50/50%|3/1/1|
|low-elev|1|66.67%|58.33%|70.00%|63.64%|8.33pp|0%|50/50/0%|1/0/3|
|low-elev|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0%|50/50/0%|0/2/1|
|low-elev|3|100.00%|100.00%|60.00%|75.00%|0.00pp|0%|100/100/0%|0/1/3|
|low-elev|4|91.67%|75.00%|90.00%|81.82%|16.67pp|50%|50/50/50%|3/1/0|
|rain|0|83.33%|83.33%|60.00%|69.77%|0.00pp|0%|50/50/0%|2/0/4|
|rain|1|100.00%|66.67%|90.00%|76.60%|33.33pp|0%|100/0/50%|4/1/0|
|rain|2|91.67%|83.33%|80.00%|81.63%|8.33pp|50%|50/50/50%|1/0/2|
|rain|3|91.67%|75.00%|90.00%|81.82%|16.67pp|0%|50/0/50%|3/0/1|
|rain|4|83.33%|75.00%|80.00%|77.42%|8.33pp|0%|50/50/0%|3/1/1|

## 15.相对D42/D45/D46

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42 original|90.56%|81.67%|81.33%|80.63%|8.89pp|23.33%|50.00%|70.00%|26/10/18|forgetting较低，性能仍不足|
|D45 global LOO|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|D50直接matched基线|
|D46 classwise LOO|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|当前最强合法开发点|
|D50 centered median|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|全部决策精确等于D45|

D50相对D45的15/15个outer prediction SHA和全部330个final argmax均相同，所有总体、场景、逐类、floor和混淆差均为0。相对D46有2/15行变化：low-elev恢复1个old正确决策但丢失1个new正确决策，回到D45；总体after`+0.56pp`、forgetting`-0.56pp`，但new`-0.67pp`、H`-0.18pp`、min-new`-3.33pp`。因此D50没有产生D45之外的联合收益。

## 16.权重与稳健证据行为

|阶段|统计|min|mean|max|
|---|---|---:|---:|---:|
|before|full权重|0.3143|0.4529|0.6819|
|before|D45锚点z0|-0.2991|-0.1913|-0.1046|
|before|median−mean平均绝对差|0.0049|0.0152|0.0336|
|before|中心化delta|-0.5694|≈0|0.8669|
|before|abs(delta)|0.0076|0.2333|0.8669|
|final|full权重|0.3218|0.5141|0.7166|
|final|D45锚点z0|-0.1690|0.0580|0.3221|
|final|median−mean平均绝对差|0.0070|0.0162|0.0286|
|final|中心化delta|-0.7270|≈0|0.7119|
|final|abs(delta)|0.0015|0.1972|0.7270|

`mean(post_log_odds)−z0`最大绝对误差为before`8.33e-17`、final`5.55e-17`，锚点闭合。final full权重均值在clear/low-elev/rain为`0.5151/0.5308/0.4963`，所以D50不是代码上退化成D45全局常数；只是这些非平凡权重变化均未跨越D45的最终决策边界。该结果反驳“只要稳健聚合类级LOO权重即可获得D46收益且恢复旧类”的假设。

## 17.B20完整训练轨迹

15个D50 int8行的逐epoch均值如下；每epoch合计query rows始终0。

|epoch|support acc|loss|grad norm|query rows|
|---:|---:|---:|---:|---:|
|1|95.14%|1.031996|1.083757|0|
|2|95.97%|0.801388|0.870572|0|
|3|97.78%|0.623484|0.690893|0|
|4|97.50%|0.500504|0.540671|0|
|5|97.78%|0.415989|0.436324|0|
|6|98.19%|0.353962|0.369829|0|
|7|98.61%|0.299062|0.315457|0|
|8|98.89%|0.260996|0.301407|0|
|9|99.03%|0.233931|0.256953|0|
|10|99.03%|0.216143|0.235860|0|
|11|99.58%|0.190273|0.220582|0|
|12|99.31%|0.174391|0.202662|0|
|13|99.72%|0.160626|0.185954|0|
|14|99.86%|0.152731|0.205840|0|
|15|99.72%|0.142408|0.173981|0|
|16|100.00%|0.131352|0.166464|0|
|17|99.72%|0.126780|0.170467|0|
|18|99.72%|0.115133|0.147418|0|
|19|99.86%|0.109940|0.131373|0|
|20|100.00%|0.102685|0.135354|0|

B20平滑收敛且与D45/D46使用同一冻结底座；D50的失败发生在head证据到决策边界这一层，不是训练发散或epoch不足。

## 18.量化、资源与协议

|项目|结果|判定|
|---|---:|---|
|matched FP32 before/final argmax变化|0/0|通过|
|FP32/int8 margin符号翻转|0|通过|
|before/final support argmax变化|0/0|通过|
|int8最大score绝对误差|min`3.759e-4`、mean`7.975e-4`、max`1.281e-3`|未改变决策|
|LDA闭式fit|36|继承D46闭合|
|LDA MAC|1,065,830,400|固定|
|可靠度评分/类级融合|6,511,104/9,826|固定|
|D50 scalar上界|1,256 MAC-equivalent|保守继承D47上界|
|总适配|1,077,329,226 MAC-equivalent|闭合|
|query MAC|6,624|单affine|
|参数/state|2,016/8,583B|≤80k/≤256KB|
|epoch/step|20/20|≤30/≤50|
|CUDA peak|22,886,912B|`cuda:0`实测|
|query fit/truth/role/quota/count/global assignment|全部0/false|通过|
|clean/source/dense graph|false/false/0B|通过|

## 19.Artifact清单

|文件|大小/B|SHA256|
|---|---:|---|
|D50_PROBE_METADATA.json|2,152|`1b3123be657edbcfeaabaa6957ea2aa0c4cef2ccac18b542c81648a969de4977`|
|RECEIPT.json|5,035|`a11daa0e6b71d8ac2457520b6dfb50f329157d24554a89f010cf651d44696f4f`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,990|`6e043c521b2b3e4fdbb73a841fbe343342a37a76b914156a7be2118f442aa678`|
|support_audit.json|313,678|`609687bbdcc2f1f10510b25d86a524b0c43e2ef8dde73f0a8f3a60377a4953cd`|
|training_log.jsonl|4,922,512|`757ad4f763ba453ecaae83b43af19b7fed671c16fd79dde0d7ba80915343a336`|
|full_performance_summary.json|61,905|`b3b70669a106f67232caa66b709a1552efa23ff7bfaf438e454be3bcb6c094ea`|

summary完整读取D50/D45/D46各105行，含全部候选、场景、类、15折、matched差、权重、20步、量化和资源；生成器为`code/scripts/summarize_d50_performance.py`。

## 20.缺陷、晋级门与停止动作

|晋级门|D50结果|判定|
|---|---|---|
|相对D45至少改变1条final预测|0/15行、0/330 argmax|失败|
|总体/三场景不退化|与D45全部相同|通过但无改善|
|new/min-new≥D46 84.67/73.33%|84.00/70.00%|失败|
|rain after≥78.33%、forget≤10pp|76.67%、13.33pp|失败|
|总体forget≤8.89pp|10.00pp|失败|
|混淆≤D42 26/10/18|24/8/16|通过|
|量化0/0/0|0/0/0|通过|

核心缺陷不是median未生效，而是“锚定且中心化的类级偏移幅度仍不足以跨越D45决策边界”；D46仅有的2个变化也被全部收回。继续扫描median缩放、trim比例或阈值会违反本轮预注册并引入development过拟合，因此本路线停止：不加变体、不跑第二seed、不formalize、不运行125。

当前最强合法开发版本仍为D46，但D46也未满足项目要求。D51必须离开“仅重加权同一full/block两个head”的局部路线，寻找能改变困难类几何同时保留单affine部署的support-only机制；在D49–D51三轮完成后必须先做强制复盘再启动D52。
