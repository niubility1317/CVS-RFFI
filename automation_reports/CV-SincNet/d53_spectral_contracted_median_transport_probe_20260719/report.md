# D53谱收缩median transport开发报告

## 1.状态

- run ID：`d53_spectral_contracted_median_transport_probe_20260719`
- operator：Codex
- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- 范围：本地receiver20-1、seed713101、K10/new5、3场景×5 folds；不访问N607、不运行125。
- 当前最强合法开发点仍为D46，不promotable。

## 2.目标与唯一公式

D51/D52共同证明median方向能修复old floor，但两个尺度均过大并伤害new。D53不直接按RMS或base norm缩放，而把median位移通过support类均值→D45判别向量的谱收缩映射转换为系数修正：

```text
M_c=mean_r(x_rc); Q_c=coordinate_median_r(x_rc); U=Q-M
M0=M-mean_c(M); W0=W_D45-mean_c(W_D45); tau=||M0||_2^2
G=U M0^T/tau; gamma_c=1-||mean_r(x_rc/||x_rc||)||_2
DeltaW=diag(gamma)G W0; W_D53=W_D45+DeltaW; b_D53=b_D45
```

## 8.完成状态

本地运行exit0，105/105行，elapsed`74.059s`；metadata验证目标行30/30、总行105、source closure不变、query未打开。receipt为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，没有formal/125权限。

## 9.七候选总体性能

|Candidate|before|after|new|H|forget|joint|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0/ProtoNet|71.11%|48.33%|52.67%|48.97%|22.78pp|0%|13.33%|3.33%|0/0/0|fallback|
|B3|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|60.00%|40.00%|33/22/19|诊断|
|D40-HNBR|85.56%|85.00%|15.33%|25.16%|0.56pp|0%|63.33%|0%|2/0/0|new崩溃|
|D41-BEC|86.11%|20.56%|78.67%|31.50%|65.56pp|0%|0%|36.67%|142/0/32|old崩溃|
|D53-INT8|92.22%|81.67%|83.33%|81.28%|10.56pp|23.33%|53.33%|73.33%|26/8/17|接近D46但略退化|
|D53-FP32|92.22%|81.67%|83.33%|81.28%|10.56pp|23.33%|53.33%|73.33%|26/8/17|与int8一致|

## 10.分场景与逐类性能

|场景|before|after|new|H|forget|joint|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|clear|98.33%|90.00%|98.00%|93.57%|8.33pp|40%|70%|90%|4/1/0|高性能|
|low-elev|88.33%|80.00%|72.00%|73.99%|8.33pp|20%|60%|50%|8/5/9|new/floor失败|
|rain|90.00%|75.00%|80.00%|76.28%|15.00pp|10%|30%|70%|14/2/8|old/forget失败|

|角色|匿名类|总体性能|主要表现|
|---|---|---:|---|
|old|O0|90.00→90.00%|稳定|
|old|O1|96.67→93.33%|稳定|
|old|O2|96.67→90.00%|小幅下降|
|old|O3|80.00→53.33%|old floor类|
|old|O4|100.00→73.33%|遗忘明显|
|old|O5|90.00→90.00%|稳定|
|new|N0|73.33%|new floor类|
|new|N1|86.67%|可用|
|new|N2|76.67%|偏低|
|new|N3|90.00%|最佳|
|new|N4|90.00%|最佳|

## 11.十五个outer行

|场景|fold|before|after|new|H|forget|joint|floor(b/a/n)|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|100.00%|90%|94.74%|0pp|50%|100/100/50%|0/1/0|
|clear|1|100.00%|83.33%|100%|90.91%|16.67pp|0%|100/0/100%|1/0/0|
|clear|2|91.67%|83.33%|100%|90.91%|8.33pp|50%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|low|0|91.67%|75.00%|80%|77.42%|16.67pp|50%|50/50/50%|3/1/1|
|low|1|66.67%|58.33%|70%|63.64%|8.33pp|0%|50/50/0%|2/0/3|
|low|2|91.67%|91.67%|50%|64.71%|0pp|0%|50/50/0%|0/2/3|
|low|3|100.00%|100.00%|70%|82.35%|0pp|0%|100/100/0%|0/1/2|
|low|4|91.67%|75.00%|90%|81.82%|16.67pp|50%|50/50/50%|3/1/0|
|rain|0|83.33%|83.33%|60%|69.77%|0pp|0%|50/50/0%|2/0/4|
|rain|1|100.00%|58.33%|90%|70.79%|41.67pp|0%|100/0/50%|5/1/0|
|rain|2|91.67%|83.33%|80%|81.63%|8.33pp|50%|50/50/50%|1/0/2|
|rain|3|91.67%|75.00%|90%|81.82%|16.67pp|0%|50/0/50%|3/0/1|
|rain|4|83.33%|75.00%|80%|77.42%|8.33pp|0%|50/50/0%|3/1/1|

## 12.相对版本表现

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D45|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|
|D46|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|
|D51|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|46.67%|70.00%|23/12/15|
|D52|90.56%|81.67%|80.00%|79.96%|8.89pp|26.67%|66.67%|66.67%|19/15/15|
|D53|92.22%|81.67%|83.33%|81.28%|10.56pp|23.33%|53.33%|73.33%|26/8/17|

D53相对D45改变5/15行：after`-0.56pp`、new`-0.67pp`、H`-0.88pp`、forget`+0.56pp`，min-new`+3.33pp`；相对D46改变5/15行，after/forget/floors持平，但new`-1.33pp`、H`-1.05pp`、old→new`+1`、new→new`+2`。相对D51，new`+1.33pp`、min-after/new各`+6.67/+3.33pp`且new→old`-4`，但after`-0.56pp`、forget`+0.56pp`、joint`-3.33pp`。因此D53没有超过D46。

## 13.谱机制、训练、量化与资源

|阶段|correction L2 min/mean/max|transport norm mean/bound mean|判定|
|---|---:|---:|---|
|before|0.0159/0.0919/0.2314|0.1630/0.4455|强收缩|
|final|0.0061/0.1248/0.7794|0.2166/0.4672|强收缩|

谱界最大超额为负（before`-0.2229`、final`-0.2085`），实现严格收缩；final correction均值比D51/D52的`0.736/1.149`明显更小。20epoch完整，epoch1/5/10/15/20 loss为`1.031996/0.415989/0.216143/0.142408/0.102685`，support acc为`95.14/97.78/99.03/99.72/100%`，query rows全0。

量化before/final argmax变化`0/0`、margin翻转0、support变化`0/0`，最大score误差`0.001588`。36次LDA fit、LDA MAC`1,065,830,400`；D53额外`430,272`，总适配`1,071,237,312`，query MAC`6,624`；参数2,016、state8,583B、CUDA peak22,886,912B；协议访问项全0/false。

## 14.Artifact与最终判定

|文件|大小/B|SHA256|
|---|---:|---|
|D53_PROBE_METADATA.json|1,790|`d77eccbe4e3ccd9f00ea151bc659ec0a96b2ce4d11599afc8d4505b716e2e440`|
|RECEIPT.json|4,845|`db4cdeebfadc35b1066a238badc6455132b79439871b681969070b6fabcdd774`|
|selection.json|2,990|`be28ec0406158adf1f6971dc73c0d3ecc047d6da03e3ce5263996d8ca92d78e8`|
|support_audit.json|313,484|`bc545cfc458e00318f31209d9c85390ba3323fe53fce1c515a6213010d4fc44b`|
|training_log.jsonl|43,056,696|`cdf9c1dc550270f8b5c4d7386a33481189b05b42a68ee7ad62690bd4c6a11c27`|
|full_performance_summary.json|84,723|`49c57c88e4c577e9f1b0e32fbde7a46661208b7e75ee8f75cf6b68b99051e264`|

D53说明谱映射成功把修正缩到安全范围并基本保持D46，但没有带来联合收益：new/H下降，rain old/forget仍失败。停止该公式，不扫描谱尺度、不加逆/正则/clip或角色门控、不跑第二seed、不formalize、不运行125。当前最强仍是D46，仍未满足项目要求。

## 15.D54成功经验合成预注册

D54仅把D53相同的谱收缩transport叠加到当前最强D46 classwise LOO底座，公式中的`W0`改为`W_D46-mean(W_D46)`；其余`U/M0/tau/G/gamma`、K1/K2 fallback和协议边界完全不变。无融合系数、扫描、clip、role/scene/query门控。目的只检验：D46的new优势能否与D53的安全几何修正共存。相同开发单元运行一次，之后必须进行D52–D54三轮回顾；失败不进入第二seed或125。

`||G||_2≤||U||_2/||M0||_2`；无pinv、ridge、rcond、alpha、阈值、clip或扫描。K1/K2在谱检查前精确D45 fallback。

## 3.协议与比较

复用同一`VALIDATED_ONCE`、`p2_min_v1`胶囊；仅support固定received IQ视图，query test-only；禁止clean/source、truth/role/count/quota/global assignment/query optimization/dense query graph。before/final同式。直接比较D45、D46、D51、D52，必须同时评价domain adaptation和new-class registration。

## 4.本地文件与验证

|文件|用途|
|---|---|
|`code/scripts/probe_d53_spectral_contracted_median_transport.py`|探针、closure、资源账|
|`tests/test_probe_d53_spectral_contracted_median_transport.py`|公式、谱界、对称性、fallback测试|
|`analysis/d53_spectral_contracted_median_transport_traceability_20260719.md`|追踪矩阵|
|本报告|执行与性能证据|

D53定向11/11通过；D45–D53联合127/127通过。`py_compile`、`git diff --check`通过。

## 5.成功与停止门

- 相对D45至少1/15预测变化；
- 总体及场景after/new/H、forget、joint、min-after、min-new联合审查，不允许old改善换new伤害；
- 至少保持D46的new84.67%、min-new73.33%，并改善old侧、遗忘或joint；
- query/role/quota/count/global/clean/source保持0/false，量化变化0/0/0；
- 失败即停止，不扫描谱尺度、不clip、不加role/scene门控、不跑第二seed、不formalize、不运行125。

## 6.计划运行与完整报告

实现提交后建立`E:\type10-7\code\snapshots\d53wt`clean worktree；runtime继续只读`d41wt`；输出为`E:\type10-7\automation_reports\CV-SincNet\d53_spectral_contracted_median_transport_probe_20260719\spectral_contracted_median_transport`；本地`ssr-gpu`、`device=auto`、单进程。完成后必须报告7候选、3场景、逐类、15 folds、相对D45/D46/D51/D52、20epoch、混淆、谱行为、量化、资源及artifact SHA，不得只报告缺陷。

## 7.执行锁与exact command

- 实现提交`284b8313`；clean detached worktree`E:\type10-7\code\snapshots\d53wt`，状态`## HEAD (no branch)`。
- 探针SHA256`7318d941ee202bba4e2b695e8e3b4ec95f0cdb51fbe59de1ef563819ad832156`；clean worktree定向11/11通过。
- 六个输入hash沿用D52已闭合值；runtime存在；输出启动前不存在。无N607连接。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d53wt\code\scripts\probe_d53_spectral_contracted_median_transport.py' `
  --d53-arm spectral_contracted_median_transport `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d53wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d53_spectral_contracted_median_transport_probe_20260719\spectral_contracted_median_transport' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
