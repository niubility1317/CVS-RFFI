# D112-SEAM-qKNN source-held G1报告（r3）

状态：`ARTIFACTS_COMPLETE / ANALYZED / HEAD_POSITIVE / CLOSE_SEAM_DA`

- run ID：`d112_g1_sourceheld_seam_20260802_r3`。
- 目标／矩阵：固定`M0/M_HEAD_GROUND/M_JOINT_SEAM`，63行／189个prediction单元；完整prediction后独立score。
- 输入：source-held archive`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`；manifest`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`；tap`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 代码：runner`f43f0532`＋字段合同修复`1b6a9711`＋只读收据修复`049c4927`；三臂理论／surface／head均不变。
- 验证：`ssr-gpu`编译通过，26项聚焦测试通过。r1/r2的prediction row、manifest、truth-open和score均为0，不含性能数据。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d112_g1_sourceheld_seam_20260802_r3\artifacts`，全新不可覆盖；本地NumPy，不使用N607/GPU。
- 裁决：完整同row old BA、seen-new、H、old floor和negative tail；负收益关闭D112，不调参、不补seed、不跑125。
- 本次为第二轮发布修复后的最后一次当前runner尝试；若技术上仍不闭合，冻结更小独立入口，不再修补此runner。

## 完成证据

|artifact|SHA256／覆盖|
|---|---|
|package manifest|`e15de8783dd70da1d37f826bf844dbf8651f33c6229d67fdb046bc208f3ef955`；21个package|
|prediction manifest|`ded399f6e6f52326a749c86b4768762911b58565d00af555254d4787b0c742a4`；63行／189单元／63个唯一receipt|
|truth-open event|`322be63b19e559b5be2bbd649e0256f895966ad3a4f639be49b3e384704d9001`|
|held scores|`00f6de8cc7b6c8ba7c8408597f095758b0e8e1fd3ba53a1df325b7e7c6463cb0`；63个performance row|
|score receipt|`432a0731fdb8d353462213f832df19bbb1644325bea0a9dd57a233d8bf329e8b`|

预测封存时`query_truth_access=false`、`target_access=false`；truth-open前已复核63/63行、189/189单元和63个唯一prediction receipt。

## 全类一般行（21行）

|范围|arm|balanced accuracy|mean row floor|correct/query|相对M0|
|---|---|---:|---:|---:|---|
|K1|M0|84.0388%|57.6720%|953/1134|基线|
|K1|M_HEAD_GROUND|85.3616%|62.4339%|968/1134|BA`+1.3228pp`，floor`+4.7619pp`|
|K1|M_JOINT_SEAM|85.3616%|62.4339%|968/1134|与head逐prediction相同|
|K5|M0|84.9896%|60.8696%|821/966|基线|
|K5|M_HEAD_GROUND|85.6108%|60.8696%|827/966|BA`+0.6211pp`，floor持平|
|K5|M_JOINT_SEAM|85.6108%|60.8696%|827/966|与head逐prediction相同|
|K10|M0|84.3915%|54.7619%|638/756|基线|
|K10|M_HEAD_GROUND|84.3915%|54.7619%|638/756|完全持平|
|K10|M_JOINT_SEAM|84.3915%|54.7619%|638/756|完全持平|

21行合并均值中，head／joint的BA为85.1213%，M0为84.4733%，增益`+0.6480pp`；mean row floor为59.3551%对57.7678%，增益`+1.5873pp`。

## K1登记行（42行）

|arm|old BA|seen-new|H old/new|old floor|all-class floor|相对M0|
|---|---:|---:|---:|---:|---:|---|
|M0|84.0388%|84.0388%|82.3063%|59.4356%|57.6720%|基线|
|M_HEAD_GROUND|85.3616%|85.3616%|84.2799%|64.0212%|62.4339%|`+1.3228/+1.3228/+1.9736/+4.5855/+4.7619pp`|
|M_JOINT_SEAM|85.3616%|85.3616%|84.2799%|64.0212%|62.4339%|与head逐prediction相同|

`JOINT_VS_M0`的42行negative tail为：old BA`6`、seen-new`2`、H`7`、old floor`1`；对应正／零／负分别为old BA`17/19/6`、seen-new`5/35/2`、H`17/18/7`、old floor`17/24/1`。以7个receiver为cluster的50,000次配对bootstrap中，均值及95%区间为：old BA`+1.3228pp[0.0000,3.1746]`、seen-new`+1.3228pp[0.0000,3.1746]`、H`+1.9736pp[0.0000,4.9765]`、old floor`+4.5855pp[0.4409,10.6702]`。收益方向为正但存在receiver异质性。

## 机制归因、资源与对比

- `M_HEAD_GROUND−M0`：一般行改变52/2856个prediction，登记行因六类角色复用改变234/6804个计数prediction；形成上述正收益。
- `M_JOINT_SEAM−M_HEAD_GROUND`：一般行与登记行均为0个prediction变化，63/63行所有性能效应严格为0。
- 这不是SEAM无功能：21个唯一package的joint state均有6个正ρ旧类，最大`rho=0.562422`、最大`alpha=0.387345`、最大anchor位移`0.014563`；只是运动未跨越任何决策边界。
- 资源：ground head／joint持久数值态均4308B、query依赖态0B、每query额外上界960MAC；ground enrollment 960MAC，joint enrollment 2880投影MAC＋90个LOO标量项，无训练、反向传播或优化器。
- 同一source-held split上，D110 M_JOINT的登记old BA／seen-new／H／old floor为82.4515%／82.4515%／79.5106%／49.3827%，而D112 joint为85.3616%／85.3616%／84.2799%／64.0212%；但D112优势全部来自ground head，不能归因于SEAM motion。

历史Target125的D92、D62、SVRN-qKNN-BCRR和D91不与本source-held G1混排：D92完整125的after old／seen-new／H为65.56%／58.93%／61.57%，D62为64.39%／59.11%／61.09%，SVRN为43.03%／23.46%／29.25%；D91只有15行development且prediction与matched D62逐哈希相同。D112尚无Target125证据，因此不能宣称超过D62或D92。

## 最终裁决

存在正收益版本：`M_HEAD_GROUND`。它是轻量静态Phase1 ground-anchor head，不是新的域适应胜利。`M_JOINT_SEAM`虽然相对M0为正，但与head逐prediction完全相同，域运动独立收益为0；关闭D112-SEAM motion，不补seed、不跑125、不调参。下一方法研发必须以`M_HEAD_GROUND`为新基础，证明独立DA增益后才进入性能实验。
