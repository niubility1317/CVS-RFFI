# D113-BCAT-qKNN真实G0报告

状态：`G0_ARTIFACT_COMPLETE / REJECT_REVISION_NO_FUNCTION / D113_CLOSED / N607_NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

## 1.身份与目标

|字段|值|
|---|---|
|实验ID|`d113_bcat_g0_real_20260802_r1`|
|日期|2026-08-02|
|operator|主agent负责理论、实现整合与功能分析；唯一Terra Max runner在本地提交后负责N607|
|目标|在真实588条Phase1 strict tap上验证BCAT的K1/K5/K10特征、score、margin和argmax是否均产生非零变化|
|比较|同fold、同support、同query的M0 Student-t qKNN|
|性能边界|G0禁止打开truth，不输出accuracy、H、floor或Target指标|

假设：六个旧类target support含有跨类共享的加性接收机偏移；固定ground投影的贝叶斯矩估计可在K1/K5/K10形成非零有效偏移，球面解析逆对全部old/new support及每条query统一恢复共同坐标。

## 2.冻结机制与门

- 理论：`analysis/d113_bcat_qknn_theory_20260802.md`，commit`d49e9a38`＋方差修正`f1dd828d`。
- 核心：`code/cvsrffi/stage2_d113_bcat_qknn.py`；source聚合：`code/cvsrffi/stage2_d113_g0_source_bundle.py`；入口：`code/scripts/run_d113_bcat_g0_one_shot.py`。
- 输入：固定588行tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- K=`1/5/10`，28fold，每K共588query；query fit/update/selection均为0。
- 每个K必须`feature/score/margin/argmax changed count>0`；任一K的argmax变化为0即`REJECT_REVISION_NO_FUNCTION`并关闭D113。
- 技术停止仅限协议/执行错误、覆盖风险、非有限数或重复确定性零prediction异常；不按性能停止。

## 3.本地验证与版本状态

|项目|状态|
|---|---|
|聚焦测试|`ssr-gpu`下6项通过；三个实现文件和测试`py_compile`通过|
|真实source aggregate smoke|6类；`tau_b2=0.0001681924623`；`sigma0=0.000407905–0.002466851`；`v_ground=0.000246369–0.001303704`；ground量化MSE=`2.62e-7–4.00e-7`|
|独立复审|首轮`P0=0/P1=1`：bundle与bank Phase1 lineage未共同绑定；最小修复后增量复审为`P0=0/P1=0/GO`|
|Git|实现尚未提交；增量复审后提交，不push|

## 4.运行面

|字段|本地真实功能run|N607发布|
|---|---|---|
|CWD|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`|本地提交后由唯一runner登记|
|环境|`ssr-gpu`|`ssr-gpu`|
|输出|`E:\type10-7\automation_reports\CV-SincNet\d113_bcat_g0_real_20260802_r1\artifacts\local_exact_g0_r2\result.json`|D113已关闭，不发布|
|GPU/PID/log|本地CPU闭式功能验证，无训练|待唯一runner补充|

本地使用冻结入口`python code/scripts/run_d113_bcat_g0_one_shot.py`，输入上述tap/receipt/checkpoint SHA，run ID=`d113_bcat_g0_real_20260802_r1_local_r2`。完整artifact SHA256=`a5924dca9c26bd6d562d4b5989eb7c5328b60401e580d04beddd4b10a53507b7`，execution root=`007eafe39a409ed8c9556072658c02cd0252b6cc23c068c1042c3f8f8f48b3f8`。

首次本地调用使用`local_exact_g0/result.json`，入口在完整计算后因父目录不存在而按不可覆盖合同退出，未生成输出文件。该问题不涉及方法、数据或数值；已记录为第1个release工程缺陷。修复仅预建精确父目录，并以新根`local_exact_g0_r2/result.json`重跑同一冻结命令。

## 5.结果表

|K|fold/query|feature变化|score变化|margin变化|argmax变化|最大`||b||`|最大state bytes|裁决|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|28/588|588|588|588|0|0.060113|2704|拒绝：无prediction变化|
|5|28/588|588|588|588|1|0.041110|2704|仅功能非零，不能单K晋级|
|10|28/588|588|588|588|0|0.040279|2704|拒绝：无prediction变化|

G0无性能值。三K共同裁决为`REJECT_REVISION_NO_FUNCTION`，`zero_argmax_k_values=[1,10]`，`g1_entry_allowed=false`。BCAT确实改变了全部feature、score和margin，且28fold均有非零`b`；但变化不足以跨过K1/K10的任何决策边界。该结果排除了“实现全回退”，同时也证明当前固定投影共同平移不是值得继续验证的强机制。

按目标文档，D113立即关闭：不调整`tau_b²`、收缩映射、variance、rho或核参数，不修runner，不发布N607，不运行source-held G1、Target25或125。下一轮必须研发新的HEAD/DA机制，不能把K5单个argmax变化当作正收益版本。
