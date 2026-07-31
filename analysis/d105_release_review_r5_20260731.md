# D105 R5独立发布审查收据

状态：`LOCAL_RELEASE_GO（仅同一Git archive的本地发布闭包）`

审查日期：2026-07-31

审查人：独立terra-max release reviewer

审查对象：`D105-CBRC+LPO-RC-qKNN`，修复提交`46a65b3af2621d23bcc0a34631f45c8be17af4dd`

协议：`p2_min_v1`

本收据替代已被R3跨平台字节闭包P0作废的R4本地结论。它只批准完全相同的Git archive进入新的Phase1 source-held资产链发布准备；不构成Phase1 formal asset、Target25、125/300/600矩阵、任何性能、稳定性或`PROMOTABLE`证据。

## 1.最终裁决

|等级|数量|裁决|
|---|---:|---|
|P0|0|未发现Git blob、runtime manifest、精确Git archive与解包执行文件之间的字节漂移；未发现清单外本地执行依赖、legacy exporter/SSDG/checkpoint-loading/model_modified可达路径、query真值/角色/配额访问或模型预检前打开。|
|P1|0|未发现阻止同版本本地发布的loader、checkpoint、LF、帮助面、无truth smoke、根/镜像报告或R3/R4追溯缺口。|
|P2|2|`model.py`仍有`torch.cuda.amp.autocast`弃用面；旧PyTorch不存在`safe_globals`时保留严格SHA绑定的兼容反序列化分支。当前审查环境实际走`weights_only_with_explicit_safe_globals`，两项均不改变本次结论。|

结论：`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`。

## 2.冻结身份与跨平台字节闭包

|对象|独立复核值|
|---|---|
|Git提交|`46a65b3af2621d23bcc0a34631f45c8be17af4dd`|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|candidate runtime manifest|`dc315ffe2860a9d76493ba5284aff6dfb9c248330613717a6614de6997da1cfc`；54个runtime文件|
|candidate method lock|`ac796d83e92ea1e8b5f0efa6e8a303f9eb989ba1f876219784cda1ac7363a030`|
|精确Git archive|`source_46a65b3a.tar`；242800640B；SHA256=`d313243c79eab306f988abadf67c2e207d380dba633f39a04e2cc63ffae7ed7a`|
|archive结构|4747项；唯一`source/`根；0个符号链接/硬链接；0个绝对或`..`逃逸成员|
|重生成核验|流式`git archive --format=tar --prefix=source/46a65b3a`与上述archive的长度和SHA256完全一致，即逐字节相同|
|54文件三面一致性|提交`code/...`blob内容SHA256=runtime manifest=archive member=解包`source/code/...`，逐项`54/54`通过|
|LF负测|54个runtime文件`CRLF=0`；`.gitattributes`对`*.py`和`*.sh`固定`eol=lf`|

## 3.解包执行闭包复核

全部下列命令均在精确archive的解包`source/`中执行，未读取Target、query真值或性能标签。

|检查|独立结果|
|---|---|
|canonical runtime/method loader|通过；runtime=`dc315ffe…a1cfc`、method=`ac796d83…3a030`、checkpoint绑定一致、实际逐文件验证54项|
|源码编译|54/54个runtime文件完成独立内存编译，0失败；不向archive源码树写入pyc|
|静态本地导入闭包|166条本地导入边均指向54个声明文件；清单外本地导入边=0；未发现`importlib`、`__import__`、`runpy`、子进程动态代码装载面|
|fresh-process真实checkpoint guard|真实checkpoint以`weights_only_with_explicit_safe_globals`加载；195个state tensor；模型为eval；factory与backbone均解析为archive内的`model_dual_cvsincnet.py`和`model.py`|
|禁止依赖|上述fresh-process构造后`SSDG`、`cvsrffi.checkpoint_loading`、`paper_reproduction`、`model_modified`、`scripts.export_phase1_jp4_tap_archive`和`scripts.export_phase1_singleobs_dual_feature_archive`均未导入|
|关键负测|archive内8项guard/preflight/query-tamper测试通过：legacy stack拒绝、tap-cache闭包、model可达性、预检先于模型、query truth/split篡改fail-closed等|
|CLI/子命令帮助面|9/9通过：5个正式D105 CLI，以及实际source-held链的`tap-cache`、`predict-source-held`、`open-truth`、`score-source-held`帮助面|

query evaluator的顺序已独立复读并由上述测试覆盖：四包authority/materialization、same-row lifecycle、runtime/method lock、qKNN、formal Phase1 asset、split与support/query物理ID/同IQ预检、checkpoint SHA与device检查全部在模型构造前完成。预测后仍强制`query_rows_used_for_fit=0`、`query_state_updates=0`、truth/role不存在，并按所有已注册类逐query独立决定。

## 4.无truth smoke与报告闭合

|对象|独立复核值|
|---|---|
|archive真实checkpoint无truth smoke|SHA256=`a915eb66c4df926e6f738a4de636026fa29cb9bf3968c5fb6a15007ffc47ce84`|
|smoke边界|`target_access=false`；`formal_phase2_evidence=false`；`performance_computed=false`；`query_truth_read=false`；`query_rows_used_for_fit=0`；`query_state_updates=0`|
|smoke技术不变量|400个source-held meta step完成；K1下`M_HEAD=M0`和`M_JOINT=M_DA`均精确成立|
|Phase1根报告与Git镜像|逐字节一致；SHA256=`d152fd2f71f7d861a36d50ffa23a844903ebb72b5ca43691ac0821e780fdf423`|
|Target25根报告与Git镜像|逐字节一致；SHA256=`6c7589342ab0900828e313e1d3375c8d8093ae32abf6b9995c187fe266c89872`|
|报告语义|两份报告均写明R3=`LANDED_PRELAUNCH_HASH_MISMATCH / NO_PERFORMANCE_RESULT`、R4已作废、R5=`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`，并保留无性能边界|
|工作区格式检查|`git diff --check`退出0|

## 5.仍然必须遵守的发布边界

1. R3永久封存：它在预启动哈希门失败且从未detach，不得恢复、重命名或当作性能运行。
2. 新的N607 release必须使用新的不可覆盖run ID、同一提交和同一archive；runner仍须先做preflight、远端archive/manifest/loader复核与本地SSH清理。
3. `LOCAL_RELEASE_GO`只允许进入Phase1 strict tap、source-held prediction、独立truth-open/score/gate、外部authority签名和formal asset验证链。任何一步失败均不得进入Target25。
4. Target25只有在同一正式Phase1资产链完整通过后，才可运行完整25job/300 scenario-arm pair/600 state prediction surface；此收据不提供任何目标域或性能结论。
