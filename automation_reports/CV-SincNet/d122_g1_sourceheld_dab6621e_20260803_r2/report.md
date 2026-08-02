# D122-RDCE×静态ground head source-held G1报告（r2）

## 1.登记与目标

|字段|内容|
|---|---|
|run ID|`d122_g1_sourceheld_dab6621e_20260803_r2`|
|状态|`LOCAL_VERIFIED`|
|时间/操作员|2026-08-03，Codex主agent＋唯一Terra Max runner|
|目标|在同一D104 source-held矩阵上分离RDCE域适应、D112静态ground head及其同坐标联合效应|
|假设|Jacobian输运后的ground head可在RDCE空间保留old/new/H/floor；完整同row结果不支持时立即关闭|
|矩阵|63行×4臂=`252`prediction单元；K=`1/5/10`；seed=`104713`|

四臂冻结为`M0`、`M_DA=RDCE＋qKNN`、`M_HEAD=identity＋static ground head`、`M_JOINT=RDCE＋同坐标static ground head`。必须报告`DA_AT_BASE`、`HEAD_AT_ID`、`HEAD_AT_DA`和交互项，不得用`M_JOINT-M0`替代独立效应。

## 2.版本、验证与r1边界

|项目|证据|
|---|---|
|设计|`e8b84afa`|
|方法实现|`d5a1892a`|
|r1技术闭合|`1d072ee0`|
|r2 release repair|`dab6621e`|
|独立审查|`MERGE / P0=0 / P1=0`|
|定向测试|`py_compile`通过；D122两份测试`14 passed`|
|本地真实无truth smoke|63行/252单元；manifest SHA256=`605bba274c67f77ff07913c6c39ab6c1ed8ab23bd461e78c245da187a9ce685a`；六旧类全激活、零fallback、新类逐bit边界全通过|

r1在0prediction、truth未打开前因远端缺少D106 wrapper技术退出，状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。r2使用全新不可覆盖run root，不重启、不覆盖、不复用r1输出；方法、参数和矩阵不变。r2仅移除无关D112 wrapper链，并同步完整静态import闭包。

## 3.输入与远端路径

|字段|内容|
|---|---|
|项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages`|
|RDCE wire SHA256|`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|tap SHA256|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|tap receipt SHA256|`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2`|
|GPU|`CUDA_VISIBLE_DEVICES=0`；无训练|

## 4.发布闭包

同步主入口及其33个静态项目内import依赖，逐文件记录local→remote映射和SHA256。启动前必须在远端项目`code`目录执行：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -c "from scripts import run_d122_g1_sourceheld_one_shot as m; print('D122_IMPORT_OK',m.CANDIDATE_ID,m.ARMS,m.K_VALUES)"
```

只有输出冻结候选、四臂和`(1,5,10)`后才可启动。exact predict/score命令、log、PID、同步清单与SHA由唯一runner在启动前补入。predict不得接受truth；63行prediction完整封存后，score才可打开`scorer_only/truth.json`与`truth_input_seal.json`。

## 5.健康停止与预期产物

只允许P0协议/安全/覆盖/错误hash/错误checkout，或至少两行同一deterministic exception fingerprint且零prediction时停止精确run-owned进程树。禁止查看accuracy、BA、H、floor决定停止。失败保留全部artifact并标`NO_PERFORMANCE_RESULT`，不得重启同ID。

预期：`prediction_manifest.json`、63个row JSON、`truth_open_event.json`、`held_scores.json`、完整log、PID/exit/cleanup receipts。

## 6.结果（TBD）

|臂|old BA|seen-new|H|old floor|all floor|correct|判定|
|---|---:|---:|---:|---:|---:|---:|---|
|M0|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_DA|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_HEAD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_JOINT|TBD|TBD|TBD|TBD|TBD|TBD|TBD|

当前没有D122性能结论。完整产物返回后由主agent完成同row、K、receiver、held-class和正确数分析并决定关闭或晋级。
