# D122-RDCE×静态ground head source-held G1报告（r2）

## 1.登记与目标

|字段|内容|
|---|---|
|run ID|`d122_g1_sourceheld_dab6621e_20260803_r2`|
|状态|`STARTUP_IMPORT_SMOKE_FAILED / NO_PERFORMANCE_RESULT`|
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

### r2唯一runner同步映射与精确命令

本地根为E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt，远端根为/home/szu2070436088/2510044040/CV-SincNet；以下33个相对路径按原路径一对一同步：

|相对路径|local SHA256|
|---|---|
|code/cvsrffi/dual_feature_forward.py|eeaca06f84f5771c90dfb92e6bbbc4980f2772e9fcdf80d54e06fee387afd815|
|code/cvsrffi/leo_weak_cache.py|19c98daafcc6f3e6f2de038883b83ea10c4d59edca62ff0e73cb509175c57ef8|
|code/cvsrffi/phase1_rb_metabias4_bundle.py|a88fafc8c948e2ecfe223baa9f84012f831d88b8423f5a3c5c5e65d80db3fb06|
|code/cvsrffi/rxid_metabias4_bundle.py|817bea937e2f5bcdf45f9a5a7db2a5c68c656666e61c30ae39bedbe99f372414|
|code/cvsrffi/rxid_metabias4_held_execution.py|571ddb448cd44131a05ff6187fbb66ad20ae115af57412447e8a92c08c39cc1e|
|code/cvsrffi/rxid_metabias4_phase1_trainer.py|e00ee30081efcfd42917c97bbe3997958fa9dfdc67cb45194f420edca4d00527|
|code/cvsrffi/rxid_metabias4_source_archive.py|dcfe6f0c8d0c49b06d7482185329a389eba3f14f542790d6d3577d8b48f3e764|
|code/cvsrffi/somph_runtime_trust.py|4b1dee1d8ffdc793f48c46c21a11b0fdf8b6ef6e3b253807cc1138011dc1f9fc|
|code/cvsrffi/stage2_d104_source_split.py|5175c3ff6b9522394445c7de4f3e50018ac024a56f39d0f400339232cb088f3b|
|code/cvsrffi/stage2_d105_cbrc.py|9f07c1db63bf518ce11089bd099462cf22d19c2c2c643731f80a5faef53203f6|
|code/cvsrffi/stage2_d105_feature_tap.py|4aab4febf63fe93fff73e9f78acf240db44cb086e71a7c368e9467d814c78a56|
|code/cvsrffi/stage2_d105_phase1_authority.py|fe81a728bcd8e1047a40069b9d9954aed2af1c89b98633489ccf2b922b4364bd|
|code/cvsrffi/stage2_d105_phase1_bundle.py|91931cb3893cb902a7eef1e509d209b232d2769225b012c9a0027c978a3ced39|
|code/cvsrffi/stage2_d106_phase1_tap.py|5a63a5935748f17a1efcbf4069d5c80c1d99a8e813330a2c3a15895483c53e9b|
|code/cvsrffi/stage2_d106_rcmr_2v_qknn.py|1c33d87af6b8729f5f79b222423baf48669a8f309989fde92df48d12ed80f805|
|code/cvsrffi/stage2_d106_rcmr_g0.py|6d496298e21a1b9a1e5398ff9f507b44fa0164ca0332c50d29a385f2b7b5a2b2|
|code/cvsrffi/stage2_d106_rdce_asset.py|e9d57245a80cdf31ae4ea5fd76cd521022399d0be512e2647a92cc2a0671da1f|
|code/cvsrffi/stage2_d106_rdce_runtime.py|9d78b83134bfb668c3b9c32053eaa86b5c9fd4d970e87aa99dc30ac2df8df946|
|code/cvsrffi/stage2_d106_train_only_predecessor_lock.py|97cfc235d7b6832337d09627f80fa984b82f3bf753514b449e12e60bd28c46e3|
|code/cvsrffi/stage2_d111_g0_source_bundle.py|6450627dbb37c32bf0f97960dc881a2b74c667e8ee7e2c5468a7951164affcd7|
|code/cvsrffi/stage2_d111_loo_gat_bundle.py|2e81fe22c90516ea9f0d78f32686a1dcd15f922cc0f52869a4b6d3aab5918492|
|code/cvsrffi/stage2_d112_g0_source_bundle.py|4938b1cb04146b5d14b1063102292b8d94bbc80df3ecdd901ea249ee2c2087d2|
|code/cvsrffi/stage2_d112_seam_bundle.py|aa640c0d10b4239070591083552dc385328fe78b8b431305732646e4ce06fe90|
|code/cvsrffi/stage2_d112_seam_qknn.py|4927c4cd505ac83f539090c05fd61fcf4391c49db2d8a8f4a7ec63f05939b903|
|code/cvsrffi/stage2_d122_rdce_ground_head.py|1d244d2ef89fc4bbd9d87c02a83f008548fd65e62bc05a579c77ad5897764197|
|code/cvsrffi/stage2_lpo_rc_qknn.py|e88a55c239c31067eb4ae01c729039f73c2a5705d5406a45f761c33d97865492|
|code/cvsrffi/stage2_rb_metabias4_qknn.py|fd051f6c13a3bec243fa9ffed3c5841becd3ba2b5dc17a9896a011764389a3fd|
|code/cvsrffi/stage2_rxid_metabias4.py|1750dd1aea775034f74e921ca4f44b12df20c60ccb82182d97dcc195ded35294|
|code/cvsrffi/stage2_zid_student_t_qknn.py|f7bc2ab7e6f9457085973099431db934edfa840ba37e904288ff4720726101e2|
|code/model.py|afc6e6266a09fd5f5be967fed85254c6c92fa0241a0336fd5ffa3eb12aa1c417|
|code/model_dual_cvsincnet.py|11b56f2a763eb49de21d0a566d8e4420538cb7af310c9338f4e8d4a21c42c235|
|code/scripts/run_d106_g1_sourceheld_one_shot.py|500d22b473b6b803706471c1aad7798bf414486ca4f103a906d72ed6fd4dbc90|
|code/scripts/run_d122_g1_sourceheld_one_shot.py|0439bd189cc46fe5c50e7c7af490e7a3d3e307d322bc63c7dd8fb2487f07d3f5|

tap和receipt另同步至run root/input，SHA沿用第3节。工作目录=/home/szu2070436088/2510044040/CV-SincNet，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，CUDA_VISIBLE_DEVICES=0，日志=<run root>/runner.log。

~~~bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d122_g1_sourceheld_one_shot.py predict --package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages --rdce-asset-wire /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --d106-tap-archive /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/input/d106_ls_strict_tap.npz --d106-tap-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/input/d106_ls_strict_tap.receipt.json --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d122_g1_sourceheld_dab6621e_20260803_r2 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/predictions
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d122_g1_sourceheld_one_shot.py score --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/predictions --truth-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth.json --truth-input-seal-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth_input_seal.json --truth-open-event-json /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/truth_open_event.json --output-json /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_dab6621e_20260803_r2/held_scores.json
~~~

## 5.健康停止与预期产物

只允许P0协议/安全/覆盖/错误hash/错误checkout，或至少两行同一deterministic exception fingerprint且零prediction时停止精确run-owned进程树。禁止查看accuracy、BA、H、floor决定停止。失败保留全部artifact并标`NO_PERFORMANCE_RESULT`，不得重启同ID。

预期：`prediction_manifest.json`、63个row JSON、`truth_open_event.json`、`held_scores.json`、完整log、PID/exit/cleanup receipts。

## r2唯一runner技术闭合

|字段|技术事实|
|---|---|
|同步核验|33/33代码文件local/remote SHA256匹配；tap与receipt SHA匹配。|
|真实import smoke|exit=1；未输出D122_IMPORT_OK，因此主入口py_compile与detached launch均未继续。|
|异常|D122导入D106链时，stage2_d106_phase1_tap的construction closure报告expected a regular non-symlink file。|
|运行与truth|main.pid不存在；0prediction；predictions与truth_open_event均不存在；truth未打开。|
|清理|GPU compute-app为空；本地ssh.exe=0；到N607:22的ESTABLISHED=0。|
|import日志SHA256|c808e2eb0bdc3b6a1351f0c9f3b2b31f9796d380f1f663f8acca9261882e774d|
|cleanup回执SHA256|b0ff0910c29d378e9c0f08cb14502953267b0b5611733a8f8aeabee10408faf6|
|最终技术状态|STARTUP_IMPORT_SMOKE_FAILED / NO_PERFORMANCE_RESULT；r2未启动且不得重试。|

## 6.结果（TBD）

|臂|old BA|seen-new|H|old floor|all floor|correct|判定|
|---|---:|---:|---:|---:|---:|---:|---|
|M0|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_DA|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_HEAD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|
|M_JOINT|TBD|TBD|TBD|TBD|TBD|TBD|TBD|

当前没有D122性能结论。完整产物返回后由主agent完成同row、K、receiver、held-class和正确数分析并决定关闭或晋级。
