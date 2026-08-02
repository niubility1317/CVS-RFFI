# D122-RDCE×静态ground head source-held G1报告（standalone r3）

## 1.登记与目标

|字段|内容|
|---|---|
|run ID|`d122_g1_sourceheld_standalone_add8f7d5_20260803_r3`|
|状态|`ARTIFACTS_COMPLETE / ANALYZED / CLOSE_D122_COMPOSITION_RETAIN_COMPONENTS`|
|时间/操作员|2026-08-03，Codex主agent＋唯一Terra Max runner|
|目标|以不依赖D106 construction链的独立入口，执行冻结的D122四臂source-held G1必要矩阵|
|假设|RDCE同坐标输运后的D112静态ground head可改善old/new/H/floor；完整配对结果不支持时立即关闭D122|
|比较目标|`M0`、`M_DA=RDCE＋qKNN`、`M_HEAD=identity＋static ground head`、`M_JOINT=RDCE＋同坐标static ground head`|
|矩阵|63行×4臂=`252`prediction单元；7 receiver；6 class；K=`1/5/10`；seed=`104713`|

必须报告`DA_AT_BASE`、`HEAD_AT_ID`、`HEAD_AT_DA`及交互项。不得从局部行挑选候选，不运行125矩阵，不因性能较弱停止健康执行。

## 2.版本与本地闭包

|项目|证据|
|---|---|
|设计冻结|`e8b84afa`|
|D122方法实现|`d5a1892a`|
|standalone入口|`add8f7d54609e9f58344ad9f4c8925e2ce80dc12`|
|独立审查|`MERGE / P0=0 / P1=0`|
|定向测试|`17 passed`；`py_compile`通过；CLI真实import通过|
|数值等价|真实RDCE wire＋21个真实package上，K=`1/5/10`的asset、basis、attenuation、state receipt、support/query transform与D106参考路径逐bit/逐receipt一致|
|静态import闭包|8个项目文件；禁止D106/D105/D104 construction、model、旧D106 runner命中0；动态import命中0|
|本地真实无truth smoke|63行、252单元、63个唯一prediction receipt、21个RDCE state receipt；global valid全通过；fallback=0；新类逐bit边界失败=0；query truth/target/fit/update/selection均为0|
|本地smoke manifest|SHA256=`39b5c24ea5ea3f22f7aa71f80fd87e5354241a56e4682f472ce5a92552feb550`|

r1、r2均为0prediction且truth未打开的技术失败，没有性能结果，也不得重启或复用。r3是按两轮release repair上限冻结的新独立入口；方法、参数、输入与矩阵均未改变。

## 3.文件与SHA256

|相对路径|SHA256|用途|
|---|---|---|
|`code/scripts/run_d122_g1_sourceheld_standalone.py`|`512132757cb631b253d620a3f8332cdb717856415a45e81095e6057b831b17d0`|唯一predict/score入口；本轮唯一需要同步的新代码|
|`code/cvsrffi/stage2_d122_rdce_ground_head.py`|`1d244d2ef89fc4bbd9d87c02a83f008548fd65e62bc05a579c77ad5897764197`|冻结D122组合核心|
|`code/cvsrffi/stage2_d112_g0_source_bundle.py`|`4938b1cb04146b5d14b1063102292b8d94bbc80df3ecdd901ea249ee2c2087d2`|安全依赖|
|`code/cvsrffi/stage2_d111_g0_source_bundle.py`|`6450627dbb37c32bf0f97960dc881a2b74c667e8ee7e2c5468a7951164affcd7`|安全依赖|
|`code/cvsrffi/stage2_d111_loo_gat_bundle.py`|`2e81fe22c90516ea9f0d78f32686a1dcd15f922cc0f52869a4b6d3aab5918492`|安全依赖|
|`code/cvsrffi/stage2_d112_seam_bundle.py`|`aa640c0d10b4239070591083552dc385328fe78b8b431305732646e4ce06fe90`|安全依赖|
|`code/cvsrffi/stage2_d112_seam_qknn.py`|`4927c4cd505ac83f539090c05fd61fcf4391c49db2d8a8f4a7ec63f05939b903`|安全依赖|
|`code/cvsrffi/stage2_zid_student_t_qknn.py`|`f7bc2ab7e6f9457085973099431db934edfa840ba37e904288ff4720726101e2`|冻结qKNN实现|

本地测试文件为`tests/test_run_d122_g1_sourceheld_standalone.py`。runner必须逐一核对上述8文件远端SHA；除新standalone入口外，不重复同步SHA已匹配的依赖。

## 4.输入、资源与精确命令

|字段|内容|
|---|---|
|远端项目根/CWD|`/home/szu2070436088/2510044040/CV-SincNet`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|package root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages`|
|RDCE wire|`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire`；SHA256=`20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795`|
|tap/tap receipt|同步至`<run root>/input/`；SHA256分别为`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`、`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3`|
|GPU|`CUDA_VISIBLE_DEVICES=0`；无训练，仅轻量特征侧predict/score|
|日志/PID|`<run root>/runner.log`、`<run root>/main.pid`|

远端真实import smoke：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -c "from scripts import run_d122_g1_sourceheld_standalone as m; print('D122_STANDALONE_IMPORT_OK',m.CANDIDATE_ID,m.ARMS,m.K_VALUES)"
```

predict：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d122_g1_sourceheld_standalone.py predict --package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages --rdce-asset-wire /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/rdce_asset/d106_rdce_gtsm.asset.wire --rdce-wire-sha256 20e44cb0eb2f5698e6d5f9029b63cf296ffbf4716edb999fed6743c8671bd795 --d106-tap-archive /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/input/d106_ls_strict_tap.npz --d106-tap-receipt /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/input/d106_ls_strict_tap.receipt.json --d106-tap-archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d122_g1_sourceheld_standalone_add8f7d5_20260803_r3 --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/predictions
```

score仅在63行完整封存后执行：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/run_d122_g1_sourceheld_standalone.py score --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/predictions --truth-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth.json --truth-input-seal-json /home/szu2070436088/2510044040/CV-SincNet/runs/d106_g1_sourceheld_b442472b_20260801_r2/packages/scorer_only/truth_input_seal.json --truth-open-event-json /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/truth_open_event.json --output-json /home/szu2070436088/2510044040/CV-SincNet/runs/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/held_scores.json
```

### r3唯一runner handoff

|字段|冻结值|
|---|---|
|唯一发布owner|Terra Max runner；主agent不并发启动同run ID。|
|发布基线|standalone add8f7d54609e9f58344ad9f4c8925e2ce80dc12；报告基线97abe058。|
|direct preflight|通过；普通N607账号、项目/Python可见、GPU compute-app为空。|
|不可覆盖检查|r3 run root在同步前确认不存在。|
|同步范围|仅standalone入口和tap/receipt；其余7个依赖只核对SHA，不重复同步。|
|启动门|8文件远端SHA全部匹配且真实import smoke输出候选ID、四臂、K=(1,5,10)。|
|执行顺序|detached predict完整封存63行/252单元后，独立score才可打开truth。|
|fresh retry|不授权；任何技术失败保留artifact并记NO_PERFORMANCE_RESULT。|

## 5.健康边界与预期产物

只允许P0协议/安全/错误hash/错误checkout/覆盖风险，或至少两个独立单元出现同一deterministic exception fingerprint且零prediction时，停止该run-owned进程树。禁止读取accuracy、BA、H或floor决定停止。

预期产物：`prediction_manifest.json`、63个row JSON、`truth_open_event.json`、`held_scores.json`、完整`runner.log`、PID/exit/cleanup receipts。成功条件为63行/252单元全部封存并完成独立score；失败必须保留artifact并标`NO_PERFORMANCE_RESULT`，不得重启或覆盖同ID。

## r3唯一runner技术闭合

|字段|技术事实|
|---|---|
|预登记提交|2af5587a|
|远端闭包|8/8代码文件SHA匹配；tap/receipt SHA匹配；standalone import和py_compile通过。|
|启动与退出|wrapper PID120965；自然退出；exit=0。|
|预测闭合|63行、252单元、63个唯一row receipt；prediction目录64个文件。|
|评分闭合|63个performance row；truth_open_after_prediction_seal=True。|
|query边界|query truth=False、target access=False、fit/update/selection=0。|
|异常fingerprint|0。|
|资源清理|GPU compute-app为空；本地ssh.exe=0；到N607:22的ESTABLISHED=0。|
|本地artifact根|E:/type10-7/automation_reports/CV-SincNet/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/artifacts/n607|
|prediction_manifest SHA256|b7d11435564f6508b77846582e4f92236fb42ef9ac8600725329c259c57eea7c|
|truth_open_event SHA256|5c066ab8c4340f2a33f2bafd873330bbb7b61b0f6b615b16901ab4470d3429c2|
|held_scores原始路径|E:/type10-7/automation_reports/CV-SincNet/d122_g1_sourceheld_standalone_add8f7d5_20260803_r3/artifacts/n607/held_scores.json|
|held_scores SHA256|cc335dfb4ea05b3b62ef59fe415de9a1494d45c27dab7cdf03bedd57adc5549f|
|runner.log SHA256|6a9f35ab256cd1fca967cbb6d4db18137637179048ffdf042036c2d8e52883b2|
|main.pid SHA256|187a8088e7f2f2b4e2d9e8c3021a98af5ed7dc367854b0cf4c5cab61beb899fb|
|exit_code SHA256|9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa|
|cleanup receipt SHA256|d590ec1ca4689c7b0088cdf46d603ee2da6588cb37593176a345090b575269e5|

以上仅为执行与artifact事实，不包含性能解释或晋级决定。

## 6.完整同row结果与判定

下表中old BA、old floor、all floor和correct按63行平均；seen-new与H按42个held-class行平均。前五项为百分比，correct为每行平均正确样本数。

|臂|old BA|seen-new|H|old floor|all floor|correct/row|判定|
|---|---:|---:|---:|---:|---:|---:|---|
|M0|83.6560|83.7772|82.2378|57.9803|56.4199|288.9683|基线|
|M_DA|83.9163|84.1404|82.6826|58.2627|56.8637|289.8889|RDCE小幅正收益|
|M_HEAD|84.7348|85.0282|83.7209|58.8255|56.5657|292.7460|平均正收益，但接收机不均匀|
|M_JOINT|84.7233|85.0282|83.7389|58.6049|56.4257|292.7143|均值提升但floor与交互不稳，不晋级|

### 6.1冻结效应

|效应|Δold BA|Δseen-new|ΔH|Δold floor|Δall floor|Δcorrect/row|关键胜/平/负|
|---|---:|---:|---:|---:|---:|---:|---|
|DA_AT_BASE|+0.2604|+0.3632|+0.4447|+0.2824|+0.4438|+0.9206|BA 25/25/13；H 17/18/7|
|HEAD_AT_ID|+1.0788|+1.2510|+1.4831|+0.8452|+0.1457|+3.7778|BA 25/27/11；H 14/20/8|
|HEAD_AT_DA|+0.8069|+0.8878|+1.0563|+0.3422|-0.4380|+2.8254|BA 25/15/23；H 17/6/19；correct 26/14/23|
|factorial interaction|-0.2719|-0.3632|-0.4268|-0.5030|-0.5838|-0.9524|H 12/12/18；correct 20/16/27|

`HEAD_AT_ID`在均值上确有正收益，但D122新增问题是head能否与RDCE稳定组合。`HEAD_AT_DA`虽然提升old/new/H和平均正确数，却降低all-class floor，负交互同时覆盖old/new/H/floor/correct，说明两组件没有形成互补增益。

### 6.2接收机与K稳定性

|held receiver|HEAD_AT_DA ΔH|Δall floor|Δcorrect/row|
|---|---:|---:|---:|
|1-1|+10.723|+0.869|+25.000|
|1-19|+0.398|+3.020|+1.000|
|14-7|-0.591|+0.848|-1.333|
|18-2|-2.870|-8.026|-4.222|
|19-2|-0.860|-1.318|-2.333|
|2-1|+0.595|+1.540|+1.667|
|2-19|0.000|0.000|0.000|

整体均值主要由receiver 1-1的大收益驱动；18-2与19-2同时损害H、floor和正确数，不能称为跨receiver稳定正收益。

|K|行数|HEAD_AT_DA Δold BA|ΔH|Δold floor|Δall floor|Δcorrect/row|
|---:|---:|---:|---:|---:|---:|---:|
|1|49|+0.8878|+1.0563|-0.6918|-1.6949|+3.1429|
|5|7|+0.9524|不适用|+6.4935|+6.4935|+3.1429|
|10|7|+0.0952|不适用|+1.4286|+1.4286|+0.2857|

K=5/10只有无held-class行，不能替代新类注册证据；唯一同时覆盖old/new的K=1层面，joint的两个floor均下降。

### 6.3最终决定

`CLOSE_D122_COMPOSITION_RETAIN_COMPONENTS`：保留D106 RDCE作为已验证的小幅正收益域适应组件，保留D112静态ground head作为正收益head路线；关闭D122的RDCE×ground head组合，不调参、不换seed、不扩125矩阵。下一轮从轻型域适应原理出发设计新的可辨识候选，不继续修补该组合。
