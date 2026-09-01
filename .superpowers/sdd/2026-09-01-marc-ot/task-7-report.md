# Task7：MARC-OT生产Support特征ABI与Phase1/Phase2一致接入

## 设计追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|T7-01|brief Binding design；设计5|固定`marc_ot.support.row.v1`、685D及精确字段顺序|support feature builder及测试|PASS|`test_builder_has_exact_685d_schema_order_and_ignores_domain_aux`|完整实现640+3+2+6+4+16+10+1+3；未缩减为`z_id`|
|T7-02|brief TDD1-2|固定IQ与DC移除/unit-RMS view；CFO/SFO仅代理；16-bin PSD与RF-lite/quality|support feature builder及测试|PASS|确定性、置换、tone/chirp/PSD、audit测试；RF/ADV相邻41项通过|复用公开`extract_rf_lite_quality`，未复制旧算法|
|T7-03|brief Binding design|唯一opaque token、K/mask、几何和有限性fail closed|support feature builder及测试|PASS|K1/2/5/10/20、mask、重复token、K漂移、非有限及aux漂移负测|不解析token语义；K10五折fit子集记录nominal K10/effective K8|
|T7-04|brief TDD3；设计5、7|encoder先做masked逐类raw mean/diag-var/norm与可用性标记，再做共享DeepSets|encoder及测试|PASS|encoder K/mask/统计/置换测试|生产forward总会返回真实逐类统计；新增诊断字段为旧校准器手工构造保留空默认值|
|T7-05|brief TDD4；设计4.4|bundle绑定feature schema/config/dim，拒绝旧160D与ABI漂移|checkpoint及测试|PASS|round-trip、schema/dim/config/old160/missing负测|顶层bundle schema保持`marc_ot_weight_bank_v1`|
|T7-06|brief TDD5；设计7|Phase1 trainer直接调用共同builder，support-only且保留outer梯度路径|trainer及测试|PASS|trainer builder spy、query隔离及outer梯度测试|builder只收到source episode support IQ/label/opaque physical token|
|T7-07|brief TDD5；设计2、5、6|Phase2真实CLI的fold/full/stage均调用共同builder；validation不进入fit builder|runner、CLI、config及测试|PASS|真实`_adapt_unit`spy、stage gradient及Task5 cross-fit负测|fold plan和stage只收到fit tokens；full-support仅在fold后重拟合；无生产`support_encoder(z_id,...)`|
|T7-08|brief TDD6|聚焦、Task1-5相邻回归、CLI help及compile|本Task代码/测试|PASS|86+153+41项通过；CLI help与`py_compile`通过|使用`ssr-gpu`串行执行；41项相邻测试仅有既存TorchScript弃用告警|
|T7-09|brief TDD7|精确stage、指定commit、push与远端OID回读|本Task实际文件|READY|提交主题固定为`feat: add MARC-OT support feature ABI`|本报告所在提交无法自引用最终OID；OID与远端回读由交付状态记录|

## 当前接口核对

- ADV3B02生产forward顶层提供`z_id`，identity分支`aux_id`提供`t_emb/f_emb`；Task7不读取`z_dom/aux_dom`。
- 既有公开`cvsrffi.stage2_m23_rfguard.extract_rf_lite_quality`提供10维尺度不变RF-lite与quality，可直接复用而不改旧模块。
- Task5 config validator维持旧精确字段集；Task7在真实CLI内校验并剥离新增ABI绑定后复用旧validator，避免越界修改Task5模块。

## TDD记录

- Builder RED：首次收集因`cvsrffi.marc_ot_support_features`不存在失败；实现固定685D builder后16项转绿。随后新增K10五折fit子集测试，旧校验因每类8行与nominal K10不等而RED；改为只允许等长且不超过nominal K的显式mask fit子集后转绿，并保留validated-unpadded严格等于nominal K。
- Encoder RED：旧3参数forward不接受`effective_mask`；实现masked逐类raw mean、diagonal variance、两项norm及4项统计可用性flag，再经共享`phi`与逐类mean/variance pooling后18项转绿。
- Checkpoint RED：旧bundle无`support_feature`绑定且接受160D encoder；加入schema/config/dim严格绑定和旧160D拒绝后17项转绿，顶层schema未改。
- Phase1 trainer RED：旧接口接收外部`support_features`，新测试要求`support_feature_model`及共同builder而失败；改为从source episode support IQ构建后9项转绿，query仍只在fast state固定后的outer objective打开。
- Runner/CLI RED：旧stage transform只收到已选identity feature，真实CLI也直接处理`z_id`且config无ABI绑定；改为传递当前model与fit IQ、共同builder输出685D rows、严格校验config/bundle绑定后转绿。
- 相邻兼容RED：Task2校准器测试按旧4字段手工构造`SupportDomainState`产生13个失败；仅给新增诊断字段增加空tensor默认值，生产encoder仍强制生成真实统计，Task1-5相邻153项全部转绿。

## 生产边界与审计

- Builder输入仅为合法support IQ、整数label、opaque physical token、冻结nominal K和显式mask/validated-unpadded声明；无query、truth、role、quota或source IQ入口，audit固定记录`query_rows_used=0`和`source_iq_rows_used=0`。
- ADV3B02只读取顶层`z_id`和`aux_id.t_emb/f_emb`，三者均严格要求有限`[N,160]`；不读取`z_dom/aux_dom`。
- 确定性view仅对同一received IQ做DC移除和complex unit-RMS数学变换，不新增physical row、token或K。
- phase-increment四坐标只作为CFO/SFO-sensitive proxy；audit明确`PROXY_ONLY`、absolute CFO physical units unavailable和physical SFO unavailable，不声称绝对CFO/SFO。
- RF-lite10维与quality复用`cvsrffi.stage2_m23_rfguard.extract_rf_lite_quality`；该固定NumPy路径不带梯度，640维model分支及其余Torch坐标保留autograd路径。
- Phase2 K10五折只把fit IQ送进builder。每类8个fit row记录为nominal K10/effective K8；validation row不参与builder、encoder、q、gate或LR。full-support在交叉拟合完成后独立重拟合。

## 最终验证与交付

- `conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_marc_ot_support_features.py tests/test_meta_support_set_encoder.py tests/test_meta_weight_bank_checkpoint.py tests/test_meta_bank_trainer.py tests/test_stage2_marc_ot_runner.py tests/test_run_stage2_marc_ot_pilot.py -q`：86项通过。
- `conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_meta_weight_bank.py tests/test_meta_support_set_encoder.py tests/test_meta_weight_calibrator.py tests/test_meta_bank_inner_loop.py tests/test_meta_bank_trainer.py tests/test_meta_weight_bank_checkpoint.py tests/test_stage2_marc_ot.py tests/test_stage2_marc_ot_runner.py tests/test_stage2_marc_ot_pilot.py tests/test_stage2_marc_ot_scoring.py tests/test_run_stage2_marc_ot_pilot.py -q`：153项通过。
- `conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_stage2_m23_rfguard.py tests/test_stage2_wiser_rf.py tests/test_phase1_adv3b02_deployment_bundle.py -q`：41项通过；仅既存TorchScript API弃用告警。
- `conda run --no-capture-output -n ssr-gpu python code/scripts/run_stage2_marc_ot_pilot.py --help`：退出码0，显示`smoke/pilot/score`。
- 对6个生产文件和6个对应测试执行`python -m py_compile`：退出码0。
- `rg`生产调用面：Phase1 trainer与Phase2 CLI均直接调用共同builder；MARC-OT生产路径无`z_dom/aux_dom`引用，无直接`support_encoder(z_id,...)`。
- `git diff --check`：退出码0。未访问N607，未改变数据验证，未运行训练或生成性能结论。
- Git阶段只纳入本表对应的13个代码/config/测试文件与本Task报告；明确排除既有`analysis/marc_ot_traceability_20260901.md`、Task6 docs、`conversation_index/`和`local_artifacts/`。提交主题固定为`feat: add MARC-OT support feature ABI`；提交后的本地/远端OID一致性由交付状态记录。
