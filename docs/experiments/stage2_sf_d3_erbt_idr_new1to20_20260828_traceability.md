# D3+ERBT-IDR嵌套新类矩阵设计追踪

|ID|设计要求|实现/证据|状态|
|---|---|---|---|
|DATA-01|复用`p2_min_v1/VALIDATED_ONCE`权威`new20`接收IQ|`stage2_nested_registration_builder.py`|LOCAL_VERIFIED|
|DATA-02|新类规模固定`1/2/3/5/10/15/20`，K=10，嵌套前缀|构建器常量与平衡检查|LOCAL_VERIFIED|
|DATA-03|每类使用全部20条query，predictor与truth物理隔离|构建器输出`predictor/`与`scorer/`|LOCAL_VERIFIED|
|DA-01|D3固定327步、单A段、4-fold support-only OOF温度|`stage2_sf_d3_erbt_plan.py`|LOCAL_VERIFIED|
|DA-02|每场景D3只拟合一次，7个注册规模复用同一delta|三场景plan与`da_split_id`|LOCAL_VERIFIED|
|ERBT-01|固定`M29-FFT96-A4/D92-E0-NORF32`|`stage2_sf_erbt_four_state.py`|LOCAL_VERIFIED|
|REG-01|旧6类REG0与旧6+N类REG1共用旧类度量|`fit_erbt_registration_pair`|LOCAL_VERIFIED|
|CAUSAL-01|输出`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`|预测器与truth-last scorer|LOCAL_VERIFIED|
|SCORE-01|报告DA、注册和difference-in-differences|`score_four_state_predictions`|LOCAL_VERIFIED|
|PRO-01|support状态冻结后才由预测器打开query|`support_state_receipt.json`先写|LOCAL_VERIFIED|
|REL-01|正式代码/config以Git提交固定并自动push|发布记录|VERIFIED|
|RUN-01|21格四状态prediction与truth-last评分|r4完整code闭包|BLOCKED：D92不支持new1非对称注册几何|

冻结实验：receiver=`20-1`、data seed=`713101`、method seed=`392002`、K=10、三种`leo_*_weak`场景、新类数`N={1,2,3,5,10,15,20}`。基础checkpoint为ADV3B02 CORE90；D3固定为327步、4-fold support-only OOF温度；ERBT-IDR固定为`M29-FFT96-A4/D92-E0-NORF32`。本轮因历史Stage2-C truth已在旧研究中使用，结论标记为`DIAGNOSTIC_NON_FORMAL`，不得直接晋级正式默认版本。
