# 源响应方向的独立 Ag：E20 完整结果

两个新 scratch 模型，固定 seed2026092701，fullL6300/U56700/V27000、E20=13L+7U/4440更新。SIG_TIME/RAW_HZ/NoExplicitCFO、joint160默认cosine头与原U/ownEMA联合训练。只替换G，raw/G/raw权重.25/.25/.50，每槽同IQ两次RDrop；不训练As。B00是同六前向匹配控制，R_LEGACY是原普通增强参照，均只读复用原产物，不算新模型或新seed。

本轮不含B11，因此没有联合收益或交互项；不把独立Ag的变化冒充已经完成双增强目标。原七视图预测固定后由独立CUDA隐藏CPU truth-last评分，目标成绩不回流选择。

## 源域决定先固定

EMP资格：FAIL。固定要求fullV相对B00至少+0.5pp、最差源RX下降≤1pp。

|行|fullV准确率 %|相对B00 pp|最差源RX %|资格|
|---|---:|---:|---:|---|
|EMP|98.214815|-0.018519|95.240741|FAIL|
|ROT|98.225926|-0.007407|95.222222|FAIL|

EMP−ROT源clean为-0.011111pp。两行source_decision均在自身target预测前冻结，完整判断只读取source文件。V中的RX已见于源训练，不称为未见RX泛化。

## 正式默认头结果

|行|clean %|low urban %|mid urban %|urban均值 %|urban宏F1 %|最差urban RX %|suburban均值 %|
|---|---:|---:|---:|---:|---:|---:|---:|
|B00|83.108929|31.316071|38.259524|39.213294|36.971955|30.115278|47.802381|
|R_LEGACY|83.001190|36.241071|44.585714|45.652976|43.109027|36.268056|55.620040|
|EMP|83.212500|31.263690|38.183929|39.165278|37.029387|29.838889|47.697619|
|ROT|83.020238|31.519048|38.615476|39.605159|37.336178|30.506944|48.365278|

|行|视图|准确率 %|宏F1 %|最差RX %|clean−本视图 pp|
|---|---|---:|---:|---:|---:|
|EMP|clean|83.212500|82.906241|75.470833|0.000000|
|EMP|practical_high|50.419048|47.531806|36.591667|32.793452|
|EMP|practical_mid|47.680952|45.004144|35.258333|35.531548|
|EMP|practical_low_suburban|44.992857|42.530893|33.633333|38.219643|
|EMP|practical_high_urban|48.048214|45.251573|35.241667|35.164286|
|EMP|practical_mid_urban|38.183929|36.088181|29.387500|45.028571|
|EMP|practical_low_urban|31.263690|29.748406|24.887500|51.948810|
|ROT|clean|83.020238|82.696239|75.754167|0.000000|
|ROT|practical_high|51.072619|47.940878|37.575000|31.947619|
|ROT|practical_mid|48.370833|45.443837|36.329167|34.649405|
|ROT|practical_low_suburban|45.652381|42.963015|34.508333|37.367857|
|ROT|practical_high_urban|48.680952|45.675874|36.216667|34.339286|
|ROT|practical_mid_urban|38.615476|36.410261|30.162500|44.404762|
|ROT|practical_low_urban|31.519048|29.922398|25.141667|51.501190|
|B00|clean|83.108929|82.774019|75.775000|0.000000|
|B00|practical_high|50.454762|47.401581|37.025000|32.654167|
|B00|practical_mid|47.801786|44.944128|35.658333|35.307143|
|B00|practical_low_suburban|45.150595|42.535483|34.012500|37.958333|
|B00|practical_high_urban|48.064286|45.128185|35.691667|35.044643|
|B00|practical_mid_urban|38.259524|36.064748|29.725000|44.849405|
|B00|practical_low_urban|31.316071|29.722933|24.929167|51.792857|
|R_LEGACY|clean|83.001190|82.674328|74.162500|0.000000|
|R_LEGACY|practical_high|58.702381|55.556505|45.116667|24.298810|
|R_LEGACY|practical_mid|55.610119|52.567318|43.695833|27.391071|
|R_LEGACY|practical_low_suburban|52.547619|49.668064|41.562500|30.453571|
|R_LEGACY|practical_high_urban|56.132143|53.065620|43.745833|26.869048|
|R_LEGACY|practical_mid_urban|44.585714|42.043795|35.375000|38.415476|
|R_LEGACY|practical_low_urban|36.241071|34.217665|29.683333|46.760119|

全部28行×18分层评分记录见formal_strata.csv，其中两条新模型252条，两个旧控制252条。单模型seed，不提供跨seed复现或独立盲测声明。每个新模型7×168000=1176000个正式决定，原IQ、SNR、场景和困难样本不变。

## 完整训练与机制

完整解析40个epoch/8880个step；两个新行的关键loss/梯度有限、跳过更新和U真值标记均为0。每行568320个G包，300个epoch×RX/day session、1800个session×TX单元全部覆盖，参数在session内跨TX一致。

源冻结panel720复用固定父模型clean正确集合，head margin统一除以30；各状态、lock、signal_to_added_noise区间准确率/F1/保持率/间隔见source_state_quality.csv。新增噪声SNR不是总SNR。这里没有As训练，源panel中的传播状态结果只能描述Ag模型对既有损伤的表现，不证明噪声/多径的单独因果改善。
EMP源low：257/720，默认头间隔/30=-0.033064049，固定父clean正确集合保持率=36.324787%。
ROT源low：257/720，默认头间隔/30=-0.032249730，固定父clean正确集合保持率=36.324787%。

## 实测成本

|行|source秒|row墙钟秒|L前向|studentU前向|EMA前向|source eval前向|panel前向|formal前向|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|EMP|1209.123469|1261.248158|3409920|396900|396900|594002|5040|1176002|
|ROT|1345.704550|1397.204394|3409920|396900|396900|594002|5040|1176002|

两新模型共40model-epoch/8880更新；本轮源/测试真实数据新增前向11958250（含共享source smoke2+父clean panel720）。矩阵墙钟1410.908880秒。完成E20表示预算用尽，不等于收敛。

B00/R_LEGACY的既有训练、父只读E20参考的20epoch/4440更新/923.560061秒、先前有限IQ安全probe的189002前向/120.415114秒分别保留，不混作本轮增量训练或继承成本。源panel构造还包含4320次物理信道处理，属于IQ生成而非网络前向。

## 判定与交付

源规则FAIL：结束本固定候选，不开展其联合训练或种子扩展。
无论源判定，所有已注册行均完成原七视图，不选择性重跑。整体goal未完成：当前没有新联合行，也没有跨seed证明。

实际文件：analysis_summary.json、seven_view_results.csv、formal_aggregate.csv、formal_strata.csv、paired_effects.csv、measured_costs.csv、complete_training_curves.csv、source_state_quality.csv、actual_G_coverage.csv。
实际图：formal_results.png、complete_training_curves.png、source_strata_and_coverage.png、source_quality_and_existing_gate.png、measured_cost.png。图和表均由全量完成产物计算，不加载IQ、权重或target truth。

## 执行身份、命令和证据

实际训练代码 b393dfe66146d6ab42236da705d743044d45f9d3，原不可变release `cvs_source_response_e20_20261016_r01`。工作树 `codex/phase1-urban-sigtime-20261014`；取回/分析代码提交45b7514997fca6e8902a5e2f75c784389ca82e16。EMP源FAIL在读取本轮任何目标accuracy/F1前，以68fff981616966ec8eea965ec20c5f4ed9cfa17f记录；后续正式小正差不改变预登记停止规则。

本地真实native图检查1536个网络前向、198个同初始化state张量、默认W/SIG梯度、EMA绑定和成对IQ均PASS；10个标签顺序及13个协议负例通过。一次限定P0/P1审查发现跨平台B00参考路径序列化问题，修复并做唯一问题定点复核PASS；无重复全量/报告审查。真实E20默认头无query smoke为launcher首步，PASS立即继续。单归档传输比对、远端编译、普通账号、全owner GPU最多3进程及发布后PID/CWD/argv/GPU/log增长均独立核实。

当前两行训练、冻结、原七视图预测及独立CPU评分全部完成；252条新分层评分记录完整。原dispatcher和两个row PID自然终止，failure为空，VERIFIED。当前看到的其他GPU进程属于其他任务，未操作。source配置依旧fullL6300/hiddenU56700/V27000，训练As为0。没有目标重跑、测试IQ/SNR/困难样本变化或额外分类头。

实际运行的 `leo_practical.channel.__file__` 和 `residual.__file__` 都在已核实依赖release的 `experiments/adv3b02_xuc/code/leo_practical/`，非根code实现。源panel使用原residual/post_sync、25MHz、referenceSNR30、EQ关闭G=1、reflect及原质量/相位/IQ补偿；正式测试继续原固定received IQ，不重新生成。具体actual configs在各行channel_execution.json，native参数在actual_native_args.json。

在工作树执行过的命令：

```
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_source_response20.train_checks
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_source_response20.harvest_checks
E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_source_response_train_20261016/publish_local.py --output E:/type10-7/local_artifacts/cvs_source_response_train_20261016/release
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_source_response20.train_harvest --output E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final
```

取回器运行中多次返回WAITING_LIVE并核实原PID，不创建部分final；完成后一次取回，完整8880步/40epoch/252条新评分及两个控制252条旧评分全部核对。5个取回边界/算术负例PASS。没有加载新IQ、模型权重或target truth。取回成功退出，五图实际生成并逐图查看。

## 收益归因和下一条源问题

EMP−B00正式clean +0.103571pp、low −0.052381pp、mid −0.075595pp、urban −0.048016pp，最差urbanRX −0.276389pp。EMP相对原普通增强参照clean +0.211310pp，但low −4.977381pp；原参照同时含普通增强和不同L前向数，不能把该差唯一归因于G。EMP−ROT clean +0.192262pp而urban −0.439881pp，不支持方向专属性的联合收益。源fullV两个方向均略低于B00，真实G非零、梯度正常、所有session覆盖完整，因此不是未启用G或预算缺失造成的假负结果。

每行300个session、1800个session×TX单元的计数及bank参数成对完全一致；这只是聚合覆盖匹配，不宣称逐样本顺序的独立证据。真实G波形MSE曲线非零但较小，loss/梯度轨迹近似，不能据此直接断言放大幅度即可恢复。旧33tapPHYS/ROT也未建立实测幅度方向优势，本固定谱形候选到此结束，不调幅救场、不换ROT晋级。

后续唯一源问题改为：在排除整体复增益、CFO及时移后，已知L-STF的非线性频率相位关联是否仍跨TX/day重复。依据来自既存源证据：模板匹配已同时优化CFO/时移，源RX3/8相关仍明显较低；因此不能把差异直接归为简单定时偏移。当前幅度谱路线无独立价值也不能推断相位路线有效。先检索复用已有复数统计，缺项时仅在合法L6300上做零模型的交叉拟合统计；不训练诊断头、不读U/目标、不使用本轮正式分数选幅度或晋级。没有可重复相位关联就放弃相位增强，不直接发新训练。

本轮未达成low>60%、suburban接近clean或跨seed复现；也没有新联合模型。整体goal保持active，未完成项如实保留。

实际图表与数据：

- [正式七视图与配对差](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/formal_results.png)
- [完整训练曲线](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/complete_training_curves.png)
- [源lock分层与覆盖](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/source_strata_and_coverage.png)
- [源状态/质量与原门控](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/source_quality_and_existing_gate.png)
- [实测成本](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/measured_cost.png)
- [全量汇总](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/analysis_summary.json)
- [全部正式分层评分](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/formal_strata.csv)
- [全部源状态/质量](E:/type10-7/local_artifacts/cvs_source_response_train_20261016/final/source_state_quality.csv)

验证成本另计：实现检查1536个合成网络前向，独立审查复跑相同1536个，合计3072；另各10包纯IQ算子检查不经过网络。上述11958250只统计本次真实源/测试运行，含这些已知合成网络检查为11961322；本地验证墙钟未统一计时，不伪造该成本。前面的独立有限IQ安全probe成本仍另列。
