# 固定 SIG_TIME 特征均值教师：完整三种子负结果

状态 **ANALYZED / VERIFIED**。固定三种子2026092701/02/03的FEATURE_MT均值 **82.4359% ± 1.5874 pp**，匹配SIG_TIME为 **82.6806% ± 1.7906 pp**；配对差 **−0.2446 ± 0.2101个百分点，0/3为正**。该固定配方没有提高默认头clean识别性能，距84%还差1.5641个百分点。源特征损失降低不能作为性能提升的替代证据。

## 身份、范围与配方

首run 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01（代码373262b51f037f03e3238471b3aefb88e45fba5b，执行6b23a5b033ee063dea97d6783d5634e220a1d93d）；剩余run 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01（薄seed封装68b3143dbdfcde27c80997f970121a23e34e5f2e，依赖技术修复729c01e878a0df790c1056d00d1b547f709a57be，实际R02执行ab2c3127e14b48dea5bc2e064ce42c0be7cfdeeb）。复现入口为各package的dispatch/publish，固定rows与完整命令/环境在同run experiment.json；审计入口experiments.cvs_feature_mean_teacher50.audit及experiments.cvs_feature_mean_teacher_seeds.audit，汇总由各score.finish产生。原始模型、预测、日志保留既定远端run/row目录，Git只交聚合数据与曲线。

ManySig clean、split392005、每row168000样本；固定SIG112→真实joint160/id_feat_joint、fθ195041参数和既有cosine6×160/scale30头，按原损失正常联合训练。没有单独训练、冻结重拟合、替换头或额外诊断分类器，没有新投影器/预测器/网络参数。源L6300/U56700隐藏TX/V27000、源RX/day/TX/physicalIDs、NoExplicitCFO和内部补偿保持，无新增星地训练/测试声明。

原batch_neighbor＋ownEMA.999＋cos、33label＋17pseudo、λUCE=.16、λentropy=.01、Lsmoothing=.01、L128/U256不变。在所有实际U的已有teacherweak/studentstrong单位joint160上加入half squared L2 mean，teacher stopgrad/eps1e−8，系数.1×clip((epoch−33)/5,0,1)。ZERO同图系数0，原硬伪标签损失对象、采样顺序、RNG和前向数量不变。理论源为[Mean Teacher论文](https://arxiv.org/abs/1703.01780)与[作者实现](https://github.com/CuriousAI/mean-teacher)；这是部署特征层变体，不能直接借用原输出一致性的性能保证。

## 完整默认头结果

|条件|clean均值±样本SD|Macro-F1均值|逐seed最差RX均值|逐seed最差day均值|
|---|---:|---:|---:|---:|
|OLD_GATE_COS50|80.5268% ± 2.3776 pp|80.0024%|67.9708%|76.0556%|
|SIG_TIME|82.6806% ± 1.7906 pp|82.1282%|70.4944%|78.5056%|
|FEATURE_MT|82.4359% ± 1.5874 pp|81.8688%|69.8833%|78.1262%|

|seed|旧门控＋cos|匹配SIG_TIME|FEATURE_MT|FEATURE−SIG（pp）|
|---|---:|---:|---:|---:|
|2026092701|80.6685%|83.6679%|83.3619%|-0.3060|
|2026092702|78.0815%|80.6137%|80.6030%|-0.0107|
|2026092703|82.8304%|83.7601%|83.3429%|-0.4173|

FEATURE−SIG累计纠正 **1660**、引入 **2893** 个错误，净少 **1233/504000** 个正确决定。对旧门控＋cos均值增加1.9091个百分点，但SIG本身增加2.1538个百分点；旧架构差异不能归因于新增SSL，机制比较必须用同架构SIG。样本SD按3个模型seed计算，不能把151.2万次决定当作独立模型重复；这些是已暴露固定基准上的结果，不是新独立盲测。

独立审计重计剩余6row的1008000决定/108指标，并与已审计首3row合并，完整9row共1512000默认决定/162指标。全部450model-epochs被解析，66600新step＋33300历史更新分开计。保存CM/accuracy/MacroF1、每row预算、truth-last、原生/feature梯度均核实；本地另从两套独立审计指标重算均值、样本SD、分组最差值、配对差与纠错净数，与冻结汇总一致。实际指标维度receiver在统计中显式映射为rx，不改变原记录。所有worker自然退出、失败0；后续分析无新目标前向、头拟合或模型更新。

## 源目标被优化，但泛化未改善

|seed|ZERO末5轮U损失|FEATURE末5轮U损失|候选/对照|下降比例|
|---|---:|---:|---:|---:|
|2026092701|0.329255|0.270479|0.821488|17.85%|
|2026092702|0.339184|0.268806|0.792508|20.75%|
|2026092703|0.332661|0.266505|0.801131|19.89%|

完整300个新模型epoch、66600条feature step的U包加权损失已从保存流聚合；每模型963900 U包次/3774 U步，前33轮U为0。三seed都降低同U增强特征差异，但主准确率三seed都下降；当前证据不支持把该损失优化称为有效DG。也不能据此直接断言特征塌缩、接收机捷径或因果梯度冲突。

六新模型的原生与feature两套观测共独立重计 **23976层记录、71928个向量**；覆盖33/34/50各222批。候选34/50的t1/t2/SIG projection effective梯度均为222/222非零，ZERO全0。第50轮跨seed平均effective/mainLCE范数比分别.017114/.018447/.009375，余弦−.004540/−.001928/−.005292。这里mainL128和U256来自不同图/样本，不足以判断同包身份梯度的冲突或累计AdamW贡献；不能据这些数直接扫更强权重。

所有6模型都通过实际id_feat_joint与feature-only导出逐元素一致、复制W置零不改z、保留特征变换和分析前后θ/buffer不变检查。各模型192492至193460个feature元素发生数值更新，默认960个W元素均更新，5个SIG pool tensor全部更新；aux对W梯度None，主损失仍正常更新W。改动确实进入可交接fθ，但这些路径证据不构成丢弃旧W后的新TX迁移收益；按最新用户约束，没有另训头来补结论。

原生U累计接受SIG46692/2891700、FEATURE46383/2891700；U真TX未打开，包次不是唯一物理coverage或伪标签准确率。V只作无梯度、无拟合健康诊断。

## 成本、曲线与技术失败

6新scratch50共300model-epochs/66600更新；3历史baseline共150epochs/33300更新另计。新增训练模型前向0，诊断反传/存储有实际成本。新原生日志epoch时长合计13687.05秒（SIG7126.53、FEATURE6560.52），历史6887.02秒；共享GPU调度不同，不能把候选较短时间作为加速结论。完整曲线图complete_three_seed_curves_and_results.png同时展示模型seed样本SD，不用单seed曲线代替多seed趋势。

三seed源CE末5均值：SIG .233425、FEATURE .234423，前5为.241454/.242234；源V末5约98.2400%/98.2410%，趋平台而CE仍波动。50轮是固定预算，不证明充分收敛，也不按目标延长或重启。

扩展R01在dispatcher import时技术FAILED，独立核实workers0/run根目录未创建，训练更新0。失败release/archive/log保留。相同汇总helper改为已包含的依赖，并在原preflight导入dispatcher；局部AST等价/导入/包路径检查与一次针对P1窄复审PASS。R02真实checkpoint无query smoke、一次archive对比、remotecompile及PID/CWD/argv/GPU/loggrowth VERIFIED。技术恢复没有更改科学配方或重启健康任务。

## 决定与后续

源资格扩展决定 ec8ea03f7810cb3eb1bcb8108cdb60b4053a59c9 已先提交并推送，随后才回收新目标数值；02/03两臂均原样完成。源资格PASS仅代表预先声明的健康/特征路径/一致性目标被优化，并不等于科学收益PASS。本冻结配方保留为完整负结果，不扫系数、门槛、轮次、seed或换头读出。

后续只保留两项优先工作：完成已健康运行的RAW/TX_DAY_CENTERED泛化机制固定三seed；在合法L、同强图、默认头和0参数更新条件下核查特征一致性梯度与真实身份CE的关系，为后续SSL机制提供源证据，不能解封U真值或按目标调参。后一项源诊断尚未实现/启动，不将计划当证据。84%目标仍未达成，不自动开展结构或Phase2开发。
