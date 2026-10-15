# 固定 SIG_TIME：接收机风险方差 E50 首种子

状态 RUNNING（启动独立核实）。恢复代码 `1bcaa4919b835601c01cc4095ab8ed869c2e4b9b`；原科学实现代码 `dcf5d36f1af3a1f29796b282c28fb3c6158ff34b`（核心实现 `6553ed11ac2513a5d4a3622481138139c3195ddf`）。本路线针对源接收机风险差异；另一对话的R-Drop随机前向一致性独立推进，不混用其契约、权重或结果。目标仍为固定ManySig clean三预定义模型种子均值≥84%，本次首seed01不能完成目标。

## 唯一比较与训练配方

两个新模型均为原SIG_TIME、joint160、195041个特征参数和原6×160默认cosine权重；正常联合学习，不冻结/重训/替换分类头。RX_BALANCED控制为当前批出现RX的身份CE均值的等权平均；RX_VREX只另加这些RX均值的population variance，系数固定1，全11100步启用。缺RX不加入均值和方差，单RX方差为0，实际各组计数保存。五RX均出现时与已完成源检验同式。理论依据为[REx](https://proceedings.mlr.press/v139/krueger21a.html)，不移植其收益保证。

三模型冻结源检验已PASS并上传afd2de872，2700前向/60层120向量、同900物理L，支持局部导数相容，不证明训练或DG。本次不叠加Sinc单位优化、R-Drop、额外教师、Fishr或新结构。均衡CE重新从零训练以保证同一新接线对照，不把旧不同执行路径的同名结果冒充严格匹配控制。

每新模型scratch50=33label+17pseudo、222步/轮、11100更新；原batch_neighbor门控/ownEMA .999/.16 UCE/.01熵/cos及AdamW lr.0002/WD.0001/全局clip5保持，label_smoothing .01。原L128/U256及尾批保持，无新增信道训练增强；新两模型NoExplicitCFO与内部逐包补偿保持。旧门控+cos历史模型仅外部参考，不能把其与SIG_TIME新骨干的整段差归因于方差。完整3row及父配置以experiment.json为准。

## 数据、资格与成本

源L6300/U56700隐藏TX/V27000，外层7/63/30，训练池标注率10%；split/data392005、模型/原增强RNG2026092701，源RX1/3/4/6/8、day1/2/3、原6TX/物理ID不变。V仅健康诊断，无梯度、校准或选模；U真值不开放。两个新模型无checkpoint继承，EMA来自自身scratch；固定原DRO首SIG checkpoint只用于无query smoke，历史oldgate模型只用于既有合法预测复用，均按实际source_proof核验。

两新模型100轮/22200新更新，各原U963900包；无新增参数、分类头或训练模型前向。CE分组计算及反向成本实际测量；每模型E1/17/33/50每16批记录，共52梯度批/208参数层/624向量，复用原图，不新增模型前向。source特征边界/导出及实际L720比较另计成本。3row各168000clean默认决策，总504000、54分层指标；历史50轮成本单列。

固定源资格：两新模型终5轮原始非均衡CE≤.5；预算/原U/真实body信号及部署特征边界通过；候选L720特征差≥1e-6；控制终5轮批内风险方差>0，候选≤其.95，且终5轮最差RX训练CE不高于控制。方差变小不能通过损害容易RX来单独判优。PASS只允许原样两臂02/03，先上传源决定再读取自身新目标汇总；FAIL保留本次所有训练/评分，不扫系数、分组、父模型、阈值或选择性重跑。

每row冻结后自动clean预测，prediction固定后独立truth-last；不等全部row结束才测试。source_evidence先保存q，再组装新目标汇总。目标测试不用于任何选模、调参或扩种子。保留逐seed/分层accuracy、MacroF1、最差RX/day、配对差、纠正/引入错误、实际特征变化、全量源曲线及成本。n1/clean不代表跨seed/新TX/卫星部署收益。

## 发布与验证

普通N607既有CVS-RFFI/fullFP32；CWD `PROJECT/releases/cvs_receiver_vrex50_20261015_r02`，命令 `python -u -m experiments.cvs_receiver_vrex50.dispatch`，独占root `PROJECT/runs/20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s01-m3-r02`。本地 `python -m experiments.cvs_receiver_vrex50.publish --output LOCAL_ARTIFACTS/receiver_vrex50_delivery_r02`；launcher首步真实checkpoint无query smoke通过即继续。2就绪训练行+1短时历史预测行；每worker CPU2线程，显存按实际准入，不先承诺估计。共用capacity锁内读全owner实际和启动中worker，每卡≤3、free≥12000MiB，优先空卡，不固定分卡、不等待聊天许可。独立评分在空CUDA环境执行。当前容量快照不替代发布时检查。

预期产物见experiment.json。仅本run协议/隐藏U/来源/输入不符、非有限值、预算不完整、无法合法预测评分、输出冲突或系统技术失败停止；保留产物，不自动重试，不停止/重启/热改健康旧任务。单发布包SHA比较、远端编译、PID/CWD/argv/GPU/loggrowth按既有最小流程执行。

16针对性负例、真实native hook、两相同初始state、原U损失原对象、各1次合成AdamW的t1/t2/SIG/W数值更新及zeroW不改joint160均PASS。唯一独立P0/P1发现构建design关键字与FP32/64复算容差两问题，已最小修复并各定点PASS；合法FP32方差180.933731079与double180.933726412接受，明显+.01/非有限值拒绝，训练公式未变。真实远端训练/源资格/目标收益仍未知。

## 技术恢复

r01 在任何 worker 启动前因公开 PROJECT 占位符比较失败退出，训练及预测为零，原目录保留。r02 只解析登记 row.config 中的 PROJECT 目录；超参数或新增字段仍严格不等，非登记读取不变。定向复现先失败后 PASS，独立定点审查 PASS，原 16 项检查 PASS。科学矩阵、系数、数据、种子、预算和源资格全部不变。

## 实际启动

执行 `b7987055f7ad9e95f5fe65f2d368ce528dbac017` 已独立核实：远端编译/真实 checkpoint 无 query smoke PASS；两个新模型分别使用 GPU 7、1，均完成首轮且日志增长，无失败行，全服务器每卡最多 3 个实验进程。首轮含源 V 诊断耗时分别 49.785、49.875 秒，不据此承诺后续半监督阶段总耗时。历史参考行已关闭，两个新模型继续固定 50 轮，训练资格和新目标性能均未知，84% 尚未完成。

## 运行中的源端证据与分析恢复

两模型前4轮各888步/113664次L观察，合计1776步已按保存的RX计数、每组CE、population variance、实际native消费损失独立复算。第4轮控制/候选原始L CE为0.319299/0.324428，风险方差0.020479/0.015939，最差RX训练CE为0.480964/0.448499；只是早期训练观察，不是泛化成绩或终5轮源资格。首轮t1/t2/SIG方差导数相对均衡CE范数约29.13%/28.85%/27.47%，余弦0.2184/0.2254/0.1205；三层均13/13批非零。

已定位离线复算缺陷：控制组重复同图GPU反传的39个特征梯度未逐位相等，最大相对L2误差2.631e-7；两模型104层312保存向量全部通过原allclose容差。旧汇总要求alpha0时torch.equal，预计会拒绝这些合法数值。仅离线代码移除位级要求，原atol1e-6/rtol1e-5保持，实际保存向量检验PASS，260个明显损坏/非有限值负例拒绝，定点独立审查PASS。远端在训代码/训练配方保持原样；冻结后使用独立分析入口与analysis_recovery_r01输出，复用原合法sources/predictions/scores，先保存源资格再汇总，不重训、不重评分或覆盖旧产物。

最近核实两组进度15/50、17/50，进程健康、无非有限值跳步；训练与新目标性能仍未完成。

离线修复代码 `0ba4dd317a31b8d34ce8a19b70273cad7a767dfa`；独立分析发布使用 `cvs_receiver_vrex50_analysis_20261015_r01`。source模式仅在两个原模型终50轮冻结后执行；先独立保存源资格及上传扩种子决定，之后summary模式复用既有独立评分。两个阶段输出到原run下新的analysis_recovery_r01，原训练/评分产物保持，未开始新模型或新测试。

独立分析代码部署 VERIFIED：发布 `cvs_receiver_vrex50_analysis_20261015_r01`、提交 `0416c8c4f7865b7bb879f9e1dba69a95dde9878d`；单包传输比较、远端编译及独立回读通过。尚未启动source/summary分析，健康原训练继续，最近实际进度 23/27 轮。没有新训练/预测/评分或读取新目标汇总。

## 半监督阶段实际执行

完整回放当时可用的native步流：控制6770步（30个完整epoch＋第31轮110步），候选7730步（34个完整epoch＋第35轮182步）。两模型每步ownEMA成功更新数等于步号，decay=.999；原33L/17U启用边界及L/U尾批均符合原配方。候选已实际执行404个U步、103292次U样本观察，1649个被原门控接收（1.5964%），U真值标志每步为0。控制在该快照尚未进入U；未把不同训练进度的覆盖率作机制优劣比较，也未更改门控或阈值。这是运行中前缀，不是完整训练预算、U伪标签精度或DG收益。
