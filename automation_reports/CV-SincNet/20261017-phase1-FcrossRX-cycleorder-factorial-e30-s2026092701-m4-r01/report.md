# 固定 CVS 的周期重排 × practical：全源 E30 四因子终态报告

状态：**ANALYZED，候选负结果；联合研究目标仍 ACTIVE**。四行均完成从头 E30、FINAL30 冻结、clean＋六种 practical 预测、逐 row 独立 truth-last 评分和完整分析。新增 Ag 未证明独立泛化或联合收益。关闭本固定周期重排候选，不改窗口、相位拟合、排列群、概率、源 RX、seed 或预算救跑。仅 1 个 model seed，不能宣称跨 seed 重现。

As 是主要收益来源。匹配 B01−B00：urban 三场景均值 +27.308532 个百分点，mid urban +26.911310，low urban +21.639286，代价为 clean −2.180357。B11 low urban=53.339881%、clean=83.182738%；没有达到 low urban>60%、clean 净正、各 suburban 与自身 clean 差≤2 个百分点的探索目标。

## 实际起点、配置、调用路径与数据权限

起点 commit 141e136a95c25e8d19ac7ec0acec2e16ea1e5ac4；训练实现 commit 0c47f4bcd49c5df43093579c7f33e6c35bc5245d。用户最新预算为 scratch E30，覆盖旧 E20/E50。每行20有标签＋10原半监督epoch、222step/epoch、6660更新，FINAL30固定选模；初始化/optimizer/EMA均来自本行scratch，父训练成本0。旧全源E20 checkpoint只作严格加载/2合成包no-query兼容smoke，未用作训练init。

F＋跨RX同TX SupCon 固定：SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、joint160、原默认cosine头/尺度30、NoExplicitCFO与原内部逐包补偿，900722参数/206state。骨干、内部门控、参考响应、默认头共同更新，无新增部署分支、分类头或拟合诊断头。RDropα1、LS.01、AdamW weight_decay.0001、原cosine日程peak.0002/min.000001。SupCon仅primary的time_down/t2.pool，temp.2/max.1/warm5/ramp10，复用已有前向。

完整L6300/U56700隐藏TX/V27000、split392005、7/63/30保持；训练RX索引[1,3,4,6,8]/day索引[1,2,3]全部参与，不留RX、不切L_dev、不把V并入L，target_loaders=0。正式RX索引[0,2,5,7,9,10,11]，实际标签1-1/14-7/2-1/20-1/7-14/7-7/8-8；day索引[0,1,2,3]，allow_source_target_day_overlap=true为原契约。索引不等于物理名称。modelseed2026092701，U TX从未解封，原VALIDATED_ONCE句柄直接复用。

采样率25MHz、长度256；dataset float32 RMS在增强前，As内部normalize_input=unit_rms，无外加postAs RMS，不反演未知TX硬件。正式每view168000包，IQ/信道/SNR/困难样本均保持原样；训练、源读出、正式测试流分离。

实际__file__逐row打印保存于rows/*/source/module_files.json：leo_practical/channel.py、batch.py、residual.py来自不可变release cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code；同code的practical_adapter.py、DataAugmentation.py、SSDG/train_ssdg.py实际运行，**不是顶层code/**。Ag来自固定P30 release的iq_order.py。

实际As为residual/post_sync/equalization_enabled=false，即G=1，并非已经均衡或弱残余替代；quality_adaptation/phase_tracking/iq_compensation均enabled，相位和细频补偿仍依实际lock状态执行。reference_snr_db30、fc2.45GHz、altitude600km、AGC enabled/limit30dB、fractional_delay_half_length24、reflect人工边界。首批propagation/residual kernel62、数值延迟24；full_channel_waveform_synthesized=false是原residual实现事实。RX硬件按物理RX/day session跨TX稳定，receiver_seed2026092701，view_seed202610172201随epoch/view刷新，无二次As或新增EQ。

接收质量为white_input_expected_power_proxy，不等于实际波形总SNR；signal_to_added_noise_db仅为新增噪声比，true_total_snr_known=false。P24已有同包控制区分移除损伤后重算状态的总效应与固定状态/补偿的受控效应，并检查功率代理、AGC和边界。本轮分层相关性不能唯一归因噪声、多径、Doppler或门控。

## 已执行设计与普通增强审计

四行共有.5 ordinary(x) primary；secondary如下。两半CE/原RDrop KL各自归一化后.5相加。每物理L四次网络包前向（两次RDrop×concat两半），两次使用同一已生成IQ；SupCon只primary。U保留原EMA/旧门控/弱强增强，不新增U Ag或额外一致性损失。所有有标签satellite包均参与CE，无学生高置信前置门槛。

|row|secondary|clean %|mid urban %|low urban %|urban均值 %|low宏F1 %|low最差RX %|
|---|---|---:|---:|---:|---:|---:|---:|
|B00|raw|85.237500|37.426190|31.755357|38.052183|30.425385|28.183333（14-7）|
|B10|Ag(raw)|85.088690|37.491667|31.826786|38.138095|30.384644|26.958333（14-7）|
|B01|As(raw)|83.057143|64.337500|53.394643|65.360714|52.824516|48.737500（20-1）|
|B11|As(Ag(raw))|83.182738|64.328571|53.339881|65.323413|52.731897|48.220833（14-7）|

raw是loader已RMS归一化的同一L波形。旧P25 ordinary-secondary不能替代本轮raw-secondary因果控制；历史raw-As E20失败保留，本轮新控制只用于固定新Ag的匹配因子，不将E30视为旧候选救跑。

ordinary_base_config.json实值：p_dac0、p_pa.14、class_signature=false、shift p.35/max32、amp p.45/[.9,1.1]、phase p.45、CFO p.35/max.0004、phase_noise p.3/sigma≤.006、AWGN p.4/20–36dB、multipath p.18/2–4taps/maxdelay4、DC p.3/max.02、bandtaper p.25/alpha .02–.1。原日程改变逐epoch实际概率，base值不是每步值；每epoch实际PrivateAugment概率/波形统计完整解析进analysis_summary.json。NoExplicitCFO是模型约束，不能据此声称普通增强CFO关闭；历史普通PA也是共有配方，不是本轮新验证的身份保持Ag。

循环移位/256点边界可能制造接缝；增益会被As的输入RMS消除；ordinary AWGN/multipath在As前可能重复叠加损伤。因此secondary明确raw→Ag→As，primary完整保留原监督。ordinary私有流20261016051与loader/dropout/U隔离。5组配对序列比较均8880条/0 mismatch，不重抽RDrop IQ、不删重复样本。

Ag固定[80:160]四个20点周期均匀S4重排，pooled lag20相位估计后phase transport exp(j·phase·(k−perm(k)))，窗口外176点保持。seed202610173001使用trainL/epoch*/step*/secondary及不透明physical eq/RX/day/sig/base/occurrence ID，不按TX选变换。B10/B11各852480个Ag IQ，6660条元信息/排列/数值统计全部一致；24排列均覆盖、0phase fallback、最大功率误差2.980233e−7。L2中位.046414/q90.280245/最大1.088486，尾部并不总轻度，仍不能据此唯一解释失败。

As训练环境low_suburban/mid_urban/low_urban，保留几何/遮挡/传播/噪声/质量/补偿/AGC依赖，B01/B11各852480次；各覆盖90个TX×RX×scene、90个TX×RX×state、270个TX×RX×scene×state组合。无新hard门控/额外分类头/目标反馈。

## 源域判据与状态分层

预注册条件：B10 fullVclean≥B00+.5pp；B11 fullVclean≥B00；B11−B01每个2draw×2urban≥+1pp。源判据先保存再读formal；首次源失败固定于dadbfdad86893af333f7058f2e9e5fc480f1d020。科学失败仍完成四行既定正式矩阵，不修改健康任务。

fullV27000：B00 98.303704%、B10 98.266667%、B01 97.885185%、B11 97.874074%。Ag独立−.037037pp（少10个正确包），joint对B00−.429630pp。B11−B01四个源urban单元为[[−.500000,+.444444],[+.222222,−.277778]]pp。**三项全部FAIL**。微小差不证明确定伤害，但预定增益条件未满足；不依据formal补改判据、扩seed或救跑。

V1800×2draw的low urban完整视图：B00 38.555556%、B10 37.638889%、B01 62.138889%、B11 62.222222%。源62%不是正式low>60%。以下同包源分层n为2draw累计；目标逐包物理状态未保存，不能称为目标状态结果。

|low源分层|n|B00 %|B01 %|B11 %|
|---|---:|---:|---:|---:|
|LOS/locked|1060|60.660377|96.037736|96.132075|
|BLOCKED/unlocked|1565|22.300319|33.610224|33.993610|
|signal/added-noise<3dB|1127|21.118012|30.079858|30.700976|
|signal/added-noise≥15dB|1376|58.575581|95.130814|95.348837|

As收益主要出现在可恢复LOS/locked/较高信号新增噪声比观测；BLOCKED/unlocked/噪声占优仍差。极小PARTIAL单元n1–2只作描述。源margin统一logits/30，固定参考clean正确集合保持率/分类间隔见source_strata.csv；margin、梯度或参数变化不能替代识别收益。保存total/backbone梯度健康及Sinc更新，**未保存逐包fusion gate或逐分支梯度**，不能补造门控因果结论。

## 正式七环境、收益归因和clean代价

每row七view prediction全部固定后独立truth-last，504个正式CM独立复算与scorer一致。表为默认头accuracy，逐RX/day/TX及宏F1另见CSV。

|view|B00 %|B10 %|B01 %|B11 %|
|---|---:|---:|---:|---:|
|clean|85.237500|85.088690|83.057143|83.182738|
|practical_high|46.350595|46.474405|81.772619|81.654167|
|practical_mid|44.589286|44.553571|79.417857|79.376190|
|practical_low_suburban|42.622024|42.453571|77.039286|77.051190|
|practical_high_urban|44.975000|45.095833|78.350000|78.301786|
|practical_mid_urban|37.426190|37.491667|64.337500|64.328571|
|practical_low_urban|31.755357|31.826786|53.394643|53.339881|

Ag独立B10−B00：clean−.148810、urban均值+.085913、low+.071429pp；clean宏F1−.154414、low宏F1−.040741pp。clean修复1564/新增错误1814，净少250；low修复5264/新增5144，净多120但F1略降。这是错误重分配，未证实跨RX身份泛化。

As主效应B01−B00：clean−2.180357、urban均值+27.308532、mid+26.911310、low+21.639286pp。附加Ag的B11−B01：clean+.125595、urban均值−.037302、mid−.008929、low−.054762pp。joint对B00：clean−2.054762、low+21.584524pp。B01低urban最差RX48.737500%（20-1），B11为48.220833%（14-7）；最差接收机不是固定同一个。

准确率交互B11−B10−B01+B00（pp）：clean+.274405、high/mid/low suburban−.242262/−.005952/+.180357、high/mid/low urban−.169048/−.074405/−.126190。clean交互略正不能抵消clean净负；urban交互均负，不称联合成功。

B11 suburban对自身clean差1.528571/3.806548/6.131548pp，仅high≤2。不能降低clean后称缩小差距为突破。固定matchedB00 clean正确集合上的low保持率B00 34.917143%、B01 58.215490%、B11 58.139372%；历史P25B00 clean正确集合low保持率B01 58.281317%、B11 58.228193%，历史集合只是统计参考，不是初始化父权重。

旧普通拼接E30 P25B01为clean82.797024%、low53.137500%、urban均值65.158333%。本轮B11直接差clean+.385714、low+.202381、urban均值+.165079pp，小幅正差如实保留；但旧secondary为As(ordinary)，当前为As(Ag(raw))，不能全归因Ag，匹配新B01反略高于B11 urban。仅1seed，无重现证明。

## 完整训练记录、成本和技术恢复

已解析120epoch/26640step、每行8880序列/6660SupCon/Sinc及1198行完整stdout；active loss均有限、0skipped、0RuntimeError/OOM标记。每行300个nan文本来自10项未启用/未执行统计各30次（sat_cos/proxy_vaccept/dom/overall_tx/strict_udu/receiver_floor/sat_mean/sat_floor/sat_strict_mean/sat_strict_floor），不是训练非有限。预算耗尽不等于收敛。

每模型source NN5407922=student_train3976920+student_eval864002+teacher_train0+teacher_eval567000；L网络3409920、loaderL852480/U567000、uniqueL6300/U56700。SupCon6660步/active5550、valid/total anchors2841600，Sinc6660。另source qualification12602NN、formal prediction1176002NN，含各自2包smoke。四新模型共120epoch/26640updates/26386106NN（含共享兼容smoke2）；Ag与As各1704960IQ view，无额外NN。

|row|source进程墙钟s|源资格读出s|正式预测s|
|---|---:|---:|---:|
|B00|1584.602|8.472|48.711|
|B10|1621.291|7.160|48.173|
|B01|2956.008|8.296|47.819|
|B11|3039.977|7.871|48.785|

source进程墙钟和9201.877492s，不是纯GPU计算时间；dispatcher启动到终态3121.128221s。历史P25两参考共60epochs/13320updates单列，未加载；P30/P31源前置诊断25204/21604NN单列（P31 native21600）。

analysis_r01读出后因main尾部重复source_readout独占创建source_strata.csv而FileExistsError；失败根/日志保留，163.545555s，0NN/IQ/CP/updates。新增full-main回归先RED复现，再仅删除尾部重复调用；4项针对性分析测试PASS，修复b612d59af616bb8e78a80f3223b3ae0b271346bc已push并独立核对remoteOID相同。新根analysis_r02自然完成，CPU2/CUDA空、stderr0，169.354847s、1112源CM/504正式CM/14图、0NN/IQ/CP/updates；未重训/重做prediction/重评分。

其他验证：4项routing/protocol-negative PASS，local/remote compile PASS，真实checkpoint no-query smoke PASS，一次限定P0/P1独立审查PASS，一次release archive local/remote SHA比较PASS；启动后PID/CWD/argv/GPU/log增长独立核验。四worker自然完成，无健康旧任务停止/重启/热改。REJECTED_EXTRA_GATE：通用TDD技能的全库测试前置超出AGENTS最小白名单，本次只运行所改行为直接相关的4项回归，不增加全库测试或重复审查门槛。

## 代码、配置、命令和实际产物

实现：experiments/cvs_cycleorder_factorial32/的design/route/observer/source/model/channel/qualification/runner/dispatch/evaluate/publish/report/preflight/checks；固定Ag复用P30 iq_order.py。分析：analysis/cvs_cycleorder_factorial_E30_results_20261017.py；回归：analysis/check_cycleorder_factorial_E30_analysis.py；公共统计复用cvs_fullsource_E30_results_20261017.py。政策配置：configs/phase1/F_crossrx_e30_baseline.json。训练commit0c47f4bc；分析修复b612d59a；本终态文档commit由Git历史标识。

run=20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01；output=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01；release=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cycleorder_factorial_E30_20261017_r01。普通N607账户，2CPU线程，GPU B00=1/B10=2/B01=3/B11=4；sharedlock下每GPU含其他owner总实验进程≤3且free≥12000MB。环境/CWD/stoprule/输入输出/矩阵精确记录见保留experiment.json与preregistration.md。

实际worker命令，row对应替换为四个固定row；不是授权读者覆盖原输出：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cycleorder_factorial_E30_20261017_r01/experiments/cvs_cycleorder_factorial32/runner.py --mode worker --row B00-s2026092701
```

独立恢复分析实际命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01/analysis_code_r02/runner.py --truth /home/szu2070436088/2510044040/CV-SincNet/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01/analysis_r02 --commit b612d59af616bb8e78a80f3223b3ae0b271346bc
```

本地产物E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017：[逐row/seed/RX/day/TX结果](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/per_row_seed.csv)、[完整统计](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/analysis_summary.json)、[训练曲线CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_curves.csv)、[源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/source_strata.csv)、[独立源判据](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/independent_source_decision.json)、[完成证据](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/completion.json)。rows/*/source镜像actualargs/resolved/module_files/ordinaryconfig/roles/init/frozen；CP/完整预测/细日志保留远端，不入Git。

七张图各PNG/PDF，全部PNG已实际查看：

- [formal_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/formal_accuracy.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/formal_accuracy.pdf)
- [training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_curves.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_curves.pdf)
- [receiver_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/receiver_accuracy.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/receiver_accuracy.pdf)
- [source_added_noise_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/source_added_noise_accuracy.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/source_added_noise_accuracy.pdf)
- [source_state_lock_accuracy PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/source_state_lock_accuracy.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/source_state_lock_accuracy.pdf)
- [training_support_B01 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_support_B01.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_support_B01.pdf)
- [training_support_B11 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_support_B11.png) · [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_cycleorder_factorial_E30_20261017/analysis_r02/training_support_B11.pdf)

## 最终回答与下一项源域问题

As显著缓解了仍有可恢复身份证据的practical混合残余观测退化；不能唯一归于多径/噪声/CFO。默认头As使low提高21.639286pp，但clean代价2.180357pp。joint low53.339881%、clean83.182738%，附加Ag没有urban收益；只有1seed，跨seed未知。固定周期路线关闭，整个研究不complete。

未完成：clean净正、low>60%、suburban保持、跨seed、损伤唯一机制归因；fusion gates/逐分支梯度和formal每包状态没保存，不补造。当前最有价值的问题是As在固定接收状态/补偿后减少了哪类损伤敏感性。下一步拟复用既有P24同包/同随机流控制，冻结当前matchedB00/B01默认头，区分总效应和固定状态效应；0训练、不另拟合头、不救回Ag、不以目标成绩选改参数。该诊断尚未登记/实现/启动。
