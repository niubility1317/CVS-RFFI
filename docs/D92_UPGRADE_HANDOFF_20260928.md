# D92优化当前交接

## 最新状态：BranchRidge完整重复基准通过，下一步独立数据确认

两run `20260929-phase2-d92-branch-ridge-repeat-rx3-m4-r01`和rx1同名均SCORED/ANALYZED，全部进程退出；终态/下载证据rx3 `readback_1790622737.json`及两组`readback_1790622753.json`。runtime commit仍`01e93386736f1919ee3f009fb1b16fb04f9a463d`。完整9648混淆记录、4800单次fullsupport fits、原D92/DG全记录一致性及4560汇总数值已VERIFIED。最终归档已执行`--write`，不得重跑同输出；两张图已目视检查。

联合任务Δ旧/新/H（百分点）：K1 +4.887/+5.418/+6.124；K5 +13.523/+6.839/+9.375；K10 +9.315/+6.103/+7.368；K20 +6.279/+6.501/+6.580。每K的四模型seed旧/新/H均提高，全部old guard通过；16个K×新增规模三指标均值均正；48个RX×scene×K的H全部正，旧/新分别47/48正，两个K1弱城市分层退化完整保留。已达到进入独立确认的预登记条件，不能称新独立数据泛化通过。主报告在rx3 `results/combined_rx4/report.md`，两run `results/artifacts.json`，统一fit审计在rx3 `results/fit_audit.json`。新增source0B，头35376至153296B；累计fit调用23.584s、提取551.847s，joint wall473.063s，非卫星耗时。

独立数据核查与采集规范已交付：`docs/D92_INDEPENDENT_DATA_AVAILABILITY_20260929.md`及同目录`D92_INDEPENDENT_DATA_METADATA_20260929.json`。confirmation_runner仅读inventory/元数据/物理ID，未读IQ或query成绩。现有未使用RX1-20/13-7缺TX、18-19最少14条，无法覆盖26TX/K20；四本轮RX最稀缺类余2/4条。最初7个RX未用余量仍UNKNOWN，不宣称整个库均无可用数据。三个capsule合计56160个不交ID仅是已核实排除范围，不代表全历史独立性证明。按原矩阵每TX/RX需180条物理记录；严格沿用rx3池大小则198条。已向用户询问新独立数据存放路径，尚未收到；当前方法/config冻结，独立验证待数据到位。不得重新调参、选择性重跑、复用旧ID伪称新数据。无运行实验；goal ACTIVE，独立验证尚未完成。以下RUNNING文字均为历史。

## 最新实时状态：BranchRidge两cohort RUNNING，禁止重复启动

实际release commit `01e93386736f1919ee3f009fb1b16fb04f9a463d`，已push并独立OID匹配。rx3 supervisor PID2590470、首export PID2590897；rx1 supervisor PID2591155、首export PID2591167；两组`readback_1790622276.json`确认live、argv/CWD及日志增长。后续`readback_1790622356.json`：rx3首模型export已完整15444条并PASS synthetic smoke，正切换第二模型；rx1两个模型export已完成、第三模型PID2592058正在导出。未读任何本轮scores。nvidia只读查询核实所属export在GPU0、约482MiB；其余GPU训练未动。

两run均已登记RUNNING。继续只读核实原run，均终态后再下载scores/startup/complete并汇总、独立算术核验。`tools/collect_d92_branch_ridge_audit.py`双spec全4800fit/成本审计已完成23项合成tests；只metadata/log不读scores。新finalizer由d92_p0_review实现中，尚未调用。goal ACTIVE，无权限阻塞。以下“尚未发布”均是历史。

## 最新状态：BranchRidge两cohort已就绪，尚未发布

新run `20260929-phase2-d92-branch-ridge-repeat-rx3-m4-r01`和rx1同名已PLANNED/launch-ready VALID。固定单view736维、全support一次ridge1、全K同式；新source0B，K1无OOF。40核心/probe、23export/entry、61编排、24summary tests通过。唯一P0/P1见`docs/D92_BRANCH_RIDGE_P0_20260929.md`，无阻断；实际checkpoint synthetic无query smoke已在export入口内，received singleton前向。资源/输出/capsule/原baseline句柄preflight VERIFIED。root唯一launch owner，GPU0每cohort一个串行export、每cohort最多4CPU lanes×2BLAS，不干预GPU3至7已有任务。

准备按两spec的launch_command发布，各release/run目录尚不存在；发布后必须独立readback再更新本段，禁止凭历史PLANNED重复启动。source_aux负责新纯metadata fit/cost collector，d92_p0_review负责新finalizer，仅合成开发中，不影响已冻结runtime。两组均完成后才下载任何scores，固定矩阵不择优重跑。goal ACTIVE，无待批准事项。

## 最新状态：support分支探查完整分析完成，下一候选正在实现

run `20260929-phase2-d92-branch-support-probe-m4-r01`已终态，全部8行/4800任务完整，1200个K1任务仅数值诊断，其余3600任务六臂物理OOF共64800次分解。独立汇总`results/support_summary/summary.json`状态COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED；分析commit `63aac180dbbd200ba3954929f4a457386aff7fd8`，runtime仍`f7f19e87042b22c499ca2cee3d01fd0334dc8d47`。所有进程已退出，禁止重跑探查或analysis输出。完整解释位于该run的`support_interpretation.md`。

合法support证据：联合任务zfft_aux相对zfft旧/新/H +5.025/+10.221/+9.464pp，相对重复背景+4.342/+9.047/+8.231pp。所有已汇总边际分层均值正，但有任务级退化，K1没有OOF；没有query结论。新source0B，墙钟121.324秒，累计probe345.523秒。已将纯support证据交给仍盲于所有query成绩的设计agent。

下一单一候选D92-BranchRidge-v1已接受实现：原view，736维固定identity/FFT+time/freq/PA分支，当前row全support平方损失+ridge1、不罚截距，一次解析解；全K同式，不增加网格，不重跑已有OOF。source_aux负责设计/config/core，confirmation_runner负责完整received特征export/evaluate，root负责编排/登记/发布；d92_p0_review待实现ready后做唯一P0/P1。尚未启动新benchmark。用户授权先复用完整旧数据，再全面改善后新增独立验证；无待批准事项，goal ACTIVE。以下均为历史。

## 当前状态：分支support信息探查已启动，正在导出/探查

最新实时证据`readback_1790620659.json`：run `20260929-phase2-d92-branch-support-probe-m4-r01` supervisor PID2574293 live，首个export PID2574301 argv/CWD匹配。实际runtime commit=`f7f19e87042b22c499ca2cee3d01fd0334dc8d47`，release=`d92_branch_support_probe_20260929_r01`。此前发布器Windows Path→反斜杠造成启动前失败，独立证据确认无run/release/process、原归档存在且一致；仅发布器改PurePosixPath（commit ce80b6afb41ac25b18c917c13227445107e4e999），显式staged-commit恢复核对原runtime无变化后首次启动。原失败landing与恢复landing均保留。不要重复发布/启动。后续按PID/artifact核对状态，等待全部8lane后汇总；以下尚未启动文字仅是预登记历史。9项编排含Windows路径回归通过；唯一P0/P1文档已完成，无阻断。

2026-09-29：新run `20260929-phase2-d92-branch-support-probe-m4-r01` 已PLANNED/launch-ready VALID，release预定`d92_branch_support_probe_20260929_r01`。实际架构元数据已核实，固定原始单view的t_emb/f_emb/pa_local作为一组。两个背景z与z+FFT分别固定baseline/duplicate/aux三臂，共六臂；全部ridge1，duplicate用于正则/能量对照。K1仅数值诊断，其他K逐物理OOF；完整4800episode、预期64800分解，0 query访问、0新source payload，不输出部署头。原型/ground/source数据不读取。设计`docs/D92_BRANCH_SUPPORT_PROBE_DESIGN_20260929.md`。

core21、export35、entry6、root编排8及旧readback8项相关测试已通过（各owner详细证据）；唯一P0/P1正在收尾，正式launch尚未执行。资源/新路径/capsule绑定预检VERIFIED：`evidence/support_preflight_1790620345048893000.json`（GPU0空闲、其余健康任务不动；未读IQ/未数据重验）。root唯一launch owner，GPU0串行冻结export与最多4 CPU lanes×2BLAS。新helper prepare/run/publish/preflight及现有read_d92_run支持该scope。source_aux另负责只读8lane完整汇总脚本（尚未交付），其余agent不访问query成绩。下一步稳定实现/唯一review完成→mirror/commit/push/OID→只启动一次→独立PID/artifact读回→全support结果分析。goal ACTIVE；以下MVRidge已完成归档。

## 当前状态：MVRidge完整结果已核验，未达到目标，无实验进程

2026-09-29后续只读架构核实已VERIFIED：`docs/D92_FIXED_PHASE1_BRANCH_METADATA_20260929.json`，四模型exact loader完整匹配；实际time/freq/PA/stats启用、DAC关闭，t_dim=f_dim=emb_dim160，joint_proj/pa_proj为320→160。eval/冻结、BN数量0；未读任何样本或成绩、0次forward。这解决下文“实际flags待核实”，尚未执行support前向一致性/信息探查。source_aux正在制定`docs/D92_BRANCH_SUPPORT_PROBE_DESIGN_20260929.md`（方案尚未交付、不得假定已存在）；root尚未登记或启动该探查。最新结果归档commit为e1e29539ba1feb0fd6474cc73faf2555f737b125，信息审查commit为a2f7fc0fca89c476e9ee1f0d417105d3e85750b4，均独立OID验证通过。保持goal ACTIVE。

最新终态：两组MVRidge r01均SCORED/ANALYZED，supervisor/children均退出。终态证据两组均`readback_1790618538.json`，下载证据均`readback_1790618554.json`；实际release仍为`71e4bef490bfae3acf907cc3f488b9ca62172a0f`。完整4800fits、15600解析调用、9648评分记录核验VERIFIED；原D92和frozen_dg逐单元记录/混淆矩阵完全一致，独立复算4560汇总数值maxerr3.33e-16。两run的`results/artifacts.json`及rx3的`results/combined_rx4/interpretation_audit.json`已写入，finalizer不可重复write，两张图已目视核对。

联合任务Δold/new/H（百分点）：K1=−1.9780/−0.4845/−0.1513；K5=+2.9896/−2.1958/−0.5371；K10=−0.1175/−4.0229/−2.7684；K20=−1.2008/−4.0356/−3.0512。K5/10/20的新类和H均0/4模型seed提高；K1新类1/4、H2/4。全部任务旧类guard在K1/K20也未通过。完整结果未达标，不晋级、不按成绩修改该候选或选择性重跑；goal ACTIVE。新source payload0B，持久数值状态12336至53456B；累计core fit124.4419秒，query score65.3643秒，均非并行墙钟/卫星延迟；最大梯度残差4.16e-13。已有received缓存占用另列，不是源域传输；模型部署情况未知、增量传输null。

盲态代码/理论审查已完成：`docs/D92_FIXED_PHASE1_INFORMATION_AUDIT_20260929.md`。结论为同一冻结骨干已计算但出口丢弃的融合前分支值得做逐row support-only验证；代码压缩瓶颈不是实际判别增量证据。静态no_dac不计DAC信息，实际flags待元数据核实。当前cache无法反解aux，补提取需要重跑合法support的冻结forward，未来同次导出才可不新增backbone前向次数。下一步A核对实际checkpoint架构/出口；B逐row数值非冗余；C K>=2严格物理OOF的单一预设探针对比，K1无独立类内holdout。未冻结下一分类器、未登记或启动下一实验。confirmation_runner正在做A的只读checkpoint配置元数据核实，预计证据`local_artifacts/d92_upgrade_20260928/fixed_phase1_branch_metadata_20260929.json`，不得假定文件已生成。两agent继续对所有query成绩盲态，root不传递本结果段。以下启动段仅保留历史，不能当实时状态。

2026-09-29：上一轮OSC全部终态。新工作先完成4800fits纯support OOF信息边界分析（docs/D92_OSC_SUPPORT_DIAGNOSTIC_20260929.md/json），未按query成绩选row/调参。盲态设计排除了Gaussian轨道矩近似及重复ground先验坐标变换，唯一新机制为`D92-MVRidge-v1`：固定identity/FFT=1:4，物理样本总view权重1，全部注册标签共同ridge判别回归，ridge系数固定1，解析解。K1利用多类标签与变换一致性，不声称新增独立观测；各fold所有拟合状态隔离，CV仅诊断。设计在docs/D92_SUPPORT_INFORMATION_DESIGN_20260929.md。

core/entry/audit分别由source_aux/confirmation_runner/d92_p0_review负责，保持前两agent目标成绩盲态；root负责集成/登记/发布。run `20260929-phase2-d92-mvridge-repeat-rx3-m4-r01`及rx1已启动；资源/新路径证据`mvridge_preflight_20260929.json`。核心9项、入口13项、collector57项及编排/评分/汇总测试通过，唯一P0/P1无阻断。release commit `71e4bef490bfae3acf907cc3f488b9ca62172a0f`已push并独立核对远端OID。两组启动readback均`readback_1790618402.json`；rx3 supervisor PID2550613、四个child argv/CWD正确，已完成约104至107/900任务；rx1 supervisor PID2551256正在验证冻结缓存。禁止重复启动。两组全部终态后才下载/读取成绩。复用现有BNNA纯冻结四相位cache和原D92基准，所有模型/算法拟合状态不继承。goal ACTIVE，无权限缺口。下一步终态核验、fit/算术/原baseline审计和完整联合报告。finalizer43项合成案例通过，尚未用于实测结果。

## 当前状态：OSC r02完整结果已核验，K5/10/20改善但K1未达标

2026-09-29最新终态覆盖下文历史：两r02均SCORED/ANALYZED，全部supervisor/children退出，无本任务运行中的实验。rx3终态证据`readback_1790616876.json`、rx1`readback_1790616803.json`，下载证据分别`readback_1790616899.json`/`readback_1790616900.json`。实际release`19a38b714f26599b6a9a074ccfedacc1c8df464b`。禁止重发r01/r02。完整4800fits、15600解析fit calls、9648score records审计VERIFIED，optimizer steps=0；原D92/DG全记录不变，4560汇总值独立复算maxerr3.33e−16。两run已有`results/artifacts.json`，rx3`results/combined_rx4/report.md`为联合报告，两图已目视检查。

联合任务Δold/new/H（百分点）：K1=−0.0029/−0.8050/−0.0947；K5=+8.7523/+1.8965/+4.5201；K10=+5.1736/+1.6861/+2.9952；K20=+2.1840/+3.5698/+3.2922。K5/10/20四个模型seed三指标全部提高；K1新类0/4提高、H1/4提高。正式all-cell old guard各K均通过（Δ+0.3870/+8.6264/+4.9403/+1.9796）。这是部分明显改善，未满足各K全面目标；不晋级、不新增独立数据，goal ACTIVE。不得选择性忽略K1或按成绩改已冻结方案。

新增source payload0B，无新特征提取、无checkpoint加载；持久数值状态49200至213200B。累计core fit412.755秒、query scoring159.287秒，均非墙钟/卫星延迟；模型部署状态未知，模型增量传输null。原始scores/full traces/compact logs保留原路径；Git仅小报告/图/审计。

仍盲于target成绩的source_aux在OSC成绩读取前独立完成后备设计`docs/D92_POST_OSC_DESIGN_20260929.md`，冻结设计commit`0127a53a918782db8f0e0b59d51ffd4322c179b9`。仅文档未实现：正则化Cauchy联合中心/协方差一次MM，明确K1无类内信息、不能保证全K，作者建议不要仅为“下一个方案”机械启动。后续方法决策须依据support/源域许可信息和理论，不向盲设计agent透露本段成绩。无权限阻塞，用户已授权先复用数据，待全面改善再独立验证。

## 当前状态：OSC r02已实际运行，禁止重复启动

2026-09-29独立读回`readback_1790616709.json`：rx3/rx1均PREDICTING，8个CPU worker的argv/CWD匹配。rx3各24/900fits，rx1各51/300fits；supervisor分别2532747/2532935。实际发布commit`19a38b714f26599b6a9a074ccfedacc1c8df464b`，远端Git OID一致已复核。r02已登记RUNNING。r01技术失败记录完整保留；旧SHA字段兼容修复及90合成tests、8真实cache只读绑定VERIFIED。下一步只读跟踪两r02，全部终态再收集评分/fit audit/统一解释。公式不改、不得重发/按性能停止，无真实OSC scores已读。goal ACTIVE。

## 当前状态：OSC r01技术失败，r02已预登记，等待缓存校验修复

已定位唯一问题：旧native `received_features/checkpoint_provenance.json`不含SHA；对应`d92_startup.json`和BNNA三份metadata均精确绑定预期SHA。最小兼容修复为旧origin显式有SHA时仍强制匹配，缺字段时由既有startup及producer/cache强绑定；科学来源/role/污染检查不变。90项相关合成测试通过，包含真实旧schema和错误SHA拒绝。8cache只读绑定检查进行中；r02尚未发布。

发布commit`7800bd72b154f6c8ded9ccbdac45bfa2cd5a5195`已push/OID VERIFIED。两r01均在缓存provenance校验阶段TECHNICAL_FAILURE，读回`readback_1790616335.json`确认全部worker/supervisor退出、无任何fit或score；已登记FAILED并保留产物。禁止重发r01。review agent正以获准缓存metadata定位字段不匹配，公式保持冻结；startup/final provenance不一致的初始假设已被8个cache证伪。两恢复run `20260929-phase2-d92-osc-repeat-rx3-m4-r02`和rx1同名已PLANNED，spec为`configs/d92_osc_repeat_rx3_recovery_20260929.json`及rx1，资源/路径证据`osc_recovery_preflight_20260929.json`。尚未launch r02。无用户许可缺口，继续技术修复→回归验证→commit/push→唯一root发布→读回。

## 当前状态：OSC实现和预登记完成，尚未发布

2026-09-29：OSC核心/入口/缓存校验及编排已实现，相关合成测试通过；两run `20260929-phase2-d92-osc-repeat-rx3-m4-r01`、`20260929-phase2-d92-osc-repeat-rx1-m4-r01`已PLANNED、launch-ready字段VALID。只读资源/路径核实见`local_artifacts/d92_upgrade_20260928/osc_preflight_20260929.json`。只复用BNNA的固定编码器四视图缓存，不复用其适应状态或成绩；CPU四model lanes×2BLAS，无checkpoint加载/GPU提取/source读取。唯一launch owner root。独立P0/P1审查完成后镜像、commit/push/OID再发布；禁止从旧记录推断已启动。

固定方案和矩阵不变，全部物理fold重估，CV仅诊断，零optimizer steps。新旧类与各K须全面明显改善，方进入新增独立数据阶段。两cohort终态前不读任何OSC scores。BNNA及更早run均终态，不能重发。用户复用数据授权持续有效；goal ACTIVE。

## 当前状态：BNNA完整重复基准未达标，全部终态已核验

**2026-09-29最新状态，覆盖下文运行中历史。**rx3/rx1均SCORED，所有supervisor/children退出；终态证据rx3`readback_1790615090.json`、rx1`readback_1790614237.json`，两run下载证据`readback_1790615121.json`。完整4800fits/9648scores、958784真实Adamsteps已核验，禁止重发原run。release仍`0fce1af680992fac2c343762af0f07915e10b647`。

联合Δold/new/H（百分点，new_count>0）：K1=−1.554/−0.617/−0.421；K5=+0.796/−7.347/−4.840；K10=−4.889/−11.565/−9.790；K20=−8.872/−14.906/−13.486。正式all-cell旧类保护Δ=−1.206/+0.919/−4.586/−8.831，仅K5通过。K5/10/20四个model seed的新类/H均下降；K1各1/4正向。未晋级，不新增独立数据，goal仍ACTIVE。

两run arithmetic/fit audit及combined interpretation均VERIFIED。独立复算4560汇总值，maxerr3.33e−16；58个原始证据不变，重复归档拒覆盖。core拟合累计5993.482s、提取累计452.549s，不是墙钟；fold691200steps、final267584steps；新增source载荷0，模型部署及增量模型传输null，完整训练checkpoint15992872至15992936bytes。持久数值状态12288至63552bytes。完整原始日志与评分保留原路径，Git仅小型报告/审计/图表。

下一步实施已在成绩读取前独立设计并提交的OSC方案，文档`docs/D92_NEXT_SUPPORT_DESIGN_20260929.md`（设计commit`a2d721e943b4369aeb8abe71ff3537c5c74d1e56`）。source_aux与confirmation_runner仍保持成绩盲态；不向其发送以上分数。当前仅设计、未实现、未预登记/启动。继续固定Phase1、support-only、完整物理fold、源数据禁入和透明重复基准。用户无需再次批准复用数据。

并行后备设计已完成：`docs/D92_NEXT_SUPPORT_DESIGN_20260929.md`，唯一OSC机制为四视图循环对齐与共享协方差Gaussian相位边缘化。由保持target成绩盲态的source_aux独立设计，未实现、未启动；BNNA保持不变。该设计在BNNA成绩读取前落盘，尚无效果结论。

后续只读进展：rx1在`readback_1790614237.json`已核实SCORED终态（2412records、4×300fits、无livechildren/supervisor），登记ARTIFACTS_COMPLETE；已收集`results/fit_audit.json`和fit_logs，完整1200fit/242496实际steps审计VERIFIED。rx3同次读回仍PREDICTING，4个CPU子进程2502694/96/97/98正常、各227至246/900fits。尚未下载或读取任一BNNA scores。不要重启rx1；继续原rx3。新归档工具`tools/finalize_d92_bnna_record.py`及19项合成测试已就绪，尚未对真实成绩执行。

**最新实际状态覆盖下文启动前历史。**发布commit`0fce1af680992fac2c343762af0f07915e10b647`，push/OID已独立核实。rx3为`20260929-phase2-d92-bnna-repeat-rx3-m4-r01`、supervisor2497991、首exporter2498011；rx1为`20260929-phase2-d92-bnna-repeat-rx1-m4-r01`、supervisor2498328、首exporter2498344。两run的`readback_1790613570.json`核实live argv/CWD，首model特征进度分别5248/15444与2176/4680，意味着实际native合成全流程smoke已通过；无真实BNNA评分读取。**禁止因压缩、超时或本文件旧文字重复发布。**下一步只读跟踪原run，全部终态后再下载评分、收集完整fit审计并统一分析。用户要求新旧类与各K全面明显改善后才新增独立数据；目前goal未完成。

2026-09-29：BNNA设计、core/config、4view exporter/predictor已实现；37项核心/入口合成检查通过，完整物理fold隔离和K1无CV已核实。runner/publisher/scorer/联合汇总已接入。新增run为`20260929-phase2-d92-bnna-repeat-rx3-m4-r01`与`20260929-phase2-d92-bnna-repeat-rx1-m4-r01`，spec对应`configs/d92_bnna_repeat_rx3_20260929.json`、rx1同名；两者PLANNED，唯一launch owner仍root。资源和新路径只读核实`local_artifacts/d92_upgrade_20260928/bnna_preflight_20260929.json`，GPU0空闲，其余任务未干预。

启动前实现核验完成：core14项、优化后入口+audit62项、编排评分103项相关测试通过，audit补充检查后独立38项通过；独立P0/P1无阻断发现，见`docs/D92_BNNA_P0_REVIEW_20260929.md`。同一C26K20纯合成配对完整fit18.985秒→5.234秒，数学等价且配置未变，只是本地成本。剩余镜像、commit/push/OID及原计划发布。固定公式/超参不因任何target结果改变。入口native smoke已扩展为8条纯合成IQ→4views→完整适应→query逐样本一致性，发生在received数据打开前。尚无BNNA真实分数，禁止从本段推断已launch或已达标。

**2026-09-29最新终态，覆盖下列运行中历史。**MVKME rx3-r02和rx1-r01均已SCORED/ANALYZED，所有supervisor/children退出；rx3终态`readback_1790611107.json`，rx1终态`readback_1790610777.json`，两run下载读回`readback_1790611135.json`。当前没有本任务运行中的实验，禁止重发已完成run。全部9648条评分逐混淆矩阵独立复算通过，4800次拟合日志审计通过，1840个summary值独立解释复核通过。完整报告为rx3-r02的`results/combined_rx4/report.md`，两run均有`results/artifacts.json`。

每K联合任务Δold/new/H(pp)：K1 +1.766/−0.297/+0.490；K5 +7.637/−1.213/+0.987；K10 +4.194/−1.341/−0.030；K20 +1.721/−1.271/−0.607。正式含old-only的old guard Δ=+2.000/+7.506/+3.750/+1.079pp，全过保护，但所有K平均new下降，未晋级，整体goal未完成，不进入新增独立数据验收。所有原始输出保留，不能据此修改已评分候选参数。

实际成本（累计进程内时间，非墙钟）：提取602.124s、core拟合370.824s、query分数计算76.741s、预测序列化写入22.107s。新source payload0，模型是否已部署未知，增量模型传输null；完整训练checkpoint15992872至15992936bytes，不是最小推理包。两图已目视检查。

下一步：未读目标结果的source_aux_feasibility正在设计ONE D92-BNNA-v1，只用当前row support估计最多8维不稳定方向并学习有界非线性收缩；设计文档计划`docs/D92_BNNA_DESIGN_20260929.md`，尚未实现或预登记/启动。confirmation_runner独立只读检查原D92数学结构，发现old-only metric在head-only LOO外预拟合导致support OOF旧新不对称风险；这是代码事实，不可据此断言目标退化原因。新方案必须每物理fold重新估计全部适应状态。两设计agent保持不读scores/results；root和解释审查者可归档评分。用户继续授权先复用数据，直到全面明显改善后才新增独立数据验证，无待批准事项。

## 以下保留已结束的MVKME运行过程，以顶部终态为准

最新实际状态：MVKME已修复并正常启动两cohort，均使用release commit`70bd88dfc6a0864e09b8fa582d2aa48b34a741fb`。rx3为`20260928-phase2-d92-mvkme-repeat-rx3-m4-r02`，spec为`configs/d92_mvkme_repeat_rx3_recovery_20260928.json`，supervisor2466165，首exporter2466183；rx1为`20260928-phase2-d92-mvkme-repeat-rx1-m4-r01`，spec为`configs/d92_mvkme_repeat_rx1_20260928.json`，supervisor2467532，首exporter2467546。两run的`readback_1790610576.json`均核实live/argv/CWD及日志增长，分别首model11648/15444与1152/4680；原rx3-r01失败终态保留。**禁止重发以上已启动run。**每cohort四model串行导出，两个cohort最多两个冻结推理进程共用GPU0，实测单进程484MiB，未干预其他GPU任务。

桥接修复已真实checkpoint+纯合成11IQ全链验证，独立故障定点审查通过，30项入口测试通过。未修改环境/核心/公式/数据/预算。修复使用Python列表双向转换，解决Torch2.1与NumPy2.2.5双数组类C接口不兼容。诊断见各run的`evidence/mvkme_numpy_bridge_diagnostic.json`。

下一步只读监控两个原run，均终态前不下载/读取新scores。全部完成后使用既有`read_d92_run.py --download scores.json complete.json startup.json`、`summarize_d92_repeated_benchmark.py`（已支持MVKME）、算术audit、`collect_d92_fit_logs.py`及新`collect_d92_mvkme_audit.py --spec ... --output <new fit_audit.json>`。新fit audit24项合成测试通过，尚未对真实run执行；它不读取scores/truth。先保留完整结果再解释，不凭实现/启动完成goal。

MVKME启动后技术状态更新：已发布commit`a208d859a0cf6e4d7beaaccfa7bb9133fc5ea7b3`并启动rx3-r01，supervisor2458913/child2458931均退出，`readback_1790609872.json`核实FAILED；无预测和评分，禁止原地重启。真实checkpoint零IQ smoke通过；后续torch转NumPy输出意外成为object dtype，核心正确拒绝。confirmation_runner在已发布代码上用纯合成IQ复现并定位转换/加载故障，不读源样本或目标结果。rx1-r01仅PLANNED尚未发布。下一步修复该技术问题、本地失败回归及原问题定点审查，rx3使用新r02，rx1更新联合引用后按原冻结算法执行。

用户最后确认“先复用数据，待性能有明显改善后再新增独立数据进行验证”。目标仍ACTIVE；无需等待新增数据。SGJoint两run（rx3/rx1）均已SCORED并登记ANALYZED，禁止重启。发布commit为`a794b71e1c93d6d4e9f3cfe1f2ccc591a685bdea`；终态证据分别为`readback_1790608523.json`与`readback_1790608322.json`。完整9648评分与4800拟合已核验。主报告在rx3的`results/combined_rx4/report.md`，两run均有`results/artifacts.json`和`fit_audit.json`。

K1/5/10/20的Δnew为−2.306/−1.041/−1.181/+0.178pp；ΔH为−1.557/+1.296/+0.129/+0.500pp。未晋级，不能称全面明显改善。联合表旧类只含new_count>0；正式旧类保护含old-only，Δold为+1.213/+7.097/+3.287/+0.973pp，均通过。原汇总保留，解释审计补足口径；新汇总器已修正。拟合累计1119.447秒；摘要numeric实际4410bytes，包8191至8383bytes，新增地面统计0。大scores和完整trace保留原路径。

下一版D92-MVKME-v1由未读目标结果的source_aux_feasibility独立设计并实现core/config/design；confirmation_runner实现exporter/predictor；d92_p0_review做一次P0/P1；root负责控制面与唯一launch。固定16个received-IQ视图、冻结identity160+FFT96、固定Fourier核均值、当前row support岭回归。K1固定，K>=2物理support九候选CV，无源样本或地面摘要拟合。目前仅实现/本地验证，尚未启动。下一步完成接口验证、预登记、资源核查和发布。四模型串行GPU提取后CPU拟合，两个cohort均终态后才读评分。

## 以下是SGJoint启动过程历史，以顶部终态为准

实际启动已VERIFIED：两run均使用release commit`a794b71e1c93d6d4e9f3cfe1f2ccc591a685bdea`。rx3 supervisor2441154、CPU子进程2441202/03/04/05在`readback_1790608341.json`仍live且argv/CWD一致、日志增长，继续原run；rx1 supervisor2441856已在`readback_1790608322.json`核实SCORED终态、4row完成、2412记录、无livechildren。尚未下载/阅读任一新scores。不要因前文PREPARING历史或观察超时重发。下一步只读核实rx3终态，两个run均终态后才下载scores并完整联合分析。

用户最新明确说“那就先复用数据”，随后要求解释；已解释固定基准复用与新独立确认的区别。等待新增数据的前一安排已被该授权替代。SGJoint公式与配置冻结，按原完整矩阵做重复基准，不因旧scores改公式、不读取source样本。子Agent负责reuse runner/publisher/scorer、两个新spec、只读输入预检和reuse delta正确性审查，主Agent是唯一launch owner。

新候选为D92-SGJoint-v1：既有量化摘要决定固定类无关归一化，identity160+同一received IQ的FFT96联合特征，注册类等先验的共享收缩LDA，K≥2仅当前row物理support CV，K1固定规则。参数来自未接触目标结果的独立数学设计，源样本/源逐样本特征/teacher/query均不参与拟合。代码、公式、固定配置和未来验收边界见`docs/D92_SUMMARY_JOINT_DELIVERY_20260928.md`及其链接。

拟执行run为20260928-phase2-d92-sgjoint-repeat-rx3-m4-r01和rx1-m4-r01，release为d92_sgjoint_repeat_rx3_20260928_r01和rx1_20260928_r01。各自复用旧SCV/r02与SFHead/r01的900/300split/model、四个相同模型、原D92基线预测和cached received features，CPU-only4lanes×2BLAS。只读preflight位于`local_artifacts/d92_upgrade_20260928/sgjoint_reuse_preflight.json`，新run/release/archive不存在，原run终态、八row/capsule/四checkpoint绑定已VERIFIED。此段写入时尚未启动；启动后以新run现状读回为准，绝不凭这一历史句重复启动。

两个cohort都完整执行，原SGJoint配置不变。两者终态前不下载/阅读新scores。每cohort内部全矩阵prediction固定后独立truth-last评分；主结论在四RX全部4800paired cells上等权汇总（含DG共9648评分记录），每K要求ΔH>0、Δnew>0、Δold≥−1pp，完整报告局部退化。`tools/summarize_d92_repeated_benchmark.py`负责严格完整联合汇总，已用9项合成测试验证3:1cohort的真实cell权重及防缺行/混搭。结果须标明REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS，不能冒充全新独立确认。无待批准权限或待新增数据阻塞。

## 下列为已保留的过程历史，以当前状态为准

最新状态：SFHead完整2412条结果已ANALYZED，supervisor2403988和所有子进程已终止，终态证据`readback_1790605266.json`。该版本每K的Δ旧/Δ新/ΔH（百分点）：K1 +0.29/−2.92/−4.54；K5 +10.88/−13.20/−8.89；K10 +4.27/−14.14/−11.28；K20 +1.34/−15.90/−12.73。未晋级，goal未完成。完整图表和CSV在本run的results/summary，2412混淆矩阵独立复算通过，1200拟合完整日志核验：1168收敛、32个K20预算达到300，所有raw/compact/trace保留。禁止重跑或根据这些目标分数调参。

下一步：`/root/source_aux_feasibility`正在做不接触目标结果的独立数学/代码方案核查，尚未写新候选或启动新实验。它没有收到SFHead结果，职责是从用户source-free约束与已有D92代码独立推导完整候选；每row只能用本row物理support holdout，K1固定规则，不借用其他row额外support，无sourceproxy。等待其完成后按实际协议判断可执行性，不能把本次评分传给研发任务。所有下面的RUNNING/待评分条目均是过程历史，当前以本段终态为准。

当前用户纠正（2026-09-28）：辅助训练禁止使用源域数据；允许地面已冻结模型、原型及少量汇总统计，结合卫星合法support。暂无硬传输上限，优先小payload并报告字节数。源TX留出辅助模型方案已被替代，未启动任何新训练；其实现草稿已归档并从正式代码撤出。当前推进source-data-free Phase2方法，禁止读取source IQ、逐样本特征、source loader或由原型重建伪源样本。允许的量化摘要按当前`项目.md`5.3.2及实际冻结来源判断；无需再次询问辅助训练授权。下列此前“等待范围答复”条目只记录历史。

新候选D92-SFHead-v1与冻结配置见`docs/D92_SOURCEFREE_AUX_20260928.md`及`configs/d92_sourcefree_frozen_20260928.json`。仅support上的taskbalancedCE+旧类KD+L2，CPU确定性L-BFGS-B，无source校准、无query选参。新增地面统计0字节，现有ground摘要只读核算8191至8383字节/模型。37项core/summary/allocation/predictor合成测试、42项runner/scorer测试通过，独立P0/P1审查无阻断。新run：`20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01`；data run：`20260928-phase2-d92-sfhead-data-manytx-r01`；release：`d92_sfhead_confirmation_20260928_r01`。RX20-19按未用于本轮评分且完整26TX可用性选取；4模型×1RX×3scene×4K×5newcounts×5supportseed=1200配对split，连同基线和DG共2412记录。每TX180物理记录（3scene各supportpool30/query30），全矩阵固定后才独立评分。

当前已VERIFIED启动：supervisor PID2403988，实际release commit `3348c8c37df402098a6f84397eaa1bf74d8d592e`。`readback_1790604450.json`核实4个D92子进程的实际argv/CWD一致，4模型已完成checkpoint无query推理检查及received特征提取。数据已ARTIFACTS_COMPLETE，capsule `residual-noeq-d0a99fede324159c5a4750fd`，4680观测、300splits，VALIDATED_ONCE。保持此run，禁止重复启动或改运行中配置。监控命令：`python tools/read_d92_run.py --spec configs/d92_sourcefree_confirmation_20260928.json --compact`。新结果尚未读取，完成后下载scores并用支持新spec的汇总器评分；不能以本地测试或启动成功标记goal完成。

本地汇总器已支持`--spec configs/d92_sourcefree_confirmation_20260928.json`，20项合成测试通过；由spec验证2412条精确矩阵覆盖，按每K的ΔH>0、Δnew>0、Δold>=−0.01判断。本地spec只补充了与发布前data_config及notes一致的`data.scenarios`和机器可读acceptance字段，未修改远端配置、候选或矩阵。终态确认后可用`read_d92_run.py --download scores.json complete.json startup.json`下载，用`summarize_d92_confirmation.py --scores <path> --spec <spec> --output <new directory>`汇总，`collect_d92_fit_logs.py --spec <spec>`收集全部compact JSONL/CSV；保留远端完整fit_trace。旧`finalize_d92_confirmation_record.py`写死SCV旧run，不可用于本轮。需新增本轮独立算术核验与最终报告，不能误写旧结果或晋级。

独立算术核验入口已补好：`tools/audit_d92_confirmation.py --spec <spec> --scores <scores.json> --output <new arithmetic_audit.json>`，9项合成测试通过。`plot_d92_confirmation.py --summary <summary.json>`现按真实方法名/矩阵/验收结果绘图，拒绝覆盖既有图；真实图尚待本轮全部完成后生成并目视检查。最新运行检查`readback_1790604980.json`仍核实supervisor与4个D92子进程活跃、argv/CWD一致，基线226至246条预测，尚未进入最终评分。上一goal turn为实现/发布进展，本轮也有指标核验工具进展与真实live进程等待，不存在权限阻塞。

- 目标：固定Phase1，全面改善新旧类与各K的H。用户明确强调新类也要好。源域结果不能完成goal；目标分数不能回流调参、选种子或选择性重跑。
- 工作树：`E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt`，分支`codex/d92-support-upgrade-20260928`。根目录不是Git仓库，主承载面其他暂存改动未动。
- 源域旧类诊断180条、注册代理2100条均完成并登记ANALYZED。K1固定FFT权重0；K≥5的support-only选择算法与grid冻结。代理6个TX均被Phase1见过，不能称真正新TX结果。
- 数据run：`20260928-phase2-d92-confirmation-data-manytx-r01`，15444观测、900划分；capsule `residual-noeq-76121e6f34363fa612ec25fb`已VALIDATED_ONCE，不重建或重验。
- r01执行在加载checkpoint前因reference contract没有classes字段失败；无目标预测/评分。保留所有记录。已修复为actual source→注册旧类→ground NPZ类序绑定；15个runner测试通过。
- 已完成运行：`20260928-phase2-d92-scv-confirmation-manytx-m4-r02`，N607 supervisor PID2354788已退出，release `d92_scv_confirmation_20260928_r02`，实际代码commit `841f189690884f0ae51ad8778099fc576e48d849`。完整7236条记录已评分、下载、汇总；不得重复启动。
- 完整矩阵：4model seeds2026092701..04 ×3RX19-1/8-14/8-7 ×3scenes ×K1/5/10/20 ×new0/2/5/10/20 ×5support seeds2026092711..15。每模型900D92+900SCV+9DG，合计7236评分行。RX与当前源/最近目标集合不重合，不宣称所有历史从未使用。
- 当前独立证据：根目录`automation_reports/CV-SincNet/<run>/evidence/readback_1790601598.json`。run已登记ANALYZED；`results/summary/`保留全K、新增类规模、seed与RX/scene、H/两侧准确率/宏F1/floor/forgetting，完整7236覆盖检查通过。4模型compact JSONL/CSV已收集，图已渲染检查。
- 科学结果：候选未通过。每K的Δ旧/Δ新/ΔH（百分点）：K1 +6.46/−7.00/−3.87；K5 +7.82/+0.26/+2.63；K10 +3.00/−0.85/+0.24；K20 −0.23/−0.04/−0.11。不得晋级或标记goal成功，不依据这些目标分数改参数/筛seed/重跑。
- 下一步的范围问题已用async问用户：允许独立源域辅助模型做真正类留出开发，还是禁止所有辅助训练仅用冻结特征。正式4个Phase1均保持冻结。用户回复前不启动依赖此授权的辅助训练。已知源代理六TX均被Phase1见过的局限是目标评分前的证据；辅助训练候选与参数必须只从源域开发选择，禁止将此目标结果传入开发选参流程。
- 已完成不接触目标结果的独立代码可行性核对。具体方案见`docs/D92_SOURCE_AUX_PLAN_20260928.md`：预先固定两组互补3TX、独立scratch辅助模型、TX筛选和连续标签映射、独立辅助导出。状态DRAFT_PENDING_USER_SCOPE，未登记或启动辅助实验。正式6TX源训练实测约19.1至20.0小时/模型，辅助耗时尚未知。待范围答复后再实施依赖步骤。
