# D92优化当前交接

## 当前状态：MVKME完整重复基准未达标，正在独立设计BNNA

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
