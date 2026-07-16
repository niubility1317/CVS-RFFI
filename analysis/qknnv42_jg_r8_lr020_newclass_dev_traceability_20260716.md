# JG_R8_LR020新类注册development追踪表

日期：2026-07-16

状态：retry7完整PASS；new5/new10/new20、3种LEO弱化场景共9个正式Stage2-C结果行已回收，最终结论为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`

依据：`E:\type10-7\AGENTS.md`、`E:\type10-7\项目.md`、用户锁定的`JG_R8_LR020`候选

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| JG-01 | `项目.md` 7.1 | Phase2仅可接收预先叠加且可验证的`leo_*_weak`密封输入；clean数据、缓存、派生信号与切换控制流均不可达 | `jg020_stage2c.py`、registered cache spec、split builder | remote-verified/PASS | exact role allowlist、detached seal、pre-open SHA256、NPZ allowlist及N607实际opened-member审计PASS | 三规模enrollment/apply均`clean_member_reachable=false` |
| JG-02 | `项目.md` 7.2 | predictor逐样本面向全部已注册类；无query role、真实批次类别数、quota、global assignment | apply-only predictor、contract test | remote-verified/PASS | predictor receipt与formal predictions确认逐样本、单view、全部注册类 | 禁止Hungarian/OT/批量重排 |
| JG-03 | `项目.md` 7.2 | 预测与评分隔离：先输出SHA256绑定的不可变prediction artifact，之后独立scorer才连接truth | apply-only predictor、isolated scorer | remote-verified/PASS | 三个`.cvspred`均`SEALED_READ_ONLY_ATOMIC_NOREPLACE`；scorer按`exact_scenario_query_token`事后连接truth | scorer输出未反馈预测或适配 |
| JG-04 | `项目.md` 7.1、7.2 | enrollment/adapter进程仅接收注册support，不得接收query路径、payload、标签或统计 | support-only CLI、enrollment package | remote-verified/PASS | enrollment为`query_member_reachable=false`、`truth_member_reachable=false`；apply为`support_member_reachable=false`、`truth_member_reachable=false` | 三规模均PASS |
| JG-05 | `项目.md` 7、8.4 | `R_t={20-1}`与`R_s`不相交；同receiver同时有真实target-old与ManyTx target-new覆盖 | registered cache spec | remote-verified/PASS | old=`ManySig.pkl`、new=`ManyTx.pkl`、receiver=`20-1`；每类K10 support+Q20正式query均构造成功 | source package count为n5:110/220、n10:160/320、n20:260/520 |
| JG-06 | `项目.md` 8.4、10.3.1 | 使用预先按覆盖锁定的真实嵌套`Y_new^5⊂Y_new^10⊂Y_new^20`，禁止按query性能选TX | 3个candidate lock、cache spec | implemented/local-verified | label顺序SHA256与5/10/20前缀锁通过 | 开发seed为713101 |
| JG-07 | `项目.md` 9.2、9.3 | 同一run保存注册前Stage2-B old结果和注册后Stage2-C old/new结果，复用同一old query | old-only adapter fit、frozen runtime、append-only prototype head | implemented/local-verified | new support不进cached full-forward/episode的sentinel测试通过 | before/after共享相同runtime与old query |
| JG-08 | `项目.md` 10.3.1 | 锁定候选为ADV3B02+ground P4+BPJG-LOPO `joint_gate`, rank8, lr0.02, 5epoch, <=50step, K10 | candidate locks、support-only trainer、launcher | implemented/local-verified | exact 6,400参数门与真实artifact cached parity证据已锁定 | N607 receipt复核steps/loss/resource |
| JG-09 | `项目.md` 8.5、10.3.1 | development cell覆盖`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，support/query场景逐row一致 | cache spec、package manifest、scorer | implemented/local-verified | 三场景exact tuple与物理token顺序检查已实现 | query固定1-view |
| JG-10 | `项目.md` 10.3.1 | 同row报告`old_acc_before_increment`、注册后`old_acc`、`min_old_class_acc`、`seen_new_acc`、`H_old_new`、forgetting、逐类与逐场景 | immutable prediction streams、isolated scorer | remote-verified/PASS | 9个正式row与逐TX同sample指标已写入report | 所有row均fail-target，不拼接边际最大值 |
| JG-11 | `项目.md` 10.3.1 | 保存完整训练loss trace和实际适配资源：参数、steps、时延、峰值显存、状态、MAC/forward | enrollment receipt、loss trace | remote-verified/PASS | 三规模均6400参数、50step、峰值31924224B；持久状态70816/80416/99616B；runtime parity 0 | scorer不允许从prediction artifact单独提出adapter资源正式声明 |
| JG-12 | `项目.md` 10.3.1 | 同sample/同view保存strict direct ADV3B02 old-only对照 | direct runtime、direct stream | remote-verified/PASS | clear/low-elev/rain direct old_acc=0.6667/0.6833/0.6167 | direct无新类头，不计算seen-new |
| JG-13 | `项目.md` 10.3.1 | development query只评分，不参与candidate、epoch、阈值或超参数选择 | candidate locks、apply package | implemented/local-verified | predictor不fit head、无optimizer/query truth路径测试通过 | 本轮仅运行锁定JG_R8_LR020，不做query sweep |
| JG-14 | `AGENTS.md` Experiment Reporting | N607前本地报告、环境/命令/路径/GPU/输出、风险、成功条件齐备；完成后回收完整日志 | `automation_reports/CV-SincNet/qknnv42_jg_r8_lr020_newclass_dev_20260716/report.md` | completed | retry7 summary、outer log、三规模receipt/predictions/formal rows已回收；最终inventory与SSH断开PASS | 另有非本实验GPU任务，未干预 |
| JG-15 | `AGENTS.md` Version Management | 本地先改、`ssr-gpu`验证、Git diff/status、提交，再SCP并校验哈希 | report、Git commit、sync manifest | completed | 最新py_compile与14/14 pytest通过；同步文件哈希已记录；最终结果准备提交 | 保留用户与并发任务的无关修改 |

## 遗漏风险

1. 旧trainer即使不使用query loss，只要CLI或进程文件allowlist可访问query，也不满足JG-04。
2. package若包含truth sidecar、raw PKL路径或legacy loader开关，不能交给adapter/predictor。
3. 严格时序实现为：仅6个旧类registered support训练JG一次，冻结runtime；新类support只追加prototype，不进梯度。before/after复用相同old query、scenario、1-view与runtime。
4. `Y_new`必须是真实ManyTx TX label且按覆盖预登记，不能在看过query结果后替换。
5. 本development cell只用于验证锁定候选的新类注册行为，不能替代5receiver×5确认seed完整矩阵。
6. 本cell仅为1-view cosine prototype注册头；无FFT96、无自适应多View，不是最终qKNNV42算法版本。
