# JG_R8_LR020新类注册development追踪表

日期：2026-07-16

状态：retry3在训练前暴露N607 Torch/NumPy ABI桥问题；兼容修复已提交并同步，远端哈希/py_compile/dry-run通过，等待retry4启动

依据：`E:\type10-7\AGENTS.md`、`E:\type10-7\项目.md`、用户锁定的`JG_R8_LR020`候选

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| JG-01 | `项目.md` 7.1 | Phase2仅可接收预先叠加且可验证的`leo_*_weak`密封输入；clean数据、缓存、派生信号与切换控制流均不可达 | `jg020_stage2c.py`、registered cache spec、split builder | implemented/local-verified | exact role allowlist、detached seal、pre-open SHA256、NPZ allowlist测试通过 | N607真实cache provenance待运行复核 |
| JG-02 | `项目.md` 7.2 | predictor逐样本面向全部已注册类；无query role、真实批次类别数、quota、global assignment | apply-only predictor、contract test | implemented/local-verified | all-registered prototype test通过 | 禁止Hungarian/OT/批量重排 |
| JG-03 | `项目.md` 7.2 | 预测与评分隔离：先输出SHA256绑定的不可变prediction artifact，之后独立scorer才连接truth | apply-only predictor、isolated scorer | implemented/local-verified | `.cvspred`只读、O_EXCL/no-replace与truth-root分离测试通过 | scorer输出不得反馈预测或适配 |
| JG-04 | `项目.md` 7.1、7.2 | enrollment/adapter进程仅接收注册support，不得接收query路径、payload、标签或统计 | support-only CLI、enrollment package | implemented/local-verified | CLI无query/truth参数，物理role-set负测通过 | apply包反向不含support/truth |
| JG-05 | `项目.md` 7、8.4 | `R_t={20-1}`与`R_s`不相交；同receiver同时有真实target-old与ManyTx target-new覆盖 | registered cache spec | configured/pending-remote-data-audit | old=`ManySig.pkl`、new=`ManyTx.pkl`、receiver=`20-1`已锁定 | 实际逐类30+样本覆盖待N607 cache build验证 |
| JG-06 | `项目.md` 8.4、10.3.1 | 使用预先按覆盖锁定的真实嵌套`Y_new^5⊂Y_new^10⊂Y_new^20`，禁止按query性能选TX | 3个candidate lock、cache spec | implemented/local-verified | label顺序SHA256与5/10/20前缀锁通过 | 开发seed为713101 |
| JG-07 | `项目.md` 9.2、9.3 | 同一run保存注册前Stage2-B old结果和注册后Stage2-C old/new结果，复用同一old query | old-only adapter fit、frozen runtime、append-only prototype head | implemented/local-verified | new support不进cached full-forward/episode的sentinel测试通过 | before/after共享相同runtime与old query |
| JG-08 | `项目.md` 10.3.1 | 锁定候选为ADV3B02+ground P4+BPJG-LOPO `joint_gate`, rank8, lr0.02, 5epoch, <=50step, K10 | candidate locks、support-only trainer、launcher | implemented/local-verified | exact 6,400参数门与真实artifact cached parity证据已锁定 | N607 receipt复核steps/loss/resource |
| JG-09 | `项目.md` 8.5、10.3.1 | development cell覆盖`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`，support/query场景逐row一致 | cache spec、package manifest、scorer | implemented/local-verified | 三场景exact tuple与物理token顺序检查已实现 | query固定1-view |
| JG-10 | `项目.md` 10.3.1 | 同row报告`old_acc_before_increment`、注册后`old_acc`、`min_old_class_acc`、`seen_new_acc`、`H_old_new`、forgetting、逐类与逐场景 | immutable prediction streams、isolated scorer | implemented/pending-run | 五条prediction stream已绑定同row | 结果待N607运行，不拼接边际最大值 |
| JG-11 | `项目.md` 10.3.1 | 保存完整训练loss trace和实际适配资源：参数、steps、时延、峰值显存、状态、MAC/forward | enrollment receipt、loss trace | implemented/pending-run | full-backbone与cached-small-path分账字段已实现 | 实测值待N607 receipt |
| JG-12 | `项目.md` 10.3.1 | 同sample/同view保存strict direct ADV3B02 old-only对照 | direct runtime、direct stream | implemented/pending-run | checkpoint类别映射顺序SHA已锁定 | direct无新类头，不计算seen-new |
| JG-13 | `项目.md` 10.3.1 | development query只评分，不参与candidate、epoch、阈值或超参数选择 | candidate locks、apply package | implemented/local-verified | predictor不fit head、无optimizer/query truth路径测试通过 | 本轮仅运行锁定JG_R8_LR020，不做query sweep |
| JG-14 | `AGENTS.md` Experiment Reporting | N607前本地报告、环境/命令/路径/GPU/输出、风险、成功条件齐备；完成后回收完整日志 | `automation_reports/CV-SincNet/qknnv42_jg_r8_lr020_newclass_dev_20260716/report.md` | implemented/preflight-verified | 本文件与预运行报告已创建；直连preflight与training inventory均PASS | N607尚未同步或运行 |
| JG-15 | `AGENTS.md` Version Management | 本地先改、`ssr-gpu`验证、Git diff/status、提交，再SCP并校验哈希 | report、Git commit、sync manifest | local-verified/commit-pending | 最新py_compile与12/12 pytest通过；待提交前执行diff-check | 保留用户已有两处未提交修改；retry4尚未SCP |

## 遗漏风险

1. 旧trainer即使不使用query loss，只要CLI或进程文件allowlist可访问query，也不满足JG-04。
2. package若包含truth sidecar、raw PKL路径或legacy loader开关，不能交给adapter/predictor。
3. 严格时序实现为：仅6个旧类registered support训练JG一次，冻结runtime；新类support只追加prototype，不进梯度。before/after复用相同old query、scenario、1-view与runtime。
4. `Y_new`必须是真实ManyTx TX label且按覆盖预登记，不能在看过query结果后替换。
5. 本development cell只用于验证锁定候选的新类注册行为，不能替代5receiver×5确认seed完整矩阵。
6. 本cell仅为1-view cosine prototype注册头；无FFT96、无自适应多View，不是最终qKNNV42算法版本。
