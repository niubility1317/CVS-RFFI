# CVS十二小时自动审查报告

生成时间：2026-06-26 13:33:42 Asia/Hong_Kong  
审查来源：GitHub发布仓库本地快照`E:\type10-7\github_publish\CVS-RFFI-repo`  
审查主体：Codex本地证据审查  
网页GPT状态：未执行，见同目录`WEB_GPT_UI_RUNTIME_BLOCKED.md`

## 1.证据边界

- 本报告只基于本次发布仓库快照、`experiment_records/CV-SincNet/`中的报告副本、自动化状态副本和已同步代码文件。
- 本轮没有访问N607、没有启动/停止/重启实验、没有重新读取远端GPU/进程/数据集，也没有把任何远端日志当作实时状态。
- 本轮没有实际调用ChatGPT Pro网页端GPT；未配置可审计的`CVS_CHATGPT_PRO_GPT_URL`和可复用的登录态/浏览器控制链路，因此不能伪造网页模型结论。
- `metrics_inventory.csv`当前为空表头。原因很可能是发布脚本为了规避Windows长路径把实验产物展平成`<hash>_metrics.json`和`<hash>_score_table.csv`，但清单收集逻辑仍按精确文件名`metrics.json`/`score_table.csv`匹配。这会让后续网页端或外部审查漏采指标，必须优先修复。

## 2.当前取得的主要成果

| 类别 | 当前状态 | 证据 |
|---|---|---|
| GitHub快照链路 | 已建立本地发布仓库、快照脚本、PowerShell入口和十二小时自动化任务 | `scripts/sync_cvs_release_snapshot.py`、`scripts/run_cvs_snapshot_cycle.ps1`、`docs/AUTOMATION_GITHUB_REVIEW.md` |
| 项目覆盖 | 已同步核心代码、工具、基线、paper reproduction、测试、协议副本和近期实验报告 | `code/`、`tools/`、`baselines/`、`paper_reproduction/`、`experiment_records/CV-SincNet/` |
| 近期H06 old/unknown诊断 | `stage2_spaceborne_h06_oldfuse_repair_20260626_122707`完成48/48，但结论是负诊断，不可提升为部署成功 | `experiment_records/CV-SincNet/reports/stage2_spaceborne_h06_oldfuse_repair_20260626_122707/report.md` |
| 发布侧基线 | 已保留发布仓库自带baseline脚本、说明和测试，避免后续快照覆盖掉发布侧文件 | `baselines/`、`tests/test_baseline_paper_launchers.py` |

## 3.主要矛盾

当前主矛盾仍是`identity-style conflict`没有被H06旧类/未知类路线解决：候选机制能通过回滚或阈值策略压低部署后的unknown FAR，但代价是旧类接收、覆盖率和可部署性同时塌缩。最新oldfuse修复矩阵48个候选全部触发rollback，`rollback_accepted_count=0`、`safe_gate_count=0`，因此它证明了这一类修复方向的边界，而不是证明了Stage2-B成功。

最典型的同一行证据是`OA_MSE_H06_OLDFUSE48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0`：整体`hmean=0.567783`、`old_acc=0.494444`、`unknown_FAR=0.333333`，但部署口径下`deployed_old_acc=0.083333`、`deployed_unknown_FAR=0.033333`且verdict为rollback。这个结果不能作为部署成功，只能说明安全回退压低了未知误接收，同时牺牲旧类可用性。

## 4.次要矛盾

| 次要矛盾 | 表现 | 风险 |
|---|---|---|
| 证据采集与外部审查输入不一致 | `metrics_inventory.csv`未采到展平后的指标文件 | 网页端GPT可能只看到报告摘要，漏掉候选级同排指标 |
| 发布镜像与发布侧维护文件的边界 | 快照脚本会重建部分目录，必须显式保留发布仓库自己的baseline文件 | 后续自动整理可能误删发布侧说明、测试或launcher |
| Stage2-B旧类校准与Stage2-C seen-new注册混线 | H06路线是old/unknown-only负诊断，不包含target-new support | 论文或报告容易误写成seen-new能力提升 |
| rollback安全与实际可用性冲突 | deployed unknown FAR低，但deployed old accuracy很低且全部rollback | 低误报不能替代可用的旧类识别能力 |

## 5.必须解决的问题

| 优先级 | 问题 | 具体要求 |
|---|---|---|
| P0 | 指标清单漏采 | 修改`collect_metrics_inventory`，同时识别`metrics.json`、`score_table.csv`和展平后的`*_metrics.json`、`*_score_table.csv` |
| P0 | 候选级联合排序不足 | 自动生成同一候选行内的`old_acc`、`unknown_FAR`、coverage、rollback、deployed指标表，禁止只引用边际最大/最小值 |
| P1 | H06后续路线需从“更多回滚校准”转向“旧类保留优先” | 新矩阵必须显式包含非回滚保留候选或oracle/upper-bound诊断，不能再只堆同族gate |
| P1 | Stage2-C主线需要单独推进 | 使用ManyTx真实non-old TX拆分`Y_new/Y_unknown`，与H06 Stage2-B诊断分开报告 |
| P2 | 优化过程可观测性 | 完整保留loss、coverage、rollback触发和adapter摘要，不能只看最终分数 |

## 6.文件级修改建议

| 文件 | 建议 |
|---|---|
| `scripts/sync_cvs_release_snapshot.py` | 修复指标文件后缀识别；在`metrics_inventory.csv`中加入`source_path`、`artifact_path`、`artifact_kind`；对展平文件保留原始basename字段 |
| `tools/analyze_h06_feature_separability.py`或新增H06聚合器 | 聚合OLDGEOM/OLDCONF/OLDBUDGET/OLDQUAL/OLDRISK/OLDFUSE所有score table，输出联合排序和负诊断对比表 |
| `tools/spaceborne_fewshot_da_matrix.py` | 下一轮Stage2-B矩阵应以old retention为主约束，明确`KOLD>0,KNEW0`、target-new support禁止、unknown query不参与阈值拟合 |
| `tools/optimizer_validate_matrix.py` | 强制Stage2-B行输出`rollback_triggered`、`deployed_old_acc`、`deployed_unknown_FAR`、coverage denominator和support/query字段 |
| `docs/source_controls/PROJECT_PROTOCOL.full.md` | 当前不建议修改协议；只有改变Stage2定义、K-shot、receiver/TX集合或指标声明时才先改协议 |

## 7.下一轮实验矩阵建议

| 路线 | 目标 | 边界 |
|---|---|---|
| H06 old/unknown保留优先诊断 | 在unknown FAR约束下最大化旧类accepted/full accuracy，减少全量rollback | Stage2-B only；`KNEW0`；不得声明seen-new |
| H06 oracle/upper-bound小矩阵 | 判断当前表征是否存在可被阈值或原型救回的上界 | 诊断-only；不得注册为部署证据 |
| Stage2-C ManyTx真实seen-new矩阵 | 用真实`Y_new/Y_unknown`拆分验证old+seen-new enrollment | 必须按receiver label对齐ManySig/ManyTx，禁止复用unknown query调阈值 |
| 自动化证据修复 | 先让快照清单稳定输出候选级指标表 | 修复后再让网页GPT读取，否则审查输入不完整 |

## 8.禁止写入论文/报告的声明

- 不得把H06 oldfuse写成部署成功。
- 不得把Stage2-B old/unknown拒识结果写成Stage2-C seen-new identity accuracy。
- 不得把clean view或rollback后的低unknown FAR写成真实在轨验证。
- 不得把所有48个候选rollback的矩阵写成可部署候选已找到。
- 不得在缺少网页GPT实际调用记录时声称“ChatGPT Pro已审查”。

## 9.自动化下一步

十二小时自动化可以继续执行快照、提交和本地审查；但要让网页ChatGPT Pro进入闭环，需要先提供稳定的网页GPT入口和可复用登录态控制方式。未满足前，自动化只能写入`WEB_GPT_UI_RUNTIME_BLOCKED.md`并保留本地Codex证据审查，不能伪造网页端输出。
