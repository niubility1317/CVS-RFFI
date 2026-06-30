# CVS GitHub发布快照

- 生成时间：2026-06-30T16:30:40Z UTC
- 本地时间戳：`20260701_003040`
- 来源工作区：`E:\type10-7`
- GitHub发布工作区：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 同步文件数：777
- 跳过文件数：58
- 生成文件清单：`docs/release_manifest_latest.json`

## 已同步范围

- 核心代码：`code/`、`baselines/`、`paper_reproduction/`、`tests/`。
- 项目控制文件：`docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。
- 自动化工具：`tools/`中允许公开的`.py`、`.md`、`.ps1`文件。
- 启动脚本：`scripts/launchers/run_*.sh`。
- 公开协议文档：`docs/PROJECT_PROTOCOL.md`、`docs/GROUND_TRAINING.md`、`docs/DEPLOYMENT_PHASES.md`、`docs/PUBLISH_SCOPE.md`。

## 本轮落实

- GitHub发布范围已收敛为CVS-only：只同步CVS相关源码、协议、工具、launcher、测试和发布说明。
- 已阻断实验记录、AI审查提示/输出、source notes、baseline历史运行产物进入发布仓库。
- RIEI/DRIFT仅作为CVS对照baseline保留，不上传paper-only队列、paper parity测试或Fedbase paper材料。

## 边界

- 不上传`experiment_records/`、`docs/source_notes/`、`docs/source_workspace_docs/`、`docs/analysis_requests/`或`docs/ai_review/`。
- 不上传WiSig/ManySig数据集、模型权重、checkpoint、原始大日志、N607凭据或本地密钥。
- 不上传baseline历史运行产物、自动化报告、服务器日志或本地snapshot。
- 不上传paper-only复现队列、paper parity测试、Fedbase paper训练器、`paper_resnet/`或非CVS paper reproduction配置。
- 指标解释必须绑定同一run或同一candidate row，不能把不同row的单项最大值拼成结论。
- clean view只能作为control/reference；Stage2-A/B不能声明seen-new identity accuracy。
