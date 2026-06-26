# CVS-only发布范围

本仓库是从`E:\type10-7`整理出的GitHub发布包，不是本地工作区镜像。

## 已包含

- `code/`核心源码：CVS训练、数据协议、模型、satellite channel、Stage2和CVS工具。
- `paper_reproduction/`CVS相关论文复现与CVS-aligned扩展代码。
- `baselines/`CVS对照实验所需的源码级baseline实现，不包含baseline运行产物和权重。
- `tests/`和`code/tests/`中的协议、paper reproduction、Meta-SSL和Stage2 smoke测试。
- `docs/`中的公开CVS协议、训练、部署和发布范围文档。
- `scripts/launchers/`中CVS相关启动脚本和根目录兼容wrapper。
- `tools/`中CVS矩阵、验证、分析和发布同步工具。

## 已排除

- WiSig/ManySig数据、`Dataset_WigSig/`和所有`*.pkl`数据文件。
- 模型权重、checkpoint和特征文件：`*.pt`、`*.pth`、`*.ckpt`、`*.npz`、`*.npy`。
- 运行产物：`runs/`、`logs/`、`outputs/`、`analysis_tmp/`、`remote_artifacts/`。
- 自动化报告、`experiment_records/`、服务器日志、远端备份和本地snapshot。
- 本地工作区笔记、历史source notes、ChatGPT/GPT审查提示和审查输出：`docs/source_notes/`、`docs/source_workspace_docs/`、`docs/analysis_requests/`、`docs/ai_review/`。
- baseline历史运行产物：`baselines/baseline_runs/`。
- paper-only复现队列、paper parity测试、Fedbase paper训练器、`paper_resnet/`、`code/SYNC_MANIFEST*`和非CVS paper reproduction配置。
- N607私有SSH配置、内网地址、账号路径、私钥路径和远端绝对路径配置。
- PPT、DOCX、XLSX、第三方论文PDF和大型渲染媒体。
- Codex/Claude本地状态、临时目录、缓存和`__pycache__`。

## 发布原则

该发布包只保留CVS相关的可审计源码、协议逻辑、测试、launcher和必要说明，删除运行环境、历史归档和本地审查材料。任何复现实验都需要研究者提供自己的WiSig/ManySig路径、GPU环境和输出目录。仓库中的命令默认是smoke、dry-run或协议检查，不直接声明论文完整复现或在轨部署成功。
