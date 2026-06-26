# 发布范围

本仓库是从`E:\type10-7`整理出的GitHub发布包，不是本地工作区镜像。

## 已包含

- `code/`核心源码：训练、数据协议、模型、satellite channel、Stage2和CVS工具。
- `paper_reproduction/`论文复现与CVS-aligned扩展代码。
- `baselines/`源码级baseline实现，不包含baseline运行产物和权重。
- `tests/`和`code/tests/`中的协议、paper reproduction、Meta-SSL和Stage2 smoke测试。
- `docs/`中的公开协议文档。

## 已排除

- WiSig/ManySig数据、`Dataset_WigSig/`和所有`*.pkl`数据文件。
- 模型权重、checkpoint和特征文件：`*.pt`、`*.pth`、`*.ckpt`、`*.npz`、`*.npy`。
- 运行产物：`runs/`、`logs/`、`outputs/`、`analysis_tmp/`、`remote_artifacts/`。
- 自动化报告、服务器日志、远端备份和本地snapshot。
- N607私有SSH配置、内网地址、账号路径、私钥路径和远端绝对路径配置。
- PPT、DOCX、XLSX、第三方论文PDF和大型渲染媒体。
- Codex/Claude本地状态、临时目录、缓存和`__pycache__`。

## 发布原则

该发布包保留可审计源码和协议逻辑，删除不可公开的运行环境细节。任何复现实验都需要研究者提供自己的WiSig/ManySig路径、GPU环境和输出目录。仓库中的命令默认是smoke或协议检查，不直接声明论文完整复现或在轨部署成功。
