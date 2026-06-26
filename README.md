# CVS-RFFI / CV-SincNet

CVS面向天基射频指纹识别中的弱标注跨接收机域泛化与在轨跨域少样本适应。项目采用`地面训练、天上部署`架构：地面端学习跨接收机稳定的发射机身份表征，部署端在目标卫星接收机域内用少量样本完成旧类校准、新类注册和未知类拒识。

本仓库是从本地实验工作区整理出的干净发布包。它包含源码、协议文档、论文复现扩展和小规模测试，不包含WiSig/ManySig数据、模型权重、N607私有自动化配置、远端日志、实验报告归档、PPT或第三方论文PDF。

## 当前协议版本

- 协议日期：2026-06-24
- 科研主线：weak-label/semi-supervised source-domain DG + spaceborne few-shot adaptation
- 地面训练：集中式训练版和元学习版并列
- 在轨部署：Stage2-A/B/C三段协议
- 声明边界：satellite/LEO stress是物理启发部署压力测试，不是真实在轨验证

## 项目边界

CVS不应被表述为普通WiSig少样本分类、普通全监督域泛化、纯few-shot learning，或真实卫星部署已完成验证。WiSig/ManySig是地面可接入代理数据体系；satellite-channel augmentation和satellite stress用于deployment-oriented validation-control。

地面阶段只允许使用源接收机域`R_s`。目标接收机域`R_t`不得参与训练、验证、early stopping、BN统计、阈值拟合、prototype构建、adapter更新或伪标签生成。一旦目标域参与模型选择，该结果必须改标为DA/TTA/few-shot adaptation，而不是source-only DG。

## 训练与部署路线

| 层级 | 路线 | 核心入口 | 协议含义 |
|---|---|---|---|
| 地面训练 | 集中式训练版 | `code/train.py --train_mode centralized` | 汇聚`R_s`内弱标注和无TX标签样本，训练CV-SincNet/CVS身份表征 |
| 地面训练 | 元学习版 | `code/train.py --use_meta_ssl_cvs --use_meta_rxday_episodes` | 在`R_s`内部用receiver/day/rx_day组织episodic source split，模拟未见接收机外推 |
| 地面评估 | source-only DG | `code/eval_feature_diagnosis.py`、`code/training_test_eval.py` | 评估strict UDU、receiver floor、satellite stress、leakage probe |
| 在轨部署 | Stage2-A | `code/cvsrffi/spaceborne_fewshot.py` | 零目标标签下旧类识别和非旧类拒识 |
| 在轨部署 | Stage2-B | `code/cvsrffi/spaceborne_fewshot.py` | 目标域旧类`K`shot校准，不声明seen-new identity accuracy |
| 在轨部署 | Stage2-C | `paper_reproduction/cvs_aligned/`、`code/cvsrffi/spaceborne_fewshot.py` | 目标域旧类校准+seen-new enrollment+unknown拒识 |

## 仓库结构

```text
code/
  train.py                     # 中心训练、联邦训练和Meta-SSL-CVS入口
  dataset_wisig.py             # WiSig/ManySig split、Meta-SSL split和协议字段
  model.py                     # CV-SincNet/CVS主干
  model_dual_cvsincnet.py      # z_id/z_dom双分支相关模型
  sat_channel.py               # 简化LEO残余信道和satellite stress
  cvsrffi/                     # Stage2、prototype、adapter、安全评估和协议工具
  SSDG/                        # source-domain DG训练入口
  federated/                   # 联邦/receiver粒度训练扩展
  tests/                       # 核心模块smoke与协议测试
paper_reproduction/
  protonet_cda/                # ProtoNet-CDA论文复现基线
  feature_separation_crossrx/  # Feature Separation跨接收机基线
  cvs_aligned/                 # CVS Stage2-B/C协议扩展评估层
  configs/                     # 脱敏示例配置
baselines/
  common/、cvcnn_ce/、drift/、riei_fd/、ra_collab/、paper_resnet/
tests/
  test_paper_reproduction_*.py
  test_spaceborne_fewshot.py
docs/
  PROJECT_PROTOCOL.md
  GROUND_TRAINING.md
  DEPLOYMENT_PHASES.md
  PUBLISH_SCOPE.md
experiment_records/
  CV-SincNet/                 # 十二小时快照、最近实验指标清单和有界报告证据
```

## 环境

本地开发环境使用Python和PyTorch。原工作区使用Conda环境`ssr-gpu`，但仓库不依赖该环境名。安装PyTorch时请按本机CUDA版本选择官方wheel。

```powershell
conda create -n cvs-rffi python=3.10
conda activate cvs-rffi
pip install -r requirements.txt
```

运行测试时设置仓库根和`code/`进入`PYTHONPATH`：

```powershell
$env:PYTHONPATH="$PWD;$PWD\code"
python -m pytest tests\test_paper_reproduction_cvs_aligned.py code\tests\test_meta_ssl_cli_defaults.py -q
```

## 最小命令

集中式训练协议检查：

```powershell
python code\train.py --help
python code\train.py --dataset wisig --wisig_pkl <path-to-ManySig-or-WiSig-pkl> --train_mode centralized --wisig_train_ratio 0.1 --epochs 1 --output_dir runs\centralized_smoke --test_eval_policy interval_final
```

元学习版协议检查：

```powershell
python code\train.py --dataset wisig --wisig_pkl <path-to-ManySig-or-WiSig-pkl> --use_meta_ssl_cvs --meta_ssl_protocol_check_only --use_meta_rxday_episodes --wisig_train_ratio 0.1
```

论文复现和CVS扩展测试：

```powershell
python -m pytest tests\test_paper_reproduction_protonet.py tests\test_paper_reproduction_feature_separation.py tests\test_paper_reproduction_cvs_aligned.py -q
```

这些命令用于验证入口、协议字段和小规模smoke，不代表完整论文复现实验或部署成功。

论文baseline审计实验入口：

```powershell
bash run_cvs_baseline_queue.sh --methods riei_fd --wisig-protocol riei_original --dry-run
bash run_cvs_baseline_queue.sh --methods drift --wisig-protocol drift_day1 --dry-run
```

完整运行前需提供真实`Dataset_WigSig/ManySig.pkl`路径并移除`--dry-run`。RIEI original使用`riei_last10`统计，DRIFT-Day1使用`drift_last5`统计；详细命令、超参数对照和声明边界见`baselines/README.md`。

## 关键声明

- `rho_label<=0.1`是地面弱标注训练的核心约束。
- 地面训练支持集中式训练版和元学习版，但两者都必须保持source-only。
- `R_t`必须与`R_s`不相交。
- `Y_old`、`Y_new`和`Y_unknown`必须互斥。
- Stage2-A/B不能声明seen-new identity accuracy。
- Stage2-C只有在目标域同时存在target-old和target-new support/query时才成立。
- clean view只能作为control/reference，不能单独作为deployment success。
- 本仓库不含真实数据、权重或远端运行证据；任何结果声明必须绑定具体run、split、K-shot、satellite/LEO view和完整同row指标。

更多协议细节见`docs/PROJECT_PROTOCOL.md`、`docs/GROUND_TRAINING.md`和`docs/DEPLOYMENT_PHASES.md`。

## 十二小时自动整理

本仓库包含自动整理脚本`scripts/run_cvs_snapshot_cycle.ps1`。该脚本从`E:\type10-7`同步核心代码、协议、工具、launcher和最近实验指标，生成`experiment_records/CV-SincNet/LATEST_SNAPSHOT.md`、`metrics_inventory.csv`和`docs/analysis_requests/latest_chatgpt_pro_prompt.md`，然后提交并推送到GitHub。详细边界见`docs/AUTOMATION_GITHUB_REVIEW.md`。

## 变更纪律

项目相关改动必须进入Git流程。改动前检查`git status -sb`，改动后检查`git diff`/`git status -sb`并运行必要验证；完成后提交到明确分支，除非用户明确要求不要提交。每次代码、配置、脚本或协议变更都要同步检查`AGENTS.md`、`docs/PROJECT_PROTOCOL.md`、README、docs和报告类Markdown是否需要更新。
