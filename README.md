# CVS-RFFI / CV-SincNet

CVS面向天基射频指纹识别中的弱标注跨接收机域泛化与在轨跨域少样本适应。项目采用`地面训练、天上部署`架构：地面端学习跨接收机稳定的发射机身份表征，部署端在目标卫星接收机域内用叠加简化LEO星地信道的少量旧类样本和新类样本完成目标域适应、旧类校准和新类学习。open-set/unknown拒识现在是Phase3备用项，不是Phase2主线。

本仓库是从本地实验工作区整理出的干净发布包。它包含源码、协议文档、论文复现扩展和小规模测试，不包含WiSig/ManySig数据、模型权重、N607私有自动化配置、远端日志、实验报告归档、PPT或第三方论文PDF。

## 当前协议版本

- 协议日期：2026-07-07
- 科研主线：weak-label/semi-supervised source-domain DG + spaceborne few-shot adaptation
- 地面训练：集中式训练版和元学习版并列
- 在轨部署：Phase2主线为Stage2-A/B/C目标域适应与新类学习；Phase3为open-set备用项
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
| 在轨部署 | Stage2-A | `code/cvsrffi/spaceborne_fewshot.py` | 零目标标签下旧类识别和target-new未注册参考 |
| 在轨部署 | Stage2-B | `code/cvsrffi/spaceborne_fewshot.py` | 叠加LEO目标域旧类`K`shot适应和校准，不声明seen-new identity accuracy |
| 在轨部署 | Stage2-C | `paper_reproduction/cvs_aligned/`、`code/cvsrffi/spaceborne_fewshot.py` | Phase2主线：叠加LEO目标域旧类校准+seen-new enrollment |
| 在轨部署 | Phase3 | `code/cvsrffi/spaceborne_fewshot.py` | 备用项：open-set/unknown rejection安全扩展 |

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
  receiver_agnostic_twostage_uda/ # Bao et al. GLOBECOM 2023 DANN+LMMD闭集UDA复现
  mitigating_receiver_impact_da/  # Liu Yang et al. IoTJ 2024 GAD/DV-KL/CPL复现
  cvs_aligned/                 # CVS Stage2-B/C协议扩展评估层
  configs/                     # 脱敏示例配置
baselines/
  common/、cvcnn_ce/、drift/、riei_fd/、ra_collab/
tests/
  test_paper_reproduction_*.py
  test_spaceborne_fewshot.py
docs/
  PROJECT_PROTOCOL.md
  GROUND_TRAINING.md
  DEPLOYMENT_PHASES.md
  PUBLISH_SCOPE.md
  RELEASE_SNAPSHOT.md         # CVS-only发布快照摘要
  release_manifest_latest.json # 发布文件清单
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

CVS扩展测试：

```powershell
python -m pytest tests\test_paper_reproduction_cvs_aligned.py -q
```

这些命令用于验证入口、协议字段和小规模smoke，不代表完整论文复现实验或部署成功。

论文baseline审计实验入口：

```powershell
bash scripts/launchers/run_cvs_baseline_queue.sh --methods riei_fd --wisig-protocol riei_original --dry-run
bash scripts/launchers/run_cvs_baseline_queue.sh --methods drift --wisig-protocol drift_day1 --dry-run
```

完整运行前需提供真实`Dataset_WigSig/ManySig.pkl`路径并移除`--dry-run`。RIEI original使用`riei_last10`统计，DRIFT-Day1使用`drift_last5`统计；详细命令、超参数对照和声明边界见`baselines/README.md`。

## 关键声明

- `rho_label<=0.1`是地面弱标注训练的核心约束。
- 地面训练支持集中式训练版和元学习版，但两者都必须保持source-only。
- `R_t`必须与`R_s`不相交。
- `Y_old`、`Y_new`和`Y_unknown`必须互斥。
- Stage2-A/B不能声明seen-new identity accuracy。
- Stage2-C只有在目标域同时存在target-old和target-new support/query时才成立。
- open-set/unknown FAR属于Phase3备用项，不能作为Phase2主线成功。
- clean view只能作为control/reference，不能单独作为deployment success。
- 本仓库不含真实数据、权重或远端运行证据；任何结果声明必须绑定具体run、split、K-shot、satellite/LEO view和完整同row指标。

更多协议细节见`docs/PROJECT_PROTOCOL.md`、`docs/GROUND_TRAINING.md`和`docs/DEPLOYMENT_PHASES.md`。

## CVS-only自动整理

本仓库包含自动整理脚本`scripts/run_cvs_snapshot_cycle.ps1`。该脚本从`E:\type10-7`同步CVS相关核心代码、协议、工具、launcher、测试和必要发布说明，生成`docs/RELEASE_SNAPSHOT.md`和`docs/release_manifest_latest.json`，然后按参数提交并推送到GitHub。脚本不会上传`experiment_records/`、本地工作区笔记、AI审查提示/输出或baseline历史运行产物。详细边界见`docs/AUTOMATION_GITHUB_REVIEW.md`和`docs/PUBLISH_SCOPE.md`。

## 变更纪律

项目相关改动必须进入Git流程。改动前检查`git status -sb`，改动后检查`git diff`/`git status -sb`并运行必要验证；完成后提交到明确分支，除非用户明确要求不要提交。每次代码、配置、脚本或协议变更都要同步检查`AGENTS.md`、`docs/PROJECT_PROTOCOL.md`、README、docs和报告类Markdown是否需要更新。

协作输出规则：DO NOT send optional commentary。
