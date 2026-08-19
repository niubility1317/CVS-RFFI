# ADVB02 CRRA改造追踪记录

## Source

- 用户设计原文：`E:/codex/home/attachments/2cf19e77-82bf-42ec-9cbe-3e24d0198789/pasted-text.txt`
- 已确认CRRA完整设计：`E:/codex/home/attachments/bd0330d7-5397-4eeb-8979-915357e07237/pasted-text.txt`
- 当前科学/协议边界：`E:/type10-7/项目.md`
- 当前代码承载：`code/model.py`、`code/model_dual_cvsincnet.py`、`code/SSDG/train_ssdg.py`、`code/baseline_origin_sat_view.py`、`code/sat_channel.py`
- 历史星地信道：`mixed_orbit`；本次候选训练与测试：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`

## 历史实现映射

下表记录旧`mixed_orbit`CRRA原型的实现状态，不构成本次LEO弱信道候选已完成的证据。

| ID | Design requirement | Target | Status | Verification |
|---|---|---|---|---|
| CRRA-01 | 稳健层位于身份路径共享Sinc/IQ与高频特征之后 | `code/model.py`、`code/model_dual_cvsincnet.py` | verified | `python -m pytest code/tests/test_advb02_crra_model.py -q` |
| CRRA-02 | 选择性复数I/Q 2x2协方差收缩白化 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-03 | 有界残差门控，初始近似恒等映射 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-04 | rank=8低秩深度卷积残差、上投影零初始化 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-05 | 条件向量由RCN统计与GAP特征生成并stop-gradient | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-06 | 源域支持门和修正能量可观测 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-07 | 域分支读取原始共享特征，PA分支保留旁路 | `code/model.py`、`code/model_dual_cvsincnet.py` | verified | `python -m pytest code/tests/test_advb02_crra_model.py -q` |
| CRRA-08 | clean/satellite成对余弦与现有satellite KL一致性（可选，首轮关闭） | `code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py` | verified | `pytest -q code/tests/test_crra_training_plumbing.py`；独立KL激活回归通过，首轮launcher显式置零 |
| CRRA-09 | 最小干预能量、gate L1和同视图干扰回归 | `code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py` | verified | `pytest -q code/tests/test_crra_adapter.py code/tests/test_crra_training_plumbing.py`；真实`mixed_orbit`单批反向冒烟 |
| CRRA-10 | E1–16恒等、E17–46渐进、E47后固定 | `code/crra.py`、`code/SSDG/train_ssdg.py` | verified | `pytest -q code/tests/test_crra_adapter.py code/tests/test_crra_training_plumbing.py` |
| CRRA-11 | 信道元数据来自同一次`mixed_orbit`生成，不重新生成第二视图 | `code/baseline_origin_sat_view.py`、`code/cvsrffi/eval.py` | verified | `python -m pytest code/tests/test_crra_mixed_orbit_metadata.py code/tests/test_baseline_origin_sat_view.py code/tests/test_concat_sat_channel_aug.py -q` |
| CRRA-12 | Phase1禁止目标接收机访问和目标adapter-only校准 | `code/SSDG/train_ssdg.py`、`code/tests/test_crra_protocol_negatives.py` | verified | `pytest -q code/tests/test_crra_protocol_negatives.py code/tests/test_crra_training_plumbing.py`；真实checkpoint clean/`mixed_orbit`无query冒烟 |
| CRRA-13 | 旧checkpoint在CRRA关闭时兼容加载 | `code/post_stage_common.py`、`code/cvsrffi/checkpoint_loading.py` | verified | `pytest -q code/tests/test_exact_ssdg_checkpoint_loading.py`；`best_joint_safe_ssdg.pth`严格重建通过，CRRA结构补入前向通过 |
| CRRA-14 | `concat_masked`clean主分支加satellite监督CE/nuisance/shell | `code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py`、launcher | verified | 聚焦训练测试通过；launcher dry-run显示`concat_masked`、`B+B`、pair/KL=0、shell=0.15 |
| CRRA-15 | 当前Phase1四角色`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，V_select选模、V_cal校准 | `code/SSDG/train_ssdg.py`、`docs/PROJECT_PROTOCOL.md` | verified | 四角色不交测试通过；launcher显式传入当前角色协议 |
| CRRA-16 | 最小`mixed_orbit`同row实验和独立评分 | launch/report artifacts | planned | run report |

## Boundary Notes

- 本记录不改变`项目.md`中的Phase1数据协议，只增加方法实现与诊断字段。
- 当前设计实现不把Phase2目标域adapter能力宣称为Phase1结果；Phase1配置明确拒绝目标adapter开关。
- 历史远端实验与新CRRA实现分离；新实验必须使用新的run ID和不可覆盖输出根目录。

## 当前CRRA-S LEO弱信道映射

| ID | 已确认要求 | 实现目标 | 状态 | 验证 |
|---|---|---|---|---|
| CRRA-L01 | 训练和测试均仅使用三种`leo_weak`场景 | `crra_training.py`、LEO launcher | local-verified | `python -m pytest code/tests/test_crra_protocol_negatives.py code/tests/test_crra_mixed_orbit_metadata.py code/tests/test_phase1_advb02_crra_leo_weak_launcher.py -q` |
| CRRA-L02 | 复用Core90三段LEO日程和E200超参数 | LEO launcher | local-verified | `python -m pytest code/tests/test_phase1_advb02_crra_leo_weak_launcher.py -q`；完整训练命令经`build_arg_parser()`解析通过 |
| CRRA-L03 | 每对I/Q独立alpha与收缩白化 | `crra.py` | local-verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-L04 | FiLM条件rank8残差与零初始化上投影 | `crra.py` | local-verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-L05 | 源域多中心对角Mahalanobis支持门 | `crra.py` | local-verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-L06 | q条件时间/频率/PA可靠度融合，PA不重构 | `model.py`、`model_dual_cvsincnet.py` | local-verified | `python -m pytest code/tests/test_advb02_crra_model.py -q` |
| CRRA-L07 | 唯一`lambda_sat_cons=0.05`，禁止KL双计 | `train_ssdg.py` | local-verified | `python -m pytest code/tests/test_crra_training_plumbing.py -q` |
| CRRA-L08 | E1–16/E17–46/E47+与CRRA LR=0.25 | `crra.py`、`train_ssdg.py` | local-verified | `python -m pytest code/tests/test_crra_training_plumbing.py -q` |
| CRRA-L09 | 合法final checkpoint不得被诊断性P0/P1阻断后测 | `train_ssdg.py`、LEO launcher | local-verified | `python -m pytest code/tests/test_crra_training_plumbing.py code/tests/test_phase1_advb02_crra_leo_weak_launcher.py -q` |
| CRRA-L10 | clean和三种LEO的独立指标与CRRA遥测 | `crra_evaluation.py`、评估器 | local-verified | `python -m pytest code/tests/test_crra_evaluation.py code/tests/test_cvsrffi_sat_eval.py code/tests/test_phase1_advb02_crra_leo_weak_launcher.py -q`；独立重建会恢复checkpoint的CRRA训练轮次；N607同row指标待产生 |
| CRRA-L11 | Phase1不访问target、不启用CRRA-C | 训练配置与负测 | local-verified | `python -m pytest code/tests/test_crra_protocol_negatives.py -q` |

## Reverse Audit

以上状态只表示本地实现与聚焦测试已验证，并不表示性能提升已经得到证实。N607训练、clean和三种LEO逐场景独立评估完成后，才在本表和同row实验报告中记录结果与最终Git提交。
