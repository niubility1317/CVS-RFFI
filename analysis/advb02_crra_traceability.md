# ADVB02 CRRA改造追踪记录

## Source

- 用户设计原文：`E:/codex/home/attachments/2cf19e77-82bf-42ec-9cbe-3e24d0198789/pasted-text.txt`
- 当前科学/协议边界：`E:/type10-7/项目.md`
- 当前代码承载：`code/model.py`、`code/model_dual_cvsincnet.py`、`code/SSDG/train_ssdg.py`、`code/baseline_origin_sat_view.py`、`code/sat_channel.py`
- 当前历史星地信道：`mixed_orbit`

## Requirement Trace

| ID | Design requirement | Target | Status | Verification |
|---|---|---|---|---|
| CRRA-01 | 稳健层位于身份路径共享Sinc/IQ与高频特征之后 | `code/model.py`、`code/model_dual_cvsincnet.py` | verified | `python -m pytest code/tests/test_advb02_crra_model.py -q` |
| CRRA-02 | 选择性复数I/Q 2x2协方差收缩白化 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-03 | 有界残差门控，初始近似恒等映射 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-04 | rank=8低秩深度卷积残差、上投影零初始化 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-05 | 条件向量由RCN统计与GAP特征生成并stop-gradient | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-06 | 源域支持门和修正能量可观测 | `code/crra.py` | verified | `python -m pytest code/tests/test_crra_adapter.py -q` |
| CRRA-07 | 域分支读取原始共享特征，PA分支保留旁路 | `code/model.py`、`code/model_dual_cvsincnet.py` | verified | `python -m pytest code/tests/test_advb02_crra_model.py -q` |
| CRRA-08 | clean/satellite成对余弦与现有satellite KL一致性 | `code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py` | verified | `pytest -q code/tests/test_crra_training_plumbing.py`；真实`mixed_orbit`单批反向冒烟 |
| CRRA-09 | 最小干预能量、gate L1和同视图干扰回归 | `code/SSDG/train_ssdg.py`、`code/cvsrffi/losses.py` | verified | `pytest -q code/tests/test_crra_adapter.py code/tests/test_crra_training_plumbing.py`；真实`mixed_orbit`单批反向冒烟 |
| CRRA-10 | E1–16恒等、E17–46渐进、E47后固定 | `code/crra.py`、`code/SSDG/train_ssdg.py` | verified | `pytest -q code/tests/test_crra_adapter.py code/tests/test_crra_training_plumbing.py` |
| CRRA-11 | 信道元数据来自同一次`mixed_orbit`生成，不重新生成第二视图 | `code/baseline_origin_sat_view.py`、`code/cvsrffi/eval.py` | verified | `python -m pytest code/tests/test_crra_mixed_orbit_metadata.py code/tests/test_baseline_origin_sat_view.py code/tests/test_concat_sat_channel_aug.py -q` |
| CRRA-12 | Phase1禁止目标接收机访问和目标adapter-only校准 | `code/SSDG/train_ssdg.py`、`code/tests/test_crra_protocol_negatives.py` | verified | `pytest -q code/tests/test_crra_protocol_negatives.py code/tests/test_crra_training_plumbing.py`；真实checkpoint clean/`mixed_orbit`无query冒烟 |
| CRRA-13 | 旧checkpoint在CRRA关闭时兼容加载 | `code/post_stage_common.py`、`code/cvsrffi/checkpoint_loading.py` | verified | `pytest -q code/tests/test_exact_ssdg_checkpoint_loading.py`；`best_joint_safe_ssdg.pth`严格重建通过，CRRA结构补入前向通过 |
| CRRA-14 | 最小`mixed_orbit`同row实验和独立评分 | launch/report artifacts | planned | run report |

## Boundary Notes

- 本记录不改变`项目.md`中的Phase1数据协议，只增加方法实现与诊断字段。
- 当前设计实现不把Phase2目标域adapter能力宣称为Phase1结果；Phase1配置明确拒绝目标adapter开关。
- 历史远端实验与新CRRA实现分离；新实验必须使用新的run ID和不可覆盖输出根目录。

## Reverse Audit

实现完成后逐项把`Status`从`planned`改为`implemented`或`verified`，并在`Verification`中填写实际测试命令、同row实验报告路径和最终Git提交。若任何条目未实现，不得将CRRA标记为完成。
