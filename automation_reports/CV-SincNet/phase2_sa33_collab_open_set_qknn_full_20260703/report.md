# SA33协同open-set qknn8全链路测试报告

## 基本信息

- 实验ID：`phase2_sa33_collab_open_set_qknn_full_20260703`
- 时间：2026-07-03
- 操作：Codex
- 目标：基于用户指定权重`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`，实现并测试多接收机协同open-set qknn8证据生成与融合；协同接收机数量覆盖`1..N`。
- 远程环境：用户明确指定N607使用`CVS-RFFI` Conda环境；本地验证仍按项目规则使用`ssr-gpu`。

## 算法设计

- 名称：`COSR-CI/qknn8`
- 主干：冻结SA33/CV-SincNet特征提取器，只在目标接收机侧维护qknn8量化support code。
- 本地接收机证据：`predicted_label`、`known_score`、`known_margin`、`unknown_risk`、`latency_ms`、`bytes`。
- 协同融合：中心端只融合低带宽证据包，不跨接收机合并support特征。
- 阈值边界：unknown query只做最终评估；阈值来源记录为`support_known_only`。
- 部署边界：支持新增seen-new support时追加int8 code和label，不需要全模型重训；本轮没有做full-model在线微调。

## 协议与数据

- checkpoint：
  - `runs/cvs_sa27_optimization_central_20260527_204005/SA33_sa27_ch2_leo3_ce0p7_r010/best_strict_udu_model.pth`
  - SHA256：`0efd38621eda7bc18adec89827f366132977ce6e183a45e0d5cf16c22d80592d`
- source receivers：`0,1,2,3,4,5,6`，对应ManySig前7个接收机。
- target receivers：`20-1,3-19,7-14,7-7,8-8`，共5个，和source receiver集合不重叠。
- old TX：`14-10,14-7,20-15,20-19,6-15,8-20`。
- seen-new TX：`1-16,1-18`。
- unknown TX：`10-1,10-10`。
- 星地信道：`simplified_leo_residual`，场景`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

## 文件变更

| 文件 | 用途 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | open-set协同证据融合评估；补齐per-K分母、metadata类别floor、完整混淆分桶 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | 从Stage2-C feature NPZ生成qknn8证据并调用融合评估 |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 融合评估回归测试 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | qknn8证据生成器测试 |

Git镜像提交：

- `02b5499 Add collaborative open-set qknn evidence builder`
- `0fd8f10 Harden collaborative open-set qknn denominators`
- `44d8bec Mark receiver-domain qknn collaboration diagnostics`

说明：`github_publish/CVS-RFFI-repo`仍领先远端，且存在非本轮未跟踪文件`code/scripts/phase2_qknn_active_support_select.py`，本轮未处理。

## 验证与同步

本地验证：

```text
conda activate ssr-gpu
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`4 tests OK`、`8 tests OK`。

远程同步：

| 本地文件 | N607目标 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/evaluation/collaborative_open_set_qknn_eval.py` |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_collaborative_open_set_qknn_eval.py` |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_collaborative_open_set_qknn_eval.py` |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_collaborative_open_set_qknn_eval.py` |

远程测试：

```text
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_collaborative_open_set_qknn_eval.py
```

结果：`4 tests OK`、`8 tests OK`。测试后GPU为`10/24576MiB`，无残留SSH连接。

## 远程命令

特征导出使用GPU0，启动前所有GPU均为`10/24576MiB`：

```text
CUDA_VISIBLE_DEVICES=0 python -u code/export_spaceborne_features.py \
  --ckpt runs/cvs_sa27_optimization_central_20260527_204005/SA33_sa27_ch2_leo3_ce0p7_r010/best_strict_udu_model.pth \
  --wisig_pkl Dataset_WigSig/ManySig.pkl \
  --new_wisig_pkl Dataset_WigSig/ManyTx.pkl \
  --out_npz runs/phase2_sa33_collab_open_set_qknn_full_20260703/features.npz \
  --feature_name z_id \
  --source_tx_ids 0,1,2,3,4,5 \
  --source_rxs 0,1,2,3,4,5,6 \
  --target_old_tx_ids 0,1,2,3,4,5 \
  --target_old_rxs 20-1,3-19,7-14,7-7,8-8 \
  --target_old_channel_view satellite \
  --new_tx_ids 1-16,1-18 \
  --new_rxs 20-1,3-19,7-14,7-7,8-8 \
  --unknown_tx_ids 10-1,10-10 \
  --target_new_channel_view satellite \
  --star_ground_channel_impl simplified_leo_residual \
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --target_new_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
  --max_samples_per_tx 400 \
  --batch_size 512 \
  --device cuda:0
```

严格同事件协同评估首先执行，结果被阻断：

```text
RuntimeError: NO_ALIGNED_COLLABORATIVE_EVENTS
```

原因：WiSig导出的多接收机target query没有共享`role+tx+day+sig+scenario`同事件键。为避免伪造同事件协同，严格模式拒绝rank-aligned伪协同。

随后执行显式诊断模式：

```text
python -u code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_sa33_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_sa33_collab_open_set_qknn_full_20260703/collab_open_set_qknn_ranked.json \
  --output_evidence_csv runs/phase2_sa33_collab_open_set_qknn_full_20260703/collab_open_set_qknn_ranked_evidence.csv \
  --collab_counts all \
  --k_shot 8 \
  --query_per_class 20 \
  --qknn_k 8 \
  --event_alignment_policy receiver_domain_ranked
```

## 结果

该表为`receiver_domain_ranked`诊断，不是严格同事件同步协同。

| 协同接收机数 | old_acc | seen_new_acc | unknown_reject_rate | unknown_FAR | known_coverage | defer_rate | bytes/event | p95 latency ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4464 | 0.4500 | 0.1500 | 0.8500 | 1.0000 | 0.0000 | 40 | 0.0618 |
| 2 | 0.4107 | 0.3500 | 0.1000 | 0.8000 | 0.9342 | 0.0521 | 80 | 0.0618 |
| 3 | 0.3750 | 0.5000 | 0.1500 | 0.7500 | 0.8553 | 0.0729 | 120 | 0.0618 |
| 4 | 0.4643 | 0.5000 | 0.0000 | 1.0000 | 0.9868 | 0.0000 | 160 | 0.0618 |
| 5 | 0.4643 | 0.4000 | 0.0000 | 0.9500 | 0.9868 | 0.0104 | 200 | 0.0618 |

完整结果：

- 本地：`automation_reports/CV-SincNet/phase2_sa33_collab_open_set_qknn_full_20260703/artifacts/collab_open_set_qknn_ranked.json`
- 远程：`runs/phase2_sa33_collab_open_set_qknn_full_20260703/collab_open_set_qknn_ranked.json`

## 结论

- 模块实现完成并通过本地/远程单元测试。
- N607已同步并用`CVS-RFFI`环境验证。
- SA33权重、星地信道、5个未见target receivers、协同数量`1..5`均已覆盖。
- 严格同事件协同因数据集缺少共享事件键被阻断，这是正确行为。
- 可运行结果为`receiver_domain_ranked`诊断，性能未达目标：unknown FAR过高，old/seen-new per-class floor为0，不能作为部署成功证据。

## 2026-07-03子agent监督签核

该签核仅覆盖SA33任务，不替代ADV3B02相邻报告。监督结论：SA33链路已完成模块实现、N607同步、`CVS-RFFI`环境验证、星地信道评估和`1..5`协同数量诊断；但严格同事件卫星群协同未成立，当前只能标为`receiver_domain_ranked`诊断负例。

|检查项|签核|证据/边界|
|---|---|---|
|指定权重|通过|使用`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`，checkpoint路径和SHA256已记录。|
|协同数量`1..N`|通过但降级|结果覆盖5个target receivers的`1..5`；因缺少共享事件键，只能解释为receiver-domain ranked诊断。|
|N607同步|通过|4个代码/测试文件已按本地优先流程scp到N607目标路径。|
|远端Conda|通过|远端测试使用`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate CVS-RFFI`。|
|星地信道|通过|导出使用`simplified_leo_residual`和`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。|
|低显存GPU测试|通过|启动前和测试后GPU记录为`10/24576MiB`级别；本轮未留下训练进程。|
|报告持久化|通过|本报告已记录实验ID、命令、路径、指标和诊断结论。|
|Git状态|部分通过|代码进入Git镜像提交；根目录不是Git仓库，镜像领先远端，未追踪文件不属于本轮，不能声明全项目Git闭环完成。|
|本地优先|通过|本地`ssr-gpu`验证后再同步N607。|
|SSH断开|通过|远端任务后检查无残留SSH连接。|
|性能声明|通过但负结论|unknown FAR高，old/seen-new floor为0；不得写作Stage2-C成功、部署成功或论文主结论。|
|下一步缺口|未完成|需要真实共享`event_id`或共享`role+tx+day+sig+scenario`的多接收机query，严格模式才可形成同事件协同证据。|

## 2026-07-03当前代码复跑

在SCORER-CVS组件隔离修复同步后，使用当前N607代码复跑SA33的`receiver_domain_ranked`全量评估。命令仍在`CVS-RFFI`环境中执行，输入为已有星地信道特征：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python code/scripts/phase2_collaborative_open_set_qknn_eval.py \
  --feature_npz runs/phase2_sa33_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_sa33_collab_open_set_qknn_full_20260703/collab_open_set_qknn_ranked_current.json \
  --output_evidence_csv runs/phase2_sa33_collab_open_set_qknn_full_20260703/collab_open_set_qknn_ranked_current_evidence.csv \
  --collab_counts all \
  --k_shot 8 \
  --query_per_class 20 \
  --qknn_k 8 \
  --event_alignment_policy receiver_domain_ranked \
  --seed 407040
```

运行前后8张RTX3090均为`10/24576MiB`，没有新增GPU显存占用。运行输出：`receiver_count=5`、`group_count=310`、`evidence_row_count=1000`。远端产物已拉回：

- `automation_reports/CV-SincNet/phase2_sa33_collab_open_set_qknn_full_20260703/artifacts/collab_open_set_qknn_ranked_current.json`
- `automation_reports/CV-SincNet/phase2_sa33_collab_open_set_qknn_full_20260703/artifacts/collab_open_set_qknn_ranked_current_evidence.csv`

当前复跑结果，`active_risk_components=["score"]`：

|协同receiver数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_FAR|unknown_reject_rate|defer_rate|known_coverage|bytes/event|p95 latency ms|缺失seen-new类|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.2211|0.0000|0.0333|0.0000|0.2167|0.5167|0.2968|0.2600|40|0.0796|无|
|2|0.0946|0.0000|0.0000|0.0000|0.0227|0.3636|0.4625|0.0765|80|0.0796|无|
|3|0.0833|0.0000|0.0000|0.0000|0.0000|0.2500|0.6350|0.0625|120|0.0796|无|
|4|0.2065|0.0000|0.0000|0.0000|0.0556|0.2222|0.5375|0.1532|160|0.0796|无|
|5|0.1600|0.0000|0.0000|0.0000|0.0500|0.0500|0.6333|0.1143|200|0.0796|`1-18`|

判定：当前代码复跑后，SA33在该Stage2-C qknn8诊断下性能仍显著低于可部署目标，尤其old/seen-new per-class floor为0。该结果只能作为`receiver_domain_ranked`负例和后续算法设计输入，不能写作严格同事件协同、Stage2-C成功或部署成功。

## 下一步

1. 若要真实卫星群同事件协同，需要导出或构造带共享事件ID的多接收机query。
2. qknn8 unknown gate需要改进，当前support-only阈值明显无法控制unknown FAR。
3. 后续可引入Gaussian prototype/Mahalanobis或EVT support-tail阈值，但阈值拟合仍不能使用unknown query。
