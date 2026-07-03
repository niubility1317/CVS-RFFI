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

## 2026-07-03SCORER-CVS增强版SA33复跑

监督子agent指出：上一轮原型/Mahalanobis/per-label阈值增强主要闭环在`ADV3B02_CORE90_SOFT_E200`特征上，不能替代用户指定的`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`权重结果。因此本节用同一SA33星地特征复跑当前增强版协同推理。

本地修正：

|文件|修正|
|---|---|
|`code/scripts/phase2_collaborative_open_set_qknn_eval.py`|新增`_qknn_label_score_matrix()`；per-label阈值校准改为使用`score_y(x)`的同类分数，而不是support LOO时top1预测类分数；evidence新增`score_threshold_source`和`base_receiver_score_threshold`。|
|`code/tests/test_phase2_collaborative_open_set_qknn_eval.py`|新增反例测试：当support LOO样本被错分到`new-a`时，`old-a`类阈值不得吸收错误top1高分。|

本地与Git镜像验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：`test_phase2_collaborative_open_set_qknn_eval.py`为32 tests OK，`test_collaborative_open_set_qknn_eval.py`为27 tests OK。Git镜像同样复测通过。

N607远端执行：先运行`tools\n607_ssh_preflight.ps1`，直连`N607`通过，项目根为`/home/szu2070436088/2510044040/CV-SincNet`；8张RTX3090均为`10/24576MiB`。同步脚本和测试文件后，在远端执行：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate CVS-RFFI
python -m py_compile code/scripts/phase2_collaborative_open_set_qknn_eval.py code/evaluation/collaborative_open_set_qknn_eval.py
python code/tests/test_phase2_collaborative_open_set_qknn_eval.py
python code/tests/test_collaborative_open_set_qknn_eval.py
```

结果：远端`CVS-RFFI`环境确认通过，单测为32 tests OK和27 tests OK。随后在`CUDA_VISIBLE_DEVICES=0`上复跑4组`receiver_domain_ranked`诊断，均使用`--collab_counts all`，覆盖5个target receiver的1到5协同数量；source receiver数量为7，仅作为训练/源域统计，不作为本评估的协同参与数。运行前后GPU均为`10/24576MiB`，SSH/SCP后本地检查无残留`ssh.exe`或22端口`ESTABLISHED`连接。

统一参数：`--k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 --support_calibration_mode leave_one_out --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only --scenario_aware --radius_norm 0.3 --fusion_policy scorer_cvs --collaboration_policy adaptive_gain --label_fusion_policy vote_margin --receiver_reliability_policy deployment_prior --event_alignment_policy receiver_domain_ranked --support_selection_policy stable_first`。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_proto2_maha1_qknnonly.json`|`638522140BD0AB56BA58257A77EDCF1BBAAF8DA6A70355325161204471457E36`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_proto2_maha1_qknnonly_evidence.csv`|`90E3CCFB368C06B42E9C90402CF49B8527F6FAEC0E43A60BD0396B7C7874BDE1`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq20_proto2_maha1_qknnonly.json`|`1B72C44E990EECBDCD391BB890B430CC1FA64ADC38FE768DDCDF1B1AE53854DC`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq20_proto2_maha1_qknnonly_evidence.csv`|`9BEF2B27B441BE82CE755F36B33460362338291388C0AD3401DB8A1F53382BBC`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq35_proto2_maha1_qknnonly.json`|`C3419244C3AD924E542760FCD030641FEB92B2D9E698D95C88D7B3BBA2CD1200`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq35_proto2_maha1_qknnonly_evidence.csv`|`9442E590C1E6C1978D4024EB1F991AC8157594F0F4D83AF6A3E173F9F943A2D1`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq50_proto2_maha1_qknnonly.json`|`AA2DFED23824454BA62DB5EFB6DCDB743494148132145429891486655FE0895F`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_classq50_proto2_maha1_qknnonly_evidence.csv`|`3F86385FCC58B02B9F674889E220EC4BD246E42284A2D82A40B3A8ADCA3C561C`|

SA33增强版结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`sa33_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.2500|0.6656|1.0000|120.0|
|`sa33_proto2_maha1_qknnonly`|2|0.1824|0.0000|0.4490|0.3793|0.1915|0.4894|0.3934|1.7746|213.0|
|`sa33_proto2_maha1_qknnonly`|3|0.2083|0.0500|0.5500|0.5000|0.1000|0.5750|0.4750|2.5650|307.8|
|`sa33_proto2_maha1_qknnonly`|4|0.4891|0.0000|0.7419|0.7273|0.1515|0.5152|0.1923|3.2692|392.3|
|`sa33_proto2_maha1_qknnonly`|5|0.3846|0.0000|0.7000|0.0000|0.1000|0.5500|0.2826|3.6413|437.0|
|`sa33_classq20_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.2500|0.6656|1.0000|120.0|
|`sa33_classq20_proto2_maha1_qknnonly`|2|0.1757|0.0000|0.4286|0.3448|0.1702|0.4894|0.3975|1.7746|213.0|
|`sa33_classq20_proto2_maha1_qknnonly`|3|0.1917|0.0000|0.5500|0.5000|0.1000|0.5750|0.4750|2.5450|305.4|
|`sa33_classq20_proto2_maha1_qknnonly`|4|0.4783|0.0000|0.7419|0.7273|0.1515|0.6061|0.1731|3.1859|382.3|
|`sa33_classq20_proto2_maha1_qknnonly`|5|0.4231|0.0000|0.7000|0.0000|0.1000|0.7500|0.2065|3.4348|412.2|
|`sa33_classq35_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.2667|0.6558|1.0000|120.0|
|`sa33_classq35_proto2_maha1_qknnonly`|2|0.1554|0.0000|0.4286|0.3448|0.1702|0.5106|0.3852|1.7705|212.5|
|`sa33_classq35_proto2_maha1_qknnonly`|3|0.1667|0.0000|0.5500|0.5000|0.1000|0.6250|0.4800|2.5500|306.0|
|`sa33_classq35_proto2_maha1_qknnonly`|4|0.4565|0.0000|0.7419|0.7273|0.1515|0.6364|0.1987|3.2179|386.2|
|`sa33_classq35_proto2_maha1_qknnonly`|5|0.4423|0.0000|0.7000|0.0000|0.1000|0.8000|0.2065|3.5000|420.0|
|`sa33_classq50_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.2833|0.6526|1.0000|120.0|
|`sa33_classq50_proto2_maha1_qknnonly`|2|0.1149|0.0000|0.4286|0.3448|0.1702|0.5532|0.3934|1.7623|211.5|
|`sa33_classq50_proto2_maha1_qknnonly`|3|0.1167|0.0000|0.5500|0.5000|0.1000|0.6500|0.4700|2.5300|303.6|
|`sa33_classq50_proto2_maha1_qknnonly`|4|0.4130|0.0000|0.7097|0.7000|0.1212|0.6667|0.2372|3.2115|385.4|
|`sa33_classq50_proto2_maha1_qknnonly`|5|0.3462|0.0000|0.7000|0.0000|0.1000|0.8500|0.2826|3.6087|433.0|

解释：在SA33指定权重下，增强版最高同row seen-new为`sa33_proto2_maha1_qknnonly`预算4，`seen_new_acc=0.7419`、`min_seen_new=0.7273`，但`unknown_FAR=0.1515`不安全；FAR最低仍为0.1000，不能作为开集部署结果。per-label阈值修正后能提高unknown reject，但没有把FAR降到可部署区间，并且`min_old`多数为0。结论仍是诊断负结果：当前SA33+receiver-domain ensemble不满足99/97/99目标，不满足严格同事件卫星群协同证明。

文献/方法子agent与联网检索给出的可落地路线：

|方向|依据|迁移到CVS-RFFI的实现建议|
|---|---|---|
|星上分布式AI约束|[On-Orbit Space AI](https://arxiv.org/html/2604.16518v1)强调星座AI需要处理动态接触图、异构算力、非IID漂移和安全审计。|当前`receiver_domain_ranked`应升级为显式contact graph和receiver health加权，不再把接收机数简单当静态ensemble。|
|通信高效协同推理|[Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks](https://arxiv.org/html/2604.04654v1)使用模型切分、流水和压缩优化LEO协同推理延迟/通信。|CVS-RFFI不需要传activation；更适合传64-128 bytes证据包，但应把`request_more_receivers()`建成早停/增量路由。|
|分布式早退|[DistrEE](https://arxiv.org/abs/2502.15735)把多节点协同和early-exit结合，在延迟和精度间自适应取舍。|按`unknown_risk`、margin、class p-value先用1-2台receiver判定，只有低置信事件才请求更多receiver。|
|邻域保形校准|[Neighborhood Conformal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25936/25708)使用embedding近邻校准样本生成自适应预测集。|把per-label阈值升级为class-conditional conformal p-value：每个receiver输出`p_y`和`p_unknown`，跨receiver做log-opinion pooling。|
|低通信微调|[Federated LoRA with Sparse Communication](https://arxiv.org/html/2406.05233v1)说明LoRA+稀疏通信适合异构低资源联邦微调。|星上实时微调不要更新主干；只同步低秩adapter、class prototype、阈值摘要和少量hard negative统计。|
|卫星FL异步/部分更新|[FedLEO引用页](https://arxiv.org/html/2411.00263v1)强调LEO场景难以完美同步，需支持partial updates。|训练/微调调度应允许异步receiver adapter更新，报告中分开写“推理协同”和“微调聚合”。|
|持续学习稳定性|[DOLFIN](https://arxiv.org/html/2510.13567v1)用LoRA和梯度投影记忆平衡联邦增量学习的稳定/可塑性。|对新接收机只训练小adapter，并用source/old prototype子空间约束避免旧TX遗忘。|

下一版算法建议命名为`SCORER-CVS-CPR`：Support-Calibrated Open-set Receiver Evidence Routing with Conformal Prototype Routing。节点本地保留冻结SA33/ADV特征器、int8 support memory、EMA prototype、Mahalanobis逆方差、class conformal校准缓存和低秩adapter；每个事件先由单receiver输出`top2 label、score_y、p_y、p_unknown、radius_z、support_density、receiver_health`，控制器按风险请求更多receiver，并用校准后的log evidence融合。训练侧只更新adapter/prototype/threshold，不回传原始IQ或query unknown，满足低显存和低通信部署边界。

查漏补缺结论：per-label阈值原先存在“真实标签阈值吸收错误top1分数”的语义漏洞，已修复并补测试；但小K时per-label阈值仍应做receiver-global收缩，且当前结果仍不是严格同物理事件协同。若用户要求“全体源接收机1..7”字面评估，需要单独定义源receiver作为协同节点的协议；本报告完成的是CVS Stage2-C target receiver 1..5协同诊断。

## 2026-07-03virtual unknown独立风险组件SA33复跑

目标：把support-derived virtual unknown从“全局抬阈值”改为独立unknown风险组件，避免上一轮virtual unknown校准强行压低known接受率。实现保持`--virtual_unknown_calibration_enabled`旧路径不变，新增`--virtual_unknown_risk_enabled`只把support原型间合成边界样本用于`virtual_unknown_risk`，不进入阈值拟合；metadata中`active_risk_components`加入`virtual_unknown`，融合器可显式投票。

本地与远端验证：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为34 tests OK和28 tests OK。N607直连预检通过，8张RTX3090均为`10/24576MiB`；同步4个代码/测试文件后，远端`CVS-RFFI`环境同样为34 tests OK和28 tests OK。三组SA33复跑均使用`CUDA_VISIBLE_DEVICES=0`，运行后GPU仍为`10/24576MiB`，SSH/SCP后本地核验无残留`ssh.exe`或22端口`ESTABLISHED`连接。

统一远端输入：`runs/phase2_sa33_collab_open_set_qknn_full_20260703/features.npz`，该特征来自用户指定权重`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`，包含`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`星地信道视图。统一参数沿用增强版：`--collab_counts all --k_shot 8 --query_per_class 20 --qknn_k 8 --candidate_class_top_m 2 --prototype_score_blend 2.0 --mahalanobis_score_blend 1.0 --support_calibration_mode leave_one_out --unknown_gate_mode support_envelope_evt --score_threshold_combine qknn_only --scenario_aware --radius_norm 0.3 --fusion_policy scorer_cvs --collaboration_policy adaptive_gain --label_fusion_policy vote_margin --receiver_reliability_policy deployment_prior --unknown_risk_threshold 0.995 --scorer_component_vote_threshold 0.25 --event_alignment_policy receiver_domain_ranked`。

拉回产物SHA256：

|产物|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk2_proto2_maha1_qknnonly.json`|`63EC1E39E97757BAD3188842A55806DD1E10EF9424351FD4B84F511C82AE19D4`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk2_proto2_maha1_qknnonly_evidence.csv`|`C78161E0A1B9E261804C87DC33CD33E3591FD75FB6D616DF83944D02938B19D7`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk4_proto2_maha1_qknnonly.json`|`80124F15DC6095F5303E78592958CC937BE15BAF551C2F751B7BC682BC500F2F`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk4_proto2_maha1_qknnonly_evidence.csv`|`EE9B65B268665B2D6068180F22A64178A9D9FD9C49EE8E4029066527795ADE99`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk4_margin03_proto2_maha1_qknnonly.json`|`FD5A8A6A3E12275693B1582C118F39424C0D97700953BC7DE6C9E907FD33D94C`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_vrisk4_margin03_proto2_maha1_qknnonly_evidence.csv`|`64E91D40A51EC5A6DA8BE652CB879A6306F336F4F9C29DD4C0C3438A6DE16456`|

结果表：

|候选|预算|old_acc|min_old|seen_new_acc|min_seen_new|unknown_FAR|unknown_reject|defer_rate|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`sa33_vrisk2_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.1667|0.6883|1.0000|120.0|
|`sa33_vrisk2_proto2_maha1_qknnonly`|2|0.0338|0.0000|0.3673|0.2759|0.0638|0.5319|0.5984|1.9590|235.1|
|`sa33_vrisk2_proto2_maha1_qknnonly`|3|0.0417|0.0000|0.2750|0.2500|0.0250|0.6250|0.6800|2.8750|345.0|
|`sa33_vrisk2_proto2_maha1_qknnonly`|4|0.1196|0.0000|0.5161|0.5000|0.0606|0.6667|0.6090|3.7564|450.8|
|`sa33_vrisk2_proto2_maha1_qknnonly`|5|0.0385|0.0000|0.4000|0.0000|0.0500|0.7000|0.6957|4.8152|577.8|
|`sa33_vrisk4_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.1667|0.6851|1.0000|120.0|
|`sa33_vrisk4_proto2_maha1_qknnonly`|2|0.0338|0.0000|0.3673|0.2759|0.0638|0.5319|0.5943|1.9590|235.1|
|`sa33_vrisk4_proto2_maha1_qknnonly`|3|0.0417|0.0000|0.3250|0.3000|0.0250|0.6250|0.6650|2.8750|345.0|
|`sa33_vrisk4_proto2_maha1_qknnonly`|4|0.1196|0.0000|0.5161|0.5000|0.0606|0.6667|0.6090|3.7564|450.8|
|`sa33_vrisk4_proto2_maha1_qknnonly`|5|0.0385|0.0000|0.4000|0.0000|0.0500|0.7000|0.6957|4.8152|577.8|
|`sa33_vrisk4_margin03_proto2_maha1_qknnonly`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.1833|0.6818|1.0000|120.0|
|`sa33_vrisk4_margin03_proto2_maha1_qknnonly`|2|0.0338|0.0000|0.3673|0.2759|0.0426|0.5319|0.5902|1.9590|235.1|
|`sa33_vrisk4_margin03_proto2_maha1_qknnonly`|3|0.0417|0.0000|0.3000|0.2500|0.0250|0.6500|0.6550|2.8650|343.8|
|`sa33_vrisk4_margin03_proto2_maha1_qknnonly`|4|0.1196|0.0000|0.5806|0.5455|0.0606|0.6970|0.5962|3.7372|448.5|
|`sa33_vrisk4_margin03_proto2_maha1_qknnonly`|5|0.0385|0.0000|0.4000|0.0000|0.0500|0.7000|0.6957|4.7717|572.6|

解释：独立virtual unknown风险组件确实改善FAR。相对上一节`sa33_proto2_maha1_qknnonly`预算4的`unknown_FAR=0.1515`，本轮预算3可降至`0.0250`，预算5为`0.0500`；但old仍极低，`min_old=0.0000`，seen-new也明显低于上一节最高`0.7419`。最佳安全折中是`sa33_vrisk4_margin03_proto2_maha1_qknnonly`预算3：`unknown_FAR=0.0250`、`seen_new_acc=0.3000`、`old_acc=0.0417`，平均`2.8650`个接收机、`343.8 bytes/event`。该结果证明风险通道方向正确但接受逻辑过保守，仍不能满足old 99%/floor95%、seen-new 97%/floor93%、unknown拒识99%目标，也不能写作严格同事件卫星群协同或部署成功。

下一步算法应转向`SCORER-CVS-CPR`：每个receiver输出class-conditional conformal p-value、Mahalanobis tail、support density和receiver health；控制器先按低风险高margin早停，只有低置信事件请求更多receiver。virtual unknown应作为`p_unknown`的一部分，而不是直接把`unknown_risk`取max后压制全部known接受。

### review closeout

多子agent审查结论已处理：

|审查点|处理|
|---|---|
|`virtual_unknown`风险组件声明但融合未接线|已修复：`RISK_COMPONENT_KEYS`、`row_values`、聚合输出和adaptive gain均接入`virtual_unknown_risk`；新增`test_scorer_cvs_can_use_virtual_unknown_component`覆盖显式组件投票。|
|support-derived virtual unknown协议边界|报告中明确：该样本只由target support原型合成，不使用target unknown query，不代表真实unknown先验；当前仅为诊断/风险通道。|
|`collab_counts all`解释|保持解释为target receiver 1..5协同诊断；因`receiver_domain_ranked`不是严格同事件同分母曲线，不把k间差异写成纯协同数量因果。|
|延迟字段|保留`latency_ms_p50/p95`、`bytes/event`和`prototype_storage`，但延迟只作为离线proxy，不写成真实星上端到端延迟。|
|性能声明|保持负结论：FAR虽下降，但old/seen-new远低于目标，不是Stage2-C成功、严格同事件协同或部署成功。|

## 2026-07-03CPR review修复后SA33固定版复跑

背景：查漏补缺子agent指出原`conformal_rescue`存在P1风险，即unknown样本若被预测成old/seen-new且p-value高，可能在多风险通道同时高风险时仍被缩放为known接受。已修复为：只有`risk_component_agreement < scorer_component_vote_threshold`时CPR才允许降低`effective_unknown_risk`；多风险通道一致高风险时fail closed。另将`class_conformal_min_support`默认值从1提高到2，防止K=1/LOO退化。

本地、Git镜像和N607`CVS-RFFI`环境均通过：

```powershell
conda activate ssr-gpu
python -m py_compile code\scripts\phase2_collaborative_open_set_qknn_eval.py code\evaluation\collaborative_open_set_qknn_eval.py
python code\tests\test_phase2_collaborative_open_set_qknn_eval.py
python code\tests\test_collaborative_open_set_qknn_eval.py
```

结果：本地和Git镜像均为36 tests OK和30 tests OK；N607使用`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`同样为36 tests OK和30 tests OK。修复版SA33复跑使用已有`runs/phase2_sa33_collab_open_set_qknn_full_20260703/features.npz`，该特征来自用户指定权重`SA33_sa27_ch2_leo3_ce0p7_r010_20260527_204104`，覆盖5个target receiver、协同数量1到5、星地信道`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。运行后GPU读数为8张RTX3090均`10 MiB/24576 MiB`；SSH/SCP后本地检查无残留`ssh.exe`或N607 22端口连接。

SA33固定版CPR结果：

|候选|协同数|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|unknown_reject|defer|avg_rx|bytes/event|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.3000|0.6558|1.0000|120.0000|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|2|0.3716|0.1000|0.4490|0.3793|0.2340|0.5319|0.1557|1.7623|211.4754|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|3|0.4333|0.1500|0.5500|0.5000|0.2250|0.6250|0.2200|2.5350|304.2000|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|4|0.5870|0.0000|0.7419|0.7273|0.3030|0.6364|0.0641|3.1859|382.3077|
|`cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed`|5|0.4808|0.0000|0.7000|0.0000|0.1000|0.7500|0.1739|3.4783|417.3913|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.1667|0.6883|1.0000|120.0000|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|2|0.3851|0.1000|0.3673|0.2759|0.1915|0.5319|0.2131|1.9590|235.0820|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|3|0.3250|0.0500|0.2750|0.2500|0.0500|0.6250|0.4750|2.8750|345.0000|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|4|0.5543|0.0000|0.5161|0.5000|0.1515|0.6667|0.1859|3.7564|450.7692|
|`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`|5|0.4038|0.0000|0.4000|0.0000|0.0500|0.7000|0.4457|4.8152|577.8261|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|1|0.0000|0.0000|0.3833|0.2750|0.2333|0.2500|0.6656|1.0000|120.0000|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|2|0.3851|0.1000|0.4490|0.3793|0.2553|0.4894|0.1885|1.7746|212.9508|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|3|0.4417|0.1500|0.5500|0.5000|0.2250|0.5750|0.2650|2.5650|307.8000|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|4|0.5978|0.0000|0.7419|0.7273|0.3333|0.5152|0.0769|3.2692|392.3077|
|`cpr_p20_s05_proto2_maha1_qknnonly_fixed`|5|0.4808|0.0000|0.7000|0.0000|0.1000|0.5500|0.2174|3.6413|436.9565|

判定：SA33指定权重在修复版CPR下仍未达到目标。最佳FAR行是`cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed`协同3或5，`unknown_FAR=0.0500`，但协同3`old_acc=0.3250`、`seen_new_acc=0.2750`，协同5`old_acc=0.4038`、`seen_new_acc=0.4000`且`min_old=0`。最佳seen-new行协同4可达`seen_new_acc=0.7419`、`min_seen=0.7273`，但`unknown_FAR=0.3030/0.3333`且`min_old=0`。该结果只能作为`receiver_domain_ranked`负例，不能写作严格同事件卫星群协同、Stage2-C成功或部署成功。

固定版SHA256：

|文件|SHA256|
|---|---|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed.json`|`719FE2202CE7120BD34A0EE87F36DA2E217DE4972F0F31F6D1B6CB3974F341D6`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p15_s05_u098_proto2_maha1_qknnonly_fixed_evidence.csv`|`257962DA462E0D7F89EB9210BBDE3EEC44A779E7035BD3592C60640413325639`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed.json`|`750DCA7B8554B48CCB81ECB0AEE67583B864FB71F934BD31659AC55206C62190`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p20_s04_vrisk2_proto2_maha1_qknnonly_fixed_evidence.csv`|`D3DD9AA7FCE4C53BDD5A4B32C7F430B9A7390DFA6313F6EE04C7399D2F0A8A2D`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p20_s05_proto2_maha1_qknnonly_fixed.json`|`1A2F7C4435055458137DBF305647582514C4CA08686213ABFE44D15B90D84872`|
|`collab_open_set_qknn_scorer_cvs_evt_adaptive_gain_vote_margin_sa33_cpr_p20_s05_proto2_maha1_qknnonly_fixed_evidence.csv`|`0E957CF78F0A345D99CBE7DA9415646F79632CB9122D680C37C87DA4EADD0A18`|
