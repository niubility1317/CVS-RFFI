# phase2_adv3b02_goal_feasibility_20260704

## 基本信息

| 项目 | 内容 |
|---|---|
| 实验ID | phase2_adv3b02_goal_feasibility_20260704 |
| 时间 | 2026-07-04 |
| 操作方 | Codex |
| 目标 | 在ADV3B02+qknn8现有evidence字段上，诊断是否存在任意风险字段组合可同时满足old/seen-new/unknown目标 |
| 底座模型 | `ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 输入evidence | `E:\type10-7\remote_artifacts\phase2_adv3b02_collab_open_set_qknn_full_20260703\collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv` |
| 结论边界 | DIAGNOSTIC_ONLY；扫描使用query标签做oracle上限估计，不能作为部署阈值选择 |

## 本地改动

| 文件 | 目的 |
|---|---|
| `E:\type10-7\code\scripts\phase2_evidence_field_separability_diag.py` | 增加目标可行性扫描、known floor约束扫描和最接近目标缺口输出 |
| `E:\type10-7\code\tests\test_phase2_evidence_field_separability_diag.py` | 增加目标可行性和known floor回归测试 |

## 本地验证

| 命令 | 结果 |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_evidence_field_separability_diag.py code\tests\test_phase2_evidence_field_separability_diag.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_evidence_field_separability_diag.py -q` | PASS，2 passed |

## 本地诊断命令

三字段全组合在本地超过180秒，因此本轮记录二字段组合诊断。该范围与既有separability报告中的主结论一致，足以判断现有字段门控是否接近目标。

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_evidence_field_separability_diag.py --evidence_csv E:\type10-7\remote_artifacts\phase2_adv3b02_collab_open_set_qknn_full_20260703\collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv --output_json E:\type10-7\local_artifacts\phase2_adv3b02_goal_feasibility_20260704\goal_feasibility_event098_combo2.json --max_combo_size 2 --modes max,mean,min --far_targets 0.01,0.05,0.10 --known_floor_targets 0.80,0.90,0.95,0.99 --goal_old_acc 0.99 --goal_min_old_class_acc 0.95 --goal_seen_new_acc 0.97 --goal_min_seen_new_class_acc 0.93 --goal_unknown_reject_rate 0.99 --max_thresholds 160
```

## 本地结果

### 目标可行性

| 约束 | 值 |
|---|---:|
| old_acc | 0.99 |
| min_old_class_acc | 0.95 |
| seen_new_acc | 0.97 |
| min_seen_new_class_acc | 0.93 |
| unknown_reject_rate | 0.99 |
| feasible | False |

最接近目标的二字段候选：

| 字段 | mode | threshold | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | defer_rate | total_deficit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mahalanobis_risk,oldness_risk` | mean | 0.608329 | 0.6650 | 0.3300 | 0.7450 | 0.7200 | 0.4950 | 0.5050 | 0.8875 | 0.0900 | 1.8750 |

### known floor约束

| floor | 是否存在同时满足old/seen-new及类floor的门控 | 结论 |
|---:|---|---|
| 0.80 | 否 | 没有OLD80+class floor可行门控 |
| 0.90 | 否 | 没有OLD90可行门控 |
| 0.95 | 否 | 没有OLD95可行门控 |
| 0.99 | 否 | 没有目标级可行门控 |

### FAR约束下最佳门控

| unknown_FAR约束 | 字段 | mode | threshold | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | known_coverage | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ≤0.01 | `margin_risk,oldness_risk` | max | 2.2106e-05 | 0.1083 | 0.0000 | 0.2150 | 0.2100 | 0.9900 | 0.1550 | unknown达标但known崩 |
| ≤0.05 | `margin_risk,evt_risk` | mean | 3.0546e-09 | 0.3250 | 0.0000 | 0.2850 | 0.1500 | 0.9500 | 0.3413 | unknown接近但known崩 |
| ≤0.10 | `margin_risk,evt_risk` | mean | 1.6402e-06 | 0.3683 | 0.1100 | 0.3250 | 0.1900 | 0.9000 | 0.3913 | 仍远低目标 |

## 解释

该诊断比单次融合负结果更强：即使允许oracle使用query标签扫描二字段风险组合，现有ADV3B02+qknn8 evidence字段也不存在满足OLD80类floor的门控，更不存在满足old99/min95、seen-new97/min93、unknown99的门控。当前瓶颈不是阈值选择，而是特征/风险字段没有形成可部署的known/unknown边界。

下一步应转向训练或轻量适配机制：

1. ground/source阶段引入virtual future class或proxy non-old open-set margin，使`z_id`在LEO扰动下保留unknown外环；
2. target support阶段训练小adapter/cache时加入class-conditional compactness和source/proxy negative separation；
3. 保留qknn8协同推理作为低带宽evidence层，但不要期待现有字段门控直接达到目标。

## N607计划

远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`  
远端环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`  
远端输出：`runs/phase2_adv3b02_goal_feasibility_20260704/`

待同步文件：

| 本地 | 远端 |
|---|---|
| `code\scripts\phase2_evidence_field_separability_diag.py` | `code/scripts/phase2_evidence_field_separability_diag.py` |
| `code\tests\test_phase2_evidence_field_separability_diag.py` | `code/tests/test_phase2_evidence_field_separability_diag.py` |

待运行远端命令：`py_compile`、目标unittest、二字段目标可行性扫描。运行前执行N607 preflight，运行后拉回结果并检查SSH清理。

## N607验证结果

| 项目 | 结果 |
|---|---|
| preflight | PASS；直连`N607`，项目根目录可见 |
| 远端环境 | `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` |
| 远端工作目录 | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端输出 | `runs/phase2_adv3b02_goal_feasibility_20260704/goal_feasibility_event098_combo2.json` |
| 拉回结果 | `E:\type10-7\local_artifacts\phase2_adv3b02_goal_feasibility_20260704\remote\goal_feasibility_event098_combo2.json` |
| GPU选择 | GPU0；运行后`nvidia-smi`显示8张GPU均为10MiB |
| 远端测试 | `py_compile` PASS；`code/tests/test_phase2_evidence_field_separability_diag.py` 2 tests OK |
| SSH清理 | SCP和SSH后本地`ssh.exe`为空；到`172.31.111.215:22`和`172.31.105.18:22`的ESTABLISHED连接为空 |

远端执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
OUT=runs/phase2_adv3b02_goal_feasibility_20260704
FEATURE=runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=0 $PY -m py_compile code/scripts/phase2_evidence_field_separability_diag.py code/tests/test_phase2_evidence_field_separability_diag.py
PYTHONPATH=code:code/scripts CUDA_VISIBLE_DEVICES=0 $PY code/tests/test_phase2_evidence_field_separability_diag.py
CUDA_VISIBLE_DEVICES=0 $PY code/scripts/phase2_evidence_field_separability_diag.py --evidence_csv $FEATURE --output_json $OUT/goal_feasibility_event098_combo2.json --max_combo_size 2 --modes max,mean,min --far_targets 0.01,0.05,0.10 --known_floor_targets 0.80,0.90,0.95,0.99 --goal_old_acc 0.99 --goal_min_old_class_acc 0.95 --goal_seen_new_acc 0.97 --goal_min_seen_new_class_acc 0.93 --goal_unknown_reject_rate 0.99 --max_thresholds 160
```

远端结果与本地一致：

| 字段 | mode | threshold | old_acc | min_old | seen_new_acc | min_seen | unknown_reject | unknown_FAR | known_coverage | defer_rate | total_deficit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mahalanobis_risk,oldness_risk` | mean | 0.608329 | 0.6650 | 0.3300 | 0.7450 | 0.7200 | 0.4950 | 0.5050 | 0.8875 | 0.0900 | 1.8750 |

## 算法结论边界

当前证据不支持继续把主要路线放在阈值微调、投票或简单qknn融合上。oracle二字段门控失败说明现有ADV3B02+qknn8 evidence字段没有形成同时保护old、seen-new和unknown reject的可分结构。

下一轮应改为“未知优先、旧类保持约束内生化”的训练/适配路线：冻结`z_id`主干或只启用轻量adapter，加入source/proxy unknown的energy/EVT/margin约束；target support阶段做旧类源原型收缩、新类K-shot注册、多接收机可靠度加权证据融合；任何在线适配如果降低old留出support准确率或class-wise floor，必须回滚。`Y_unknown`只能用于query评估，不能用于阈值拟合。

## 下一轮算法方案：SCORPION-CVS

SCORPION-CVS的目标不是再做一次投票，而是在星上部署约束内把unknown reject、old retention和seen-new enrollment写进同一个决策流程。建议保留`CV-SincNet/z_id`主干冻结，只更新每个未见接收机的轻量校准、原型和阈值统计。

### 模块

| 模块 | 作用 | 部署边界 |
|---|---|---|
| 接收机轻量校准 | 学习对角或低秩仿射`z'_r=A_r z+b_r`，抵消未见接收机偏移 | 只用target support，不能用unknown query |
| 旧类源原型收缩 | 旧类原型锚定source prototype，避免K-shot覆盖旧类结构 | 旧类准确率下降触发回滚 |
| 新类K-shot注册 | 对`Y_new`support建立新类原型和类内尺度 | 不混入`Y_unknown` |
| unknown-first门控 | 融合Mahalanobis、energy、margin、KNN密度和EVT尾概率 | 先拒识再分类 |
| 多接收机证据融合 | 只交换top-M类别证据、unknown log-odds和接收机可靠度 | 不上传原始IQ |
| old retention shield | 旧类候选需通过源头分类器和目标原型一致性检查 | 不允许用unknown FAR改善换old崩溃 |

### 核心公式

嵌入和旧类收缩：

```text
z=f_theta(x), z'_r=A_r z+b_r
alpha_c=n_c/(n_c+lambda_old)
p'_{r,c}=alpha_c mean(S_{r,c})+(1-alpha_c)p^0_c, c in Y_old
p'_{r,c}=mean(S_{r,c}), c in Y_new
```

接收机本地类别分数和unknown风险：

```text
D_{r,c}(x)=(z'_r-p'_{r,c})^T Sigma^{-1}_{r,c}(z'_r-p'_{r,c})
s_{r,c}(x)=-D_{r,c}(x)/tau_r+beta_{r,c}
E_r(x)=-tau_r log sum_c exp(s_{r,c}(x)/tau_r)
g_r(x)=w_1 min_c D_{r,c}+w_2 E_r-w_3 margin_r-w_4 log rho_knn+w_5 EVT_tail
```

unknown-first接受条件：

```text
accept_r(x)=1[min_c D_{r,c}<=theta_D and g_r(x)<=theta_G and margin_r>=theta_M]
```

多接收机融合：

```text
L_c(x)=sum_{r in O(x)} q_r log P_r(c|x)
L_U(x)=sum_{r in O(x)} q_r logit P_r(unknown|x)
if L_U>=Theta_U or max_c L_c<Theta_C: reject/defer
else y_hat=argmax_c L_c
```

旧类保持约束：

```text
Acc_old_after >= Acc_old_before - epsilon
min_class_old_after >= min_class_old_before - epsilon_cls
```

### 资源估计

| 项目 | 估计 |
|---|---:|
| 单接收机证据上报 | top-5类别ID、分数、unknown log-odds、可靠度，约80-160B/样本 |
| 可选embedding上报 | 128维INT8约128B/样本 |
| 原型存储 | `(C_old+C_new)*d*4B`；100类、128维约50KB |
| 对角协方差/尺度 | 与原型同量级 |
| 单样本计算 | `O((C_old+C_new)*d)`；100类、128维约1.3万乘加 |
| 在线更新 | prototype/temperature/bias/threshold为主；默认禁止full fine-tuning |

### 最低验证矩阵

| 验证项 | 必须报告 |
|---|---|
| qknn8 baseline对照 | 同一`R_t`、同一`Y_old/Y_new/Y_unknown`、同一K-shot、同一LEO view |
| unknown门控 | unknown reject、unknown FAR、score distribution和EVT尾部拟合质量 |
| old保持 | old_acc、min_old_class_acc、accepted old accuracy、coverage、回滚次数 |
| seen-new注册 | seen_new_acc、min_seen_new_class_acc、K-shot敏感性 |
| 协同数量 | `k=1..N_receiver`；strict event不足时必须标注为receiver-domain ensemble诊断 |
| 资源 | bytes/event、latency、prototype storage、GPU/CPU memory |

### 方法依据

该设计组合了四类已验证方向：energy OOD detection用于替代softmax过度自信问题；prototype+EVT用于open-set SEI尾部拒识；open-world RFFI的增强半监督思想用于保护已知类；source-free cross-receiver RFFI用于未见接收机适应。对应参考包括Energy-based OOD Detection（NeurIPS 2020，https://arxiv.org/abs/2010.03759）、Open-Set Specific Emitter Identification Based on Prototypical Networks and Extreme Value Theory（Applied Sciences 2023，https://www.mdpi.com/2076-3417/13/6/3878）、Open-world Radio Frequency Fingerprint Identification via Augmented Semi-supervised Learning（AAAI 2025，https://ojs.aaai.org/index.php/AAAI/article/view/32003）和Cross-Receiver Radio Frequency Fingerprint Identification: A Source-Free Adaptation Approach（Sensors 2025，https://www.mdpi.com/1424-8220/25/14/4451）。
