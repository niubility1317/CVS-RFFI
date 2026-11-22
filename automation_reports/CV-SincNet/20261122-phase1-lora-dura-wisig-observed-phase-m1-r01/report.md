# LoRa DURA 观察相位对照

运行名 `20261122-phase1-lora-dura-wisig-observed-phase-m1-r01`。父对照为同一物理数据、同一源动作缓存和同模型种子的 [原坐标八阶段版本](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/report.md)。原坐标版本粗恢复的源校验显示准确性严重下降，同时四个 practical 场景的输出能量降至配对 clean 的约一半或以下。单纯整体相位校正对识别的源反事实只有 +1.53 个百分点。这次检查的假设是训练不应主要消耗容量去拟合与实际观测不一致的绝对相位，且保留已验证的原网络结构和阶段损失。

## 唯一训练变量

在合法源 L/U IQ 的每个已知配对上，以内积求单位复数 `p = exp(i·angle(sum(y·conj(x))))`。训练目标使用 `p*x`，输入始终是未加工的同一 `y`。幅值与 chirp 形状保持原量。完全相同的 clean 配对及近零相关情形逐点保留原参考。作用监督同步将主系数乘共轭 `p`、镜像系数乘 `p`，因此算子作用后的无噪观测相同。噪声方差、FIR、采样顺序、优化器、损失系数、教师训练和各阶段 4,940 步不变。全新训练，不加载其他训练态。只读复用 `20261122-phase1-lora-dura-wisig-lineage-m1-r02` 已完成的 LoRa L/U 作用代理缓存。

## 正式对照和运行

- 种子 2026092702，N607 普通账户，程序只用现有 CUDA 环境和独立 release。训练阶段只读取源 IQ，不读 F、TX 真值或目标。
- 八阶段从粗恢复到 R69 共 39,520 步。与原坐标实验使用同一冻结 Tri-set，15 台未见接收机、30,000 个物理包、clean 和六景 practical。原版完整输出与同输出加既有盲门控各一条，加八条冻结对照，共 70 个评分单元。与原坐标版的同读出结果在各自全部预测冻结后匹配比较，报告净纠错、clean 和每景/每 RX。
- 路由可用 GPU 0 至 7，按当时进程和显存状态分散，保留所有已有健康进程。同一时刻仅一个本运行 GPU worker。技术停止规则为源数据、动作来源、冻结权重或固定对照不符，真实源 smoke 失败，训练非有限或预测矩阵缺项。
- 发布入口为 `python -m experiments.pamf_lora_main.full_dura_lineage_publish ... --reuse-actions ORIGINAL_SUBMIT`。固定根下取 `smoke.json`、训练日志、每阶段权重、源 V 波形诊断和完整盲预测。由本地 `full_dura_score` 等待并独立评分。首轮用于单种子验证，最终限差仍需三种子确认。

[已冻结的粗恢复归因](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/source-stage-analysis.md)

发布 **VERIFIED**。代码提交 `e0f430b587859dd48a69abcebf4466bde7a4184d`，发布归档 SHA 比对和远端编译通过。真实源 smoke 的动作梯度范数 8.0118。独立复核证明 2,048 条真实 clean chirp 和恒等系数逐字节不变，六景无噪声动作重放最大误差为 `5.44e-7`。N607 控制器、GPU 3 训练进程、工作目录和源作用缓存实际链接已校验。新 run 完成粗恢复首轮 247 步。原坐标 run 在 GPU 0 继续训练作用估计。两套诊断分别等待冻结源权重，两套正式评分分别等待七景预测冻结。

[发布记录](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_release/submit.json)

## 冻结粗恢复的源域分解

使用同一固定源 V 面板和同种子冻结 Tri-set。观察相位版全量粗恢复在六景的正确数由原坐标版的 186/720 增至 238/720，仍低于直通 395/720。城区低强度的输出能量恢复到参考的 97.62%（原坐标版 38.84%），正确数为 32/120（原坐标版 26/120，直通 44/120）。观察相位版 clean 的功率比为 0.988，波形 NMSE 约 0.00339，正确数只有 44/120（原坐标版 57/120，直通 94/120）。

在接收 chirp 的观测相位与频率坐标中，将实际冻结修正正交分解为沿载波方向和垂直于载波的波形变化。保留同一批输入，每行只改变加入的修正项，冻结 G 与 F 都不更新。

| 读出 | Clean 正确数（共 120） | Practical 六景正确数（共 720） |
|---|---:|---:|
| 直通 | 94 | 395 |
| 粗恢复全量 | 44 | 238 |
| 粗恢复半量 | 64 | 335 |
| 仅载波修正 | 94 | 399 |
| 仅非载波修正 | 44 | 234 |
| 半量非载波修正 | 64 | 330 |
| 四分之一非载波修正 | 75 | 372 |

Clean 中非载波修正占修正总能量的约 91%。仅用载波修正时，clean 正确数保持，六景净多 4 次；仅用非载波修正时，几乎复现全量的识别损伤。单纯减小修正幅度能够缓解，但四分之一非载波仍损失 19 个 clean 与 23 个 practical 正确判定。该源域证据把当前粗恢复的主要识别副作用定位到非载波波形，尚需同批冻结作用和 R17 验证展开更新能否恢复这些细节。

[逐景读出和修正能量](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/coarse_component_readout.json) · [粗恢复与直通的完整源对照](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/coarse_frozen_tri_set.json)

## 冻结作用估计的配对归因

观察相位版作用阶段完成 20 轮、4,940 步。继续使用原版六步软求解和相同的冻结 Tri-set。两个作用估计器的 full 探针比较使用各自训练时的参考坐标，均在同一个实际带噪观测和同一虚拟探针轨迹下评分。

| 场景 | 直通正确数 | 原坐标作用后 | 观察相位作用后 | Full 学习算子 NMSE 原坐标 → 观察相位 |
|---|---:|---:|---:|---:|
| Clean | 94 | 80 | 68 | — |
| 郊区高 | 85 | 73 | 65 | 1.519 → 0.010 |
| 郊区中 | 77 | 66 | 65 | 1.630 → 0.005 |
| 郊区低 | 56 | 49 | 59 | 1.686 → 0.011 |
| 城区高 | 75 | 60 | 65 | 1.487 → 0.007 |
| 城区中 | 58 | 43 | 54 | 1.424 → 0.058 |
| 城区低 | 44 | 31 | 45 | 1.312 → 0.050 |
| 六景总计 | 395 | 322 | 353 | — |

观察相位版六景相对原坐标版净多 31 次，仍比直通少 42 次。弱城区输出能量从原坐标作用后 0.535 到观察相位作用后 0.994，识别从 31/120 到 45/120。郊区高与 clean 的正确数仍比原坐标版低。Full 弱城区对两个学习算子都额外校正整段复增益后，NMSE 从原坐标版 0.2272 到观察相位版 0.0107。改进同时涉及整体相位和其余动作误差。用相位一致的 residual 代理再次复算，弱城区探针误差从 0.5949 到 0.0384，方差平均绝对误差从 0.727 到 0.380 dB。

首次 residual 代理统计对观察相位模型误用了原坐标参考。已在诊断中同步旋转参考与系数后重新计算。训练和冻结预测均按预登记运行，只有源域比较统计需要重算。

[作用后的同批源识别](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/action_frozen_tri_set.json) · [Full 探针复算](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/full_action_probe_gauge.json) · [修正参考后的代理探针](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/action_gauge_corrected.json)
