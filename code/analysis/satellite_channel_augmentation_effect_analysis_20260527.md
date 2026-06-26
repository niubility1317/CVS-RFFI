# 星地信道增强效果不明显的证据链分析

- Timestamp: 2026-05-27 17:05 Asia/Hong_Kong
- Operator/agent: Codex
- Scope: CVS-RFFI / federated CVS-RFFI satellite-ground channel augmentation under `E:\type10-7\code`
- Main evidence:
  - Code paths: `sat_channel.py`, `baseline_origin_sat_view.py`, `concat_sat_channel_aug.py`, `train.py`, `federated/fed_trainer.py`
  - Completed report: `automation_reports/CV-SincNet/cvs_backbone_stability_ceonly_central_20260527_120817`
  - Live full-log parse through remote time 2026-05-27 16:57 CST:
    - `logs/cvs_backbone_dsq_followup_central_20260527_150427`
    - `logs/cvs_sat_leo_only_diag_central_20260527_153452`
    - `logs/cvs_sat_leo_multival_central_20260527_154407`
  - Federated config/report checks:
    - `automation_reports/CV-SincNet/20260527_104701_fl82_stylebank_targetrx_zdom/report.md`
    - `automation_reports/CV-SincNet/20260527_111318_fl82_stylebank_methods_ablation/report.md`

## 结论先行

星地信道增强不是没有注入，也不是代码完全失效。它主要表现为一个弱到中等强度的鲁棒正则项：能推动某些卫星场景指标上升，也能在配合 domain-DSQ 后帮助 clean/UDU 路线，但目前没有形成“干净泛化”和“星地鲁棒”同时显著上升的稳定机制。

核心原因是：当前增强只把卫星信道作为 label-preserving view 喂给 TX CE 或完整本地目标，它并没有强制模型学到 clean-sat 不变的 transmitter feature；更糟的是早期 full-concat/baseline_view 会把卫星样本带入 domain/adv/cons/Fishr/groupCE 等完整 DG 损失，用原始 receiver/day 域标签标记一个已经被卫星信道改变过的样本，导致域损失语义被污染。

## 实现语义

`sat_channel.py` 的信道模拟是有实际扰动的：它包含自由空间路径损耗、elevation/slant range、LOS/LOO/Rayleigh 状态、大气衰落、多普勒/CFO、相位噪声、AGC、AWGN、IQ imbalance，可选 multipath。这里的关键不是“没变换”，而是幅度项会被 mild AGC 和常见 IQ 归一化管线削弱，真正留下来的是频偏、相位、噪声、IQ 失衡、多径等 nuisance 变化。

`BaselineOriginSatViewAugment.expand()` 是严格的 clean+sat 2B 拼接：`x` 和卫星视图拼接，`y` 复制，`d_raw` 也复制。这适合朴素监督分类，但放进 CVS-RFFI 的 full DG 目标后，会让卫星视图同时参加域分类、GRL、consistency、Fishr 等损失。

中央训练现在有两条路径：

- legacy concat/full-objective：`--use_concat_sat_channel_aug` 但不加 `--concat_sat_ce_only`，卫星样本进入主 CVS 损失栈。
- CE-only concat：`--use_concat_sat_channel_aug --concat_sat_ce_only`，干净样本走完整 CVS，卫星视图单独 forward，只加 `concat_sat_ce_weight * TX CE`。

联邦训练也有类似分叉。`fed_trainer.py` 中 `fl_sat_aug_mode=baseline_view` 且 `fl_baseline_view_ce_only=false` 时仍是 full 2B 本地目标；只有显式打开 `fl_baseline_view_ce_only` 时，卫星视图才只做 baseline-view CE。当前看到的 FL82 targetrx/zdom 日志/命令没有 `fl_baseline_view_ce_only`，`[FED-CONFIG-SAT]` 也只显示 `use_sat=1 mode=baseline_view train=clear_leo lambda_cls=0.0 lambda_cons=0.0`，因此这类联邦结果不能等同于后来的中央 CE-only 改良路径。

## 定量证据

### 完成的中央 CE-only + backbone 稳定性实验

完成集 `cvs_backbone_stability_ceonly_central_20260527_120817` 的结论很清楚：

| branch | best primary | strict UDU | final-primary SAT avg/min | 解释 |
| --- | ---: | ---: | ---: | --- |
| `SA11` anchor | 82.97 | 80.73 | 43.49 / 38.62 | CE-only 星地增强 anchor |
| `SA16` domain DSQ | 84.45 | 82.78 | 43.66 / 39.56 | clean/UDU 最强，卫星只小幅改善 |
| `SA14` ID phase+DSQ | 82.58 | 79.80 | 47.17 / 40.99 | 卫星更强，但 clean/UDU 下滑 |
| `SA17` all phase+DSQ | 82.42 | 79.80 | 46.55 / 40.67 | 同样是鲁棒专用分支，不适合作默认主线 |

这说明增强“有效”的方向和主指标方向分离：domain DSQ 提升 clean/UDU；ID phase/DSQ 更像卫星鲁棒路线，但会吃掉 primary/strict UDU。

### 进行中的 DSQ follow-up，截至 2026-05-27 16:57 CST

这些日志还没跑完，以下是 full-log parse through latest complete epochs：

| branch | latest parsed | best primary | strict UDU at best primary | latest/curve-best SAT avg | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| `SA18` domain DSQ ch2 | E109/170 | 85.39 | 83.86 | 41.93 / 43.57 | clean 更强，卫星没有同步升高 |
| `SA20` domain phase+DSQ | E110/170 | 83.70 | 81.78 | 44.39 / 44.85 | 卫星稍强，primary 低于 SA16 |
| `SA23` DSQ CE weight 1.5 | E102/170 | 83.24 | 81.30 | 43.01 / 45.13 | 加大卫星 CE 有鲁棒收益，但伤主指标 |
| `SA24` ID phase+DSQ CE 0.7 | E130/170 | 83.00 | 80.51 | 45.35 / 46.29 | 仍是鲁棒路线，主指标不够 |

这进一步支持“不是没注入，而是目标冲突”：提高卫星压力或引入相位/频谱稳定性可抬 satellite avg，但 primary/strict UDU 往往掉；降低压力或 domain-only DSQ 可保主指标，但卫星增益很有限。

### LEO-only 分支，截至 2026-05-27 16:57 CST

这些分支用于纠正“baseline 只用 LEO”这一对比口径。当前也没有看到显著早期胜出：

| branch | latest parsed | best primary | strict UDU | latest/curve-best LEO SAT avg | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| `SA26` LEO3 CE1.0 | E076/170 | 82.23 | 80.53 | 42.01 / 43.97 | LEO-only 没自动解决问题 |
| `SA27` ch2 LEO3 CE1.0 | E072/170 | 83.60 | 82.15 | 44.12 / 44.99 | 比 SA26 好，但尚未超过 SA16 LEO 子集 |
| `SA29` LEO3 CE0.7 | E066/170 | 84.28 | 82.59 | 44.40 / 44.97 | 当前最平衡，但 LEO avg 仍低于 SA16 LEO 子集 45.64 |
| `SA30` clear-only train, LEO3 eval | E066/170 | 80.62 | 78.92 | 42.43 / 43.78 | 只训 clear_leo 不泛化到低仰角/雨衰，且主指标弱 |

所以“只把场景改成 LEO”不是充分条件。即使对齐 baseline 的 LEO 口径，当前 CE-only 星地视图仍更像 regularizer，而不是决定性鲁棒学习机制。

## 为什么效果不明显

1. 星地增强没有增加 transmitter ID 信息，只是在已有样本上加 nuisance 扰动。对 RFFI 来说，模型要抓的是发射机硬件指纹；卫星信道大多是传播链路、接收扰动、相位/频率噪声。它能逼模型别过拟合某些链路，但不能凭空补足跨 RX/day 的指纹可分性。

2. 当前信道模拟的幅度信息被有意边界化。`sat_channel.py` 用 mild AGC 把样本 RMS 拉回目标范围，和数据预处理里的归一化方向一致。因此路径损耗这种“很强的卫星物理信号”不会直接变成可学习的大幅度差异，剩下的是更难学、更容易和硬件指纹纠缠的相位/频偏/噪声结构。

3. legacy full-concat 路径会污染 DG 损失语义。2B 拼接把卫星样本复制同一个 `d_raw`，但这个样本已经多了 satellite style。domain classifier、GRL、Fishr、group CE 等看到的是“同域标签下混入卫星风格”，会把传播扰动当作 receiver/day 域内部噪声或错误域证据处理。这解释了为什么早期“严格拼接复刻 baseline”并没有自然带来 baseline 那样的卫星优势。

4. CE-only 修正了损失污染，但信号太单薄。CE-only 只告诉模型“卫星视图也要分对 TX”，没有显式约束 clean 和 satellite 的 `z_id` 要贴近，也没有对不同 satellite scenarios 做对比/不变性建模。于是它更像额外带噪监督样本；在 0.1 train ratio 下，能帮一点，但很难把 strict satellite UDU 从 40 多直接拉到 60。

5. 主指标和卫星鲁棒性存在真实 tradeoff。已完成和 live 分支都显示：`SA16/SA18/SA29` 这类路线保 primary/strict UDU；`SA14/SA17/SA24` 或更强 satellite CE 会抬 SAT avg，但 primary/strict UDU 下滑。也就是说，当前表征容量/损失配方下，模型在“接收机/日期泛化”和“卫星信道鲁棒”之间还没找到共享表征。

6. 场景覆盖不是越多越好，也不是只训 clear_leo 就好。all-five rotation 会稀释每个场景的训练密度；LEO3 比 all-five 更公平，但当前没有明显突破；clear-only 对 low_elev/rain 泛化不足。训练场景需要 schedule 或 curriculum，而不是简单扩大/缩小集合。

7. 联邦下更容易被稀释。每个 client 是 receiver，本地 batch 本来就缺少 centralized BEX02 的同 step 多域对照；星地 baseline_view 如果还是 full objective，会进一步把卫星风格混入本地 DG/Fishr/GRL 估计。当前 FL82 clean strict 还在 60-70 区间时，卫星 strict 到 60 更难，先要把联邦 clean 主线稳住。

## 目前该怎么判断“有没有用”

应避免二分说“有用/没用”。更准确的判断是：

- 作为单独增强，它对 satellite robustness 有弱到中等帮助，但不足以成为主解决方案。
- 作为 full-concat 输入完整 CVS/DG 损失，它很可能是有害或低效的。
- 作为 CE-only view，它更干净，但主要提供额外监督，不提供强不变性。
- 配合 domain DSQ，它能保住 clean/UDU；配合 ID phase/DSQ，它能抬卫星鲁棒；但两条路线尚未统一。

## 建议的下一步

1. 不再把“星地信道增强”单独当作核心创新点。把它定义为受控辅助 view，主线放在表征不变性和 receiver/day DG 上。

2. 对中央训练保留两条候选：
   - primary 路线：`SA18` 或 `SA29`，看完整跑完后是否稳定优于 `SA16`。
   - robustness 路线：`SA24` / `SA14` 类，作为卫星鲁棒专用分支，不直接替换主线。

3. 增加轻量 clean-sat alignment，而不是回到 full-concat。优先试 stop-gradient `z_id(clean)` vs `z_id(sat)` 的 late-ramp consistency 或 supervised contrastive：从 epoch 60/90 后小权重启动，避免早期破坏 clean feature。

4. 做 satellite CE schedule，而不是固定 epoch-1 全强度。候选：`0.3 -> 0.8` late ramp、`1.2 -> 0.7` anneal、或只在 validation plateau 后加权。

5. 联邦实验如果要继承中央 CE-only 结论，必须显式打开 `--fl_baseline_view_ce_only` 并记录 `[FED-CONFIG-SAT]` 中的 ce-only/weight 字段；否则仍是在测 legacy full-objective baseline_view。

6. 报告里必须分开三类指标：clean strict UDU、LEO-only SAT、all-scenario SAT。不要把 LEO-only baseline 与 all-five mixed/storm 结果混在一个平均数里比较。

