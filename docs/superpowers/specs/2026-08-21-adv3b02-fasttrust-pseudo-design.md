# ADV3B02 FastTrust伪标签与高效训练设计

日期：2026-08-21

## 1.目标与边界

本设计修订MUSE-SSDG的U侧身份学习，同时保持当前Phase1数据协议和ADV3B02 CORE90星地增强口径不变。目标有两个：减少错误伪标签的确认偏差；在不降低每epoch U_s覆盖率的条件下缩短训练时间。

本轮不使用既有target结果选择结构或超参数。所有候选使用同一source划分、seed和训练代码，只允许预登记的单因素差异。最终target clean与三种LEO弱信道结果只用于冻结候选的一次性诊断，不反馈重训。

## 2.FastTrust伪标签路由

E1–E16为严格S1。训练代码不得建立U侧身份loss图，不执行global/local/prototype融合，不更新temporal memory或U prototype，并记录：

```text
pseudo_enabled=0
u_identity_selected_count=0
u_identity_loss=0
u_identity_gradient_norm=0
u_satellite_identity_selected_count=0
```

E17以后，EMA teacher提供global概率，local head和classification prototype分别提供第二、第三路证据。三路证据经现有prior alignment后计算reliability。唯一身份伪标签仅授予：

```text
U_H = high reliability ∩ temporal stable ∩ three-head agreement ∩ class-balanced cap
```

`U_H`使用hard CE。`U_M`只使用融合soft target或candidate-set loss；`U_L`不产生TX身份梯度。默认每batch的hard上限为25%，全部身份loss样本上限为50%，每个预测类使用同一可靠度排序和限额公式。任何路由不得读取U_s真实TX。

运行期每epoch记录H/M/L、hard/soft/candidate/no-id数量、逐预测类和逐source receiver选择数、identity loss、首批identity分项梯度norm、prototype更新数及三头一致率。精确分项梯度只在每epoch首批和E1/E16/E17边界计算，避免每batch重复反向。

## 3.星地信道增强

L_s严格沿用`ADV3B02_CORE90_SOFT_E200`：clean+satellite拼接、satellite仅承担TX CE、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`，日程固定为：

```text
E1–40   0.30 × leo_clear_weak
E41–90  0.60 × {leo_low_elev_weak,leo_rain_weak}
E91–200 0.80 × {leo_clear_weak,leo_low_elev_weak,leo_rain_weak}
```

U_s只有`U_H`获得唯一伪身份后才加入星地身份监督。U侧使用同一场景和概率日程，伪标签detach，卫星CE权重为`lambda_u(epoch)×0.68`；`U_M/U_L`不使用hard satellite CE，也不启用satellite consistency或channel-invariance。已有paired nuisance/satellite view必须复用，不能为U satellite CE再增加一次模型前向。

## 4.训练速度设计

主配置将U侧batch从128提高到256，L_s batch保持128。U loader仍在每epoch完整遍历，避免通过丢弃U样本取得表面提速。GPU3设置U batch128/384配对，测量速度和显存边界。

student strong与paired nuisance/satellite输入合并为一次拼接前向，再按batch边界拆分输出。EMA teacher使用AMP下的inference-only前向。S1跳过完整身份融合和伪标签图；权重为0或能力关闭的分支不生成view、不执行前向。训练记录每epoch耗时、samples/s、各类前向样本数、峰值显存和optimizer step数。

两条实验在同一GPU并发时，完整训练wall time只用于资源记录。速度结论使用同卡配对吞吐和启动前非阻断单进程microbenchmark，不能跨GPU直接比较。

## 5.稳定性与完成条件

所有候选使用AdamW、E1–5线性warmup、E6–160 cosine decay、E161–180 backbone LR×0.2、E181–200 backbone LR×0.05和`max_grad_norm=5`。数值异常、strict checkpoint重建失败或输出碰撞属于技术失败；低性能不得停止训练。

训练、最终评测和deployment bundle导出三者解耦。只要`final_ssdg.pth`已完成且可严格恢复，launcher必须继续执行clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`测试。bundle导出失败单独记录，不能把已完成训练改写为`TRAIN_FAILED`，也不能阻断同GPU另一条候选。

## 6.发布矩阵

机器可读矩阵为`configs/phase1_adv3b02_fasttrust16_s392002_20260821.json`。16条实验均使用seed392002、E200、`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`和不可覆盖输出目录。

| GPU | slot A | slot B | 配对目的 |
|---:|---|---|---|
| 0 | R0_SCRATCH_CONTROL_U256 | R1_ADV_INIT_CONTROL_U256 | 隔离ADV初始化贡献 |
| 1 | R2_FAST_HML_U256 | R3_FAST_HML_UPROTO_U256 | 隔离U prototype更新 |
| 2 | R4_FAST_FULL_U256 | R4_NO_U_SAT_ID_U256 | 隔离伪身份U星地增强 |
| 3 | R4_FAST_FULL_U128 | R4_FAST_FULL_U384 | U batch速度/显存边界 |
| 4 | R4_NO_PROTO_EVIDENCE_U256 | R4_NO_U_PROTO_UPDATE_U256 | 拆分prototype证据与U更新 |
| 5 | R4_NO_TEMPORAL_U256 | R4_NO_PRIOR_U256 | temporal与prior单因素 |
| 6 | R4_NUISANCE_DETACHED_U256 | R4_NO_NUISANCE_U256 | nuisance梯度路径单因素 |
| 7 | R4_NO_CROSSRX_U256 | R4_NO_CLASS_CAP_U256 | cross-RX与class cap单因素 |

R0为新协议下from-scratch控制；其余候选均从历史`best_joint_safe_ssdg.pth`初始化，不启用额外冻结教师蒸馏，从而只隔离初始化而不混入第二个teacher因素。R2不使用U prototype更新或U satellite identity；R3只增加U prototype更新；R4为FastTrust完整候选。

## 7.验证要求

实现按TDD完成，至少覆盖：S1零identity梯度；hard必须同时满足稳定性、三头一致和class cap；mid/low不能进入hard CE；U satellite CE只消费`U_H`；ADV3B02 CORE90日程逐边界一致；U batch256完整覆盖；拼接前向与分离前向数值等价；关闭分支不执行对应前向；bundle失败不阻断四场景评测和同卡队列。

矩阵发布不等于代码已实现或实验已启动。进入N607前仍须完成本地TDD、一次真实checkpoint无query smoke、一次P0/P1审查、不可覆盖run预登记和资源preflight。
