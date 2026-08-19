# ADVB02 NTRS LEO弱信道增强设计

## 目标与版本边界

本版本命名为`ADVB02_NTRS_LEO_WEAK_E200`。NTRS表示Nuisance-Tangent Robust System。它是与`ADVB02_CRRA_S_LEO_WEAK_E200`并列的独立Phase1候选，不修改CRRA方法、checkpoint、运行目录或结果。

NTRS以`ADV3B02_CORE90_SOFT_E200`为身份骨干，完整落实用户指导中“第一版最值得实施的配置”：快慢干扰表示、近恒等物理粗校正、全局成对差分干扰切空间、160维身份端有界残差、raw/robust双头、可校正性与开放集安全门，以及拼接卫星CE为主、弱KL和类别几何保持为辅的训练目标。

指导中的Phase2类共享接收机映射、无标签query适配、接收机链路模拟器扩域和困难增强挖掘属于后续独立候选，不进入本次Phase1模型或运行。尤其不得因为实现NTRS而读取target support/query或改变固定LEO_WEAK评测信道。

## 冻结协议和Core90继承项

- 数据角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，四个角色均为source receiver，物理样本ID两两不交。
- 随机种子固定为`392034`；训练200 epochs；label/pseudo=`130/70`；AdamW；基础学习率`2e-4`；`weight_decay=1e-4`；`lite_d`；`branch_ablation=no_dac`；`domain_enhancer=rcn_stats`；保留Core90的EMA、伪标签和开放世界几何目标。
- 训练和最终测试只使用`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。训练日程为E1–40：clear、`p=0.30`；E41–90：low-elev/rain、`p=0.60`；E91–200：三场景、`p=0.80`。
- `mixed_orbit`仅是历史复现入口，本候选的训练、选模和测试命令均不得使用它。
- 完整训练后必须使用冻结final checkpoint分别保存clean和三种LEO弱场景结果；聚合均值不能替代逐场景指标。
- Phase1不访问target receiver、target support、target query、target truth或Phase2运行状态。

## 网络结构

### 分组物理描述符与快慢上下文

从原始256点I/Q计算固定、数值安全的分组描述符：幅度/噪声、I/Q链路、相位/频率、频响/多径四组，总维数控制在32–48维。描述符本身stop-gradient。

`q_fast`由当前样本描述符编码。`q_slow`按source `rx_day`域保存EMA上下文，衰减系数为`0.95`，只允许训练source样本更新；评估、validation、teacher和query路径只读。若同一次LEO增强提供有效CFO、SNR、多径、相位噪声、AGC和IQ不平衡元数据，则经valid mask编码为`q_meta`；缺失元数据使用零值和显式缺失标志，不重新生成信道。

最终上下文为`q=P([q_fast,q_slow,q_meta])`，并输出不确定度`u`。接收机、日期和clean/LEO信道三个轻量头分别读取对应上下文，TX去泄漏头通过GRL读取`q`。

### 有界广义复数粗校正

使用长度`L=3`的复数FIR、共轭FIR和相位斜坡：

```text
x_corr[n]=exp(-j*phi[n])*(sum h_l*x[n-l]+sum g_l*conj(x[n-l]))
```

所有系数由`tanh`约束并近恒等初始化：`h_0=1`、其余`h_l=0`、`g_l=0`、相位频率/曲率为0。校正幅度由上下文门控且受固定上界保护。

原始身份路径仍生成`z_anchor`和`raw_logits`。校正视图只进入第二次共享身份骨干前向：时域路径读取`x_corr`；频域路径使用raw/corrected的有界双视图残差；PA路径强制读取原始I/Q；domain骨干始终读取原始I/Q。这样物理粗校正不能重构PA指纹或domain证据。

### 全局干扰切空间与有界身份残差

训练时从同一source物理样本的clean和单一LEO视图得到`z_anchor_clean`与`z_anchor_sat`，以差分更新只读EMA协方差和rank-8正交基`U_nuis`。评估路径不得更新该基。

修正系数由`[z_anchor,z_phys-z_anchor,q]`预测，最终修正严格写成`delta_z=U_nuis*a`。因此`(I-UU^T)delta_z`在数值容差内为0。幅度为`alpha<=0.20`，经过`tanh`和分阶段门控，初始严格为零：

```text
z_rob=normalize(LayerNorm(z_anchor-rho*alpha*tanh(delta_z)))
```

### raw/robust双头、可校正性和开放集安全门

raw头沿用Core90分类器。robust头是独立CosFace头，初始化为raw头原型权重的副本。可校正性头读取`q`、`u`、修正能量、raw margin和source支持距离；source支持统计只由训练source样本更新。

训练标签为`robust CE < raw CE-epsilon`。基础门为：

```text
rho=r_corr*exp(-u)*exp(-d_support^2/tau)*stage_scale
```

最终安全门还要求修正能量不过界，并在无真值推理时要求raw/robust预测一致。`unknown_rescue_disabled_by_default=true`：分歧样本保持raw结果，稳健头不能单独把低knownness样本拉入已知类。输出raw→robust四类转移、类别吸引方向余弦、子空间残差、gate、energy和支持距离遥测。

## 训练目标与阶段

拼接卫星CE是主监督。逐样本KL仅为`0.01`；类别margin保持为`0.03`；批次关系蒸馏为`0.02`。不启用强点对点MSE，不把所有Core90损失机械复制到卫星半批。

NTRS新增目标包括：

- receiver/day/channel分解与q的TX去泄漏；
- 类别条件`z_id/z_dom`去相关，权重`0.01`；
- 最小修正能量，权重`1e-3`；
- 切空间残差，权重`0.02`；
- correctability BCE，权重`0.02`；
- raw/robust known-score稳定和类别吸引抑制。

阶段严格为：

- S1，E1–16：NTRS gate为0，只训练Core90身份锚点；
- S2-a，E17–40：训练快慢上下文、三类domain头和全局干扰基，gate上限线性到0.10；
- S2-b，E41–68：启用有界残差、margin、relation和条件去相关，gate线性到0.20；
- S3，E69–200：gate固定0.20，启用correctability和开放集安全目标；NTRS学习率降低，但骨干与NTRS保持`1:5`学习率比例。

## 评估和声明边界

独立评估器从checkpoint恢复NTRS训练epoch和所有EMA/basis/support buffer，在`eval()`与`no_grad()`下运行，不更新任何状态。每个场景保存准确率及NTRS遥测：raw/robust/fused准确率、四类转移、gate、alpha、energy、支持距离、correctability、uncertainty、物理校正能量、子空间残差、类别吸引余弦和分支路径标志。

实验只能证明当前source-only Phase1协议和模拟LEO弱信道下的代理鲁棒性；不能声明真实在轨性能、完全消除接收机干扰、Phase2适配成功或真实unknown拒识成功。

