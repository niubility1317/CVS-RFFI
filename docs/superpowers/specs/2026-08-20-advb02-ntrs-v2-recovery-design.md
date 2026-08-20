# ADVB02 NTRS-V2星地性能修复设计

## 目标与证据边界

本设计根据用户提供的《ADVB02/NTRS星地性能回退诊断》修复NTRS-v1的公共训练路径，并以最小因果矩阵验证修复是否成立。目标不是继续调整`alpha_max`、rank或零散loss权重，而是优先消除四个已确认的公共路径混杂：主干低学习率、非恒等LayerNorm、独立robust分类头和随机双前向。

本轮只形成Phase1 source-only、模拟LEO_WEAK条件下的代理证据。结果不能声明真实卫星链路、Phase2适配、真实unknown拒识、多节点协同或在轨验证成功。

## 当前事实与冲突处理

已完成矩阵给出以下同行结果：M0的LEO均值为`70.457%`，M1完整NTRS-v1为`51.618%`，下降`18.839`个百分点；M1的clean准确率下降`3.224`个百分点。六组均完成E200有效训练和独立最终测试，因此本设计把回退视为真实算法负结果，而不是运行失败。

用户报告中“ADVB02正式默认训练增强仍应区分mixed_orbit”与当前`项目.md`冲突。科学与数据协议以`项目.md`为准：新建Phase1训练和最终星地测试默认只使用`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，`mixed_orbit`不进入本轮训练、测试、选模或晋级。

## 冻结公共条件

- base candidate：`ADV3B02_CORE90_SOFT_E200`。
- seed：`392034`。
- Phase1角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- source-only：不得读取target receiver、target support、target query、target truth或Phase2状态。
- 训练轮数：200；label/pseudo=`130/70`；AdamW；基础学习率`2e-4`；其他Core90训练条件保持不变。
- LEO日程：E1–40为clear且`p=0.30`；E41–90为low-elev/rain且`p=0.60`；E91–200为三个LEO_WEAK场景且`p=0.80`。
- checkpoint selection：保持现有final-only规则；所有正式结果来自本run的E200最终checkpoint。
- 最终测试：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别完整测试，不用聚合均值代替逐场景结果。

## 方案选择

采用分阶段因果修复，而不是直接堆叠完整稳健系统：

1. 用现有M1 checkpoint补做raw/robust/fused和gate只读诊断。
2. 用D1验证NTRS外壳严格无侵入。
3. 用D2与D3正交分离低主干LR和v1结构损害。
4. 用V2-1验证共享头、无LayerNorm、同主干LR的最小有界残差。

不采用“只跑一个V2结果”的路线，因为它不能区分LR修复与结构修复；不采用“一次加入teacher basis、快慢上下文、物理校正和安全损失”的路线，因为它会再次失去因果可解释性。

## 配置接口

训练器增加两个正交控制接口：

```text
ntrs_variant = v1 | v2_min
ntrs_core_lr_mode = baseline | v1
ntrs_identity_bypass = true | false
```

launcher对外只暴露预注册profile，避免任意组合产生未登记实验：

|profile|variant|core LR|identity bypass|用途|
|---|---|---|---|---|
|`v2_identity_bypass`|`v2_min`|baseline|true|D1严格旁路|
|`v2_identity_bypass_v1_lr`|`v2_min`|v1|true|D2低LR诊断|
|`v1_fair_core_lr`|`v1`|baseline|false|D3结构诊断|
|`v2_min_shared_head`|`v2_min`|baseline|false|V2-1最小稳健机制|

现有`full`和四个v1消融profile保持历史行为，不追溯改写旧checkpoint、命令或报告。

## 严格恒等契约

### 模型级契约

V2路径定义：

```text
delta_z = bounded_residual(z_anchor, q_fast)
z_rob = z_anchor - stage_scale * delta_z
raw_logits = H(z_anchor)
robust_logits = H(z_rob)
fused_logits = raw_logits + safe_gate * (robust_logits - raw_logits)
```

`H`是原ADVB02共享CosFace头。V2不得实例化或训练独立robust prototype集合，不得在`z_rob`上附加LayerNorm或额外L2 normalize；CosFace内部已有归一化。

必须满足：

```text
delta_z == 0  => z_rob == z_anchor
safe_gate == 0 => fused_logits == raw_logits
identity_bypass == true => tx_logits、z_id、raw_logits逐元素等于原始ADVB02路径
```

数值验收为最大绝对误差小于`1e-6`。identity bypass必须在`model.eval()`和固定输入下对真实模型前向验证，不以源码文本检查代替。

### D1/D2旁路语义

D1和D2实例化NTRS-V2模块及其checkpoint字段，但前向直接复用raw身份路径：不执行第二次身份骨干前向，不改变`z_id`，不改变分类logits，不更新NTRS support、basis或其他状态。两臂唯一差异是core LR日程。

这使D1测量“外壳和参数分组是否无侵入”，D2测量“在同一无侵入结构上，v1低LR日程造成多少损害”。

## 主干与NTRS优化日程

### baseline core LR

D1、D3和V2-1的core参数组在E1–E200始终使用基础学习率：

```text
lr_core(epoch) = 2e-4
```

不得因启用NTRS而把core降为`0.2×`或`0.1×`。

### v1 core LR

D2保留历史v1日程，作为单一因果变量：

```text
E1–16: 1.0×
E17–40: 0.2×
E41–68: 0.2×
E69–200: 0.1×
```

### V2-1残差日程

V2-1从头训练时，core全程保持baseline LR；NTRS残差遵循成熟锚点后启用原则：

```text
E1–90: stage_scale=0，NTRS LR=0
E91–130: stage_scale从0线性升至1，NTRS LR=base LR
E131–200: stage_scale=1，NTRS LR=base LR
```

E1–90期间V2-1的模型输出必须与raw路径一致。该日程让身份主干先完成90轮全学习率训练，再在三个LEO_WEAK场景联合阶段逐步打开残差；core仍可继续适应新场景。

## V2-1最小有界残差

### 输入和结构

V2-1只使用一次raw身份骨干前向得到`z_anchor`。上下文只保留当前样本的确定性40维物理描述符和`q_fast`编码，不读取slow/source support，不使用metadata，不执行物理IQ粗校正，也不执行第二次随机身份骨干前向。

残差头读取`[stop_gradient(q_fast), z_anchor]`并输出与`z_anchor`同维的候选残差。候选残差先经`tanh`，再按样本限制为相对锚点范数不超过`alpha_max=0.20`：

```text
raw_delta = tanh(residual_head([z_anchor, q_fast]))
delta_z = raw_delta * min(1, alpha_max * ||z_anchor|| / (||raw_delta|| + eps))
```

残差头末层零初始化，因此初始`delta_z`严格为0。V2-1不使用v1随机更新的global tangent basis；确定性EMA teacher basis属于后续V2-2，只有V2-1达到本设计晋级门槛后才进入下一轮。

### 最小训练目标

V2-1保留Core90原有主损失和拼接LEO卫星CE，只增加：

1. robust共享头CE；
2. 直接最小修正损失：`mean(||delta_z||²/(||z_anchor||²+eps))`；
3. clean/satellite共享头logit KL，保持历史权重`0.01`；
4. clean/satellite类别margin保持，使用同一个共享头prototype坐标系。

V2-1不启用receiver/day/channel分解、TX对抗、条件去相关、shared receiver offset、correctability BCE、score stability、class attraction、slow support、物理IQ粗校正或unknown rescue。这样首轮只回答“共享头严格恒等有界残差能否改善LEO闭集鲁棒性”。

### 融合策略

训练期和评估期均使用连续有界门：

```text
safe_gate = stage_scale * correction_energy_ok
```

其中`correction_energy_ok`只由固定上界决定，不使用raw/robust argmax一致性，也不使用尚未训练的correctability和support。门为0时严格回退raw；门大于0时共享头logits线性插值。V2-1不声称开放集安全性。

## 现有checkpoint只读诊断

对M1的E200 checkpoint重新运行现有独立评估器，启用`eval_ntrs_telemetry`，保存clean和三个LEO_WEAK场景下的：

- raw、robust、fused准确率；
- raw/robust disagreement；
- rescued、harmed、both-correct、both-wrong；
- `ntrs_gate`、`ntrs_safe_gate`、`ntrs_alpha`；
- correction energy、physical correction energy；
- support distance、correctability、uncertainty；
- `P(safe_gate>0.01)`、`P(safe_gate>0.05)`、`P(safe_gate>0.10)`；
- class-attraction cosine。

诊断只能解释既有M1，不参与V2阈值、超参数、checkpoint或候选重排。

## 实验矩阵

|实验|新训练|比较目标|预注册解释|
|---|---|---|---|
|D0|否，复用M0同行结果|控制基线|LEO均值`70.457%`|
|M1-DIAG|否，只读重评M1|raw/robust/fused分解|解释v1回退，不晋级|
|D1|是|D1−D0|外壳与参数分组零侵入|
|D2|是|D2−D1|v1低core LR的净影响|
|D3|是|D3−D1|v1结构与损失的净影响|
|V2-1|是|V2-1−D1及V2-1−D0|最小共享头残差收益|

四个新训练run均使用独立、不可覆盖的run ID和output root。单seed结果只用于首轮可证伪和候选筛选；只有V2-1达到门槛后，才进入至少3个seed的正式重复。

## 指标和晋级门槛

### D1恒等门槛

相对D0：

```text
abs(delta clean) <= 0.5 pp
abs(delta LEO mean) <= 0.5 pp
max_abs(logits_bypass - logits_raw) < 1e-6
max_abs(z_bypass - z_anchor) < 1e-6
```

D1未通过时，V2外壳不得晋级，但已启动的其他合法矩阵臂继续完成并报告，不因低性能中止。

### V2-1闭集门槛

相对D0：

```text
delta LEO mean >= +1.0 pp
delta clean >= -0.5 pp
clear、low-elev、rain均不得低于D0超过0.5 pp
rescued > harmed
```

同时报告Net Rescue Rate：

```text
(rescued - harmed) / N
```

V2-1只有达到以上全部条件才进入多seed或V2-2。D2、D3是诊断臂，不以性能高低单独晋级。

### 开放集边界

V2-1不启用安全损失，因此不报告unknown FAR、OSCR或真实unknown成功。开放集安全损失是否恢复只在后续冻结known-retention和unknown-FAR双门后研究，不能由本轮闭集准确率代替。

## 评估artifact

每个新训练run必须保存：

- `final_ssdg.pth`；
- 200行`metrics_epoch.csv/jsonl`；
- `phase1_terminal_status.json`；
- `phase1_resource_summary.json`；
- `independent_final_eval/final_eval.json/txt`；
- clean和三个LEO_WEAK逐场景日志；
- raw/robust/fused、gate和transition遥测。

训练结束而缺少任一必需最终测试时，不得标记`ARTIFACTS_COMPLETE`。

## 错误处理与安全边界

- parser拒绝未知profile、`mixed_orbit`、非`392034`seed和错误角色比例。
- `v2_min`与独立robust head、物理IQ corrector、slow support或v1安全损失同时启用时立即报错。
- identity bypass路径若产生非零修正、更新NTRS状态或改变raw输出，聚焦测试失败并阻止发布。
- 训练只因协议/seed/场景错误、错误checkout/CWD、输出碰撞、无checkpoint、无prediction闭合或重复确定性技术异常停止；不得因性能较低停止。
- 不覆盖或删除v1的checkpoint、日志、metrics和报告。

## 测试策略

实现严格遵循测试先行：

1. 先新增D1真实模型恒等测试，并观察它在v1代码上因LayerNorm/独立头而失败。
2. 新增共享头测试，证明V2没有独立prototype参数。
3. 新增LR模式测试，证明baseline模式E1–E200的core LR不变，v1模式保持历史日程。
4. 新增V2阶段测试，证明E1–90输出严格raw、E91–130连续ramp、E131–200全启用。
5. 新增minimum-correction测试，证明零残差loss为0且只随`delta_z`变化。
6. 新增launcher profile测试，验证四个profile、角色比例、三LEO场景、final eval和不可覆盖路径。
7. 扩展评估测试，验证gate阈值比例和raw/robust/fused transition序列化。
8. 运行NTRS核心、模型、训练、评估、协议负测和launcher聚焦回归；随后进行真实checkpoint无query smoke。

## 设计追踪

|ID|来源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|R01|报告第3、11节|NTRS候选core LR与M0公平|`code/cvsrffi/ntrs_training.py`、训练器|verified|LR测试及D1/D2/D3完整曲线|D2单独保留v1 LR，D1/D3/V2-1全程baseline LR|
|R02|报告第4、11节|零修正严格恒等且移除LayerNorm|`code/ntrs.py`、模型|verified|真实模型`<1e-6`恒等测试及D1三头一致|V2路径限定；D1单seed性能等价门因正向偏差仍未满足|
|R03|报告第5、11节|删除独立robust head并共享CosFace|模型、checkpoint兼容|verified|参数身份、logit测试及V2-1严格加载|v1历史路径不变|
|R04|报告第6、11节|首轮避免随机双身份前向|模型V2 forward|verified|hook调用次数测试|V2-1只前向一次|
|R05|报告第7节|minimum correction直接约束`delta_z`|NTRS loss bundle|verified|零值、单调性测试及E1–200曲线|不再比较LN输出|
|R06|报告第10节|补做M1三头与gate只读诊断|评估器、M1诊断run|verified|M1 clean及三LEO_WEAK完整JSON|raw/robust/fused与转移已保存；三个gate阈值比例字段未序列化，报告明确限制|
|R07|报告第11、12节|实现共享头最小V2-1|NTRS模块、模型、训练器|verified|聚焦回归、真实smoke及E200独立测试|科学结果为负，不含teacher basis之后模块|
|R08|报告第12节|发布D1、D2、D3、V2-1矩阵|launcher、实验报告|verified|四组E200、六行clean＋三LEO_WEAK矩阵|D0复用现有同行结果，M1只读诊断|
|R09|报告第13节|按恒等、闭集和rescue门槛晋级|实验报告/scorer|verified|同行最终指标与门槛逐项判定|D1等价门及V2-1四项晋级门未通过，停止V2-2至V2-6和多seed|
|R10|当前`项目.md`|固定source角色和三LEO_WEAK|launcher、协议负测|verified|现有协议及本设计复核|优先于旧报告|
|R11|报告中的旧默认描述|继续以mixed_orbit为ADVB02默认|无|rejected|与`项目.md`冲突|本轮禁止mixed_orbit|
|R12|报告第11.4、11.5、12节|确定性teacher basis、fast/slow support、物理IQ校正、安全损失|后续V2-2至V2-6|deferred|仅在V2-1晋级后设计|避免首轮重新堆叠|

## 完成定义

本轮技术完成要求R01–R09均达到`verified`，R10保持`verified`，R11保持有理由的`rejected`，R12保持有条件的`deferred`。科学完成不由代码或启动替代：只有D1、D2、D3和V2-1均完成E200及clean＋三LEO_WEAK最终测试，才可报告矩阵闭环；只有V2-1达到预注册门槛，才可声明最小NTRS-V2在本Phase1代理协议下晋级。
