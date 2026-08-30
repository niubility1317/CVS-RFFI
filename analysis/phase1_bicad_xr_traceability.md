# ADV3B02-BiCAD-XR设计追踪表

来源：用户提供的《ADV3B02架构级域泛化重构》报告

规格：`docs/superpowers/specs/2026-08-30-adv3b02-bicad-xr-design.md`

状态：规格冻结阶段，训练代码尚未修改

| ID | 来源章节 | 验收要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| P01 | 总体结构 | 保留ADV3B02双骨干、共享Sinc/HF、RCN、CosFace和160维`z_id/z_dom` | `model_dual_cvsincnet.py` | verified | 现有`DualCVSincNetDisentangle`代码映射 | 不以HCF-DG单骨干替代 |
| P02 | 协议冲突 | 使用`concat_sat_ce_only+LEO_WEAK`而非`mixed_orbit`单前向 | config、trainer、launcher、协议负测 | verified | 用户于2026-08-30明确确认 | E3 pair默认关闭 |
| P03 | 4.1 | 把唯一`rx_day`域头拆为receiver/day/channel因素化头 | `heads.py`、`trainer.py` | pending | 聚焦头与路由测试 | channel含clean和三种LEO弱场景 |
| P04 | 4.2 | 对`L_s`使用真实TX one-hot构造160×C CDAN条件映射 | `heads.py`、`losses.py` | pending | 形状、梯度、无标签负测 | 禁止预测标签替代真值 |
| P05 | 4.4 | 普通batch尽量TX×receiver平衡 | `sampler.py` | pending | 覆盖率与缺cell测试 | 不复制或伪造样本 |
| P06 | 5.1 | `z_dom`增加TX adversary并使用独立GRL | model、`heads.py`、trainer | pending | 梯度方向测试 | 仅`L_s`有TX监督 |
| P07 | 5.2 | V1保持严格`z_id/z_dom`二分解 | config、checkpoint runtime | pending | 配置锁测试 | `z_e+z_int`明确deferred |
| P08 | 6 | 普通正交替换为TX条件cross-cov | `losses.py` | pending | 独立/相关/小组数值测试 | `lambda_orth=0` |
| P09 | 7.2 | 每receiver动态ridge donor分类器，稳定求解且停止梯度 | `xdc.py` | pending | 解、梯度、条件数测试 | 不显式求逆 |
| P10 | 7.3 | donor质量加权并蒸馏到公共CosFace头 | `xdc.py`、trainer | pending | 权重、KL、stop-gradient测试 | 统一公式，无ID特例 |
| P11 | 7.4 | XDC每4步执行一次且向量化 | config、trainer、`xdc.py` | pending | 调度与调用计数测试 | 禁止逐样本循环 |
| P12 | 8.1 | E3实现小规模clean/satellite配对不变性 | `losses.py`、trainer | pending | pair采样、cosine、JS测试 | 复用concat输出，不新增全batch前向 |
| P13 | 8.2 | 同packet跨receiver配对仅在可靠packet ID存在时启用 | config、sampler | deferred | 当前无可靠packet ID证据 | 不恢复26D内容键 |
| P14 | 9.2 | 维护类条件receiver EMA中心并做top-K SVD | `tangent.py` | pending | source-only中心与SVD测试 | 默认K=4 |
| P15 | 9.3 | F1 factual shift和F2最坏方向shift可区分 | `tangent.py`、trainer | pending | 扰动约束和margin测试 | Stage3后启用 |
| P16 | 10 | 实现三层组EMA风险与0.6/0.3/0.1 CVaR margin-tail | `losses.py`、`metrics.py` | pending | 分组、CVaR、置换测试 | 不加权域损失 |
| P17 | 11.1 | 监控并控制两类对抗梯度比 | `gradients.py`、metrics | pending | 范数与区间控制测试 | identity0.15–0.25，TX-adv0.05–0.10 |
| P18 | 11.2 | shared-stem域正向梯度防火墙=0.05 | model、`gradients.py` | pending | 参数级梯度测试 | 域后半段不缩放 |
| P19 | 11.3 | D6每4步执行局部任务保护投影 | `gradients.py`、trainer | pending | 冲突/非冲突梯度测试 | 只作用登记模块 |
| P20 | 12 | 实现常驻、稀疏和后期三类总损失路由 | config、trainer | pending | 候选损失可达测试 | 非loss soup并非每步全开 |
| P21 | 13.1 | 75%普通batch，batch_size96，TX/RX平衡 | config、sampler | pending | batch组成测试 | day近似平衡 |
| P22 | 13.2 | 25%结构化`6×4×2`batch，缺cell mask | `sampler.py` | pending | 无placeholder负测 | 实际有效数可小于48 |
| P23 | 13.3 | E3每4步抽8–12对 | config、trainer | pending | 调度和上限测试 | V1关闭 |
| P24 | 14 | 实现Stage0–4进度调度与末段GRL/LR衰减 | config、trainer | pending | 边界测试 | Stage3前不冻结Sinc/首块 |
| P25 | 15 | 默认关闭FastTrust、pseudo、CSD、HCF、26D LODO、HDRO、open loss、Fishr、MixUp/MixStyle | config、launcher | pending | 冲突开关负测 | 候选必须fail closed |
| P26 | 16 | 实现D0–D6、E0–E4、F0–F3单因素候选矩阵 | config、launcher | pending | config diff测试 | E0/F0父候选写入runtime |
| P27 | 16 | 实现`ADV3B02-BiCAD-XDC-V1`冻结别名 | config | pending | 精确开关测试 | D5+E1+tail |
| P28 | 16 | 快筛支持fold1/fold8×seed392001/2/3×5000 updates | launcher | pending | plan dry-run | 当前不启动N607 |
| P29 | 17.1 | 保存类条件receiver probe输入artifact | `metrics.py`、trainer | pending | artifact schema测试 | 在线域准确率不能替代 |
| P30 | 17.2 | 保存`z_dom` TX probe输入与域分类证据 | `metrics.py`、trainer | pending | artifact schema测试 | 同时报告环境分类 |
| P31 | 17.3 | 输出完整donor→query迁移矩阵 | `metrics.py`、`xdc.py` | pending | 矩阵维度与mask测试 | 不只报告均值 |
| P32 | 17.4 | 输出pair干预的表示、预测和margin变化 | `metrics.py` | pending | E3 artifact测试 | E3以外写N/A |
| P33 | 17.5 | 输出Q0.1 margin和最差组合组 | `metrics.py` | pending | 分位数与组最差测试 | 明确TX/RX/day/channel |
| P34 | 最终评估 | strict checkpoint重建后分别评估clean和三种LEO弱场景 | launcher、评估入口 | pending | 真实checkpoint无query smoke | 缺任一场景不得闭合 |
| P35 | 推理边界 | 训练辅助模块不进入部署推理图 | model、checkpoint runtime | pending | inference graph测试 | 最终只读`z_id→TX` |
| P36 | 科学边界 | Phase1不访问目标receiver、Phase2、support、query或truth | trainer、launcher负测 | pending | 禁止参数与路径扫描 | 目标结果不得反馈研发 |
| P37 | 资源 | 记录吞吐、显存、GPU-hours和额外前向比例 | `metrics.py` | pending | metrics schema测试 | 性能收益必须同时报告成本 |
| P38 | F3 | 实现source-LORO低风险窗口SWAD | trainer、checkpoint | pending | 窗口与平均状态测试 | 只在F3启用 |

## 当前遗漏风险

1.现有`train_ssdg.py`很大，若新机制直接散落其中，容易出现“CLI有开关但训练路径未调用”；规格要求用独立package和最小钩子避免该问题。
2.现有双骨干已具备全局GRL和可选TX adversary，但没有receiver/day/channel类条件路由；不能把“已有头”误报为P03/P04/P06完成。
3.`concat_sat_ce_only`本身禁止额外卫星一致性进入默认总损失；E3必须是显式候选，不能偷渡到V1。
4.XDC缺cell、ridge奇异和donor低质量必须通过mask/跳过闭合，不能复制样本或数值清洗掩盖错误。
