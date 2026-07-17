# D21胶囊轻型适应与注册开发追踪

范围：只复用已封存的K10/new5、rx20-1、seed713101三场景Phase2胶囊；本记录不扩展协议准入。

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| D21-DEV-01 | 用户目标、项目.md§7.4 | 直接开展域适应与新类注册，不重复前期审查 | `run_capsule_fast_adapt_dev.py` | verified | K10/new5三场景真实胶囊运行完成 | 仅快速读取已有manifest以取得class registry |
| D21-DEV-02 | 项目.md§7.2 | 每个query独立面对全部注册类，无role/quota/global assignment | `predict`子命令 | verified | truth-free prediction SHA=`36626331...b4fa6` | predictor不读取truth sidecar |
| D21-DEV-03 | 项目.md§7.2 | query标签仅由预测后的隔离scorer读取 | `score`子命令 | verified | 两条独立命令；score绑定prediction/truth SHA | scorer不生成或修改模型状态 |
| D21-DEV-04 | 用户子任务 | 比较identity、radius、old guard、2原型、fixed-IQ FFT96 top1、int8、对角metric和old-lock CVaR | support-only候选集合 | verified | support LOO与真实query结果 | 全部超参数只看support；FFT作用于同一received IQ且不增加K |
| D21-DEV-05 | 用户子任务、项目.md§10.3.1 | 报告三场景before old、after old/new/H、逐类floor、forgetting | `score_k10_new5_l7_final.json`、`report.md` | verified | 三场景及聚合表 | 同row、同old query计算遗忘 |
| D21-DEV-06 | 用户子任务、项目.md§10.3.1 | 报告状态bytes、MAC、latency | predictor receipt、`score_k10_new5_final.json` | verified | GPU实测及逻辑状态审计 | backbone固定；单列分类器增量MAC |
| D21-DEV-07 | 父任务M2 | A0上开发≤4.5k参数低秩度量，B旧类拟合、C从B初始化，支持集选rank/reg | `predictions_k10_new5_m2_final.npz`、`score_k10_new5_m2_final.json` | verified | rank4、2,304参数、20epoch、三场景真实query | query不参与选择或拟合；完整loss/resource/per-class证据已保存 |
| D21-DEV-08 | 父任务M4 | 在current单观测胶囊合法复跑D1：A1、共享对角和全注册类权重 | `run_m4_d1_current_capsule.py`、M4 prediction/score/loss | rejected | 3,456参数、20epoch；聚合old/new/H=0.6000/0.7267/0.6573 | support过拟合，合法负证据；旧跨场景D1结果未复用 |

设计边界：这是开发seed真实query诊断，不是独立确认矩阵或正式成功声明；选择只使用K10 support LOO，query scorer结果不反馈候选。

状态汇总：verified=7，deferred=0，rejected=2，blocked=0。int8 support状态已证明几乎无损；L6/L7进一步提升均值和遗忘，但old/new绝对值与floor仍低于目标。L8 class-mean融合因support-only选择`alpha=1`而被拒绝。M2低秩残差降低遗忘并提高new floor，但牺牲old floor和seen-new均值，保留为诊断Pareto。M4/D1 current-capsule虽仅6,912B状态，但support过拟合且真实old/floor显著退化，拒绝晋升。本轮不再扩展机制。
