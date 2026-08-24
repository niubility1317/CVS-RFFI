# CAPTA-P0设计报告追溯表

设计来源：`pasted-text.txt`（面向CVS Phase2星上部署的轻型快速／多步域适应设计）

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| CAPTA-01 | 一、六、十八 | 独立CAPTA-P0路径，冻结主干且零反向传播 | `code/cvsrffi/stage2_capta/` | verified | 原型层8项聚焦单测与真实checkpoint无query smoke通过 | 不修改late-block算法文件 |
| CAPTA-02 | 6.4 | support目标中心与冻结原型做有效样本量收缩 | `prototype_transport.py` | verified | 手算球面收缩fixture通过 | 只用target support |
| CAPTA-03 | 6.3、6.4 | 共享目标域平移并保持类间结构 | `prototype_transport.py` | verified | 共享平移手算fixture通过 | A2 |
| CAPTA-04 | 5.2、18 | rank-4低维域码迁移 | `prototype_transport.py` | verified | rank上限、输入不变、确定性单测通过 | 使用support残差近似；不声称地面学习`U,V` |
| CAPTA-05 | 6.6 | 源/目标双路径安全门控，DA0路径永远保留 | `safe_source_target_gate.py` | verified | leave-one-out、并列回退源路径单测通过 | 系数在query打开前冻结 |
| CAPTA-06 | 17 | source/target分数和迁移审计分开记录 | `runtime.py`、runner | verified | runtime行为、CLI schema与真实checkpoint smoke通过 | 不覆盖CosFace头 |
| CAPTA-07 | 17 | query只读、逐样本面对全部注册类 | `runtime.py`、runner | verified | 状态不变、顺序无关和无禁止API单测通过 | 无batch-global分配 |
| CAPTA-08 | 14.1 | A0/A1/A2/A3/A6同row最小可证伪矩阵 | experiment report | verified | N607 Target5完成45份prediction与45份paired score，失败0 | 三个候选均未晋级 |
| CAPTA-09 | 5.1、18 | 地面对角方差、半径、温度、跨域原型胶囊 | none | rejected | `项目.md`5.3/5.3.1协议核对 | 当前白名单只允许冻结类原型及映射，禁止类条件source统计 |
| CAPTA-10 | 6.3、7.3 | 无标签软分配、unknown sink | none | deferred | 协议核对 | Stage2-B本轮无独立合法无标签target流，不能把query用于拟合 |
| CAPTA-11 | 6.5 | query kNN图细化 | none | rejected | `项目.md`5.2/5.4协议核对 | query及其view不得更新状态或跨query重排 |
| CAPTA-12 | 八、九、十八P1 | 物理校准、SSF、rank残差、梯度和安全步数 | none | deferred | P0性能门槛 | 设计报告明确要求P0稳定收益后才进入P1 |
| CAPTA-13 | 十、十八P2 | 持续记忆、EMA、上下文库 | none | deferred | 后续因果协议 | 不属于当前`p2_min_v1`单row验证 |
| CAPTA-14 | 十六 | mean、floor、资源和状态写入报告 | report/scorer | verified | v2报告含15-row等权mean、全矩阵floor、资源审计与结论 | REG0的新类/unknown指标为N/A |

## 反向审计规则

- `implemented`必须给出可达代码路径；`verified`必须给出实际命令或artifact。
- `rejected`和`deferred`保持明确协议/阶段理由，不因低性能改写。
- 若CAPTA-04最终只能实现support残差近似，最终报告必须标为设计近似而非严格地面域基实现。
