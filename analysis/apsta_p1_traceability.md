# APSTA-P1设计报告追溯表

设计来源：`pasted-text.txt`（对提交`b6791bc242e86b6e654a4086d1a7aeb04820c284`的深度复盘）

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| APSTA-01 | 一、六、二十 | P1不再等待P0正收益，真实部分主干梯度更新 | `stage2_apsta_time_robust.py` | verified | 48项合并回归、真实checkpoint无query smoke | 首个正式主检 |
| APSTA-02 | 七、十九 | 冻结teacher，训练`t3+t_proj+fuse`学生路径 | runtime、runner | verified | 真实checkpoint仅8个选中参数张量变化，其他参数/buffer不变 | 频率分支冻结 |
| APSTA-03 | 八、十九 | rank-4目标CosFace头、temperature、bias | none | rejected | `项目.md`5.3.1 | 禁止可训练判决状态/持久分类头 |
| APSTA-04 | 11.1 | 可微leave-one-out原型损失 | runtime | verified | 手算fixture与自身排除测试通过 | 只用target support |
| APSTA-05 | 11.2 | worst-class平滑tail目标 | runtime | verified | 类置换与弱类风险测试通过 | 对齐floor |
| APSTA-06 | 11.3 | 删除特征MSE，使用L2-SP和topology | runtime | verified | topology fixture、loss trace和真实smoke通过 | 不拉回冻结特征 |
| APSTA-07 | 11.4 | 目标物理一致性 | none | deferred | 设计报告范围核对 | 缺少绑定checkpoint的增强校准范围 |
| APSTA-08 | 十二 | `0/10/30/100/300`多步checkpoint | runtime | verified | 正式配置锁定与checkpoint trace测试通过 | query不得参与选择 |
| APSTA-09 | 12.4、13.3 | robust LOO+worst-margin Pareto安全回退 | runtime | verified | 风险改善但margin下降时回退step0的fixture通过 | step0永远保留 |
| APSTA-10 | 十三 | class/sample软门控 | none | deferred | 首候选范围核对 | 先验证representation适配 |
| APSTA-11 | 十四、十九 | 时间/融合首候选，雨衰频率候选第二 | config/report | implemented | 正式配置解析为15个Target5 row | 第二候选仅在首候选后决策 |
| APSTA-12 | 十五 | 多域地面原型包 | none | rejected | `项目.md`5.3.1 | 需要新prototype-only协议 |
| APSTA-13 | 十六 | 地面episodic元训练适配器 | none | deferred | 后续Phase1研究 | 不属于当前p2_min_v1运行时 |
| APSTA-14 | 十七 | 逐row/逐类、选择步数、翻转和场景诊断 | scorer/aggregator/report | implemented | truth-last聚合测试通过 | 不反馈predictor；最终诊断待N607结果 |
| APSTA-15 | 十八、十九 | 单seed Target5首个正式主检 | config/matrix/report | implemented | 15-row配置与不可覆盖矩阵测试通过 | 15/15 paired score待N607 |
| APSTA-16 | 七、十三 | 冻结teacher路径和student分数分开保存 | runner/artifact | verified | runner schema与query零状态变更测试通过 | 支持离线归因 |
| APSTA-17 | 一、十 | 资源指标记录但不作为研发硬门槛 | audit/report | verified | 真实checkpoint计数`76,736/1,049,665=7.311%` | 按复盘报告放宽旧≤1%门，不允许全量更新 |

## 反向审计规则

- `implemented`必须有真实可达路径，`verified`必须有实际命令或artifact。
- `rejected/deferred`必须保持协议或阶段理由，不得伪装为已实现。
- 当前实现若不含可训练目标头，最终报告必须明确写作协议安全近似，而非设计严格复现。
- `REJECTED_EXTRA_GATE`：不增加独立设计审批、执行方式审批、重复review或完整Target25前置门。
