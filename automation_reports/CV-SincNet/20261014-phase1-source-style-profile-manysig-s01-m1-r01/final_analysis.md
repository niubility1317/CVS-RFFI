# 浅中层源风格诊断完整结果

run：20261014-phase1-source-style-profile-manysig-s01-m1-r01；实际执行提交：d594ac63a95bf5c654d10dbc9e9a82608fe447b3。首种子 2026092701，固定旧门控＋cos final50/11100 来源；没有新训练、目标预测或分类头拟合。

完整合法 L_s 6300 包，50 批、12600 包次，CPU 8.52 秒。t1/t2 只捕获实际层输出并作固定 eval 干预；模型参数与 buffers 逐元素不变、无 .grad 累积或优化器 step。U/target 未读取；共享缓存曾 materialize V，立即丢弃且无 V 前向或求导。

| 描述量 | t1 | t2 |
|---|---:|---:|
| RX 条件均值/标准差变化占总中心平方和 | 32.6554% | 28.2094% |
| TX 条件均值/标准差变化占总中心平方和 | 64.4622% | 66.9365% |
| 实际启用批次/50 | 29 | 23 |
| 可配对包/6300 | 6273 | 6273 |
| 实际修改包次 | 3650 | 2801 |

统计条件分别为同 TX/day 的 RX 和同 RX/day 的 TX。这是描述量，不是因果噪声分解、捷径诊断、特征秩或迁移证明；两种比例不构成互斥分解。

固定 p=.5、alpha=.1、strength=1，同 TX、同 day、不同源 RX 且不同真实物理 ID 配对。每次实际非自身 donor 由运行代码检查身份规则；离线完整记录重新核对 donor 范围、启用/未启用格式、批次与总计，未把单独索引当作独立身份证明。

实际 joint160 干预平均 L2=1.482290、cosine=0.979574。默认头 clean 正确 6264/6300，干预后 6268/6300，预测改变 19；修正 11、破坏 7，另 1 个错误改变为另一错误。头和骨干均未训练。源错误差是冻结干预观察，不能当作训练效果，亦不能预判 learnable augmentation 无效。

结论：有实际浅中层跨 RX 统计差和进入部署 z 的接线证据，支持最小身份保护 MixStyle 训练反证。单一首模型 seed 不冒充多种子；固定 t1/t2 及 p/alpha，不根据目标结果调参数。下一项使用旧门控＋EMA＋默认 cosine 正常联合训练，独立 scratch50 对照，不叠加 PAIRS/Sinc/TIME，也不另训头。

用户新贴的时序覆盖×分支 SiLU 矩阵已完整完成，保留原结果，不重复启动；浅中层风格机制与另一对话的单基时序路线不同。

发布状态 VERIFIED：单 release archive 本地/远端 SHA 一致、远端 compile 与 real checkpoint no-query smoke PASS，owned PID/CWD/argv 已核对，最终 completion 与 47 字节日志确认终止，无 failure。完整 results.json 与 launch_evidence.json 同目录。
