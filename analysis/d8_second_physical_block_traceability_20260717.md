# D8第二物理样本块development验证追踪

范围：在不消费`3-19/7-14/7-7/8-8`确认query、也不复用既有`20-1/713101`物理根的前提下，为冻结的D7a机制准备并执行一个`20-1/713201/K10/new5`development-only proxy。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D8-01 | `项目.md`§7.1、§7.1.1 | Phase2只接收已叠加一次LEO_weak的IQ；每个物理样本只属于一个scenario | cache spec、cache builder、coverage audit | blocked | builder在overlay前因覆盖不足停止；无cache文件产生 | 禁止通过降低覆盖或复用旧根绕过 |
| D8-02 | 主任务补充边界 | 不消费4个未来确认receiver的713102-713106 query，也不使用其713101调参 | D8 report/spec | verified | 固定`20-1/713201`；未打开其他receiver数据 | 未占用确认矩阵 |
| D8-03 | 主任务补充边界 | 新块与既有`20-1/713101`全部120/TX物理根零重叠 | cache builder、reference cache exclusion audit | verified | 新增引用已验证cache按role+dataset SHA+source record排除；6项测试PASS；真实构建在不足120时fail closed | 排除机制已验证，实际新块未生成 |
| D8-04 | `项目.md`§8.4、§10.3.1 | old/new真实TX，TX/day分层，三scenario各40/TX，共120/TX | cache spec、cache set | blocked | old6各余2880；new19类各余30，`4-10`余6 | 不足120，甚至不足K10+Q20三scenario所需90 |
| D8-05 | `项目.md`§7.2 | D7a结构、operator集合、支持集选择门限完全冻结；query不参与拟合/选择 | D7a method lock、support COMMIT | blocked | 因cache覆盖门失败未启动support lock | 未读取query |
| D8-06 | `项目.md`§7.2 | K10 support与query物理独立；先COMMIT后预测，再由隔离scorer连接truth | D8 predictor/scorer artifacts | blocked | support/prediction/scorer均未启动 | truth join=false |
| D8-07 | AGENTS.md实验报告 | 保存清单、逐类/逐scenario结果、资源、日志、Git状态和阻塞 | automation report | verified | report、cache spec、coverage audit已固化 | 根目录非Git；未提交 |

## 预登记冻结项

- receiver：`20-1`
- development-only seed：`713201`
- K：`10`
- new class count：`5`
- old TX：`14-10,14-7,20-15,20-19,6-15,8-20`
- nested new TX前5：`1-16,1-18,18-10,14-11,8-3`
- D7a operator集合：`base,dc_rms,dc_rms_spec15`
- D7a类条件选择与全局回退门：保持现有实现，不根据D8 query结果修改

## 高风险项

引用cache排除机制已落地并验证。剩余最高风险转为数据覆盖硬上限：ManyTx在`20-1/eq1/days0-2`每个new TX通常仅150条，旧cache已使用120条，无法再形成独立的三scenario K10+Q20行。D8按`LOCAL_DATASET_EXTENSION_REQUIRED`停止，不得构建预测或评分。
