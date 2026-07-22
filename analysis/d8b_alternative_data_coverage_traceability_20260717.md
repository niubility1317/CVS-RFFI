# D8b替代数据覆盖只读盘点追踪

范围：不构建cache、不启动实验，仅在N607只读盘点ManyRx、SingleDay、ManyTx，寻找不占用4个未来formal确认receiver、排除`R_s`、且old6与至少new5各自可提供不少于120条独立物理样本的development-only receiver/subset。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D8B-01 | AGENTS.md N607 SSH | direct preflight后使用短连接只读命令，结束后清理SSH | D8 report、remote audit | verified | preflight PASS；2次`ssh ... python3 -`；结束后ssh进程0、TCP22连接0 | 未复制大文件、未启动任务 |
| D8B-02 | `项目.md`§8.4 | 排除`R_s`与formal 5 receivers | remote inventory filter | verified | source/formal集合在远端脚本中显式过滤 | 不占用确认receiver |
| D8B-03 | 主任务 | old6与至少new5每TX均≥120 | ManyRx/SingleDay/ManyTx metadata | verified | 逐receiver、逐TX、eq1计数 | 只读pickle元数据/shape |
| D8B-04 | `项目.md`§8.4 | ManyRx只能作为control，且必须有真实old/new覆盖 | ManyRx inventory | rejected | 所有非源/非formal receiver均缺`6-15`，old6不完整 | new5足够但不能补齐old6 |
| D8B-05 | `项目.md`§8.4 | SingleDay只能作为single-day smoke/control | SingleDay inventory | verified | `13-13,2-20,8-13`均old6+22new，800/TX，但仅1天 | 不满足多日分层主路线 |
| D8B-06 | 主任务补充 | 若前两者不足，评估ManyTx非formal receiver development control | ManyTx inventory | verified | 6个receiver满足；`1-20`逐日核验old6/new5均200、4天各50 | 推荐首选 |
| D8B-07 | `项目.md`正式主线边界 | 清楚声明ManyTx target-old偏离ManySig target-old主线 | report/recommendation | verified | 标为development control/sensitivity | 未更新`项目.md`前不得晋升正式证据 |
| D8B-08 | 主任务继续指令 | 本地构建ManyTx@1-20 old6+冻结new5单观测cache，三scenario各40/TX | cache spec、cache set、coverage audit | verified | cache set重载；逐TX/day/scenario exact40；总1320条 | 实际只使用预登记day0-2，不使用第4天 |
| D8B-09 | 用户单观测约束 | 三scenario物理根互斥，每个物理样本只叠加一次LEO_weak | cache set lineage | verified | 三组集合交集0；与20-1参考交集0；lineage single-observation PASS | clean=false，additional state=false |
| D8B-10 | 主任务继续指令 | 只生成K10/new5 enrollment-only sealed package，不打开query truth | support-only offline packager/package seal | verified | strict K10 bundle loader复验；每类每scenario实际可达10条；forbidden成员空 | before/after均密封，不预测、不评分 |
| D8B-11 | 资源/内存约束 | 两role同指ManyTx时复用同一只读pickle对象，避免重复4.2GB加载 | cache builder | verified | dataset cache参数接入；组合32项测试PASS；实际构建12.3秒完成 | 不改变样本选择语义 |
| D8B-12 | 用户统一K-shot边界 | manifest K必须与每类实际可达support精确相等；K1/K5只能为同一K10包前缀 | strict K10 builder、反例测试、verification | verified | 旧20-per-class包标`PROTOCOL_INVALID_KSHOT_REACHABILITY`；当前before=`6×10`、after=`11×10`；组合53项测试PASS | K20未来必须单独密封 |
| D8B-13 | query隔离协议 | 参数化support candidate COMMIT后构建Q20 query-only；predictor无truth/role/quota，truth仅在scorer根 | query control builder、seal、selection audit | verified | D9结构包before=`6×20`、after=`11×20`；support交集0；old query前后复用；组合55项测试PASS | D9 floor仅0.1–0.4，包标`STRUCTURAL_ONLY_NOT_SELECTED`，未预测、未评分 |

## 推荐

首选`ManyTx.pkl/receiver=1-20/equalized=1`：

- receiver不属于`R_s`，也不属于formal 5 receiver；
- old6与固定new5=`1-16,1-18,18-10,14-11,8-3`均为200条/TX；
- 4天各50条，可选120条并按TX/day分层分配至3个互斥scenario；
- receiver不同于旧development cache的`20-1`，物理根包含receiver字段，因此不会与旧cache复用同一物理IQ样本；
- 代价是target-old来自ManyTx而非正式主线ManySig，只能作为development control/sensitivity。

备选SingleDay的`13-13`仅适合单日smoke，不用于多日稳定性判断。ManyRx因缺old TX`6-15`淘汰。
