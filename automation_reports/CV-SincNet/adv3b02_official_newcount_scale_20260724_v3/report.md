# ADV3B02官方对比方法新类数量实验v3

## 实验登记

- 实验ID：`adv3b02_official_newcount_scale_20260724_v3`
- 日期：2026-07-24
- 操作：Codex主代理；N607唯一发布与运行所有者：`no_leo_n607_release`
- 目标：保持ADV3B02、CSIL和MoPC-HR官方核心方法及完整训练过程不变，仅改变新类数量，并对新类物理样本叠加一次固定LEO弱信道观测。
- 对比假设：新类数减少后，可区分算法本身的类别规模敏感性与先前低性能；不以中途性能决定停止或选点。
- 当前状态：`LOCAL_VERIFIED / WAITING_N607_RELEASE`

## 冻结矩阵

| 方法 | 新类数量 | 论文对应 |
|---|---|---|
| CSIL | 1、3 | 规模诊断 |
| CSIL | 20 | 官方每阶段新增20类；使用既有同协议完整结果并纳入联合汇总 |
| MoPC-HR | 1 | 额外规模诊断 |
| MoPC-HR | 3、5、10、25 | 官方代码增量规模`[25,10,5,3]` |

所有新运行覆盖5个目标接收机、5个确认seed、`K={1,5,10,20}`和
`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。CSIL为200个cell、600个场景行；
MoPC-HR为500个cell、1500个场景行；合计700个cell、2100个场景行。

## 数据和方法边界

- 旧类固定6类；new25采用冻结嵌套前缀，较小规模均为其前缀。
- 只对新类样本叠加LEO星地信道；对比方法不受阶段二方法边界和项目数据协议约束。
- 查询不更新方法状态；评价独立计分；不得以查询真值、类配额或中途性能修改方法。
- ADV3B02不冻结，按其官方完整方法训练；CSIL与MoPC-HR以论文和官方仓库实现为准。
- base26供CSIL，base31供MoPC-HR的new25分类容量。

## v2失败修复与本地验证

v2因远端`code/export_spaceborne_features.py`版本过旧，真实
`_build_wisig_dataset`不接受`exclude_source_record_indices`，首wave 8/8同指纹失败，
关闭为`NO_PERFORMANCE_RESULT`。v3不恢复或覆盖v2。

| 检查 | 结果 |
|---|---|
| 真实模块接口契约测试 | 新增并通过 |
| 相关pytest | `42 passed` |
| Python编译 | PASS |
| `git diff --check` | PASS |
| v2→v3数据spec语义 | 25/25仅run ID变化 |
| schema/命令/parity/规范化哈希 | 25/25 PASS |
| ManyTx路径 | 25/25指向`CV-SincNet/Dataset_WigSig/ManyTx.pkl` |

冻结运行依赖：

| 文件 | SHA256 |
|---|---|
| `code/export_spaceborne_features.py` | `9e0ed8cefd8c652abd0b57a0e3baeebddc03cef16f215e3cf1004f1270552662` |
| `code/scripts/build_cvs_leo_weak_iq_cache.py` | `0c0684d17b1c5fc2e0b71f0dc0c059ee983721f819468062ba316a50d0b1bb33` |
| `cache_specs_manifest.json` | `0976f2d4e12455f6937ceca396c40e05c085e6c77a7ffaad7fa67e06622d42ae` |

## N607发布计划

- 环境/CWD：`ssr-gpu`；`/home/szu2070436088/2510044040/CV-SincNet`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v3`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v3`
- release：`paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v3_release`
- 顺序：25套cache→25套既有new20奇偶校验→base26/base31→两方法真实smoke→700-cell完整矩阵→完整日志与artifact回收。
- 远端放行门：两个运行依赖哈希匹配、真实模块导入及签名检查通过、v3 run/log均不存在。
- 技术停止：仅P0协议/安全错误，或至少两个不同row在预测前出现同一确定性异常指纹；绝不因准确率低停止。
- 预期产物：cache/parity收据、base状态、plan、逐row prediction/score/exit、health state、方法汇总和完整日志。

## 结果表

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting/per-class old | loss摘要 | coverage | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| 待运行 | — | — | — | — | — | — | — | — | — | — | — | 0/700 cells | `WAITING_N607_RELEASE` |

完成后必须在本报告补齐同row结果、异常、解释和下一实验建议。

## 独立审查关闭

v3独立审查为`P0=0,P1=1,P2=0 / NOT_APPROVED`。25份spec对`target_old`和
`target_new`都执行LEO叠加，与“仅新类样本叠加LEO”冲突。v3未同步、未启动、无性能结果，
关闭为`LOCAL_RELEASE_REJECTED / NO_PERFORMANCE_RESULT`。修复使用新的v4 run ID。
