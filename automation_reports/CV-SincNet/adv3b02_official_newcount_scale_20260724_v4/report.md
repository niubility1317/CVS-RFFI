# ADV3B02官方对比方法新类数量实验v4

## 登记与状态

- 实验ID：`adv3b02_official_newcount_scale_20260724_v4`
- 日期：2026-07-24
- 操作：Codex主代理；N607唯一运行所有者：`no_leo_n607_release`
- 目标：保持ADV3B02、CSIL和MoPC-HR官方核心方法及完整训练不变，比较新类数量；只有新类物理样本叠加LEO弱信道。
- 状态：`LOCAL_VERIFIED / WAITING_INDEPENDENT_REVIEW`

## 冻结实验

| 方法 | 新类数量 | 论文口径 |
|---|---|---|
| CSIL | 1、3 | 规模诊断 |
| CSIL | 20 | 官方每阶段新增20类；联合既有完整同方法结果 |
| MoPC-HR | 1 | 额外规模诊断 |
| MoPC-HR | 3、5、10、25 | 官方代码增量规模`[25,10,5,3]` |

新运行覆盖5接收机×5seed×`K={1,5,10,20}`×3个LEO弱场景：
CSIL 200 cells/600场景行，MoPC-HR 500 cells/1500场景行，共700 cells/2100场景行。
旧6类和new25嵌套前缀固定；base26供CSIL，base31供MoPC-HR。

## 用户边界的实际实现

- `target_old`：保留ManyTx中的原始物理接收IQ，`overlay_applied=false`，
  `channel_view=unmodified_received_iq`。
- `target_new`：调用`apply_sat_channel_for_scenario`，`overlay_applied=true`，
  `channel_view=rx_base`。
- 每份spec显式锁定`target_old.apply_leo_overlay=false`和
  `target_new.apply_leo_overlay=true`；loader要求mask与role完全一致。
- parity只比较既有前20个新类的sample ID和LEO后IQ哈希；旧类因不再叠加LEO，不与旧实验LEO IQ强行相等。
- 对比方法不受阶段二主方法协议约束；查询仍只用于独立评价，不以中途性能选点或停止。

## 本地验证

| 检查 | 结果 |
|---|---|
| 混合role真实builder测试 | 旧类未调用LEO、新类调用LEO，IQ和mask逐行通过 |
| WiSig真实接口契约 | `exclude_source_record_indices`存在 |
| 相关pytest | `49 passed` |
| Python编译 | PASS |
| `git diff --check` | PASS |
| spec/schema/命令/parity | 25/25 |
| ManyTx路径 | 25/25为`CV-SincNet/Dataset_WigSig/ManyTx.pkl` |

v1、v2均为系统性技术失败且无性能结果；v3在本地独立审查被拒绝且未发布。v4不恢复、
不覆盖任何旧run。

## N607发布

- 环境/CWD：`ssr-gpu`；`/home/szu2070436088/2510044040/CV-SincNet`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v4`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v4`
- release：`paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v4_release`
- 顺序：运行时哈希/真实import→真实混合role cache smoke→25 cache→25 new20 parity→
  base26/base31→CSIL/MoPC-HR真实smoke→700-cell完整矩阵→artifact回收和完整分析。
- 技术停止只允许P0或至少两个不同row在预测前产生同一确定性异常；禁止按性能停止。

## 结果

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting/per-class old | loss摘要 | coverage | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| 待运行 | — | — | — | — | — | — | — | — | — | — | — | 0/700 cells | `WAITING_REVIEW` |

v4在远端发布前发现sample view policy名称仍错误沿用`leo_weak_only_no_clean_access`，
未进入独立审查和N607，关闭为`LOCAL_SUPERSEDED / NO_PERFORMANCE_RESULT`。
