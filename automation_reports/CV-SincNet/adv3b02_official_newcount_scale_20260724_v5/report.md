# ADV3B02官方对比方法新类数量实验v5

- 实验ID：`adv3b02_official_newcount_scale_20260724_v5`
- 日期：2026-07-24
- 操作：Codex主代理；N607唯一运行所有者：`no_leo_n607_release`
- 目标：保持ADV3B02、CSIL、MoPC-HR官方核心方法和完整训练不变，仅研究新类数量；只有新类物理样本叠加LEO弱信道。
- 状态：`LOCAL_VERIFIED / WAITING_INDEPENDENT_REVIEW`

## 冻结矩阵

| 方法 | 新类数 | 规模来源 |
|---|---|---|
| CSIL | 1、3 | 规模诊断 |
| CSIL | 20 | 论文每阶段新增20类，联合既有完整结果 |
| MoPC-HR | 1 | 额外诊断 |
| MoPC-HR | 3、5、10、25 | 官方代码增量规模 |

新运行固定5接收机×5seed×`K={1,5,10,20}`×3场景，共700 cells/2100场景行：
CSIL 200/600，MoPC-HR 500/1500。旧6类和new25嵌套前缀固定。

## 数据作用范围

| role | IQ | `overlay_applied` | view |
|---|---|---|---|
| `target_old` | ManyTx原始物理接收IQ | `false` | `unmodified_received_iq` |
| `target_new` | 在原始接收IQ上叠加对应LEO弱信道 | `true` | `rx_base` |

sample view policy为`target_old_received_iq_target_new_leo_weak`。25份spec均显式锁定
旧类不叠加、新类叠加；loader逐row验证role/mask/view。parity仅锁定既有前20新类的
sample ID和LEO后IQ哈希。对比方法不受阶段二主方法协议限制。

## 本地证据

| 检查 | 结果 |
|---|---|
| 真实混合role builder测试 | 旧类未调用LEO且IQ不变；新类调用LEO |
| WiSig真实import/signature | PASS |
| 相关pytest | `49 passed` |
| Python编译/`git diff --check` | PASS |
| spec/hash/schema/cache/parity命令 | 25/25 PASS |
| release manifest SHA256 | `b3d5ae74193d423a7dd561734a2f014602910b8ffe4a53001b840e3b48b31313` |

v1/v2技术失败、v3审查拒绝、v4预审前发现policy命名与内容矛盾，均不作为性能结果且未被
v5覆盖。

## N607计划

- CWD/env：`/home/szu2070436088/2510044040/CV-SincNet`；`ssr-gpu`
- run/log：`runs/adv3b02_official_newcount_scale_20260724_v5`；
  `logs/adv3b02_official_newcount_scale_20260724_v5`
- 顺序：哈希/import→真实cache smoke→25 cache→25 new20 parity→base26/base31→
  CSIL/MoPC-HR smoke→700-cell完整矩阵→回收分析。
- 仅P0或两个不同row预测前同一确定性异常允许技术停止；禁止按性能停止。

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting | coverage | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| 待运行 | — | — | — | — | — | — | — | — | — | — | 0/700 | `WAITING_REVIEW` |

v5独立审查为`P0=0,P1=3,P2=2 / NOT_APPROVED`，问题为parity 1300旧硬编码、
spec字节/规范化哈希语义不清和mixed policy/audit未完全fail closed。未发布到N607，
关闭为`LOCAL_RELEASE_REJECTED / NO_PERFORMANCE_RESULT`，修复进入v6。
