# ADV3B02官方CSIL/MoPC-HR新类数量扩展实验v2

## 基本信息

- 实验ID：`adv3b02_official_newcount_scale_20260724_v2`
- 日期：2026-07-24
- 操作者：Codex主代理；N607发布子代理`no_leo_n607_release`
- 当前状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LANDED`
- 目标：减少新类数量并覆盖论文增量数量，正式新类support/query均叠加LEO弱星地信道。
- 前序失败run：`adv3b02_official_newcount_scale_20260724_v1`

## v1失败继承与唯一修复

v1已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
25份spec哈希通过，但12次可观测缓存尝试均因冻结ManyTx路径不存在而失败，
cache成功0，parity/base/smoke/full均未启动。

只读服务器证据确认实际文件为：

`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl`

v2仅修正以下内容：

- run/log/cache/parity/plan路径中的run ID由v1改为v2；
- `manytx_pkl`补上`CV-SincNet/`路径段。

方法实现、类集合/顺序、receiver、seed、K、LEO场景、CSIL base26、
MoPC-HR base31、优化参数、官方`drop_last`和矩阵规模均不改变。

## 冻结矩阵

| 分支 | 方法 | 新类数 | 论文口径 | base容量 | cell | 场景行 |
|---|---|---|---|---:|---:|---:|
| CSIL-reduced | CSIL官方仓库语义 | `1,3` | 既有正式`new20`覆盖论文每增量20类 | 26 | 200 | 600 |
| MoPC-paper-scale | MoPC-HR官方仓库语义 | `1,3,5,10,25` | `3,5,10,25`对应官方增量数量 | 31 | 500 | 1500 |
| 合计 | 两方法 | — | — | — | 700 | 2100 |

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713101-713105`
- K：`1,5,10,20`
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- CSIL smoke：`new1/K1,new3/K20`
- MoPC-HR smoke：`new1/K1,new25/K20`
- 不运行无LEO矩阵。

## 固定类列表

- 旧6类：`14-10,14-7,20-15,20-19,6-15,8-20`
- 新25类：
  `1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6,13-19,18-14,20-4,20-16,11-10`
- 所有较小新类集合使用严格嵌套前缀。

## v2本地规格

- release目录：
  `paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v2_release`
- cache spec：25份
- cache build命令：25条
- parity命令：25条
- scope：`external_comparison_registered`
- ManyTx：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl`
- target cache root：
  `runs/adv3b02_official_newcount_scale_20260724_v2/target_cache_new25`
- parity root：
  `runs/adv3b02_official_newcount_scale_20260724_v2/cache_parity`
- reference root：
  `runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/target`

## 发布硬门槛

1. 25份spec规范化哈希全部匹配；
2. 25个cache set全部生成；
3. 25个parity收据全部PASS，旧6+前20逐项sample ID和信道后IQ哈希一致；
4. CSIL base严格为容量26；MoPC-HR重建base严格为容量31和classifier`[31,160]`；
5. 两方法smoke的prediction/scorer/cell receipt全部完整；
6. 真实smoke收据生成formal authority后才允许700cell矩阵；
7. P0单次，或两个不同row在prediction前同一确定性异常指纹时，
   停止dispatch并终止身份匹配的run-owned进程组；
8. 不因低准确率或其他性能值提前停止；
9. fresh-run自动重试未授权。

## 预期路径

- N607 cwd：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v2`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v2`
- base31：`runs/adv3b02_official_newcount_scale_20260724_v2/base31/official_repo_base_state.pt`
- CSIL输出：`runs/adv3b02_official_newcount_scale_20260724_v2/csil_reduced_leo`
- MoPC-HR输出：`runs/adv3b02_official_newcount_scale_20260724_v2/mopc_paper_scale_leo`

## 本地验证和审查

沿用v1同一release实现：

- release实现commit：`3b8e9988f2213df1435b4a63e5e88b0e7b77a8ff`
- 报告commit：`5d0a7fd0efbc99bd1d3c9f30fc566a3140f14861`
- 相关测试：`41 passed`
- Python编译：PASS
- `git diff --check`：PASS
- 独立release审查：`P0=0,P1=0,P2=0 / APPROVE`

v2需对修正后的25份spec重新完成规范化哈希、schema、ManyTx实际路径和
“除run ID/ManyTx路径外无语义漂移”检查，并单独Git提交后方可发布。

## 结果表占位

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting/per-class old | loss摘要 | coverage | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| v2-cache-wave1 | — | — | — | — | — | — | — | — | — | — | — | `0/8 cache` | `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT` |

## N607运行闭环（2026-07-24）

### 最终状态

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

v2在首个8-cache wave中触发重复确定性异常指纹。余17套cache、25套parity、
base26/base31、CSIL/MoPC-HR smoke和700-cell完整矩阵均未启动；未读取任何性能值，
未创建或启动v3。

### 发布门与启动收据

| 检查项 | 结果 |
|---|---|
| direct preflight | PASS |
| v1活跃进程 | 0 |
| v2 run/log预检查 | 均不存在 |
| ManyTx真实路径 | `test -f .../CV-SincNet/Dataset_WigSig/ManyTx.pkl` PASS |
| 磁盘 | `/home`可用7.5TB |
| delta commit | `016ec587911dacedc2662a68b1e717446064138e` |
| 方法release commit | `3b8e9988f2213df1435b4a63e5e88b0e7b77a8ff` |
| v2 manifest字节SHA256 | `a4f2cd1777ed51a1a1a723136a5699ec48a3956904100ef69ffb23ff34911166`，远端匹配 |
| cache spec | 25/25 canonical hash PASS；25/25 schema PASS |
| v1/旧错误路径残留 | 0 |
| 既有运行脚本哈希 | 与v1冻结release值一致 |

首wave固定执行manifest前8条命令，最大并发8，每张GPU一个v2 cache builder。
主PID为`1303191`，PID、启动时间、完整cmdline和CWD
`/home/szu2070436088/2510044040/CV-SincNet`已真实写入
`cache_wave1_launch_receipt.txt`。直属builder PID为
`1303195,1303197,1303199,1303201,1303202,1303205,1303207,1303208`，
设备分别为`cuda:0`至`cuda:7`。

### 系统性技术失败

| 观测项 | 结果 |
|---|---|
| launched/completed/succeeded/failed | `8/8/0/8` |
| cache set | 0 |
| prediction/score | 0/0 |
| 停止后v2活跃PID | 0 |
| 停止后GPU | 8卡均0%利用率；另一个正式run的既有上下文约693MiB/卡，未干预 |
| 日志 | `cache_wave1.log`，SHA256=`2808e0596c0b64f2cd93c771c2460cbb8d875ac793434549e1929ff98c99127b` |
| launch receipt | SHA256=`6e9adbc23b27fc45a4e854b1c81deb8b91603012825ca853f2e86afcc2c4225e` |

8个不同spec均在生成cache/prediction前产生同一异常：

```text
TypeError: _build_wisig_dataset() got an unexpected keyword argument
'exclude_source_record_indices'
```

只读根因定位确认：

- 同步后的`code/scripts/build_cvs_leo_weak_iq_cache.py`调用
  `_build_wisig_dataset(...,exclude_source_record_indices=...)`；
- 本地Git承载版本`code/export_spaceborne_features.py`支持该参数，
  SHA256=`9e0ed8cefd8c652abd0b57a0e3baeebddc03cef16f215e3cf1004f1270552662`；
- N607现存`code/export_spaceborne_features.py`不支持该参数，
  SHA256=`dfcb3bac4b8974ecc9b41b73fb0b4c3c020b80f867b59870f78181f7a8257ee7`。

因此v2失败原因是缓存builder与其远端运行时依赖版本不一致，不是ManyTx路径、LEO数据、
CSIL或MoPC-HR方法性能问题。调度器和8个子进程均自行退出，无需发送SIGTERM/SIGKILL；
所有run/log产物保留。

### 阶段状态

| 阶段 | 状态 |
|---|---|
| preflight/sync/hash/schema | PASS |
| cache wave1 | FAIL，8/8同指纹 |
| 剩余17套cache | NOT_STARTED |
| parity 25/25 | NOT_STARTED |
| base26/base31 | NOT_STARTED |
| CSIL/MoPC-HR smoke | NOT_STARTED |
| formal 200+500 cells | NOT_STARTED |

fresh-run自动重试未授权。若需继续，必须把完整缓存builder运行时依赖纳入本地审查、
Git冻结和精确同步清单，再使用新的非覆盖run ID；不得恢复或覆盖v2。
