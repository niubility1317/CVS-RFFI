# AffineJoint 独立分析器的数组缓存

本改动只减少独立汇总器反复读取相同 NPZ 的开销。数学核验、数据权限、方法配置、训练实现和发布配置保持原语义；不缓存“核验通过”结果，也不改变真实训练任务。

实现位置为[汇总器的 StateResolver 和 lane 释放](../tools/summarize_d92_affine_joint_probe.py)，合成检查为[缓存测试](../tests/test_d92_affine_joint_archive_cache.py)。现有[独立汇总接口](D92_AFFINE_JOINT_SUMMARY_20261001.md)仍适用。

## 范围和资源依据

仅修改 StateResolver 的机械缓存，以及 `summarize` 完成每个 lane 后的缓存释放和资源计数输出。核、截距、Schur、伴随、梯度、损失、继承、coverage 和 archive inventory 的核验函数不在本改动范围。

主 Agent 已提供完成目录的数值清单元数据：最大单档案 numeric bytes 为 14,315,408 B，中位约 0.35 至 0.48 MB，99 百分位不超过 7.59 MB。本任务没有读取真实 NPZ、query、outer score 或真实分析结果。这些数字仅用于分析器资源安排，不是算法参数或性能指标；本任务不重算这些清单统计。

## 缓存接口

```python
resolver = StateResolver(root, cache_budget_bytes=64*1024*1024, max_entries=64)
uncached = StateResolver(root, cache_budget_bytes=0)
arrays = resolver(ref)
resolver.finalize()
resolver.clear_cache()
stats = resolver.cache_statistics()
```

默认预算为 **64 MiB 的实际驻留数组字节**，并保留 64 个 entry 的上限。`cache_budget_bytes=0` 关闭驻留；`max_entries=0` 同样不驻留。两个参数是分析器缓存配置，不进入 frozen method config，也不改变训练或 scoring。

LRU 以 archive 相对路径为 entry，命中时移到最近使用端。加载之前依据 manifest 每个数组的 `nbytes` 总和预淘汰旧 entry，直到新 entry 的预计数组字节和 entry 数量都能容纳。驻留时记实际已加载数组的 `nbytes`，不用压缩文件大小估算内存。

超预算单档案会先释放冲突的驻留缓存，随后完整加载、执行原有数组内容检查并返回；该档案不驻留。再次请求会重新加载和检查，不截断、不按需跳过数组、不放宽数值精度或核验条件。

预算约束的是缓存持有的 ndarray 字节，不是整个 Python 进程 RSS。NPZ 解压、复制、数学核验的临时矩阵、调用方持有的数组、Python 对象和文件系统缓存均可能产生额外瞬时内存，不能把 64 MiB 称为实测峰值内存。

## 文件变化和核验语义

每次请求先检查 exact manifest reference、路径边界和文件 size。首次访问记录 `mtime_ns`；随后每次请求都与首次值比较，cache hit 也不例外。miss 在加载和原有数组检查后再次核查文件状态。size 或首次 mtime 变化会作为技术错误拒绝，不能返回过时数组。

数组仍只读。cache hit 只复用已经加载的数值数组，不替代任何数学验证函数。`verified` 是原有 archive inventory 记账集合，不是跳过数学检查的条件。complete `verify_record` 在缓存开启或关闭时应有相同输出和相同数学核验调用次数。

本任务没有增加文件哈希、receipt、authority 链、数据重验、权限流程或性能 gate。文件 size/mtime 检查属于缓存新鲜度条件；不是模型或数据来源证明。

## lane 释放和计数

每个 lane 完成 `finalize()` 和既有 marker/archive 字节核对后，立即 `clear_cache()`。该调用释放缓存字典及其大小记录，把驻留字节和 entry 数清零；保留 manifest、refs、used、verified、首次 mtime 和累计计数。已完成 lane 的数组不会因 `resolvers` 字典继续存在而一直驻留。

每个 `state_archives` row 新增 `cache` 字段，来自 `cache_statistics()`：

| 字段 | 口径 |
|---|---|
| `numeric_byte_budget`、`entry_cap` | 配置预算和 entry 上限 |
| `load_count` | 实际 `np.load` 调用次数，包括失败的加载尝试 |
| `hit_count` | 文件状态核查通过后返回驻留数组的次数 |
| `loaded_numeric_bytes` | 完整加载并通过原有数组检查的实际数组字节累计，可因重复加载而重复计入 |
| `eviction_count`、`evicted_numeric_bytes` | LRU 预淘汰的 entry 数和实际驻留字节；不包括显式 clear |
| `oversized_load_count` | 正预算下超过预算的完整加载次数；零预算关闭缓存另归 uncached |
| `uncached_load_count` | 完整加载后未驻留的次数 |
| `peak_numeric_bytes`、`peak_entries` | 缓存实际驻留数组字节和 entry 数的最高值，不是进程峰值内存 |
| `clear_count`、`cleared_entry_count`、`cleared_numeric_bytes` | 显式释放的次数、entry 数和实际数组字节 |
| `resident_numeric_bytes`、`resident_entry_count` | 查询计数时仍驻留的字节和 entry；完成 lane 后为零 |

load、hit 和 eviction 描述分析 I/O，不是 head fit、triangular solve 或训练 FLOP。数学工作计数保持原口径，不能因缓存命中而减少其核验调用或算法工作计数。

## 合成验证和交付

新增测试覆盖按字节 LRU 预淘汰、entry cap、零预算、oversized 完整验证但不驻留、lane clear 后 inventory 和首次 mtime 保留、预热后 size/mtime 修改拒绝，以及现有 exact reference 的路径/形状/norm 篡改拒绝。

完整 synthetic `verify_record` 分别使用零预算和默认缓存，比较全部输出及 `verify_head`、`verify_companion`、`verify_objective`、`verify_coordinate_map` 的调用次数，并检查 `np.load` 次数下降。另在预热数组缓存后修改 CE 数学证据，确认仍进入数学核验并拒绝。

开发 Agent 只做 AST、UTF-8 和链接静态检查。主 Agent 已在项目环境串行执行缓存测试：9 项通过，另有现有摘要器 10 项回归通过；见[分析验证](D92_AFFINE_ANALYSIS_VALIDATION_20261001.md)。合成检查不代表真实数据分析完成、真实 I/O 加速比例或真实性能结论；真实执行由主 Agent 单独管理。
