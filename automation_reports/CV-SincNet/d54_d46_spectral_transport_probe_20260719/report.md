# D54 D46谱transport开发报告

## 1.状态与目标

- 状态：`IMPLEMENTED_AND_TESTED_PRE_RUN`；operator Codex。
- 开发单元：receiver20-1、seed713101、K10/new5、3场景×5fold；本地运行，不访问N607、不运行125。
- 目标：保持D46 classwise LOO的new优势，同时用D53安全谱transport改善old/rain；不新增任何尺度或门控。

## 2.公式与协议

公式与D53相同：`G=U M0^T/||M0||2^2`，`DeltaW=diag(gamma)G W0`，但`W0`来自D46。K1/K2在谱检查前精确D46 fallback。复用`VALIDATED_ONCE p2_min_v1`胶囊；support-only；query/role/quota/count/global/clean/source/dense query graph禁止。

## 3.文件、验证与停止门

- `code/scripts/probe_d54_d46_spectral_contracted_median_transport.py`
- `tests/test_probe_d54_d46_spectral_contracted_median_transport.py`
- D54＋D46联合27/27、`py_compile`通过。
- 必须至少保持D46的new84.67%、min-new73.33%、after81.67%，并改善H/forget/joint/floor之一且不产生场景交换伤害；失败即停止，不扫尺度/clip/第二seed/formal/125。
- 完成后报告7候选、3场景、逐类、15fold、D45/D46/D51/D52/D53比较、20epoch、混淆、谱、量化、资源、artifact SHA，并执行D52–D54三轮回顾。

## 4.执行锁

- 实现提交`0b06631e`；clean worktree`E:\type10-7\code\snapshots\d54wt`；探针SHA`f99427ba8606c0905c7e0f82534cb40519dede4b5c44d80f1dc2ae3d007541a0`；输出启动前不存在。
- exact command与D53报告第7节逐参数相同，仅作以下3处确定替换：脚本为`probe_d54_d46_spectral_contracted_median_transport.py`；arm为`--d54-arm d46_spectral_contracted_median_transport`；输出为`E:\type10-7\automation_reports\CV-SincNet\d54_d46_spectral_transport_probe_20260719\d46_spectral_contracted_median_transport`。所有seal/envelope/manifest/binding hash、runtime、device、mode和candidate-set不变。
