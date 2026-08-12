# D92 NewGuard v2量化保护诊断报告

## 1. 身份与目标

- run ID：`d92_e0_full_bidirectional_newguard_hard11_20260812_v2_diag`
- 状态：`LOCAL_VERIFIED / READY_TO_LAND / NO_PERFORMANCE_RESULT`
- 目的：仅重跑冻结K>2真实checkpoint truth-free smoke，保留v1丢失的部署保护细分收据，判定D42量化后具体失败约束。
- 声明：`DIAGNOSTIC_ONLY / NO_PERFORMANCE_RESULT`；不启动shard、不运行scorer、不读取query性能。
- 协议：`p2_min_v1`；复用`VALIDATED_ONCE`，不重复数据验证。

## 2. v1证据与假设

v1在第一个K>2 performance outer `rx_7_7__seed_713106__k_10__new_5`的三个场景均安全回退E0，原因`deployment_protection_failed`。FP32原始保护必然已通过，D42量化调用成功，部署新类行byte-exact，但旧收据丢弃了其余部署约束值。

可证伪假设H1：逐旧类D42量化主要破坏严格等式闭包（新support内部残差、旧组零和或`envelope==tau`），而部署后的旧类tail和新类margin不等式仍安全。若任一部署tail或新类margin小于冻结负容差，则H1被否证。

## 3. 唯一改动与验证

- 诊断提交：`863beed8`。
- 改动只把已经计算的完整`_protection_receipt`合并进exact-E0 fallback；不改候选、阈值、不等式、回退决定、query接口或输出头。
- TDD：诊断字段缺失先RED；Task1四组聚焦测试、`py_compile`和`git diff --check`通过。
- 独立P0/P1复核：`APPROVE`，`P0=0/P1=0`；确认只保留诊断字段，保护判定与exact-E0回退字节均未改变。

## 4. 发布三件套

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `d92_newguard_runtime_closure_863beed8.tar.gz` | 5,033,720 | `1dcc24f2b5401789602b832937caa50080f1e9569cae108773a15654a50c8e53` |
| `stage2_d92_full_bidirectional_newguard_hard11_v1.json` | 6,345 | `d41b116b2bb7fb8be1fb56512e9e47e7915e94b5fae57776ced9c875ceb5f523` |
| `launch.sh` | 3,065 | `e3ff12b678254cefa91b81a7cde279fbbd87b2959d9501e1952ec27876715403` |

## 5. N607冻结路径与命令

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v2_diag`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_bidirectional_newguard_hard11_20260812_v2_diag`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_bidirectional_newguard_hard11_20260812_v2_diag`
- exact command：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_newguard_source_snapshot_20260812_v2_diag &&
nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

launch只执行prepare和一个K>2 smoke，并记录smoke exit code；明确不含shard循环。fresh retry=`false`。

## 6. 预期证据

取回三个场景的以下支持集侧字段：部署新support内部残差、部署旧组零和残差、逐旧类tail margin变化、部署新类最小margin变化、部署旧类包络最大变化及其相对`tau`误差、保护容差与最终fallback。任何性能字段均不读取、不报告。
