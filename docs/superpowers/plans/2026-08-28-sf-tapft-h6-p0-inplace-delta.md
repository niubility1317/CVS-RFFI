# SF-TAPFT H6 P0原位适配与轻量部署实施计划

> 当前会话直接执行；每个任务按RED→GREEN→聚焦回归推进。不得增加白名单外发布门。

## Task 1：建立原位训练所有权和最小锚点

**Files**

- Modify: `tests/test_target_only_progressive_deploy.py`
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`

1. 先写失败测试：显式原位入口返回同一模型对象；默认入口仍返回副本；原位结果只保存许可参数锚点；非许可参数和buffer不变。
2. 运行聚焦测试并确认RED来自缺少新接口。
3. 提取共享内部实现，新增显式原位入口；删除原位路径的完整模型和完整`state_dict`复制。
4. 运行聚焦测试至GREEN。

## Task 2：稳定前缀/后缀API与FP16安全复核

**Files**

- Modify: `tests/test_target_only_progressive_deploy.py`
- Modify: `code/cvsrffi/target_only_progressive_adapt.py`

1. 先写失败测试：公开前缀/后缀API在随机norm/head扰动及1/10/100步后与完整路径一致；cache字节准确。
2. 新增FP16支持集安全结果结构和失败条件测试，证明只消费support。
3. 实现`encode_h6_prefix`、`forward_h6_suffix`、`H6SuffixTrainer`和FP32 full-path复核。
4. 实现锚点恢复与FP32缓存fallback，运行聚焦回归。

## Task 3：建立checkpoint+delta真实加载闭环

**Files**

- Modify: `tests/test_target_only_progressive_runner.py`
- Modify: `code/cvsrffi/target_only_progressive_runner.py`

1. 先写失败测试：原位训练后delta非零；从基础checkpoint materialize后logits等于适配模型；错误参数名/shape/class IDs fail-closed。
2. 改造delta payload使用训练前许可参数锚点，不再比较已被原位修改的base对象。
3. 增加严格delta loader/materializer和`emit_clean_single_bundle`兼容开关。
4. 运行runner聚焦测试和既有query-closure回归。

## Task 4：实现隔离资源基准

**Files**

- Create: `code/cvsrffi/sf_tapft_deployment_benchmark.py`
- Create: `code/scripts/run_sf_tapft_deployment_benchmark.py`
- Create: `tests/test_sf_tapft_deployment_benchmark.py`

1. 先写失败测试：预热不进入统计；10次样本生成median/P90/max；CPU RSS与CUDA allocated/reserved字段齐全；cache/delta字节来自artifact而非估算。
2. 实现单进程单GPU基准，测量常驻推理基线和适配额外峰值。
3. 输出结构化JSON供正式报告直接解析。
4. 运行聚焦测试。

## Task 5：配置、真实checkpoint smoke与一次独立审查

**Files**

- Create: `configs/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828.json`
- Modify: `tests/test_target_only_progressive_deploy.py`
- Modify: `docs/experiments/stage2_sf_tapft_h6_p0_inplace_delta_20260828_traceability.md`

1. 冻结P0A/P0B/P0C、seed、support句柄、输出路径和技术停止规则。
2. 在`ssr-gpu`环境运行聚焦协议负测和一次ADV3B02 CORE90真实checkpoint无query smoke。
3. 执行一次独立P0/P1正确性审查；若有直接P0/P1，只修复原问题并定点复审一次。
4. 精确stage代码、测试、配置和设计追踪，commit、自动push并核对远端OID。

## Task 6：N607发布、闭合、评分与报告

**Files**

- Create: `automation_reports/CV-SincNet/<run-id>/report.md`（根目录正式报告）
- Create: `docs/experiments/<run-id>_report.md`（Git镜像）

1. 完成直接SSH preflight、资源清点、单一release归档SHA本地/远端比对和远端编译。
2. 发布P0A/P0B/P0C；资源基准保持单GPU单进程，性能prediction使用不可覆盖run root。
3. 启动后核对PID/CWD/cmdline/GPU/log增长；之后只做短连接监控，不因低性能停止。
4. prediction完整后由独立scorer连接Q180 truth，报告BA、floor、逐类准确率、NLL、ECE、配对翻转及全部资源数据。
5. 更新根报告、Git镜像和设计追踪，只stage本轮文件，commit、push并独立核对远端OID。
6. 若P0B晋级，立即建立P1新capsule D0–D4最小预登记并继续执行；否则按瓶颈选择P0C或保留P0A，不跳到P1。
