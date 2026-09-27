# IR-EG / BR-IR-EG开发交付

已完成独立研究执行包，未替换默认基线或启动远端训练。

实现覆盖原生完整EG、IR参考版/缓存版、BR历史预测、事务与恢复、源侧诊断、实验配置和登记。

| 验证 | 测试数 | 失败 |
|---|---:|---:|
| CPU IR验收 | 110 | 0 |
| CUDA IR验收 | 110 | 0 |
| 旧版相关回归 | 106 | 0 |

实际命令：

```text
python code/scripts/check_ir_acceptance.py --device cpu --output acceptance/cpu
python code/scripts/check_ir_acceptance.py --device cuda --output acceptance/cuda
python -m pytest tests/test_acceptance.py tests/test_native_joint.py tests/test_response_games.py tests/test_response_reaudit.py -q -o addopts= --junitxml=acceptance/legacy_regression.xml
python code/scripts/profile_ir_acceptance.py --device cpu --output acceptance/cpu
python code/scripts/profile_ir_acceptance.py --device cuda --output acceptance/cuda
```

两次验收失败均已保留：`acceptance/cuda_initial_failure/`记录原生非确定性卷积；`acceptance/cuda_resume_device_failure/`记录测试反序列化导致AdamW步数计数器设备变化。后者改用现有生产CPU加载入口，生产逻辑本来就保留正确设备。所有修复按原容差验证。

独立审查提出的冻结参数、导入来源、非有限损失和旧配置兼容性问题均已修复并有针对性测试。无性能晋升结论。

合成profiling只报告本地实际调用、用时和内存；包含求解器snapshot、CG及外部提交，排除真实数据/教师/评估。冷启动和并行干扰的早期计时另存，不用于比较。当前测量不能证明完整训练提速。

24行计划配置均`launch=false`。尚未完成真实checkpoint探针、三seed训练、真实数据消融、source收敛成本及目标预测评分。逐项范围见[需求追溯](traceability.md)。
