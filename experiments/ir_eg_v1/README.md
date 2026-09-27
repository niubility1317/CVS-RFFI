# IR-EG / BR-IR-EG研究执行包

本包从`acf0a6407c4a2cd114851e9f60d0cd346e4b2c77`中的冻结响应代码复制，在独立分支开发。历史证据目录保持不变。原始文件来源及SHA-256保存在[baseline_manifest.json](acceptance/baseline_manifest.json)，新增代码相对该基线的diff保存在验收目录。

实现包括完整EG原路径、IR参考版、缓存版、加权CE-GGN/两次CG、AdamW冻结二阶矩响应度量、单步事务、BR上一接受梯度历史、源侧隔离诊断及准备态实验矩阵。IR是局部仿射近似，BR是另一种近似算法；不宣称精确隐式AdamW或全网络收敛。

新包各比较臂统一设置`cudnn.deterministic=True`、`cudnn.benchmark=False`，避免原生卷积非确定性被AdamW放大而破坏对齐。未启用全局`deterministic_algorithms=True`，因为原生DAOT包含该模式不支持的CUDA median操作；[核查证据](docs/cuda_kernel_diagnostic.json)说明具体边界。

## 入口

以下命令在本目录、已验证的`ssr-gpu`环境执行。

```text
python code/scripts/check_ir_acceptance.py --device cpu --output acceptance/cpu
python code/scripts/check_ir_acceptance.py --device cuda --output acceptance/cuda
python code/scripts/profile_ir_acceptance.py --device cuda --output acceptance/cuda
python code/scripts/train_response_games.py --config configs/ir_rows/IR_s392005.json --output runs/ir_eg_v1/IR_s392005
```

最后一条只解析配置，返回`VALIDATED_NOT_LAUNCHED`。本次没有远端训练。未来真实训练须绑定合法source manifest及独立输出，并按实验记录执行；本README不提供额外启动授权。不要修改原生200epoch校验来延长运行。

## 资料

- [设计规范](docs/IR_EG_DESIGN_SPEC.md)与[实施计划](docs/IR_EG_IMPLEMENTATION_PLAN.md)：用户提供的原文。
- [需求追溯](docs/traceability.md)：逐项状态和验证。
- [实验协议](docs/experiment_protocol.md)：seed角色、预算审计、源侧收敛与truth-last。
- [CPU验收](acceptance/cpu/result.json)与[CUDA验收](acceptance/cuda/result.json)：机器可读结果；`environment.json`记录实际导入和环境。
- [数学报告](docs/math_report.md)、[集成报告](docs/integration_report.md)与[独立审查](docs/final_review.md)。

真实RFFI的各seed准确率、消融收益、全训练时间和收敛成本尚无结果，不能从合成验收或局部计时推断。参考版用于语义验收；缓存版和BR的实际收益需在未来冻结实验中评估，不自动替换现有默认方法。
