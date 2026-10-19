# 非PAMF冻结模型统一推理成本预登记

本run补齐已固定设计中的推理成本，不新增识别训练或目标测试。69个冻结模型配置：F0 native full/CE 24、ManyRx native full/CE 24、generic ResNet-18五路线15、历史ResNet-18/50六个；每配置沿用原三个模型seed。原性能范围仍是186训练实例及6补测，69成本行单独计数。

## 测量定义

严格加载原source-selected／fixed-final checkpoint及原联合训练默认头，不重拟合头。统一现有Python3.8、torch1.11、NumPy1.23；历史ResNet-18真实checkpoint在此环境已严格加载PASS，不改变原训练或预测环境。输入仅为驻留GPU的FP32零IQ，batch×2×256，不读真实IQ、标签或truth；关闭TF32，固定cuDNN设置。

batch1和128各预热100次、同步计时1000次。报告各model seed的中位数、P95、吞吐和峰值显存，再做三seed等权均值及样本SD。1000次计时不能当1000次独立训练。F0、ManyRx及generic分表；相同部署图的其他fold、源数量及标签率不逐项计时，不增加准确率结论。

MAC口径为Conv1d/2d/3d点积和mm/bmm/addmm的FLOPs/2，不含BN、激活、池化、softmax及其他逐元素工作；twice_macs只表示该乘加口径的两倍，不称完整FLOPs。身份参数排除不执行的辅助头；峰值显存包含完整加载模型仍注册的辅助参数。整个测量不得更新模型参数或buffer。

## 依赖与资源

一个owner和不可覆盖root。CPU真实IB checkpoint smoke通过后立即等待；扩展suite实际PID退出，且completion.json证明原24＋新162训练＋6补测全部ANALYZED，才开始成本GPU工作。仅从物理0/1选择一张空闲卡，整批固定同卡；外部owner占用时等待，不借其他GPU，不停止、重启或热改健康任务。

首个技术错误或检测到计时竞争时保留产物、标PARTIAL，停止未启动成本行；不自动复用失败root或按性能重跑。此run明确只测合成输入成本，零新训练，不追加目标预测评分。

## 版本、路径、验证与完成

当前PLANNED。实际code commit由publisher写入release_commit.txt及本地package.json；精确命令、环境/CWD、checkpoint来源、GPU、输出/log和逐行69配置见experiment.json及local_artifacts/phase1_external_dg_cost_20261019/runtime.json。预期产物：smoke.json、state.json、69个ROW/result.json、summary.json、report.md、completion.json。

一次独立P0/P1审查PASS；四项聚焦检查通过，覆盖三seed矩阵、资源/预算/目标输入拒绝、可手算Conv/Linear MAC、上游真实退出及完整矩阵依赖、计时重复不混淆model seed。Py3.8和remote脚本语法通过。证据verification.json和uniform_cost_environment_probe.json保存在本地artifact目录。

发布只做一次归档传输SHA比较和一次remote compile；post-launch核实实际PID/CWD/argv、CPU smoke及等待状态，SUBMITTED回执单独不足为VERIFIED。最终69行必须各有两batch完整测量、原头/契约及未更新参数证据，并完成汇总；等待进程存在或单个smoke通过不代表成本完成。整体目标还需要原性能矩阵和本成本产物全部完成。
