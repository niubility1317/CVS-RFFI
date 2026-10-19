# 非PAMF扩展队列：数据和checkpoint准备结果

核验时间：2026-10-19T12:43:36+08:00。代码8c50d20ec。状态：VERIFIED；dispatcher实际存活，WAITING_PARENT，当前没有新增GPU实验行。

## 数据构建

17个源数据契约、18个测试任务的既有builder结果均已写出，manifest为VALIDATED_ONCE；本次只读回最小handle和元信息，没有重验物理数据、重算成员hash或再建验证链。F0继续复用原L_s/V和七视图。

| 协议 | 源RX数 | L_s | U_s（隐藏不用） | V | 主测试包数 |
|---|---:|---:|---:|---:|---:|
| F0 | 5 | 6300 | 56700 | 27000 | 168000 |
| F1 | 5 | 6300 | 56700 | 27000 | 168000 |
| F2 | 5 | 6300 | 56700 | 27000 | 168000 |
| chrono | 5 | 6300 | 56700 | 27000 | 168000 |
| ManyRx | 16 | 6533 | 58828 | 28023 | 123021 |
| count-c1-m1-natural | 1 | 1260 | 11340 | 5400 | 168000 |
| count-c1-m3-natural | 3 | 3780 | 34020 | 16200 | 168000 |
| count-c1-m3-fixed | 3 | 1260 | 36540 | 16200 | 168000 |
| count-c1-m5-fixed | 5 | 1260 | 61740 | 27000 | 168000 |
| count-c2-m1-natural | 1 | 1260 | 11340 | 5400 | 168000 |
| count-c2-m3-natural | 3 | 3780 | 34020 | 16200 | 168000 |
| count-c2-m3-fixed | 3 | 1260 | 36540 | 16200 | 168000 |
| count-c3-m1-natural | 1 | 1260 | 11340 | 5400 | 168000 |
| count-c3-m3-natural | 3 | 3780 | 34020 | 16200 | 168000 |
| count-c3-m3-fixed | 3 | 1260 | 36540 | 16200 | 168000 |
| label01 | 5 | 630 | 62370 | 27000 | 168000 |
| label05 | 5 | 3150 | 59850 | 27000 | 168000 |

chrono另含源RX未见D3的30000包冻结诊断；它与源训练物理ID互斥，不能冒充未见RX主测试。所有任务包含clean和六practical。ManySig共享存储共288000物理ID：原168000观测复用，只新增120000；ManyRx为123021个目标ID的七视图。相同物理ID在各fold使用相同received观测，不把七视图计成七倍独立物理样本。

## 真实checkpoint检查

23个训练结构组合通过真实L_s一批梯度与默认头checkpoint严格往返检查，覆盖6／10类、1／3／5／16源RX所需路线；不读query，不把smoke权重用作正式初始化，也不将其计入完成的研究模型数。

SDG的该项检查覆盖身份前向和默认头checkpoint；其原生增强／对比训练目标保留在原训练函数，不将此检查描述为完整SDG目标验收。原F0的SDG完整训练已在运行。

6个历史ResNet-18／50 checkpoint以原torch2.1环境完成真实严格加载；scratch来源、源契约、既有默认头联合训练和CPU无query前向均PASS。未重新训练或替换分类头。

## 调度与未完成项

新dispatcher实际PID／CWD／argv核实，CUDA_VISIBLE_DEVICES为空；完成数据和检查后自动等待原24行dispatcher。父进程实际退出且24行全部ANALYZED后，新队列才可在物理GPU0/1逐行启动；其他owner占用时等待，不借用其他卡或干预健康进程。

原队列最新为9行ANALYZED、2行RUNNING；新增162个训练和6个补测尚未启动GPU行。当前证据证明准备闭合及依赖等待，不证明全部实验完成。后续仍须逐row训练／冻结／全部任务预测／独立truth-last评分，并由最终分析复算覆盖、预算和三seed统计。

证据：local_artifacts/phase1_external_dg_20261019/preparation_complete_verified.json；原SDG full第二seed完成证据为同目录sdg_full_seed02_completion.json。原始IQ、checkpoint、日志、预测和私有运行配置不进入Git。
