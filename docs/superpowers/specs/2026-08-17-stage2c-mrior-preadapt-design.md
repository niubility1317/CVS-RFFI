# Stage2-C MRIOR-SDA预适应类增量对照设计

## 目标

在不改变CSIL与MoPC-HR类增量机制、超参数、LEO输入和support/query划分的前提下，先使用MRIOR-SDA对同一target receiver的旧类K-shot support执行监督目标域预适应，冻结所得backbone，再执行新类注册。新结果与`adv3b02_unfrozen_paperfull_ci_20260723_v7`无预适应结果按同一`receiver/seed/K/new-count/scenario`配对。

同时，ERTB-IDR在其独立Target125矩阵中增加严格`DA0_REG1`反事实：不使用target-old support更新状态，只使用不可变Phase1旧类聚合知识与target-new K-shot support注册新类。ERTB实现与发布由分支`codex/stage2c-ertb-da0-20260817`承担。

## 方案选择

### 采用：各方法现有矩阵内的配对反事实

- CSIL/MoPC-HR保留v7的5 receiver、5 seed、4个K和4个new-count，共800个类增量cell、2400个LEO scene row；只新增MRIOR-SDA预适应状态。
- MRIOR-SDA预适应按`receiver×seed×K×scenario`计算一次，共300个冻结backbone artifact；不同new-count和两种类增量方法复用同一artifact。
- ERTB-IDR沿用其5 receiver×5 seed×5 slice的Target125，共125 outer、375 scene row，新增无target-old适应的注册臂。
- 预适应效应只在同方法同row内计算；三种方法若seed或底层package不完全相同，只作描述性汇总，不作严格paired胜负声明。

### 未采用：统一重建三方法共用的新矩阵

该方案需要重新生成并封存三种方法共同的target package、seed和旧类聚合知识，科学上最整齐，但会显著延迟本次用户要求的实验启动，且不是回答“预适应是否改变类增量表现”所必需。

### 未采用：只跑一个receiver的smoke结果作为性能结论

smoke只用于真实checkpoint、状态绑定和技术健康验证，不能用于选择方法或声称性能改善。正式运行仍覆盖完整冻结矩阵。

## 状态与数据流

CSIL/MoPC-HR每个row的状态链为：

1. `DA0_REG0`：原ADV3B02 Phase1基座，未读取target support。
2. `DA1_REG0`：MRIOR-SDA读取source LEO weak cache和当前row的target-old K-shot support，执行200个adapt step；query未打开。
3. `DA1_REG1`：冻结`DA1_REG0` backbone；CSIL或MoPC-HR只用当前row的target-new K-shot support执行原冻结参数的类增量流程。
4. 预测模型及其哈希封存后才打开old/new query；每个query面对全部注册类，truth、role、quota和global reassignment均不可见。

MRIOR预适应artifact键固定为：

```text
rx_<receiver>__seed_<seed>__k_<K>__scene_<leo_scene>
```

该键不得包含`new_count`或下游方法名。artifact必须绑定：原checkpoint SHA、source cache-set SHA、target package seal SHA、old support token SHA、receiver、seed、K、scene、200步参数锁以及query-open=false收据。

## 方法锁

- MRIOR-SDA：200 adapt steps；Adam；learning rate`0.0006`；estimate steps`7`；target CE weight`1.0`；DV-KL weight`0.005`；`mu=0.5`。
- CSIL：3 epochs；batch size`20`；`0.01/(1+0.01×iteration)`；momentum`0.9`；L2 factor`0.05`；KD weight`0.2`；EWC weight`1.0`。
- MoPC-HR：20 epochs；batch size`16`；SGD learning rate`0.01`；momentum`0.9`；weight decay`0.0002`；prototype noise std`0.05`；alpha`0.97`；beta`1.0`；lambda max`1.0`。
- source cache固定为N607既有`adv3b02_three_da_leoweakonly_20260715_v1/phase1_caches/source/cache_set.json`，使用前必须哈希和scope核验；不得从query构造或更新预适应状态。

## 实现边界

- 新建`paper_reproduction/cvs_aligned/adv3b02_mrior_preadapt_ci.py`，只负责MRIOR预适应、artifact序列化和严格加载。
- 新建`paper_reproduction/scripts/build_adv3b02_mrior_preadapt_ci_plan.py`，从已授权v7矩阵生成300个预适应job和800个类增量cell绑定。
- 新建`paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py`，先闭合预适应artifact，再调用现有truth-free类增量predictor。
- 修改`run_adv3b02_paper_full_ci_truth_free_predictor.py`，仅为两个新method ID加载已冻结MRIOR backbone；原CSIL/MoPC路径保持不变。
- 测试必须先证明缺失绑定、错误support token、错误checkpoint/source-cache SHA、query提前打开、跨new-count错误复用和方法名漂移会失败。

## 输出与判定

正式输出保留同row的`old_acc_before`、`old_acc_after`、`seen_new_acc`、`H_old_new`、forgetting、old floor、训练步数、wall time、峰值显存和artifact SHA。预适应收益报告为同方法同row的`MRIOR预适应－无预适应v7`差值及按receiver、seed、K、new-count、scene的分层汇总。

技术健康只判断协议、状态、artifact和完整矩阵闭合；不得因中途准确率差而停止。完整性能结论只能在800/800 cell、2400/2400 scene row闭合并独立评分后形成。
