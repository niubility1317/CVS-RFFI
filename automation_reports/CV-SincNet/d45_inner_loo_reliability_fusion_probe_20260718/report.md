# D45冻结outer-B20的head-only LOO可靠度融合探针报告

## 1.身份与目标

- 实验ID：`d45_inner_loo_reliability_fusion_probe_20260718`
- 操作者：Codex`/root`
- 当前状态：`PREREGISTERED_PENDING_LOCAL_15_FOLD_PROBE`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问5个confirmation seeds，不生成125结果。

D44固定1:1融合取得当前最高development seen-new/H并改善最低after-old，但rain after-old/forgetting和1个量化翻转失败。D45只改变一个因素：由support内部预测可靠度连续确定full/block全局权重，其他特征、B20、LDA、RMS、量化器、数据和外部门不变。full组件直接复用D42锁定的sklearn解并仅删除类公共仿射项，避免在高维低秩inner折中用另一求解器重构系数；block组件继续使用D43的3-block结构。

## 2.数学机制

定义严格限定为`frozen_outer_b20_head_only_loo`：B20只在outer fit上训练一次，此后冻结；inner折仅重拟合full/block LDA head和各自RMS。因此inner CE只是合法support-derived融合权重，不宣称为全链路nested无泄漏泛化估计；真正无泄漏评价仍由outer-held physical rows承担。

对每个outer fit、每个组件`g∈{full,block}`，在合法fit support内部按support-row rank做leave-one-out。每个inner折只用其余`K−1`个physical support/类拟合组件和RMS，再对held rank的全部类打分。先用稳定log-softmax计算逐fold、逐类inner-held交叉熵`CE_{g,c}`，再令：

`L_g=(1/C)Σ_c CE_{g,c}`，`w_g=exp(−C L_g)/Σ_h exp(−C L_h)`。

最终完整support fit的score为：

`δ_D45=w_full δ_full/s_full+w_block δ_block/s_block`。

具体锁定`log_evidence_g=−C L_g`，再对两个log-evidence做稳定softmax；不clip、不加temperature、不设阈值。乘数`C`来自逐类正确似然的乘积，不是搜索温度。权重对当前fit所有类共享；不读取outer-held、query、old/new角色、handle或场景ID，不扫描weight、threshold、rank、lr、epoch或shrinkage。K1无法形成inner折，但full/block都退化为同一unit-covariance分类器，因此固定1:1是决策等价回退；K2的inner-train为K1，必须由两组件CE证据得到`1:1±1e-12`，否则fail closed。

## 3.资源口径

每阶段包含2个完整组件fit；K>1时再包含每组件K个head-only leave-one-rank-out fit。before/final合计`4+4K`次closed-form LDA；本development outer fit的K=8，因此每条D45行计36次LDA。MAC按`2[L(C_old,K)+L(C_all,K)]+2K[L(C_old,K−1)+L(C_all,K−1)]`逐fit组闭合，不能用单一平均值粗乘。metric B20仍只训练一次、20 epoch/20 optimizer steps；最终只持久化一套融合int8/FP16 state，query MAC仍按一套state。host FP64 covariance峰值继续标记未实测。

## 4.预注册门

相对D42 original固定基准，必须同时满足：协议/lifecycle/source/ground/state/resource闭包；inner train-held互斥且所有support row恰好held一次；类别置换对称；权重有限、严格为正且和为1；K1/K2均为1:1；before在首次new support读取前物化并保持不可变；聚合before/after/new/H与最低before/after/new、joint均不退化，forgetting不增加，至少一个final floor严格改善；clear/low-elev/rain各自before/after/new/H/joint不退化且forgetting不增加；before/final int8-FP32 argmax变化和margin翻转均为0；final old→new/new→old/new-new不超过26/10/18。探针强制identity且禁用full-K10，即使全过也需另行实现正式candidate。

## 5.文件与执行计划

- 探针：`code/scripts/probe_d45_inner_loo_reliability_fusion.py`。
- 单测：`tests/test_probe_d45_inner_loo_reliability_fusion.py`及D42–D44回归。
- 追溯：`analysis/d45_inner_loo_reliability_fusion_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d45_inner_loo_reliability_fusion_probe_20260718\inner_loo_class_likelihood`。
- 环境：本地`ssr-gpu`串行执行；当前不访问N607。

本地预运行验证：

- `python -m py_compile`通过。
- D42–D45定向回归`67 passed`，pytest退出码0。
- pytest退出后出现本机临时目录`pytest-current`的既知`WinError 5`清理噪声，不影响测试结论。
- 初版随机低样本测试暴露D43 full-control在高维低秩折中重复求解的argmax漂移；已改为full组件直接复用D42锁定解，仅删除类公共仿射项，随后回归通过。
- 输出verifier不信任自报布尔字段：它从held索引重算exact-once覆盖，从逐fold CE重算逐类CE、macroCE、log-evidence与权重，并从四个fit组的行数、类数和D42公式重算全部LDA MAC；对应tamper测试通过。

当前仅完成预注册和代码草案，不构成性能正结果、正式candidate、125结果或目标完成。
