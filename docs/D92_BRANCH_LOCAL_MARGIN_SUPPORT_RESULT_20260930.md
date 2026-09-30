# BranchLocalMargin完整support结果

LocalMargin在标准物理OOF中较LocalRidge小幅提高，但单样本proxy没有稳定改善，因此不替代LocalRidge，不启动其query基准。完整4800组、8行已由独立汇总工具校验，无query或源域样本访问。所有差值为百分点，不宣称统计显著；proxy很小的负差不表示普遍失败。

旧类固定6个，新增类0/2/5/10/20，K=1/5/10/20。practical residual沿用既有post_sync/noeq数据，未重建信道。标准OOF每个物理held样本只出现一次，K是分折前总support数；proxy每次训练1个anchor，parent K仅描述原support池，不等同正式K1。

三阶段口径：真实A适应前query准确率、真实B仅旧类适应后query准确率和正式C注册后query准确率均N/A，本轮未启动query评分。下表提供support诊断的N0旧类参照和注册后完整旧/新值；N0与N>0分别拟合，不能宣称继承B状态的顺序训练。表中不把N0参照差值伪装成已验证三阶段适应或遗忘指标。

## 相对LocalRidge的逐K变化

| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |
|---|---:|---|---:|---:|---:|---|
| standard | 5 | local_ridge | +0.632 | +0.542 | +0.660 | True |
| standard | 10 | local_ridge | +0.358 | +0.541 | +0.511 | True |
| standard | 20 | local_ridge | +0.365 | +0.642 | +0.551 | True |
| oneshot_proxy | 5 | local_ridge | +0.024 | -0.020 | +0.011 | False |
| oneshot_proxy | 10 | local_ridge | -0.028 | -0.023 | -0.026 | False |
| oneshot_proxy | 20 | local_ridge | -0.049 | -0.033 | -0.046 | False |

差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。
真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。


## physical_oof完整K×新增类数

| K | 新类数 | 方法 | 旧类N0参照 | 注册后旧类 | 注册后新类 | 新旧绝对差 | H |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 0 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 2 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 2 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 5 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 5 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 10 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 10 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 20 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 20 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 5 | 0 | local_ridge | 63.361 | 63.361 | N/A | N/A | N/A |
| 5 | 0 | local_margin | 63.847 | 63.847 | N/A | N/A | N/A |
| 5 | 2 | local_ridge | 63.361 | 60.056 | 38.833 | 21.222 | 43.261 |
| 5 | 2 | local_margin | 63.847 | 60.861 | 39.625 | 21.236 | 44.183 |
| 5 | 5 | local_ridge | 63.361 | 55.250 | 43.217 | 12.033 | 47.036 |
| 5 | 5 | local_margin | 63.847 | 55.847 | 43.467 | 12.381 | 47.488 |
| 5 | 10 | local_ridge | 63.361 | 51.736 | 42.508 | 9.228 | 45.507 |
| 5 | 10 | local_margin | 63.847 | 52.528 | 42.983 | 9.544 | 46.182 |
| 5 | 20 | local_ridge | 63.361 | 48.653 | 38.863 | 9.790 | 42.187 |
| 5 | 20 | local_margin | 63.847 | 48.986 | 39.513 | 9.474 | 42.779 |
| 10 | 0 | local_ridge | 70.701 | 70.701 | N/A | N/A | N/A |
| 10 | 0 | local_margin | 71.243 | 71.243 | N/A | N/A | N/A |
| 10 | 2 | local_ridge | 70.701 | 67.437 | 47.083 | 20.354 | 53.989 |
| 10 | 2 | local_margin | 71.243 | 67.722 | 47.562 | 20.160 | 54.444 |
| 10 | 5 | local_ridge | 70.701 | 62.889 | 51.383 | 11.506 | 55.689 |
| 10 | 5 | local_margin | 71.243 | 63.271 | 51.933 | 11.338 | 56.260 |
| 10 | 10 | local_ridge | 70.701 | 59.972 | 49.892 | 10.081 | 53.841 |
| 10 | 10 | local_margin | 71.243 | 60.354 | 50.408 | 9.946 | 54.339 |
| 10 | 20 | local_ridge | 70.701 | 57.625 | 47.481 | 10.144 | 51.380 |
| 10 | 20 | local_margin | 71.243 | 58.007 | 48.098 | 9.909 | 51.901 |
| 20 | 0 | local_ridge | 74.649 | 74.649 | N/A | N/A | N/A |
| 20 | 0 | local_margin | 75.024 | 75.024 | N/A | N/A | N/A |
| 20 | 2 | local_ridge | 74.649 | 71.649 | 53.406 | 18.243 | 60.485 |
| 20 | 2 | local_margin | 75.024 | 71.972 | 53.948 | 18.024 | 61.000 |
| 20 | 5 | local_ridge | 74.649 | 67.854 | 57.258 | 10.596 | 61.766 |
| 20 | 5 | local_margin | 75.024 | 68.184 | 57.754 | 10.430 | 62.211 |
| 20 | 10 | local_ridge | 74.649 | 65.330 | 56.150 | 9.180 | 60.068 |
| 20 | 10 | local_margin | 75.024 | 65.809 | 56.950 | 8.859 | 60.741 |
| 20 | 20 | local_ridge | 74.649 | 63.139 | 54.546 | 8.593 | 58.162 |
| 20 | 20 | local_margin | 75.024 | 63.465 | 55.278 | 8.187 | 58.732 |

## support_oneshot_proxy完整K×新增类数

| K | 新类数 | 方法 | 旧类N0参照 | 注册后旧类 | 注册后新类 | 新旧绝对差 | H |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 0 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 2 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 2 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 5 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 5 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 10 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 10 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 1 | 20 | local_ridge | N/A | N/A | N/A | N/A | N/A |
| 1 | 20 | local_margin | N/A | N/A | N/A | N/A | N/A |
| 5 | 0 | local_ridge | 51.132 | 51.132 | N/A | N/A | N/A |
| 5 | 0 | local_margin | 51.163 | 51.163 | N/A | N/A | N/A |
| 5 | 2 | local_ridge | 51.132 | 47.833 | 29.177 | 18.656 | 30.689 |
| 5 | 2 | local_margin | 51.163 | 47.872 | 29.167 | 18.705 | 30.712 |
| 5 | 5 | local_ridge | 51.132 | 43.806 | 31.929 | 11.876 | 34.379 |
| 5 | 5 | local_margin | 51.163 | 43.788 | 31.942 | 11.847 | 34.400 |
| 5 | 10 | local_ridge | 51.132 | 39.917 | 30.906 | 9.010 | 33.109 |
| 5 | 10 | local_margin | 51.163 | 39.955 | 30.848 | 9.107 | 33.089 |
| 5 | 20 | local_ridge | 51.132 | 36.066 | 26.902 | 9.164 | 29.369 |
| 5 | 20 | local_margin | 51.163 | 36.104 | 26.880 | 9.224 | 29.389 |
| 10 | 0 | local_ridge | 52.103 | 52.103 | N/A | N/A | N/A |
| 10 | 0 | local_margin | 52.164 | 52.164 | N/A | N/A | N/A |
| 10 | 2 | local_ridge | 52.103 | 48.762 | 27.683 | 21.079 | 31.420 |
| 10 | 2 | local_margin | 52.164 | 48.721 | 27.597 | 21.124 | 31.328 |
| 10 | 5 | local_ridge | 52.103 | 44.977 | 32.481 | 12.495 | 36.003 |
| 10 | 5 | local_margin | 52.164 | 44.987 | 32.480 | 12.507 | 35.993 |
| 10 | 10 | local_ridge | 52.103 | 40.981 | 31.144 | 9.838 | 34.133 |
| 10 | 10 | local_margin | 52.164 | 40.945 | 31.106 | 9.839 | 34.118 |
| 10 | 20 | local_ridge | 52.103 | 37.108 | 27.133 | 9.975 | 30.261 |
| 10 | 20 | local_margin | 52.164 | 37.062 | 27.166 | 9.895 | 30.275 |
| 20 | 0 | local_ridge | 52.293 | 52.293 | N/A | N/A | N/A |
| 20 | 0 | local_margin | 52.290 | 52.290 | N/A | N/A | N/A |
| 20 | 2 | local_ridge | 52.293 | 48.956 | 28.829 | 20.127 | 32.977 |
| 20 | 2 | local_margin | 52.290 | 48.926 | 28.707 | 20.219 | 32.864 |
| 20 | 5 | local_ridge | 52.293 | 45.123 | 32.404 | 12.719 | 36.302 |
| 20 | 5 | local_margin | 52.290 | 45.058 | 32.380 | 12.677 | 36.254 |
| 20 | 10 | local_ridge | 52.293 | 40.939 | 31.088 | 9.850 | 34.311 |
| 20 | 10 | local_margin | 52.290 | 40.888 | 31.080 | 9.808 | 34.287 |
| 20 | 20 | local_ridge | 52.293 | 37.143 | 27.204 | 9.939 | 30.557 |
| 20 | 20 | local_margin | 52.290 | 37.094 | 27.228 | 9.866 | 30.558 |

## 成本与后续

完整运行墙钟11398.950秒，包含4个方法对照及诊断开销，不能直接称为单次部署训练耗时。LocalMargin累计52800次候选拟合、204068488次行块求解；三个ridge对照共158400次分解。优化器步数是行块求解次数，不是神经网络梯度步。

实测逐阶段成本（秒；工作量之和不等于并发墙钟）：

| scope | 方法 | 指标 | 次数 | 总和 | 均值 |
|---|---|---|---:|---:|---:|
| support_oneshot_proxy | local_margin | fit_seconds | 42000 | 12374.718170 | 0.294636 |
| support_oneshot_proxy | local_margin | score_seconds | 42000 | 2659.995095 | 0.063333 |
| support_oneshot_proxy | local_margin | solve_seconds | 42000 | 12215.648330 | 0.290849 |
| support_oneshot_proxy | local_ridge | fit_seconds | 42000 | 172.465370 | 0.004106 |
| support_oneshot_proxy | local_ridge | score_seconds | 42000 | 2658.884439 | 0.063307 |
| support_oneshot_proxy | local_ridge | solve_seconds | 42000 | 4.652204 | 0.000111 |
| support_oof | local_margin | fit_seconds | 10800 | 12938.196308 | 1.197981 |
| support_oof | local_margin | score_seconds | 10800 | 937.559712 | 0.086811 |
| support_oof | local_margin | solve_seconds | 10800 | 11938.454608 | 1.105412 |
| support_oof | local_ridge | fit_seconds | 10800 | 1016.321789 | 0.094104 |
| support_oof | local_ridge | score_seconds | 10800 | 935.794186 | 0.086648 |
| support_oof | local_ridge | solve_seconds | 10800 | 5.034320 | 0.000466 |

新增源域payload为0；未更新encoder、未传输新地面训练权重。当前成本来自N607 CPU环境，不是星载硬件实测；功耗和星载峰值显存N/A。完整资源统计、内存和payload字段保留在summary.json，不能用参数少推断总算力低。

下一步结合原LocalRidge注册机制诊断，设计能够在新类注册时共同训练新旧竞争、明确继承旧适应状态的候选；不把margin目标的小幅OOF提升泛化为所有K有效。下一版仍必须报告真实A/B/C、K与新增类数，缺失阶段N/A。

证据：[完整汇总](../automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-margin-support-m4-r01/results/support_summary/summary.json)、[完整分层CSV](../automation_reports/CV-SincNet/20260929-phase2-d92-branch-local-margin-support-m4-r01/results/support_summary/by_parent_k_newcount.csv)。运行5b7319acd，分析67e452901。
