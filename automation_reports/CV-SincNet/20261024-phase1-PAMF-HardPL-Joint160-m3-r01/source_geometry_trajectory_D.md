# 源域互补分支诊断 D

VERIFIED：Joint／Complement／Path 各三个既有 own E30 学生，18 个固定 source role 行，CPU2 用时 20.490 s，分析代码 `3e7a7293bb1d16134bcbba718dca21c0d5d908cc`。模型、归一化、EMA、原型、分类头更新均为0；未读目标或U真值，无新GPU任务。

三个方法都复用同 seed 原 E20 的伪类、0.95接受掩码和同伪类跨RX伙伴；U_s每seed960、V每seed720。下面只报告伙伴有效子集：U指标是对固定原E20伪类的一致率，V是真实类别正确率，不能把两者混称为U正确率；所有样本及七干预逐seed详情在JSON中。CE按每个seed实际有效样本数加权。

| 方法 | 源域角色 | 干预 | 有效样本×3seed | 一致率／V准确率（%） | 标签CE |
|---|---|---|---:|---:|---:|
| JOINT_E30 | U_s | joint | 2703 | 100.000000 | 0.004193 |
| JOINT_E30 | U_s | without_reference | 2703 | 99.926008 | 0.011575 |
| JOINT_E30 | U_s | without_signature | 2703 | 94.339623 | 0.279053 |
| JOINT_E30 | U_s | core_only | 2703 | 89.789123 | 0.454561 |
| JOINT_E30 | U_s | reference_swap | 2703 | 99.963004 | 0.006410 |
| JOINT_E30 | U_s | signature_swap | 2703 | 99.963004 | 0.006424 |
| JOINT_E30 | U_s | late_swap | 2703 | 99.963004 | 0.007995 |
| JOINT_E30 | V | joint | 1991 | 99.849322 | 0.012872 |
| JOINT_E30 | V | without_reference | 1991 | 99.799096 | 0.020326 |
| JOINT_E30 | V | without_signature | 1991 | 95.680563 | 0.263526 |
| JOINT_E30 | V | core_only | 1991 | 90.507283 | 0.429980 |
| JOINT_E30 | V | reference_swap | 1991 | 99.849322 | 0.016307 |
| JOINT_E30 | V | signature_swap | 1991 | 99.849322 | 0.016707 |
| JOINT_E30 | V | late_swap | 1991 | 99.849322 | 0.019331 |
| COMPLEMENT_E30 | U_s | joint | 2703 | 100.000000 | 0.001970 |
| COMPLEMENT_E30 | U_s | without_reference | 2703 | 99.926008 | 0.007282 |
| COMPLEMENT_E30 | U_s | without_signature | 2703 | 99.223085 | 0.037306 |
| COMPLEMENT_E30 | U_s | core_only | 2703 | 97.521273 | 0.083632 |
| COMPLEMENT_E30 | U_s | reference_swap | 2703 | 99.963004 | 0.003636 |
| COMPLEMENT_E30 | U_s | signature_swap | 2703 | 100.000000 | 0.003043 |
| COMPLEMENT_E30 | U_s | late_swap | 2703 | 100.000000 | 0.004381 |
| COMPLEMENT_E30 | V | joint | 1991 | 99.849322 | 0.011331 |
| COMPLEMENT_E30 | V | without_reference | 1991 | 99.748870 | 0.016953 |
| COMPLEMENT_E30 | V | without_signature | 1991 | 98.895028 | 0.041106 |
| COMPLEMENT_E30 | V | core_only | 1991 | 97.589151 | 0.085071 |
| COMPLEMENT_E30 | V | reference_swap | 1991 | 99.849322 | 0.014391 |
| COMPLEMENT_E30 | V | signature_swap | 1991 | 99.849322 | 0.013678 |
| COMPLEMENT_E30 | V | late_swap | 1991 | 99.849322 | 0.016707 |
| PATH_E30 | U_s | joint | 2703 | 100.000000 | 0.001940 |
| PATH_E30 | U_s | without_reference | 2703 | 99.963004 | 0.003771 |
| PATH_E30 | U_s | without_signature | 2703 | 99.519053 | 0.034080 |
| PATH_E30 | U_s | core_only | 2703 | 98.557159 | 0.047550 |
| PATH_E30 | U_s | reference_swap | 2703 | 99.963004 | 0.003443 |
| PATH_E30 | U_s | signature_swap | 2703 | 100.000000 | 0.002928 |
| PATH_E30 | U_s | late_swap | 2703 | 100.000000 | 0.004014 |
| PATH_E30 | V | joint | 1991 | 99.849322 | 0.010671 |
| PATH_E30 | V | without_reference | 1991 | 99.849322 | 0.013124 |
| PATH_E30 | V | without_signature | 1991 | 98.945254 | 0.037446 |
| PATH_E30 | V | core_only | 1991 | 98.543446 | 0.051575 |
| PATH_E30 | V | reference_swap | 1991 | 99.849322 | 0.013610 |
| PATH_E30 | V | signature_swap | 1991 | 99.849322 | 0.012992 |
| PATH_E30 | V | late_swap | 1991 | 99.849322 | 0.015852 |

几何指标、绝对散布、两种归一化及分支相对范数保留完整JSON；散布比例不是因果解耦证明。这里的冻结特征重组也不保证物理可实现性或目标收益。该诊断不改变已发布Complement／Path／ReliableAs配方、父权重、卡位或测试矩阵。
