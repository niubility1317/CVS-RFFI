# 条件随机差分交换：未发布原型

用户最新要求只优化 SIG_TIME＋RDrop，重点为 RDrop 机制。当前 experiments/cvs_sigtime_exchange20 将两次随机特征拆为中心与差分，再跨包交换差分并改变 CE，是独立特征增强假设，因此在登记和发布前暂缓，publisher 已明确拒绝执行。没有新远端运行、真实 IQ、checkpoint、训练或目标访问。

原型的 CPU/CUDA 合成检查各前向896包，26个负例与5个资格布尔情形通过；198初值张量三条件一致，实际增强后输入捕获、默认头复用、梯度和随机数预算已检查。SELF 与 RDrop 的 FP32 目标最大差 CPU4.7683716e−7/CUDA0，所选四组梯度最大差 CPU0/CUDA4.0978193e−8。其发布草案和 source 验证仍未经独立运行审查，不能当成已启动或已批准实验。

文献边界：RDrop约束随机子模型的输出分布；[SFA](https://homepages.inf.ed.ac.uk/thospeda/papers/li2021sfa.pdf) 已探索特征高斯扰动与类条件协方差，[DSU](https://arxiv.org/abs/2202.03958) 已探索统计量不确定性，[Manifold Mixup](https://proceedings.mlr.press/v97/verma19a.html) 已探索隐特征插值。条件差分交换的具体区别不构成全球创新证明，亦无跨RX效果证据。本轮不继续这条路线。

接下来的唯一新假设收敛到 RDrop 的反向传播：保持 CE/前向/骨干/默认头不变，对比原 KL、KL仅作用特征、匹配投影范数的特征KL，以及只向两次特征差分反传的KL。须独立最小预登记与源RX留出验证，不把当前84.340079%的目标分数当作选权重、系数或重跑反馈。
