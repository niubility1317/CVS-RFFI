# A1-Fast-V2：源域从零训练、运行效率与教师改进实验

## 当前设计（实现前冻结）

用户要求继续优化A1加速版，提升性能、加快训练并发布实验验证，每卡一个实验。以原A1-Fast为主线，不混入R3结构；全部随机初始化，不加载旧checkpoint、冻结teacher或继承其他run状态。沿用seed392005和既有ManySig物理划分，不依据已见target分数调参。性能改进仅是待检验假设。

默认矩阵在空闲GPU4—7各一行；GPU0论文复现、GPU1—3既有R3训练保留。若用户选择8组新实验，则在当前任务退出后接续对应GPU，不并放两个训练任务、不停机腾卡。当前待用户可选安排；无回复采用4组补满空闲卡。

|行|GPU|执行优化|教师改进|预算|
|---|---:|---|---|---:|
|F0_FIXED_BASE|4|原identity_sequential|共同EMA缓存正确性修复|E200|
|F1_RUNTIME|5|合并梯度/路由回读、RC4身份教师|同F0|E200|
|F2_WARM_EMA|6|同F1|成功更新计数驱动EMA启动期平均|E200|
|F3_COVERAGE_KL|7|同F1|归一化后按有效共识覆盖缩放logit蒸馏|E200|

三个对照分别为F1−F0执行优化、F2−F1教师启动、F3−F1蒸馏覆盖，不把不同变化混为单因素。新A1共同修复旧EMA `.data`更新导致的CosFace缓存不刷新的问题，旧运行release不改变。批量DAOT教师、学生并批、R3重构、clean梯度截断不加入本矩阵。

## 不变条件

M/lite_d/dual、160维、6个TX、256点IQ；ManySig equalized=1，source RX=1,3,4,6,8/day=1,2,3，target RX=0,2,5,7,9,10,11/day=0,1,2,3；L/U/V=0.07/0.63/0.30，6300/56700/27000，单一只读V。L batch128、U batch256，E200 final-only，RC4身份从E21启用、E1/E21/E41/E91/E161校准。训练输入契约不变，无历史权重来源，EMA只是本轮学生的内存派生状态。既有opaque最终测试输入不重构、不重验；prediction完成后独立score，target不参与选择或重跑。

## 实现要求与追踪

|ID|来源|要求|目标|状态|验证|
|---|---|---|---|---|---|
|V2-01|用户要求/协议4.4|原A1主线、随机从零、无外部checkpoint、source-only|runner/报告/guard|LOCAL_PASS|真实parser/负测、四行随机初值完全一致|
|V2-02|旧A1已复现缺陷|EMA参数版本更新使缓存有效失效|train_ssdg|LOCAL_PASS|真实CosFace缓存自动刷新与强制刷新一致|
|V2-03|执行瓶颈定位|梯度finite/norm成组回读，保留FP32逐参数操作与Python顺序|runtime模块/训练循环|LOCAL_PASS|CPU/CUDA有限、非有限、裁剪前后逐值一致；整体速度待训练|
|V2-04|RC4真实预算路由|权重一次回读，原排序、预算边界与贪心行为不变|muse_ssdg|LOCAL_PASS|随机mask、预算边界、caps、权重逐项一致|
|V2-05|RC4双弱教师|保持当前批次形状，仅省略未使用域支路|train_ssdg|LOCAL_PASS|B512 FP32/AMP输出、route、BN和RNG精确相等|
|V2-06|EMA数学|d_t=min(d,(t−1)/t)，t只按成功更新递增|EMA更新/状态|LOCAL_PASS|t=1、跳步、上限、计数日志与checkpoint字段；本轮不提供resume|
|V2-07|DAOT归一化数学|q=mean(consensus×confidence)，归一化后缩放orbit_logit|daot_training|LOCAL_PASS|零/部分/全共识、scale状态不变、梯度恰按q缩放|
|V2-08|实验发布|每卡一训练、唯一run、Git与独立读回、最终自动评分|runner/release|LOCAL_VERIFIED|70项聚焦测试通过，独立P0/P1审查PASS；远端记录见run报告|

参数权重、batch、数据、种子和原课程不通过target反馈调整。执行优化只在等价检查通过后保留；不通过的优化在发布前排除并记录，不降低容差掩盖失败。性能项不增加模型前向，性能结论待同row E200及source/target分开分析。

技术失败保留全部产物，低性能不停机，不自动改变参数或重跑。每GPU仅一项训练；既有任务退出且其后续测试完成后才接续。最终报告区分代码已实现、机制实际非零、ARTIFACTS_COMPLETE和科学有效性。
