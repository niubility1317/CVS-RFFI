# D111-LOO-GAT形式化知识包实现记录

## 结论

`D111-B01`至`D111-B08`已完成，独立Terra Max复审结论为`CODE_ACCEPTANCE_GO / P0=0 / P1=0 / P2=0`。该结论只证明知识包实现与设计一致，不构成域适应性能证据。当前没有新增真实性能结果。

## 信任与数值闭环

实现先把`core`、逐类basis、逐域逐类中心和radius统一为连续FP32副本；签名摘要和后续计算只读取这组副本，消除了“摘要忽略FP64低位、计算却使用低位”的不一致。来源manifest必须由发布版本固定公钥表中的Phase1 authority作Ed25519签名，并同时绑定aggregate内容、checkpoint和离线生成阶段。

组件生成后仍是`PENDING_OUTER_JOINT_SEAL`。外部authority签署的第二层Ed25519封印覆盖component content root、manifest SHA、NPZ SHA、checkpoint、method lock、class registry、source aggregate、source manifest以及生成代码／配置SHA。loader复验固定公钥和全部payload后，才在只读运行时视图中给出`FORMAL_D111_OUTER_JOINT_SEALED`。生产公钥表当前为空，因此不存在把测试签名误当正式资产的通路。

`B`与`epsilon`使用FP16向上取整，构建端和载入端均验证解码值不低于未量化上界。共享Top3投影按class-projector的规范字节顺序累加；测试使用六个真实不同的逐类子空间，验证非恒等类别置换下的共享projector、`B`和`epsilon`不变，`g`和`v_g`随类别同步置换。

## 验证

|范围|结果|说明|
|---|---:|---|
|D111定向测试|10 passed|含RFC8032验签向量、空keyring拒绝、签名/内容/协议/成员篡改、谱隙、不可变、资源和类置换|
|相邻知识包回归|208 passed|`D111`、Phase1 center-lowrank bundle、D105 Phase1 bundle|
|编译与格式|通过|`py_compile`与`git diff --check`无错误|
|独立复审|P0=0/P1=0/P2=0|代码验收GO；未修改文件、未访问N607|

资源回执按实际INT8/FP16数值成员计算payload字节，并给出生成矩阵乘上界、160维特征分解维度、Phase2解码MAC和临时数组峰值上界。它不冒充实测RSS或真实性能。

## 后续边界

真实formal资产仍需release authority登记source/outer公钥并完成外部签名。该外部前置只阻止G0/G1发布，不再阻止`D111-S01`至`D111-S03`评分核研发。下一步只实现固定32步Weiszfeld、primal-dual gap、3/5共识资格、单位质量anchor/support混合和不变性测试；不调rank、包络、权重，不启动N607。
