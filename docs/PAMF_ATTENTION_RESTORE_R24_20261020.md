# R24：全分辨率局部卷积与注意力先验原型

状态：LOCAL_PROTOTYPE_VERIFIED，尚无源训练、远端发布或性能结论。R23 正常运行，不更改其代码、矩阵或资源。该备选依据 R17/R22 的源域结构与结果提出，不使用任何正式测试成绩设计。

用户要求比较多种网络。已比较的 U-Net、TCN、时频、扩散/流匹配及解析约束各有独立记录。本轮新增另一结构假设：逐样点全分辨率 token 的内容相关注意力，配合局部卷积，可能补充多尺度卷积的修正能力；这只是待证伪假设，不预先宣称保留了更多 TX 指纹。

备选矩阵为同一冻结 R07 上下文下的 `unet_control` 与 `attention_prior`。均为三个不共享先验、每步两次 warm CG、coarse 锚定、最后输出同一份 IQ；原 PAMF 再一致计算 FFT/PA/统计/时域量。只继承合规 R02/R07，不继承 R23 coarse 更新或任何后续训练 prior。两种结构先验头均零初始化，因此完整初始输出相同；隐藏参数形状不同，不能称完整初始权重相同。

注意力先验接收原九通道 `[received,coarse,current,normalized innovation,log variance]`。9→128 逐点投影，全长256，不做 patch、下采样或跨样本混合。固定0.1倍正弦位置量；6个 pre-norm block，每个含4头样点自注意力、5点深度卷积＋逐点混合、扩展率2的SiLU MLP，三条残差各缩放0.5；最终LayerNorm和零初始化128→2头。三个stage分别持有独立参数。训练使用非重入activation checkpoint减少保留的内部激活，增加重算；推理不重算。没有Dropout、BatchNorm或持久输入统计。位置量是固定buffer，不使用RX、场景、TX、F、clean或其他样本信息。

本地原型只验证数学与接口。后续真实候选须另行完成完整runtime/schema、训练/评分路由、一次独立P0/P1、登记/commit、真实checkpoint首步与不可覆盖发布，再可训练。当前不把这些未来工作当作R23阻断条件。预期仍为全6300L/56700U、TX无标签模拟配对IQNMSE、0训练好F/TX/target、原冻结F50/defaulthead源七视图资格和条件正式确认；最终预算与矩阵在任何新运行前固定。无准确率目标放宽，正式low urban≥70%，其他五场景各≥79.6761904762%。

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|T24-01|多网络设计|全分辨率注意力＋局部卷积，单IQ输出|models.py|verified|形状/零头/参数检查|报告思路扩展，并非原六类精确复现|
|T24-02|样本独立|注意力只在单包时间轴，无跨包或持久更新|models.py|verified|逐包/置换/状态检查|不使用全局query关系|
|T24-03|可训练性与成本|checkpoint前后输出/梯度一致，三个stage可达|models.py/tests|verified|实际反传/逐层梯度|增加重算成本，参数量实报|
|T24-04|无监督权限|仅接入原冻结R07、clean只训练目标|完整runtime/training|deferred|尚未接入真实训练|当前随机输入原型不等于数据契约验证|
|T24-05|实验验收|登记/真实checkpoint/源与条件正式评分|完整pipeline|deferred|尚无运行|不阻断R23|
|T24-06|身份真实性|逐TX/RX/day及独立物理族/多seed|后续合格实验|deferred|尚无性能证据|不能由接口测试证明|

当前3 verified、0 pending、3 deferred、0 rejected、0 blocked。13项聚焦检查及compile通过，仅证明本地原型接口/梯度/单包属性，不证明真实数据或识别效果。N607 Torch2.1和batch256内存/速度未验证，完整pipeline和一次候选审查留待正式登记前完成；不阻断R23。

实际随机初始化结构参数：{"unet_control": {"parameters": 4665073, "trainable": 2507526, "position_buffer_elements": 0}, "attention_prior": {"parameters": 4863217, "trainable": 2705670, "position_buffer_elements": 98304}}。注意力三个prior共2705670可训练参数，比UNet三个prior2507526增加约7.9%；全部模型参数分别4863217与4665073。新增位置buffer共98304个FP32值（393216字节），不作为可训练参数隐去，也不等于实测峰值显存。
验证命令：`python -X utf8 -m pytest tests/test_pamf_attention_restore.py -q --maxfail=1`、`python -X utf8 -m compileall -q experiments/pamf_attention_restore`。先观察缺少实现的失败，再实现并通过13项检查；未运行源数据训练或新正式评分。
