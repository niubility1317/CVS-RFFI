# SF-TAPFT t3.norm＋D92组合设计追踪

## 需求到实现映射

|用户设计项|实现约束|验证证据|
|---|---|---|
|R3双delta思想＋仅`t3.norm`＋D92-in-loop|两个平衡support子集从共同anchor产生delta；只聚合`t3.norm`；support交叉拟合D92风险参与选择；不对D92公式求梯度|R3聚焦单测、support-only审计、四状态结果|
|D0/H6 Compact的`t3.norm`delta＋D92|复用Compact H6缓存训练轨迹；临时target head丢弃；只部署1152个`t3.norm`元素|D0聚焦单测、delta键与字节数、资源记录|
|S02长程`t3.norm`＋D92|复用S02长程配置和support-only选择；只部署`t3.norm`；D92保持E0去RF32|S02聚焦单测、训练步数、资源记录|
|真实注册比较|同一row输出四状态，REG0的新类指标记为N/A；prediction先闭合，scorer后连接truth|四状态prediction目录、scorer汇总|
|最大规模方向|首轮先按最小可证伪矩阵验证2/10/20类；通过后再扩展1/3/5/15及完整确认|本run 27格结果与晋级判定|

## 固定边界

- 协议：`p2_min_v1`、`VALIDATED_ONCE`，只核对`capsule_id/split_id`。
- support：旧类6类×K10；注册态追加`N_new×K10`。
- query：只用于逐样本推理；不更新模型、D92、温度、选择或回滚状态。
- D92：`D92-E0-NORF32`，等先验LDA，注册任务平衡共享协方差公式不变。
- 持久域适应状态：仅identity backbone的`t3.norm.weight/bias`；真实checkpoint路径为`model.id_backbone.t3.norm.weight/bias`，短路径只用于兼容测试模型；domain backbone、target head、optimizer、support cache均不持久化。
- 标签对称：不允许类别ID专属权重、阈值或分支。

## 设计澄清

`D92-in-loop`定义为support-only外层选择环：训练fold拟合域适应候选和D92注册头，heldout support计算注册任务风险，用于选择R3的delta组合。它不是对NumPy/sklearn D92求梯度，也不改变D92-E0公式。
