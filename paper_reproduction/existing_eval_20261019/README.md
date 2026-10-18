# 历史三个冻结模型补充评估

本入口仅评估 `20260930-phase1-three-original-manysig-m3-r01` 中已经由源验证选择并冻结的 ASKNet、WaveMLP、DIFL 模型，不训练、替换或另拟合分类头。父实验从零训练，DIFL 教师和学生来自同一源数据契约，旧目标未评分；当前请求授权补齐目标评分。源 provenance 由原 launch plan、原 resolved config、物理角色及实际 checkpoint 检查，不使用旧索引状态推断运行或权重合规。

ASKNet/WaveMLP 使用原 ManySig round0：9源RX的day1/2，非均衡前100包，70/30源划分，seed2023；目标RX为1-1、1-19、2-1的day3/4，共3600包。直接复用已冻结的 IB-RFF round0 匿名query导出，其物理包和均功率归一化一致。DIFL 为均衡day1、每TX/RX由seed300随机取200包、源80/20 seed32；目标为相同3个RX的day1，共3600包，不套用原始IQ或day3/4。

每个模型先对真实source小批作checkpoint无query烟测，随后预测并独立评分，再转下一模型。使用原父release的模型构造函数与原checkpoint。WaveMLP logits在返回tuple第0项，DIFL logits在第1项，ASKNet直接返回logits；严格保留原头。预测器仅读匿名IQ/ID、源契约元数据和固定权重，不读取query真值；真值仅由准备器保存和预测完成后的独立scorer读取。

测试是3个既有固定行的补测，单折/单seed，不能替代完整论文矩阵。DIFL fc1特征取点尚未核准，ASKNet/WaveMLP属于结构基线；所有结果逐行报告，不跨任务混排或在看到分数后重训/重跑。作者公开完整矩阵均值不能直接作为此单折同条件目标。
