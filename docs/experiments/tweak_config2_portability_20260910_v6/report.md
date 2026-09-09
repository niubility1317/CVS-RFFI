# Tweak配置可移植性复现实验V6：mini-batch全量严格hard mining预登记

- run_id：`tweak_config2_portability_20260910_v6`
- 当前状态：`LOCAL_DIAGNOSTIC_VERIFIED_RELEASE_PENDING`
- 唯一launch owner：Codex主Agent

## 依据与单变量假设

- V5已纠正论文Eq.(1)平方L2并使对角准确率由约10—28%提高到27—47%，但其embedding仍塌缩，说明平方公式不是唯一失配。
- 论文IV-A明确要求在mini-batch经过网络后选出“triplets”中`A`比`P`更接近`N`的hard triplets。V5的每anchor一个随机正/负候选是未公开默认，且只覆盖batch内25,088个有效有向三元组中的64个；它不能代表论文所述mini-batch hard-mining集合。
- V6唯一修改：对全部有效有向三元组使用平方L2距离，选择`dAN<dAP`者并平均其triplet loss；无严格hard triplet时跳过optimizer step。batch=64下最多25,088项，使用pairwise距离与boolean mask，不改变样本、标签或梯度权重的定义。
- 不变项：Config2源训练、Config1—4/设备1—10、全记录seeded 75/25、模型拓扑、LeakyReLU/BN、margin=.1、SGD momentum=.9、五个fresh 1epoch LR probe、选中LR fresh100epoch、N=11,718、M=10、图13b/14矩阵与seed。V1—V5均保留且不复用输出根。

## TDD与受控诊断

- RED：新增4样本批测试，枚举得到6个strict-hard有向三元组及平方损失均值9.6；实现前以`ImportError`失败。
- GREEN：在`triplet.py`增加pairwise平方距离和3D布尔mask的全量miner；official runner和通用训练入口均改用它。30项相关Tweak/config测试、三个模块编译通过。
- 本地真实Config2受控诊断（不写正式输出）：以V5相同seed、模型、SGD、LR=.001、物理75/25和`N/M`，只训练3,000个source batch后，10设备×102个M=10 held-out决策取得568/1,020=`55.686%`，loss从0.5870到0.1019。它不能替代完整100epoch/论文矩阵，但首次提供了不再接近随机的、方向正确的V6依据。
