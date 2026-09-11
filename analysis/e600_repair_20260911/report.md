# E600数值风险修复与重新发布

用户授权：修复并重新发布两个E600实验，版本由本次选择。新run为`a1_e600_repair_s392005_20260911_r1`；旧run、权重和日志完整保留。

## 已定位的故障与边界

旧`X2_E600`在E260触发`RC4_SYSTEMIC_NONFINITE_BATCH_GUARD`，仅完成259条epoch记录，没有E600最终checkpoint。完整源域记录显示E73开始显著退化、后续接近六类随机水平；这不是训练正常完成。

旧run首个异常包是E1/batch1：loss有限，但`id_backbone.sinc.low_hz_`有24个NaN梯度，FP16 GradScaler从65536降为32768。本地不加载任何历史权重，使用旧E600配置与实际训练入口，在合成source-only数据上复现了相同参数、相同NaN数量和相同scale下降；FP32对照完成4次实际AdamW更新，没有首个异常包或scale下降。证据见[精度对照](precision_reproduction/result.json)、复现脚本`analysis/reproduce_e600_precision.py`及完整只读诊断`diagnosis.json.gz`。

此次采用FP32绕开已复现的FP16首批梯度溢出路径。**未证明该首批异常就是E73塌缩及E260系统性异常的充分原因；未完成600epoch稳定性验证。**不修改模型归一化、GRL或伪标签规则来猜测根因；保留现有非有限梯度技术保护。

## 固定的两行矩阵

|row|公共配置|差异|
|---|---|---|
|B1_FP32_E600|DAOT+RC4、runtime优化、FP32、连续cosine学习率、600epoch|关闭ECRS|
|B1_CLEAN_ECRS_FP32_E600|同上|clean跨接收机ECRS，weight=0.05，margin=0.2|

两行都关闭R3，seed=392005、从随机初始化开始，teacher为本run的student EMA；不读取旧checkpoint、teacher或其他继承状态。两行除候选名和ECRS权重外的训练options完全相同。选择依据是已有source证据、已复现的数值风险和可解释的配对对照，不按本次目标域得分排名选择。用户允许选择版本，因此这是一组明确记录方法变化的新探索实验，不能作为旧X2的严格单变量修复比较。

600epoch沿用200参考时钟按比例拉伸，包括L/U/V角色、阶段和clean+三种LEO场景；学习率使用已有continuous模式，避免尾段额外骨干倍率压低。已有数据capsule和split未改变，不重新验证数据。

## 输入权限和结果使用

训练为source-only，`from_scratch=true`、`a1_scratch_only=true`，baseline/teacher checkpoint均空。实际入口检查将任何checkpoint读取或target-loader构造直接设为失败。协议仍按当前`项目.md`执行；本次没有新增数据角色、query适配或truth训练权限。

E300至E600每20epoch按既有流程独立输出完整target prediction后评分，预定16个评价点，不依据目标结果调整、早停或重跑。历史target结果已经可见，因此这些是`exploratory_periodic_target`，不宣称新的干净泛化确认或科学晋级；600epoch完整权重和全部672000条/评价点产物闭合之前不能声称完成。

## 验证与资源

- 本地`ssr-gpu`实际CUDA训练入口：两行各4次成功AdamW更新；无checkpoint读取、无target输入、无首个数值异常。ECRS关闭行没有ECRS调用，开启行有有效anchor和正损失。[执行结果](local_execution.json)
- 聚焦pytest：`test_a1_e600_repair.py`、`test_a1_fast_v2_matrix.py`、`test_a1_extended_budgets.py`共8项通过。覆盖scratch/FP32/600epoch/连续LR、唯一差异、日程、固定GPU兼容性、繁忙GPU等待、尚无CUDA上下文的新子进程占位。
- 新dispatcher从GPU0–7选择低占用空位，所有CUDA进程保守计入每卡最多2个名额；无空位即等待。没有终止其他任务或为新实验抢占名额。单launch owner，唯一release和run-root禁止覆盖；发布前先Git提交和远端OID核对。
- 独立P0/P1审查已完成，无阻断项；检查了实际GPU环境映射、context前预留、AMP开关、scratch权重拒绝、唯一输出路径和prediction先于scorer。审查未运行第二轮GPU检查，不把静态审查当作长期稳定性证据。远端发布读回结果在交付补充中记录。

本地短程通过仅证明配置和已复现异常的缓解，不保证长期性能、显存峰值或所有晚期目标均无异常。同一技术指纹若在本次修复后再现，保留产物并报告，不盲目重复重启；低准确率不触发停机。
