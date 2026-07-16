# qKNNV42 K1 support信赖域适应追踪表

更新时间：2026-07-16

## 目标与边界

本轮只解决`K=1`下target梯度适应相对`P4 identity`轻微负迁移的问题。基座固定为`ADV3B02＋P4`，统一K候选更新`id_gate.0＋joint_proj.0`的rank8 LoRA，共6,400个参数；K1为5epoch、5个optimizer step。所有训练、缩放选择和回退判断仅使用已注册`LEO_weak`support及其标签；不得读取query、truth sidecar、old/new角色、类别配额、query批统计或clean/clean-derived信号。

|ID|需求/证据|实现或artifact|状态|验收|
|---|---|---|---|---|
|K1T-01|`JG_R8_LR020`是K10 source winner，但不是target注册结论|既有v21同row证据|verified|88.8354%与source receiver/K10/6类绑定|
|K1T-02|v23所有K1梯度候选相对P4 identity为负|既有v23同row证据|verified|`JP8_LR005`为-0.0321pp，不写成适应成功|
|K1T-03|基座固定`ADV3B02＋P4`|candidate capsule/manifest SHA|pending|拒绝checkpoint/P4漂移|
|K1T-04|只更新`id_gate.0＋joint_proj.0`rank8|adapter audit|pending|精确6,400参数，其余参数冻结|
|K1T-05|5epoch、K1自然5step|loss/resource receipt|pending|不得超过5epoch/5step|
|K1T-06|训练与缩放选择仅使用support|support-only API与spy测试|partially_verified|缩放API不存在query/role/quota参数；enrollment open ledger待注册主线验证|
|K1T-07|未参与梯度的接收侧增强用于信赖域验证|augmentation split receipt|pending|fit/eval变换集合不重叠且均从LEO_weak support派生|
|K1T-08|按View×类计算留一Viewprototype margin|`k1_support_trust.py`|verified|类置换等变；不读取old/new角色|
|K1T-09|从预注册缩放网格选择最大安全`alpha`|support scale decision与LoRA state scaler|verified|仅support规则选择；只缩放`lora_b`，避免`alpha²`|
|K1T-10|无安全非零缩放时回退`alpha=0`|negative测试|verified|组合LoRA residual精确为0；真实P4 merge parity待集成|
|K1T-11|加入新类前后使用同一适配状态|enrollment artifact|blocked|等待注册主线接入|
|K1T-12|K1适应后相对P4 identity非负、相对strict direct显著为正|独立scorer paired结果|blocked|target正式矩阵前不得声明达标|
|K1T-13|不同K遗忘率不劣于identity|K1/5/10/20 matched ledger|blocked|候选锁定后确认，不回流选参|
|K1T-14|轻量资源|resource receipt|pending|≤50k参数、≤20epoch、≤256KiB；本轮更严为3,840/5/5|
|K1T-15|无Oracle/配额/global assignment|API/schema/预测测试|pending|逐样本all-registered argmax|

## 方法合同

1. 使用v21已证实K10最强的`joint_gate,rank8`统一候选，不按K切换层组；缓存冻结backbone产生的`feat_id/feat_pa/frozen feat_joint`，训练时仅重算`id_gate＋joint_proj`小子图。
2. 将每个注册类的3个`LEO_weak`View分成训练变换和未参与梯度的验证变换。验证变换只能进行身份保持的低幅接收侧扰动，不接触clean样本。
3. 对缩放后特征`z(alpha)`，用其他View构造同类原型，按`(view,class)`计算正确类与最强负类的cosine margin。
4. 在固定网格`alpha in {0,0.125,0.25,0.5,0.75,1}`中选择满足平均margin不下降、最差组下降不超过预注册容差、特征漂移不超过上限的最大非零值；若均不满足则选择0。
5. `alpha`只缩放FP16 LoRA delta；合并后每query新增LoRA MAC为0。预测侧不得重新拟合或改变`alpha`。

## 输入与输出

|环节|输入|输出|
|---|---|---|
|K1快速适应|密封K1 support、ADV3B02 checkpoint、P4、固定增强/优化配置|原始6,400参数FP16 delta、5epoch loss trace|
|support信赖域|P4特征、候选特征、support标签、未参与梯度的增强View|选择的`alpha`、逐View×类margin、回退原因|
|enrollment发布|缩放后的delta、old-only/all-registered support状态|密封delta/head/resource receipt|
|独立验证|不可变prediction artifact与truth sidecar|注册前后old、floor、new、H、遗忘、direct/identity配对差|

## 当前结论边界

当前方法是待实现假设。它可以把K1负适应上界限制为安全回退，但不能在没有独立target query结果时声称带来正收益或达到目标。正式性能仍须由同一run的注册前后指标和独立scorer给出。

## 2026-07-16本地机制验证

- 新增`paper_reproduction/cvs_aligned/k1_support_trust.py`和`tests/test_k1_support_trust.py`。
- `ssr-gpu`下定向pytest为7/7通过，随后`py_compile`通过。
- 已验证：API无query/role/quota入口、留一View逐类margin的类置换等变、最大安全非零缩放、全非零候选不安全时回退0、`alpha=0`组合LoRA residual为0。
- 尚未验证：真实ADV3B02＋P4＋JG-R8 artifact的逐alpha前向、真实target性能和注册后遗忘。
