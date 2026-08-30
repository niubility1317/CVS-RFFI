# Phase1 HCF-DG报告落地追踪表

源报告：`E:\codex\home\attachments\44046c4c-b541-4639-83f8-84cec9433dc5\pasted-text.txt`

状态定义：`pending`、`implemented`、`verified`、`deferred`、`rejected`、`blocked`。

|ID|源章节|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|HCF-001|第三节|单`lite_d`identity backbone和160D`z_id`|`phase1_hcfdg/model.py`|verified|`test_model.py`|推理只保留公共路径|
|HCF-002|第三节|48D receiver/day/channel环境编码器|`phase1_hcfdg/model.py`|verified|shape、stop-gradient测试|各16D|
|HCF-003|第三、八节|环境物理键只进入环境分支|`phase1_hcfdg/model.py`、`metrics.py`|verified|输入隔离及5因子真实卫星smoke|不得形成身份捷径|
|HCF-004|第四节|rank-4公共—特定低秩头和0.5 dropout|`phase1_hcfdg/model.py`|verified|训练/推理分流测试|选模只用`W0`|
|HCF-005|第五节|严格排除query domain的LODO原型分类|`phase1_hcfdg/losses.py`|verified|数值单测+A4真实路径smoke|核心DG目标|
|HCF-006|第五、八节|内容键及无近邻回退|`phase1_hcfdg/losses.py`、`metrics.py`|verified|匹配/回退测试|V2启用|
|HCF-007|第六节|单融合层有界低秩反事实传输|`phase1_hcfdg/model.py`|verified|边界及梯度测试|V2启用|
|HCF-008|第六节|CF-ID、CF-INV、CF-ENV和style loss|`phase1_hcfdg/losses.py`|verified|逐项数值测试|V2启用|
|HCF-009|第六节|receiver→day→channel/联合swap课程|`phase1_hcfdg/config.py`、`trainer.py`|implemented|`code/tests/phase1_hcfdg/test_config.py`|Task1冻结A6/A7课程模式；trainer集成后再verified|
|HCF-010|第七节|分层HDRO与父级收缩|`phase1_hcfdg/losses.py`|verified|组风险测试|关闭Group CE/FISHR/REx|
|HCF-011|第九节|`6×4×4=96`矩形batch|`phase1_hcfdg/sampler.py`|verified|sampler测试+A4真实路径smoke|至少3个receiver|
|HCF-012|第九节|0.65/0.225/0.125 episode比例|`phase1_hcfdg/sampler.py`|verified|长序列频率及完整因素留出测试|receiver/day/channel轮换，support不含query因素的其他组合域|
|HCF-013|第十节|六类以内核心总损失及冻结权重|`phase1_hcfdg/config.py`、`losses.py`|verified|配置及损失组合测试|关闭项精确零，启用项缺输入fail closed|
|HCF-014|第十节|条件receiver对抗和`z_env`TX对抗|`phase1_hcfdg/model.py`、`losses.py`|verified|GRL上限及标签测试|GRL≤0.05|
|HCF-015|第十节|关闭旧loss soup|`phase1_hcfdg/config.py`、launcher|verified|配置负测及旧ADV3B03回归|旧入口行为不变|
|HCF-016|第十一节|70%clean+30%mixed_orbit单前向|`phase1_hcfdg/satellite.py`、`trainer.py`|verified|调用次数、比例及A0/A1/A2/A4/A5真实路径smoke|仅HCF-DG专用入口覆盖现行默认|
|HCF-017|第十一节|channel参数监督`z_channel`|`phase1_hcfdg/satellite.py`、`model.py`|verified|标签schema与真实metadata测试|CFO/phase/SNR/multipath/elevation完整进入环境分支|
|HCF-018|第十二节|4000-update V1和6300-update V2|`phase1_hcfdg/config.py`、`trainer.py`|verified|精确update测试|V1/V2计数与StageBudget一致|
|HCF-019|第十二节|双LR、5%warm-up、cosine、margin课程|`phase1_hcfdg/trainer.py`|verified|边界update测试|末端LR约`1e-6`|
|HCF-020|第十二节|50%后冻结Sinc和首个时域块|`phase1_hcfdg/trainer.py`|verified|requires_grad测试|固定点，不读外部域|
|HCF-021|第八、十二节|U_s只用于廉价环境学习|`phase1_hcfdg/trainer.py`|verified|数据访问负测|不做身份伪标签|
|HCF-022|第十五节|A0–A5快速筛选矩阵|launcher、matrix config|verified|launcher合同测试及多候选真实路径smoke|36行不可变matrix rows|
|HCF-023|第十五节|A6–A9深层外推矩阵|launcher、matrix config|implemented|`code/tests/phase1_hcfdg/test_config.py`|Task1提供24行报告顺序和6300预算；launcher门槛后再verified|
|HCF-024|第十五节|A10–A12残差矩阵|launcher、matrix config|implemented|`code/tests/phase1_hcfdg/test_config.py`|Task1提供18行父候选绑定和v2_passed门槛；launcher门槛后再verified|
|HCF-025|第十六节|5-fold×3-seed确认及最终8-seed|launcher、scorer|implemented|`code/tests/phase1_hcfdg/test_config.py`|Task1冻结三seed注册表和矩阵行schema；确认launcher/scorer后再verified|
|HCF-026|第十七节|五类新诊断指标|`phase1_hcfdg/metrics.py`|verified|公式及结构化same-row单测|含`R_drift`和margin|
|HCF-027|第十三节|完整资源遥测|`phase1_hcfdg/trainer.py`、report|verified|字段与有限值测试+smoke artifact|GPU-hours为实测字段|
|HCF-028|项目协议|Phase1 source-only角色和外部域隔离|launcher、trainer负测|verified|禁用字段及真实source-LORO smoke|与Phase2无关|
|HCF-029|项目协议|final checkpoint clean+三LEO场景闭合|launcher、evaluator|verified|A0/A1/A2/A4/A5真实checkpoint smoke|每场景独立JSON+log|
|HCF-030|第十九节|HCF-DG-V1完整实现|上述V1文件|verified|完整聚焦测试+多候选smoke|结构验证版|
|HCF-031|第十九节|HCF-DG-V2完整实现|上述V2文件|pending|V2聚焦测试+smoke|正式主候选|
|HCF-032|第十九节|HCF-DG-V3小残差实现|独立residual模块|pending|门控/bypass测试|只在V2通过后|
|HCF-033|第二十节|按报告顺序推进，不堆叠未过机制|matrix policy|verified|配置及launcher门控测试|先发布A0–A5；A6–A12仍受前序source gate约束|
|HCF-034|声明边界|只声明代理数据Phase1 DG|正式报告|pending|逆向审计|不得冒充Phase2/在轨/Phase3|

## 当前计数

- verified：27
- implemented：4
- deferred：0
- rejected：0
- blocked：0
- pending：3

当前未闭合项：`HCF-031`、`HCF-032`和`HCF-034`。其中V2和小残差按报告顺序受A0–A5 source gate约束；正式性能声明等待N607矩阵结果。HCF-DG专用入口已隔离单前向增强，旧ADV3B02/ADV3B03回归保持通过。
