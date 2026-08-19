# ADV3B02-MUSE-SSDG需求追踪

设计权威：`docs/superpowers/specs/2026-08-19-adv3b02-muse-ssdg-design.md`

|ID|来源|规范化要求|实现目标|状态|验证|备注|
|---|---|---|---|---|---|---|
|MUSE-001|用户确认设计|以ADV3B02双表征与部署接口为底座|`code/SSDG/train_ssdg.py`；新增MUSE模块|pending|checkpoint兼容与state导出测试|不重写主干|
|MUSE-002|当前`项目.md`|固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`且物理ID互斥|数据参数、launcher、协议测试|pending|真实checkpoint无query smoke|不得读取target|
|MUSE-003|指导文档|Epoch 1起利用全部`U_s`的域/GRL/自监督/扰动监督|训练循环、日程模块|pending|阶段边界单元测试|S1无伪标签分类梯度|
|MUSE-004|指导文档|Epoch 17起启用EMA多证据软伪标签|证据融合、EMA调度|pending|融合与日程测试|教师只读weak/source视图|
|MUSE-005|指导文档|全局/局部/原型三头按`0.50/0.25/0.25`融合|证据融合模块|pending|概率归一化、有限值测试|局部头训练期限定|
|MUSE-006|指导文档|以置信、间隔、JS、原型距离和稳定性形成可靠度|可靠度路由模块|pending|单调性与边界测试|禁止使用`y_u`|
|MUSE-007|指导文档|将`U_s`互斥路由为`U_H/U_M/U_L`|可靠度路由模块|pending|互斥、完备与空集合测试|三类覆盖率必须记录|
|MUSE-008|指导文档|`U_H`硬CE，`U_M`soft CE，`U_L`候选集或无身份梯度|未标注损失模块|pending|梯度与权重归一化测试|低置信禁止熵最小化|
|MUSE-009|指导文档|候选集累计质量至少`0.75`且最多3类|候选集损失模块|pending|可达/不可达测试|不可达时身份损失为0|
|MUSE-010|指导文档|`z_dom`回归卫星扰动参数|扰动回归头与Huber损失|pending|shape、mask、finite测试|只使用模拟器元数据|
|MUSE-011|指导文档|学生按样本ID与epoch确定性选择strong或单一satellite视图|卫星视图路由|pending|确定性和概率日程测试|避免三学生前向|
|MUSE-012|指导文档|稳定`U_H`只以`0.05-0.10`更新分类prototype|分类prototype bank|pending|稳定次数与权重上限测试|默认要求连续3次|
|MUSE-013|项目协议与指导文档|`U_s`不得生成proxy unknown或更新开放集几何|训练循环与开放集接口|pending|协议负测|直接失败，不静默忽略|
|MUSE-014|用户确认设计|实现M0/M1/M2/M3单seed同协议矩阵|新launcher与配置|pending|dry-run命令快照|首轮不扩成完整S0-S8|
|MUSE-015|AGENTS.md|训练完成后必须分别测试clean和三类`leo_weak`|launcher、评测闭环、报告|pending|dry-run与artifact闭合测试|缺任一项不得完成|
|MUSE-016|用户要求|保存相关配置、训练日志、逐场景测试日志和checkpoint身份|run report与输出目录|pending|artifact枚举测试|不可覆盖run ID|
|MUSE-017|设计|训练期局部头、自监督头和扰动头不得进入deployment bundle|模型导出过滤|pending|state-key负测|保持Phase2兼容|
|MUSE-018|设计|首轮记录H/M/L覆盖、有效权重、三头分歧和泄漏诊断|telemetry|pending|CSV/JSONL字段测试|隐藏标签仅训练外诊断|

## 当前汇总

- 总要求：18
- `verified`：0
- `implemented`：0
- `pending`：18
- 当前状态：设计已获用户确认，尚未修改生产代码或启动实验。
