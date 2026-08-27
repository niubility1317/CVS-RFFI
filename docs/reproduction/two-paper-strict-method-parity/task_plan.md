# 两篇论文严格方法一致性任务计划

## 目标

在不处理原始数据资产的范围内，完成论文明确方法的端到端实现，并用严格配置阻止未公开细节被误称为一比一一致。

## 阶段

- [x] 现状审计：三路独立审计完成，见`docs/audits/2026-08-27-paper-reproduction-traceability.md`。
- [x] 严格方法设计：用户确认采用论文一致＋未公开默认值路线。
- [x] 实施计划：写入逐测试、逐文件的实现计划。
- [x] Tweak严格pipeline：测试先行实现训练、校准、聚合与评测。
- [x] Hu严格pipeline：测试先行实现结构、增强、训练、微调与评测。
- [ ] 反向审计、提交与远端核验。

## 约束

- 保持paper reproduction和CVS extension分离。
- 不启动训练、不访问或下载数据集、不连接N607。
- 未公开方法细节必须使用写明理由的`UNPUBLISHED_DEFAULT`，不能归因于作者设定。

## 已知错误

|错误|处理|
|---|---|
|`cmd.exe`读取PowerShell故障目录失败|改用已验证Python UTF-8读取。|
|Python输出`项目.md`时触发GBK编码错误|改用`sys.stdout.buffer.write(...encode('utf-8'))`。|
