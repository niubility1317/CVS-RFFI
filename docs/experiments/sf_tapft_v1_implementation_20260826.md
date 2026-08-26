# SF-TAPFT V1实现记录

## 结论

已按设计报告实现SF-TAPFT V1的本地诊断版本，包括目标域原型初始化分类头、A/B/C渐进解冻、leave-one-out原型损失、L2-SP、可选选择性KD、分组target-train内部选择、zero-adapt回退和checkpoint平均。实现入口不接收source、target eval或query，输出不可覆盖，并固定标记为`DIAGNOSTIC_NON_FORMAL`。

该标记不是实现缺口。报告原版要求训练并持久化分类头，而当前`p2_min_v1`正式Phase2边界禁止将冻结原型扩展为持久分类器。因此本实现用于验证报告机制，不得用于正式Phase2晋级或性能声明，也不修改`项目.md`。

## 设计落地

- `target_only_progressive_adapt.py`实现报告核心算法。真实`DualCVSincNet`只在`id_backbone`中挂接rank-16恒等初始化时间adapter，优先读取`aux_id.feat_joint`，并使用`id_backbone.cls_head.head.weight`初始化旧类分类权重；`dom_backbone`与频域路径不参与更新。
- A阶段更新目标分类头与允许的归一化仿射参数；B阶段增加时间adapter；C阶段增加最后时间块。学生保持eval模式，避免BN running statistics漂移。
- 优化目标由类别均衡交叉熵、label smoothing、leave-one-out prototype CE、L2-SP和默认关闭的选择性KD组成。默认训练预算为500/1500/2500步，使用AdamW、warmup+cosine、梯度裁剪和CUDA混合精度。
- grouped CV只切分target-train；存在采集/session group时保持group不重叠，缺少真实group时按label执行stratified fallback。只有多数fold不降、平均NLL改善且accuracy或margin改善时才选择adapted，否则整体回退到zero-adapt。
- top-k大于1时必须提供与inner train物理ID不相交的target inner validation，并按balanced accuracy、NLL、margin和源模型距离选择快照。OOF选择后的诊断bundle固定采用seed确定的第1fold部署候选，不使用target eval，也不在全量target-train上按训练损失重选top-3。
- runner只接受固定角色的support NPZ和checkpoint，强制核对`p2_min_v1`、`VALIDATED_ONCE`、非空`capsule_id/split_id`并写入`cvs.sf_tapft.v1`bundle；已存在的输出目录会被拒绝，避免覆盖。

## 本地验证

- 新增18项聚焦测试，覆盖目标域角色、物理ID唯一性、旧/新类原型初始化、LOO与K=1回退、L2-SP、三阶段allowlist、group不重叠、无group分层回退、OOF组合门、target inner validation选top-k、zero-adapt、checkpoint平均、复现性、双分支冻结、最小Phase2数据绑定、非覆盖输出和正式权限拒绝。
- 联合既有meta-adapter回归共56项测试通过。
- 两个新模块和两个CLI完成Python编译检查；单次适配与分组选择CLI的`--help`均可启动。
- 框架只在既有`code/model.py`路径报告一个与本次实现无关的AMP弃用提示，本次新增路径无告警。
- 一次性独立P0/P1审查最初发现安全回退组合条件、top-k验证选择和无group分层回退3项P1；定点修复后唯一一次定点复审确认3项全部闭合，无遗留P0/P1。

## 交付状态与下一步

当前状态为`LOCAL_VERIFIED`。下一步是将本提交作为唯一代码/配置版本发布，然后用真实ADV3B02 CORE90 checkpoint执行一次无query smoke，验证checkpoint维度、support schema、GPU混合精度和诊断bundle闭合。smoke通过不等于取得性能结果；性能只能在prediction完成后由独立truth-last scorer给出。
