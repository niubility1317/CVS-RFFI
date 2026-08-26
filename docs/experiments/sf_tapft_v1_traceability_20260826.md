# SF-TAPFT V1设计追踪

## 实现边界

本记录将用户提供的“SF-TAPFT：无源样本目标域锚定式渐进微调”报告映射到代码与验证。报告原版包含目标域可训练分类头，与当前`p2_min_v1`冻结原型和禁止持久分类头的正式Phase2边界冲突。因此原版实现必须标记为`DIAGNOSTIC_NON_FORMAL`，不得用于正式Phase2晋级声明；本次不修改`项目.md`。

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|SF-01|3.1输入约束|适配入口只接收checkpoint模型和target train，不接收source loader/cache或target eval|`code/cvsrffi/target_only_progressive_adapt.py`、测试|implemented|聚焦测试通过|函数签名无source/eval/query入口；runner强制核对最小Phase2数据绑定|
|SF-02|3.2模型结构|冻结教师、目标学生；频域和BN running statistics冻结|同上|implemented|冻结性测试通过|教师仅在可选KD中读取target输入；学生保持eval以冻结running statistics|
|SF-03|3.2模型结构|A/B/C阶段渐进解冻head+norm、time adapter、最后时间块|同上|implemented|allowlist测试通过|真实模型映射到`id_backbone`，不更新`dom_backbone`|
|SF-04|3.3分类头初始化|源头权重与目标原型按`rho`插值，新类使用目标原型|同上|implemented|原型头测试通过|原版持久头，仅诊断|
|SF-05|4损失|类别均衡CE、label smoothing=0.05、prototype CE scale=8、L2-SP|同上|implemented|损失与L2-SP测试通过|默认不启用KD|
|SF-06|4.2原型损失|leave-one-out prototype不包含当前样本，K1回退到初始化头权重|同上|implemented|LOO/K1测试通过|不使用target eval|
|SF-07|5选择性蒸馏|实现target-input selective KD，默认权重0，T=2、gamma=2|同上|implemented|代码路径与配置校验通过|仅旧类target support触发，默认关闭|
|SF-08|6训练预算|A=500、B=1500、C=2500，共4500步；AdamW、warmup+cosine、clip=1|同上|implemented|默认配置与缩步测试通过|启用CUDA混合精度；测试使用1/1/1步|
|SF-09|7内部模型选择|grouped inner CV；balanced accuracy→NLL→margin→fold variance→距离源模型|同上|implemented|分组OOF与分层回退测试通过|存在真实group时严格分组；缺少group时按label执行stratified fallback|
|SF-10|7.4安全回退|OOF证据不足时选择zero-adapt|同上|implemented|fallback组合条件测试通过|要求多数fold不降、平均NLL改善且accuracy或margin改善|
|SF-11|7.4/6|top-3 checkpoint averaging|同上|implemented|同构state与inner-validation测试通过|top-k>1强制使用物理ID不相交的target inner validation，按balanced accuracy、NLL、margin、源距离排序|
|SF-12|11代码修改|提供独立模块、CLI/runner入口和审计结果|`code/cvsrffi/target_only_progressive_adapt.py`、`code/scripts/run_target_only_progressive_adapt.py`、测试|implemented|CLI帮助与编译通过|不扩展旧8D selector|
|SF-13|11审计测试|source/eval不可访问、冻结参数不变、L2-SP初值为0、group不重叠、LOO、复现、fallback|测试|implemented|56项聚焦与回归测试通过|其中18项为本方法新增测试|
|SF-14|项目协议冲突|正式Phase2不得使用原版持久分类头|本记录、CLI输出|rejected|`项目.md`5.3.1|`REJECTED_EXTRA_GATE`不适用；这是直接数据/方法权限边界|

## 首轮实现判定

- 严格设计一致性目标：报告原版诊断实现，不把它改写成冻结原型正式方法。
- 本地实现完成条件：所有`implemented`项均有RED→GREEN证据，CLI可启动，独立P0/P1审查闭合。
- 实验发布条件：另行完成真实ADV3B02 checkpoint无query smoke；在此之前最高交付状态仅为`LOCAL_VERIFIED`，不能证明实际checkpoint与support数据联调成功。
- 最高风险：现有ADV3B02模型的`aux_id.feat_joint`与`id_backbone.cls_head.head.weight`必须严格同维、同类别顺序；实现已显式绑定该路径，仍需真实checkpoint smoke验证。
