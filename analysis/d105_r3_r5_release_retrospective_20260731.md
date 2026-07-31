# D105 Phase1 R3/R4/R5三轮release回顾

状态：`RETROSPECTIVE_COMPLETE / NEXT_CANDIDATE_FROZEN / NO_PERFORMANCE_RESULT`

## 1.回顾范围与目标复核

本次回顾在第四次N607 release前完成。证据范围包括当前长期目标、2026-07-20版`项目.md`、刷新后的1181条项目会话索引、D105 Phase1主报告、Target25报告、R3/R4/R5完整runner handoff、R4与R5完整pipeline日志、当前Git实现及本地真实checkpoint最小复现。

当前硬目标保持不变：K10下`old_acc_after_increment≥92%`、`min_old_class_acc≥85%`、new5/new10/new20的`seen_new_acc≥92%/90%/86%`；K5/new20相对matched K10/new20的联合性能下降不超过5pp；K1/new20必须相对D92同时改善H、旧类floor和总正确数。域适应、旧类保持、弱类floor与新类注册保持同等优先级，不因Phase1技术修复而改变。

## 2.三轮事实表

|轮次|最远阶段|确定性失败|正式产物|性能结论|
|---|---|---|---|---|
|R3 `2eaa1b11`|落地后prelaunch|Windows工作树CRLF哈希被写入manifest，而Git archive为LF，24/54 runtime字节不一致|pipeline未detach；strict tap/prediction/score/component均为0|`NO_PERFORMANCE_RESULT`|
|R4 `d23469ba`|prelaunch全过、唯一detach|N607 PyTorch2.1.0与NumPy2.x边界上的`torch.from_numpy(batch)`异常|exit=1；零strict tap、零prediction、零score、零component|`NO_PERFORMANCE_RESULT`|
|R5 `9f608e8b`|prelaunch全过、唯一detach|Phase1把D105双backbone模型接入GRB旧strict tap，`z_dom`必然为空|exit=2；零strict tap、零prediction、零score、零component|`NO_PERFORMANCE_RESULT`|

三轮都在任何Target访问、query真值打开、authority签名或formal seal前停止。没有一轮产生D105 Target指标，也没有可用于方法选择的中间准确率。

## 3.R5根因闭包

R5完整日志只有一个fail-closed出口：`strict tap must expose byte-bound z_id/pre_relu and z_dom`。静态调用链和真实checkpoint复现共同把根因缩小为唯一接线错误：

1. `stage2_d105_phase1_bundle._strict_forward`调用`stage2_grb_jp4_adv_drqknn_bcrr.strict_zid_with_hook`。
2. 该GRB eager路径只执行`model.id_backbone`，并仅在其aux中搜索`feat_dom/feat_domain/z_dom`。
3. D105冻结模型的合法域特征路径是独立的`model.dom_backbone.feat_imp→model.dom_enhancer`；权威实现已存在于`stage2_d105_feature_tap.extract_d105_feature_tap`。
4. `ssr-gpu`本地真实checkpoint复现中，旧Phase1路径得到`z_id=[2,160]`、`pre_relu=[2,160]`、`hook_exact_bytes=true`，但`z_dom=None`；同一模型、同一IQ进入D105专用tap时得到`z_dom=[2,160]`。

复现记录为`analysis/d105_r5_strict_tap_real_checkpoint_reproduction_20260731.json`。因此R5不是数据问题、数值随机性、CUDA差异或方法性能失败，而是Phase1与D105权威特征出口未统一。

## 4.为什么既有本地门没有发现

- Phase1 bundle测试主要从人工物化的strict-tap archive继续验证下游资产链，没有直接执行`export_d105_phase1_strict_tap`的真实模型路径。
- 真实checkpoint smoke验证的是D105专用feature tap及四臂无truth机械闭环，不是Phase1 `_strict_forward→export_d105_phase1_strict_tap`入口。
- NumPy修复审查确认旧C-API桥已消失，但没有把“实际Phase1调用的tap实现”与“审查通过的D105 tap实现”做调用图同一性检查。
- R5通用错误把多个字段条件合并，不能从日志直接区分`z_dom=None`、shape、dtype或ReLU字节关系；这降低了首次定位效率，但不是根因。

## 5.拒绝路线

|路线|决定|原因|
|---|---|---|
|复用、恢复或远端修补R5|拒绝|违反不可覆盖run和本地先修复规则，R5已永久关闭|
|删除`z_dom`要求或退回id-only域适应|拒绝|会改变D105方法语义，使CBRC域编码失去合法输入|
|从`id_backbone`伪造或推断`z_dom`|拒绝|不等价于冻结训练时`dom_backbone.feat_imp→dom_enhancer`路径|
|只把通用错误拆细后再release|拒绝|只能改善诊断，不能修复功能接线|
|跳过Phase1直接启动Target25|拒绝|缺少source-held资格、独立score、authority和formal seal|
|根据R3/R4/R5中间状态改方法或挑row|拒绝|三轮均无性能证据，不能用于候选选择|

## 6.下一候选冻结

冻结技术候选`D105-FTU1`，仅统一D105正式特征出口，不改CBRC、LPO-RC、qKNN、矩阵、阈值、seed或性能门：

1. Phase1 `_strict_forward`改为调用`extract_d105_feature_tap`，或将两者收敛到一个D105专用、双backbone、同IQ、一次调用的权威实现。
2. 保留`z_id/pre_relu`字节绑定，并显式要求`z_dom`来自`dom_backbone.feat_imp→dom_enhancer`。
3. 增加真实checkpoint入口级回归：必须直接执行Phase1 `_strict_forward`和`export_d105_phase1_strict_tap`，验证`z_id/pre_relu/z_dom`、archive/receipt及零truth访问。
4. 将字段级失败原因保留在fail-closed诊断中，但不得把错误放宽为fallback。
5. 修复后必须重新运行统一测试、真实checkpoint no-truth smoke、精确Git archive smoke和独立`P0=0、P1=0`审查，再预登记新run ID；不能重启R5。

## 7.协议与指标完整性复核

|边界|复核|
|---|---|
|LEO弱观测|Target25仍只接受已封存`leo_*_weak`固定接收IQ；不生成第二观测|
|Phase2 clean/source|D105-FTU1只影响Phase1资产构建；Phase2不增加clean/source运行时入口|
|query|query仅逐样本测试，fit/update=0；不读取truth，不跨query重排|
|role/quota|不传old/new真实角色，不使用类别配额、Hungarian或全局重分配|
|标签置换|不增加TX/class ID专属分支、阈值或白名单|
|旧类与新类同等任务|Target25仍需同row before/after旧类、`seen_new_acc`、`H_old_new`、逐旧类准确率、floor和forgetting；缺一不可晋级|
|当前证据等级|仅根因技术闭包；D105仍无Phase1 formal asset和Target性能|

## 8.下一步决定

第四次release尚未获准。先由独立Terra Max功能研发子agent实现`D105-FTU1`，主agent负责调用图、真实checkpoint复现、协议与结果分析；再由另一独立Terra Max审查者判定`P0/P1`。只有本地闭环和Git提交完成后，才为新的非覆盖run创建报告、冻结archive和安排唯一N607 runner。
