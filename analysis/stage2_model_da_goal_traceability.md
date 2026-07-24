# Stage2快速模型域适应目标修订追踪

日期：2026-07-24
目标文件：`docs/STAGE2_METHOD_RESEARCH_GOAL.md`
依据：用户对Stage2-B/C、地面压缩原型、星上快速模型域适应、K1正收益、MRIOR超越和联合协同的最新定义。

|ID|来源要求|规范化需求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|G01|域适应不同于分类头|DA必须训练或调节基底模型推理路径；纯metric/head/score不计模型DA|`docs/STAGE2_METHOD_RESEARCH_GOAL.md`|verified|文档反向审计|明确允许LoRA、FiLM、normalization、轻量adapter和IQ前端|
|G02|Phase2分Stage2-B/C|定义Phase1状态、Stage2-B旧类support适应状态和Stage2-C旧新support联合状态|同上|verified|章节与状态机检查|同row报告注册前后旧类|
|G03|使用三类合法知识|候选同时使用地面压缩原型、target-old support和target-new support|同上|verified|输入表检查|query始终只用于测试|
|G04|使用地面压缩原型|只允许共同封存且由bundle/method lock显式启用的INT8多样本聚合旧类知识，禁止成员级或clean状态|同上|verified|协议章节检查|要求`ground_old_multiprototype_enabled=true`；地面知识不替代新类support|
|G05|K1正收益|K1不得默认恒等；总体正收益、逐receiver不负，并相对direct形成配对证据|同上|verified|性能门检查|不可辨识参数需由地面先验收缩|
|G06|优于MRIOR|Stage2-B K10必须在同row matched比较中显著优于MRIOR-SDA|同上|verified|基线和门槛检查|要求paired 95% CI下界大于0|
|G07|qKNN统一分类|适应后由qKNN对全部注册类逐query统一决策|同上|verified|因果臂检查|禁止第二head替代qKNN|
|G08|其他互补机制|OTHER只解决遗忘、注册、floor、hubness、校准或量化剩余误差|同上|verified|方法卡字段检查|metric/RDA/BCRR归入OTHER或辅助，不冒充模型DA|
|G09|联合1+1>2|隔离模型DA、无ground DA、OTHER和JOINT贡献，并计算`I_syn`|同上|verified|消融表与公式检查|天然耦合方法允许等价干预|
|G10|星上资源稀缺|限制训练参数、step、state和int8生命周期，并要求相对MRIOR资源优势|同上|verified|资源门检查|删除optimizer后只保留部署delta|
|G11|完整性能证据|报告old-before/after、gain、new、H、BA、floor、min、forgetting、混淆和分层|同上|verified|指标清单检查|不得拼接跨run极值|
|G12|实验驱动直到达标|设计冻结后快速实现；每个candidate/revision只运行单个预注册seed的25-row矩阵；通过后用新seed再运行一份25-row确认，负结果进入下一revision|同上|verified|研发顺序与run矩阵检查|25-row不得用于反向选参或补丁式调参|
|G13|当前路线重分类|RBSC/C-id/RCHM/SVRN等metric路线只作OTHER/reference，不算快速模型DA主线|同上|verified|当前起点检查|下一波必须优先模型适应方法卡|
|G14|DA纳入一切合法可用信息|候选必须审查domain branch、`z_dom`、地面压缩原型、domain basis、聚合统计、adapter先验和同received-IQ合法表征|同上|verified|合法资产清单检查|允许选择性使用，但必须说明未采用理由|
|G15|每次发布充分利用N607八卡|正式矩阵确定性分片并动态调度到GPU0–7；安全可用时8卡齐用，每卡不超过2个训练进程且不干预已有任务|同上|verified|发布与调度条款检查|资源不足时排队等待，不得缩窄矩阵；报告逐GPU分配与利用情况|
|G16|最新K10绝对门|A-old≥92%、min-old≥85%、new5≥92%、new10≥90%、new20≥86%|同上|verified|性能门逐项检查|min-old以全部实际旧类的最低准确率定义|
|G17|K5衰减限制|K5/new20相对matched K10/new20的A-old、min-old、seen-new和H衰减均不超过5pp|同上|verified|paired切片门|不能以平均H掩盖弱类坍塌|
|G18|K1必须提升|K1的M_DA或M_JOINT相对同rowM0必须在H、A-old或min-old上产生严格正收益，且old/new保护项不恶化|同上|verified|K1同row因果门|不再接受整体identity作为目标完成证据|
|G19|优先域适应并复用D62/D92|下一revision优先改变模型/表示DA，以D62、D92和统一qKNN作为头部对照，不继续仅调分类头|同上|verified|候选卡与四臂检查|关注弱类、ground压缩知识与target-old/new support联合使用|
|G20|无远端Git发布|只允许本地Git版本化和N607文件同步；不push、不建PR、不上传GitHub|同上|verified|版本与发布条款检查|本地commit用于复现，不属于上传操作|
|G21|新类support参与适应|Stage2-C允许在固定Phase1 basis/rank/schedule上用old/new类等权support进行一次有界继续适应，随后统一重编码全部support并重建qKNN|同上|verified|S_B→S_C状态机检查|不得用query、新类数量或角色专属公式改变自由度|

## 反向审计

- [x] 所有G01–G15均在目标文档中有明确落点。
- [x] 目标文档没有把纯metric、协方差或score修改称为模型域适应。
- [x] Stage2-B和Stage2-C的数据输入、状态转换、预测边界与指标均闭合。
- [x] K1、MRIOR、ground-off、模型DA、OTHER和JOINT门均可证伪。
- [x] 资源限制覆盖训练期、持久state、量化和query路径。
- [x] 每个候选/revision的正式性能发布均为单seed 25-row；通过后以新seed再做25-row确认。
- [x] 每次正式发布在安全边界内尽量并行使用N607八卡，且不以并行不足缩窄冻结矩阵。
- [x] K10绝对门、K5≤5pp衰减门和K1严格提升门均已写入。
- [x] 仅本地Git版本化与N607同步，禁止push、PR和GitHub上传。
- [x] Git提交范围只包含本次目标文档和追踪记录，未纳入并发RBSC实现文件。
