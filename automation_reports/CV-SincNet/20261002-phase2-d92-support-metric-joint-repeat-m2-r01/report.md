# SupportMetric 联合 LocalRidge 完整复用基准预登记

状态CONFIG_ONLY，入口、控制与独立scorer正在准备，尚未发布或启动。现有同方法support诊断尚无完整结果；本轮query矩阵与方法现在固定，不按该诊断或任何query成绩选择参数。实现数学定义沿已验证runtime3c3，不改变GGN/Fisher/U物理度量、步长接受规则或资源限制。

固定Phase1 source-only scratch final200与合法地面模型/原型摘要，practical residual/post_sync/noeq/25 MHz。Phase2仅使用当前split合法support拟合；每split旧类B从零开始，C继承该split实际B并冻结旧条件；new0复用B，不再次拟合。源域样本、源逐样本特征、历史adapter/head、query拟合和评分反馈均不参与。R0对照为原BranchLocalRidge，C独立重拟合，必须与顺序候选清楚区分。

固定2模型×2cohort共4 row、2400物理episode。rx3每模型900，rx1每模型300；旧类6，K=1/5/10/20，新增0/2/5/10/20，support seeds2026092711至2026092715；receivers19-1/8-14/8-7/20-19，三个practical场景。现有VALIDATED_ONCE capsule/split及物理IQ不变，方法变化不触发数据重验。此为透明的重复基准，不能称为全新独立确认。

每个query仅按自身只读特征计算全部已注册类分数和固定预测，不读truth/role/真实query类数或配额，不做全局重排。五streams A/R0_B/R0_C/B/C与同split实际B及所有声明row完整固定后，原不可变release中的独立scorer才可一次连接truth，结果不回流拟合、选择或重跑。未齐时不对子集评分；技术失败保留产物和健康row，不自动重试，性能弱不能停机。

报告同物理旧query及旧support配对的A旧准确率、B旧准确率、C旧/新准确率，注明K、旧类6和新增类数，保留完整K×新增类数及receiver/scenario/model分层。逐parent先算适应提升、注册后下降、绝对新旧差和H再聚合，不由汇总均值重算H。理想目标仅作为逐步改进方向，不新增每轮必须达到的硬准入门槛。

每row仅构建一次固定ground Q的U basis，归档普通方法状态并计入真实成本；prepare绑定/Gram与阶段训练、推理分别记账，不重复收取构建成本。保留CVS风格详细文本和compact JSONL/CSV实际损失、权重、梯度、方法状态与耗时；source validation因Phase2禁读为N/A。实际更新坐标、解析head状态、CPU硬件、训练/推理耗时、RSS、常驻状态与新增传输字节单列；星载实测、能耗、native wire及其他未测量项N/A。参数少不等于训练总计算少。

只修正登记文字：cohort引用为benchmark.cohorts，feature cache producer来自各row实际branch_features路径，属于原始冻结表征供support拟合及只读query推断；没有继承历史适应状态。算法、矩阵、数据和角色不变。入口34、scorer70、control43个distinct用例已通过；57文件精确source bundle隔离导入成功且无数据/encoder读取，独立P0/P1审查闭合。状态READY，尚未发布、启动或评分。最初合成测试失败与修复后原始输出全部保留；详情见D92_SUPPORT_METRIC_QUERY_SOURCE_VALIDATION_20261002.md。

完整query r01已唯一发布并启动，runtime c7c46558d5eb8f34bb57fb09c12adc7ac2c5109f；2026-10-02T00:10:28Z独立/proc argv/cwd/start_ticks/environment读回VERIFIED：supervisor1327506/start14743342，rx3 PID1327618/1327622 PREDICTING，rx1两行PREFLIGHT_COMPLETE等待。全四行直接预检完成，global marker无，尚未truth/评分。原公式、数据、六项资源和矩阵不变。

独立只读fixed-output评分入口已准备，12项定向测试通过并由独立source P0/P1审查闭合。正常原控制器可能因np.float64类型拒绝在producer exit0后记FAILED，原状态不修改；新入口仅接受此精确错误或原合法COMPLETE行，逐行从真实marker核全4/2400并用原独立数值校验器及二次读回闭合后，最后才接truth。新独占closure/summary不是authority，不重预测、不fit、不接受partial。r01计划已登记，原query仍运行，评分尚未发布或启动。
