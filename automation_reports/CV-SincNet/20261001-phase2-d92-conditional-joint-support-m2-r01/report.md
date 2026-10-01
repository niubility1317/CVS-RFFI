# 旧点函数约束的条件核LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-conditional-joint-support-m2-r01`
- group_id：`d92-conditional-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；独立读回时间：2026-10-01T02:27:27+00:00。训练release固定为`7204161b126a9d3a17096aee5c75bce8ccfee6ec`。

## 目的与对照

固定Phase1及既有合法support缓存，B用解析Affine LocalRidge；C在actual B函数上用全列旧点零残差条件核注册，完整隐式梯度与固定坐标球优化类RMSCE；结构改动而非参数扫描。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

VERIFIED：独立完整KKT证书63项及核心与5ENTRY联合37项合成检查通过；C q-gauge审计、JSON标量口径和Training.audit_dict ABI缺口已修，原失败保留。固定Phase1、合法support缓存的4row/160parent声明已预登记，只读元数据preflight核实缓存/来源/输出未占用，不读取query或源样本。新方法独立summary/analyzer/report/collector仍在实现；没有publish/launch或真实性能结果。A与B−A仍N/A，R0不得替代A；理想目标为soft，目标ACTIVE。

VERIFIED：新Conditional独立summary/analyzer31项、完整报告/训练诊断18项和隔离白名单bundle4项检查通过。真实合成训练trace在禁candidate数学/score条件下独立重建通过；首轮归一化original几何和namespace问题已修、原失败保留。唯一独立P1发布传递依赖已补齐并回归验证，未发现P0。实际4row/160parent/1800path支持集实验保持预登记，尚未publish/launch；真实性能/A及B−A仍N/A。只root可启动，新方法不修改健康Affineanalysis，目标ACTIVE。

RUNNING/VERIFIED：ConditionalJoint已唯一发布启动，实际训练release为7204161b126a9d3a17096aee5c75bce8ccfee6ec。独立读回核实supervisor578658/start6923770和rx3 workers578671/578672/start6923795的PID、argv、CWD及输入身份；两个rx3训练、两个rx1排队，CPU2lane/BLAS2，不使用GPU、不加载checkpoint、query/源样本0。完整独立summary31、report/collector18、隔离bundle4及核心/ENTRY37、完整KKT63检查已通过。记录中的旧Affine展示字段已按实际Conditional命令与probe预算更正，冻结launch spec及健康进程不变。尚无真实性能，A及B−A仍N/A，旧support恒等式不保证query零遗忘，目标ACTIVE。

当前只做support信息诊断：4个model/cohort row、160个parent、1800条物理顺序路径；旧类6，K=1/5/10/20，新增类0/2/5/10/20，每cohort两个预声明receiver/scenario组合。本轮尚未执行query评估，当前不存在新的最终准确率。完整结束后只对已有产物进行一次独立分析，报告完整矩阵、B/C旧新准确率、H、旧类下降及资源；A需要真实地面classifier包及行映射绑定，不能用R0补齐。

TRAINING_COMPLETE/VERIFIED：独立control读回核实4个row、160parent、每row40与所有训练进程退出，actualruntime7204161不变。已预登记唯一独立analysis release d92_conditional_joint_analysis_20261001_r01，尚未启动分析或读取真实性能；原spec/训练/状态未修改，A仍N/A。

TRAINING_COMPLETE / ANALYSIS_RUNNING / VERIFIED：预登记Conditional独立分析r01已唯一启动，PID617023/start7407753/source0145234，local handle55338；argv/CWD及原训练runtime7204161经独立只读核实。尚无summary或错误，无部分性能读取。不得再次analysis/publisher、停止或热修改；完整独立分析后才生成结果报告/采集训练档案。
