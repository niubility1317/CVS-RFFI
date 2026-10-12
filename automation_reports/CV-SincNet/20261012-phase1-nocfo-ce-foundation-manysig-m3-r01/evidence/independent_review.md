# 独立 P0/P1 发布审查

**PASS：未发现未闭合的 P0/P1。** 日期：2026-10-12。运行：`20261012-phase1-nocfo-ce-foundation-manysig-m3-r01`。本次仅审查前三行无显式 CFO 的纯 CE 基座，不代表后续 24 行已发布或已通过审查。

独立核对 DESIGN、配置与登记、模型与来源、实际训练目标和 source 入口、预检、冻结预测/评分、调度/运输及收集代码。复用此前相同训练和控制路径的审查结论，并核对本批差异。仅写本文件；未修改其他实现或登记，未使用 GPU、操作远端、启动实验或读取目标评分产物。

## 核验

- **纯 CE 与来源。** 三个 seed 的原生随机学生均不加载初始 checkpoint，初始化与教师来源列表为空。配置与实际分支均关闭 U：load_source 不打开 U 缓存，不创建 U stream，不读取 peer bank；调度已移除教师准备流程。教师、EMA、卫星增强、辅助网络、Fishr、SAM、hidden mix 和风格前向均不执行。主目标仅合法 L 的 CE，label_smoothing=0；零权重之外没有保留无用辅助计算。
- **CFO 一致性。** 唯一实际构建函数在随机初始化和自身 checkpoint 加载时均安装同一输入策略：参考响应第 27 坐标恒零，29 维接口及其余 28 个量保留，内部逐包 CFO 估计/补偿原样执行。预测只使用这一路学生构建和严格自身权重，不引入历史老师。源码与实测均支持显式坐标屏蔽，不将其解释为原始 IQ 不再含频率信息。
- **源角色与预算。** L6300、V27000 分别核对合法源契约中的完整唯一物理 ID，检查无交集；V 仅在 E5/10/15/20 无梯度健康验证。固定 scratch20、4440 次 AdamW、每步 L128，共 568320 次 L 读出/行；三行 13320 次更新。cosine 峰值 2e-4 至 1e-6、weight decay 1e-4。每行所有 U/辅助/SAM/hidden 计数应为零，末尾逐项核验，无 V/目标选模。
- **严格冻结和评分。** final20 checkpoint 保存完整自身 config、run/row/seed、4440 更新、scratch 来源与空教师祖先；模型和预测来源验证均检查。source_frozen 完成后同 PID exec 串联 clean 预测和独立 scorer。prediction 完成前不读取 truth，全部六类逐样本竞争。每行 168000 query、三行共 504000 决策；逐行与总评分完成条件均适配三行，collector 不再要求不存在的教师缓存。历史 P0 只作评分附带参考，输入与训练器差异使其不能承担 CFO 单变量因果对照。
- **发布与保护。** 三行生成 config 与登记完全相等。只读 configure 检查 63 项归档路径均存在，包含实际依赖；远端 preflight/dispatcher/scorer 绑定本包。预检用新建的临时学生和合法 L 做有限前后向，不访问目标或加载历史教师。共享锁统计全部 CUDA 进程和启动占位，每卡最多 3 个；唯一 owner、独立 run/release、逐行目录和日志拒绝覆盖。未知启动不自动重试，技术失败保留产物并停止待启动项，健康任务不受干预。

## 证据与边界

已读取登记 evidence/nocfo_foundation_cuda.json：三 seed 原生 CUDA 精确纯 CE、有限梯度、一次 AdamW、CFO27 为零、无 U/教师/辅助前向、实际 source.train 初始化至首更新边界均 PASS；内嵌本地检查覆盖三行完整预算、禁止初始化 checkpoint 读取和严格自身 checkpoint 往返。对本包全部 Python 文件的只读语法解析也通过。上述为局部验证，不代表正式训练或 clean 测试已完成。

收尾定点核验：主代理已将输入策略说明改为 no_explicit_CFO_training_and_prediction，并明确历史 checkpoint 训练策略为 None，关闭该非阻断说明残留；移除的 prepare 函数/CLI 不影响纯 CE 路径。新增 finish 在所有逐行评分及 aggregate require_scored 验证后执行，完整源日志分析与三种子结果分析成功后生成报告、更新原 spec 为 ANALYZED 并追加事件，最后才写总 completion。读取仅为已冻结源日志和评分 JSON，不再打开 IQ/truth，不回流训练。资源索引已核对真实 source.measured_training_resources 和 prediction.inference_seconds，缺 GPU 数值按 N/A 报告。

独立执行一次仅临时目录的合成 finish 贯通检查为 PASS：运行实际 collect.analyze、final_report.build、报告/证据落盘及 spec/events 更新。测试模拟已完成的源日志分析结果，没有重跑源日志扫描或声称完成真实训练，也没有读取真实评分、IQ 或使用 GPU。该检查覆盖新增收尾代码的字段和调用衔接；其余已通过的训练路径未重复测试。正式发布、远端实际输入预检和启动读回仍由唯一 launch owner 完成。本次没有增加审批、数据重验或性能门槛。
