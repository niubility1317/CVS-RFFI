# PAMF single-view paper experiments

状态：LOCAL_VERIFIED／PREPARED，尚未声明远端运行。

用户 2026-10-27 指定默认单视图，普通 PL 保持；网络分支纯 CE 监督与 CE＋PL 均需，全部采用可读学术名称。

唯一120行矩阵见 experiment.json.matrix，方法与数据解释见 docs/PAMF_SINGLE_VIEW_PAPER_20261027.md。108新F、12精确匹配复用：11设置增强主比较66、Main split无信道对照6、8结构×监督/PL×3seed48。新G及恢复缓存0。Main split完整单视图Sup/PL六模型和严格CE单视图/无路径各三模型已核实完成；不重训。

固定 E50/11100，20+10+20，seed2026092701/02/03，split392005。主比较保留完整L骨架；分支CE+0.68信道槽，LS0，无CRIC/KL/EMA。PL0.95/weight1，U256，E14启用；监督不构建U。原数据/原默认头/源最终轮/隐藏U标签保持。Main split6300/56700/27000，其他10契约沿原VALIDATED_ONCE。source-only训练配置不接目标或truth。

有增强clean+六practical，无增强clean；三群体cross_date/cross_rx/cross_both，raw/DURA-L/DURA-LU各路由全部预测固定后，独立CPU truth-last。复用合法已有决策并按物理ID核对，只填缺失预测。保留所有场景/seed/RX/TX/day、Macro-F1、净纠错、实际L/U/PL计数和成本。源数量5源点由Main相同seed模型的共同6目标冻结计数生成，不新训；主7目标分列。

N607普通账户、原Phase1 Python/Torch、FP32/TF32off。CWD=PROJECT/releases/20261027-phase1-pamf-single-view-paper-m120-r01-release；output=PROJECT/runs/20261027-phase1-pamf-single-view-paper-m120-r01，logs位于其logs/；私有实际路径/解释器在local_artifacts/pamf_single_view_paper_20261027/runtime.json和远端runtime。命令：PYTHON -u -m experiments.pamf_single_view_paper.runner dispatch --config RELEASE/runtime.json。代码身份为发布Git commit，执行后记录真实值，不要求额外代码或报告SHA。

GPU0-7动态剩余容量，一卡至多一个本run训练/预测、全卡总数至多两个、free>=12000MiB，CPU scorer至多1。已执行一次direct-only普通账户只读preflight，资源/路径可用；不停止或修改其他owner。首个实际checkpoint无query smoke通过后自动进入矩阵。首次技术/协议/输入/输出冲突/非有限/无法产生合法预测即停新派发，健康worker继续，低分不停止；不覆盖旧root，不依据结果改参数、挑seed或重跑。

## 本地验证与唯一独立审查

6项聚焦协议测试PASS，覆盖120矩阵/12复用、纯CE与U权限、禁止truth/EMA/另训头/外来checkpoint/改预算、CE+PL确实取U256、clean-only保留原七视图capsule。

唯一独立P0/P1审查PASS。发现并修复两项直接接口缺陷：F15模型ID导致loader误关U；clean-only行误将输入capsule场景契约改为空。两项各一次定点复核通过，无剩余P0/P1；没有额外全量复审。汇总比例统一乘100输出百分数。真实checkpoint和实际单视图/十类/L-U backward在launcher首步执行，不以本地元数据检查冒称通过。

远端真实启动、smoke、PID/CWD/argv/GPU/日志增长和实际执行commit见后续launch_evidence.json；在独立读回前状态为PREPARED，无新性能结果。
