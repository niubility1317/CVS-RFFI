# 跨接收机身份参照的源梯度剂量诊断

PLANNED，仅构造合法可见 L_s 固定参照和计算源梯度，不创建 optimizer 或更新 checkpoint，不读取 V、目标 IQ/标签或 U 真值。活动目标不变，科学机制与阈值背景见 [活动计划](../../../docs/goals/PAMF_RESTORED_IQ_SSL_PLAN_20261025.md)。

固定原 F0 ordinary-PL own E20+10 的三 E30 教师（2026092701/02/03），seed01 同权重临时 student 和同契约原 G-LU。既有来源 reader 核实 E20/own10、原默认头、native U/EMA 关闭与全部原物理角色；复用 VALIDATED_ONCE，不重验数据或新建 hash 链。

## 固定参照与真实权重 smoke

launcher 先对 L_s 的2包核对实际部署G、三个默认头以及临时student的160维 `z_id` 与 own E30 reader.features 一致。PASS后立即继续。完整 L6300 用三冻结E30提取归一化特征，分别按TX/RX均值后归一化，得到三套 own-seed 参照。统计仅来自可见L，不跨seed平均特征坐标，不从U/V/target更新参照，不拟合新身份头。references.pt为私有源训练统计，不提交Git。

U的新增关系分数对各类别等式使用 `mean_{other RX} cosine(z,p[class,RX]) / 0.1`，排除样本自身接收机的参照，以三冻结raw E30平均概率q计算交叉熵。q只接受三教师同类且各confidence≥.95，U标签必须仍为−1。所有新增分支额外排除第一个排序源RX；原L/raw普通PL主项照常。该留出只针对新增分支，E30与原主项见过这些源RX，不能称为从未见过的接收机。

## 12批无更新梯度

seed01固定shuffle，12批L128/U256，共L1536/U3072曝光。原L保持六次真实前向、LS.01、As KL0、ordinary/raw RDrop与浅/中层跨RX；原U主项保持自身eval的argmax/confidence≥.95、原始输入CE、weight1、完整batch平均，不让新共识门控替换主项。no_native_U仅为将L与原U主梯度分别测量，临时student不构造optimizer。

| 新增变体 | 两次输入 | 公共条件 |
| --- | --- | --- |
| RawReference | raw/raw | 同一q、mask和完整U batch归一 |
| RestoredReference | G(scene_k)/G(scene_k) | 同上 |
| PairedReference | G(scene_k)/G(scene_k+1) | 同上，两次损失先平均 |

场景依次 low suburban/mid urban/low urban，k=(step−1)%3；固定信道seed20261016421、硬件seed20261016423、namespace `PAMF/restored_dose/U/fixed`，与原三场景缓存相同。G固定reuse=True/graph=True/native=False，FP32/TF32off。每变体前恢复同一post-primary RNG和临时buffers，批后恢复全部named_buffers，包括非持久bank；所有参数不变。总计60次autograd.grad，不对V求梯度。原分类头主梯度必须非零，新增关系约束只通过已有身份特征求梯度、不产生新增头目标。

剂量在看到本run结果前固定为一个共同值：`min(.25,.10*median(primary_norm)/max_v median(extra_v_norm))`。三个变体用最大中位额外范数确定同一系数，保证匹配且各中位梯度剂量不超过主项10%。Sinc两坐标梯度除25000，其余原单位，匹配CENTER_UNITS的全局clip5物理坐标；这是preclip梯度代理，不是AdamW实际更新保证。记录全部cosine，无临时目标反馈或隐藏的阈值重排。有限/非零12批不足则技术失败，不生成有效剂量。

预算：参照F包前向18900，raw教师9216，student主项及三新增梯度33792，G6144；真实L2 smoke另列。独立物理记录与这些重复前向次数分别说明。F/G冻结，student最终全部参数/registered buffers不变，refs更新/新头拟合/optimizer/V/U真值/目标读取均为0。

## 发布、停止与产物

命令：`PYTHON -u -m experiments.pamf_channel_pl.receiver_reference_source dispatch --config PROJECT/releases/20261025-diagnostic-PAMF-RXReference-dose-release-r01/runtime.json`。N607既有普通账户Python/Torch2.1，CWD为该不可覆盖release，输出 `PROJECT/runs/20261025-diagnostic-PAMF-RXReference-dose-m1-r01`。GPU0一条本run worker，准入时CUDA/pending总数<2且空闲≥10000MiB，CPU2threads；保护所有其他owner。

独立P0/P1审查发现源NumPy元数据不能直接依赖N607 Torch的NumPy桥，已改为显式tolist后构造long tensor，并增加相应边界回归；一次定向复审在发布前记录。相关模块compile PASS，聚焦数学/协议负向测试与兼容回归按实际结果记录。Git commit为代码/config身份；一次archive SHA传输比对、远端compile、启动后独立PID/CWD/argv/GPU/loggrowth核实。首个来源/角色/特征smoke/非有限/持久状态变化或技术错误停止本run，保留部分产物，不自动重提，不操作健康任务。

预期smoke.json、references.pt、progress.json、gradient_complete.json、complete.json、pipeline_state.json及本run公开源梯度汇总。该run不能证明目标提升或跨接收机迁移；通过后仍须按已公开计划另行固定源侧新增分支留出训练矩阵，并保留全部负结果。

发布前验证结果：六项聚焦数学/协议负向与NumPy桥兼容检查PASS；独立审查的唯一P1已修复，针对该问题的一次定向复审PASS。真实权重smoke留在launcher首步执行。
