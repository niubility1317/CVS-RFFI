# 任务缩放参照的匹配剂量源验证

PLANNED。四个方法剂量对照，均为同一student seed2026092701，不是四个种子。本run没有目标句柄，不启动目标预测或三种子扩展。原raw clean+1pp、raw六景>1.5pp、同G六景正收益联合目标保持。

## 已有依据与新矩阵

上一轮GTaskScaled源留出三景+0.222222pp（净增36次）仍未达原+0.3pp（至少49次）；其源clean+0.011111pp。全部失败产物保留，不放宽门槛。新的冻结L诊断显示，保留全部L样本的关系分数只微小改变参照正确率，没有足够依据增加结构。本轮只改变既有系数，原梯度缩放公式/特征/默认头保持。

| 方法 | 原extra路线 | 系数因子 | 最终新增系数 |
| --- | --- | ---: | ---: |
| GTask2x | GTaskScaled | 2 | 0.06515361984430072 |
| GTask3x | GTaskScaled | 3 | 0.09773042976645108 |
| GUniform3x | GUniform | 3 | 0.09773042976645108 |
| RawTask3x | RawTaskScaled | 3 | 0.09773042976645108 |

系数固定为min(.25,factor*0.03257680992215036)，base来自已完成的12批source wholemodel校准，不声称新probe。GTask两剂量在原校准表达式中的加权最大中位梯度比为20%/30%，同3x系数GUniform约13.47%、raw约2.35%；它们是匹配系数控制，不是相同梯度预算，也不是AdamW更新或目标收益保证。444步升温、cap32、两次同输入前向损失先平均及完整U batch归一保持。

每条从原own ordinaryPL E30/6660独立追加20轮/4440到E50/11100，累计13L+37LU。原L完整T1、raw普通PL阈值.95/weight1、普通增强/RDrop/As真实CE且KL0、LS.01、浅中层ORD跨RX .1、默认头联合训练、nativeU/EMA off保持。原L6300/U56700/V27000合同不变；q和三景G恢复U缓存、每seed自己的visibleL中心均只读复用，不更新教师/G/参照，不读取U真值或增加头。

新增U分支继续排除首个源RX1，原L/raw主项保持全部RX。G每步使用原低郊区/中urban/低urban缓存轮换，raw控制用原IQ；GTask3x/GUniform3x使用完全同源恢复输入和相同新增接受流。每条预期U曝光1134000、held曝光226800、新增接受838780、额外学生前向2268000，动态scale sum/max/cap及真实系数/route/factor均报告。原父/G/可见L已见RX1，这只是新增分支迁移诊断，不称为未见baseRX。

## 源评估与选择

各条固定final后立即完整V27000八视图，与既有sourceShift的同seed PLControl按完全相同物理顺序/场景比较；native无状态V健康推理允许，新objective不读V，所有V状态更新0。原source namespace PAMF/restored_source/V/fixed，所有channel/G契约保持。

四条全完成后，仅通过原heldRX三个G平均差≥+.3pp且整体V rawclean≥−.1pp的候选可被选中；按heldG增益最大、精确同分名字母序。无通过者不扩展。前一轮GT10/普通G10/raw10为已完成剂量参照，不作为新父权重，也不重复训练。所有失败/负结果保留，本控制器不自动启动未来三种子或目标测试。

## 发布与停止

加载前原E20/ownE30实际来源和sourcecontract、oldvisibleL refs/cache/G、base sourcecalibration状态核对保持。真实最高剂量GTask3x E30/L2/联合L128U256 backward smoke是第一个GPU步骤，通过立即继续；原L768包、AsKL0、有限头梯度/机制计数核实。3条新配置/源标签/route/系数/prepare smoke聚焦检查已通过，相关compile和一次独立P0/P1审查结果发布前追加。

N607既有普通账户Python/Torch2.1、FP32/TF32off，不安装依赖。GPU0至7均可准入，按全局CUDA+pending负载最少、空闲最多分配，每卡总<2、free≥10000MiB，最多4条本run train/eval，各CPU2threads；保护所有其他owner。命令PYTHON -u -m experiments.pamf_channel_pl.task_dose_training dispatch --config PROJECT/releases/20261026-phase1-PAMF-TaskRefDose-sourceHold-release-r01/runtime.json；CWD该release，输出PROJECT/runs/20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01。

代码身份为runtime.code_commit，一次archive SHA传输比较、remote compile、启动后独立PID/CWD/argv/GPU/loggrowth检查。首次技术/协议/非有限/smoke/预算/状态错误停止本run新启动，健康已启动子进程继续，保留输出，不自动重提、不热改或干预其他任务。科学负结果完成全部四条。

新增训练为80 F轮；历史三E30前缀90轮共享，原PLControl和GT10等对照的已有尾部成本单列复用，不算作新训练或零成本。私有权重/IQ/逐包预测/日志/runtime不提交Git，公共同row聚合和代码/config提交。

本候选唯一一次独立 P0/P1 审查 PASS，无 P0/P1。三条新聚焦检查与相关 compile PASS；实际最高剂量联合 backward smoke 由 launcher 首步执行，通过立即继续。
