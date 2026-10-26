# 恢复前后跨接收机身份关系：源证伪

PLANNED；四候选只有一个学生 seed2026092701。原联合目标保持：同一候选三个学生种子的 raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%。本run无目标句柄，源通过不代表目标达成。

## 源依据与待检验机制

ResidualMix四条完整E50/sourceV已核实；MixBase/MixDose/PairBase/PairDose的留出G3差为+.129630/+.080247/+.129630/+.104938pp，均未达原+.3pp。提高剂量扩大参与RX拟合，未扩大留出迁移。同系数G/G控制比较也没有稳定优势。负结果保留，不继续仅靠全局增权。

待检验假设是：恢复分支CE让模型学会判别源恢复输入，却没有明确把恢复前后不同RX的身份关系连接起来。新损失直接作用于当前学生现有joint160，原L与raw普通PL保持；无固定旧特征参照、不更新G或另训头。不将此假设写成已证明原因。原U/REPL已有RDrop、浅中层跨RX和raw joint160对比，本轮只检验不同实际输入的关系结构，不宣称通用对比学习是新机制。

| 方法 | 两个额外学生输入 | 配对范围 |
| --- | --- | --- |
| SeparateRG | raw、同包G(H_k(raw)) | 只在各自视图内 |
| BridgeRG | 与SeparateRG完全相同 | 只在两个视图之间 |
| BridgeHG | 同包H_k(raw)、G(H_k(raw)) | 只跨视图 |
| BridgeGG | 同包G(H_k(raw))、G(H_(k+1 mod3)(raw)) | 只跨视图 |

G/H复用原缓存三景lowSuburban/midUrban/lowUrban，k按原step顺序轮换；GG为不同已缓存场景，不是给同一确定IQ重复取新名称。所有记录仍按同一物理key读取，不交换不同包波形。

## 损失、权限与剂量

原三冻结E30 raw教师全部同类且confidence≥.95的记录接受，q的argmax为伪类。U真值隐藏。附加监督去掉RX1，原L/rawPL使用全部源RX。两视图都提取当前学生特征，受理集合联合计算detached均值/std（floor1e-4），再L2归一（eps1e-4）；拒绝及留出RX不进入标准化或任意配对。

正对为同伪类且不同RX，负对为不同伪类；同物理包/同RX同类不作正对。Separate只允许同视图，Bridge只允许跨视图。对有正/负对的anchor，loss=logsumexp(allpositive+negative scores)−mean(positive scores)−log(numberpositive)，cosine温度.2。总loss为有效anchor和除以2*原完整U batch数，不按受理数放大。SeparateRG/BridgeRG输入、共同标准化、伪类、曝光、总正负pair/anchor数量完全匹配；仅关系方向不同。两学生前向都保留feature梯度，附加classificationweight梯度为0，现有featurefusion norm/gain仍训练；默认头由原主项联合训练。

先12批原L128/U256无更新源校准，primary为原完整L+rawPL梯度，CENTER_25K/25000物理换算保持。共同系数一次固定min(.25,.10*medianPrimary/max_method medianExtra)，全部四条使用同一系数及444步升温；不根据V/目标选剂量。初始梯度代理不保证后续AdamW或目标收益。记录每route范数/cos、分类weight梯度0及相同pair计数，所有学生/教师/G参数和registered buffers不变、temporary buffers恢复、optimizer/V/目标/Utruth读取或更新0。

四条均own原ordinaryPL E20+10 E30独立+20/4440步至E50/11100，累计13L37LU；不继承任何研究F50。原SIG_TIME_X3/CENTER_25K、ordinary与raw RDrop、As CE且KL0、LS.01、浅中层ORD128跨RX.1、默认头、raw selfPL .95/weight1保持，nativeU/EMA off。每row U1134000、held226800、附加受理838780、额外学生FW2268000；有效anchor及正负pairs按真实批记录，不能把受理数冒称全部有效anchor。新增80F轮；共享三E30历史90轮及原q/H/G缓存成本单列。

## 固定源判断与后续

先真实E30/defaulthead/G的L2 smoke，再12批校准；随后BridgeGG原生L128/U256联合backward smoke，通过立即串联四条训练。每row final后自动sourceV27000八视图，保留raw/Graw/三H/三G与TX/RX/day/held/trained及净纠错。native eval-only V健康检查允许，objective不读V、不更新V状态。全四条完成后才按原heldRX1 G3≥+.3pp、overallV rawclean≥−.1pp筛选；最高合格heldG、精确同分按方法名。无合格者不扩三种子/目标。RX1只从附加分支留出，父模型/L/rawPL/G已见它，不称全网络未见RX泛化。

通过只支持另行冻结正式三种子raw/同G七视图矩阵，不改变默认方法。所有源负结果和缺项保留，不拼接不同方法最优项。

## 发布、环境与停止

N607普通账户，既有Python/Torch2.1，FP32/TF32off，每进程CPU2线程，无安装；CWD PROJECT/releases/20261026-phase1-PAMF-RestorationBridge-sourceHold-release-r01。命令PYTHON -u -m experiments.pamf_channel_pl.restoration_bridge_training dispatch --config该release/runtime.json。不可覆盖root PROJECT/runs/20261026-phase1-PAMF-RestorationBridge-sourceHold-m4-r01。仅打包minimal source_context，无目标句柄。

GPU0..7均可按全局CUDA+pending<2、free≥10000MiB准入，最多4条本run训练/源评估，保护其他owner。一份archive一次本地/远端SHA比较、remote compile、独立PID/CWD/argv/GPU/loggrowth。首次技术/协议/非有限/预算/smoke错误停止本run新派发，健康已启动任务继续；保留产物，不自动重提/热改/干预外国owner。科学负结果完成全矩阵。

8项聚焦检查与local compile通过；唯一一次独立P0/P1检查PASS，无P0/P1。源码/config与公共聚合提交Git，IQ/权重/逐包预测/日志/runtime私有。无额外设计审批、seals或重复审查。
