# 同投影统计形状／有序面积：条件式剩余两种子

本登记是已固定首seed源PASS后的原样扩seed准备，已于 2026-10-14 发布并核实运行。seed2026092702/03 × OLD_GATE_COS50、SIG_TIME、COV_SHAPE、AREA_SHAPE，共八行：两个同seed历史baseline只读复用、六个独立scratch50。首seed及其健康运行保持原计划，firstseed不重训、不替换、不按目标筛除。

机制严格沿用报告路线一：单6D局部路径、四窗，共同mean6/variance6，最后15分量是offdiagCov或skewArea；同固定27→28lift、112→160、LayerNorm、gain初始化0.05/上限0.2。三新arm各新增18976参数，Cov/Area全部初始trainable参数与fixedlift一致，native/body/defaultW也同初态。保持t2 postDropout取点，未叠加新kernel/SiLU/Sinc/PAIR/Dropout前取点/跨窗层级/U权重。

默认cosine6×160scale30随骨干正常联合训练，禁止任何另训身份或RX/day/channel探针；使用原已核实native fits-disabled scope。旧batch_neighbor＋自身EMA0.999＋cos，33label＋17pseudo、AdamW lr2e−4→1e−6/wd1e−4、lambda_U0.16/lambda_ent0.01/smoothing0.01固定。L6300/U56700/V27000、split392005、RX/day/TX和物理ID保持，U真值隐藏，V只健康eval不选模；NoExplicitCFO和内部补偿保持，无新星地训练。

新增成本6×50=300model-epochs/66600更新；历史baseline100轮单列复用。合并原首seed时9个新模型共450/99900，3历史150复用；不从首seed权重续训，没有父/teacher checkpoint继承。模型seed固定02/03，不扫描/换seed或按目标早停。

发布仅依原首seed预登记source规则PASS：三个模型finalLCE≤0.5，新参数实际更新，真实joint160精确导出且W独立，固定L720 AREA−COV zL2≥1e−6。首seed四行必须完成既定闭环，但这里只读completion元数据，不读target analysis/metrics决定扩seed。FAIL保留历史结果，六新模型不启动。

实现提交d14fdcfacaa61f10a4e389c7b29de3596f35934b。入口python -u -m experiments.cvs_moment_area_seeds.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_moment_area_seeds_20261014_r01；不可变run/output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s0203-m8-r01。共享capacity锁计所有owner每GPU≤3/free≥12000MiB，按实时容量分配，保护健康旧任务。已有六真实模型context/同初态/同参数/原512c10a首seed四config一致/foreignconfig拒绝、两seed COV→AREA汇总、source gate负例、发布scriptcompile均PASS；一次限定新增执行P0/P1审查PASS，不重审旧核心。

发布时Git代码/config、此最小登记、一个archive及一次local/remoteSHA比较、remotecompile；首步原source PASS和旧seed02真实CP zeroIQ/no-query smoke，PASS直接运行。复用既有VALIDATED_ONCE handles，不增加数据重验、签名、receipt、报告hash或审批。

Stop：本run自身角色/来源/非有限数/预算/default head/export/物理ID/prediction/truthlast技术失败停止新增派发，保留健康进程与产物，无自动重试；不按目标改参数/父权重/seed/停止规则。六个新模型固定final50/11100。

逐row冻结后自动clean168000默认头逐包prediction，prediction固定后独立truth-last。剩余八行1344000决策/144指标；合并原firstseed为12行/2016000/216指标，三seed平均、样本SD、各seed配对差/正差数、MacroF1、最差RX/day、修正/引入错误及完整训练曲线/U接受/时间。只有完整预定义三seed候选均值≥84%才达目标，不用两seed/最高seed/source或另训头替代。

输出release preflight/source_expansion_evidence；run launch/queue、每新行initial/final/frozen/resolved/actual_args/initialization/roles/50epoch/11100step/source_feature_witness、逐行default prediction/complete、scores/metrics/completion、source_features/analysis/combined_three_seed_analysis/completion。IQ、weights、rawlogs和大型prediction留产物目录；代码、报告、必要聚合Git上传并核对remoteOID。当前已启动六个新模型，尚未完成，不声称性能提高、Phase2、新TX或卫星部署。


首seed source PASS已核实：LCE 0.248262/0.259459/0.250530，AREA−COV L720 L2=3.069351；真实导出/W独立和新增参数实际更新通过。四行已完成默认头预测和truthlast，扩seed决定未读取新的target指标，详见 source_expansion_decision.json。据此原样发布剩余两种子。


发布 VERIFIED：执行提交 8bd34db31787cd80c4ad53ee7327ff6a55daee72，dispatcher PID 2116969；六个新进程 CWD/argv/GPU/resolved config 核实，首轮日志均增长 5171 至 5175 字节，epoch=1，failure=null。一个 archive local/remote SHA 比较及 remote compile 通过，旧 seed02 真实 checkpoint no-query smoke PASS，无训练继承/另训头。详见 launch_evidence.json。
