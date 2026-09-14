# FULL9已完成六组E200独立测试

用户2026-09-14明确要求对已完成实验执行测试并给出详细测试结果。本轮固定六组F-A1/F-M14/F-M11/F-M05/F-M08/F-M12；其余F-M00/F-M07/F-M13仍健康训练，不加载中间epoch，不改变训练进程、配置或原owner。

来源run=phase1_adv3b02_xuc_full_s392005_20260913_r1，执行commit=e7dbe643f5b85062e8f4e2de6d711a5447c7a422；来源release=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_xuc_full_e7dbe643f5。六行各自final_ssdg.pth、E200/44400、scratch/无外部初始化、source角色一致性在加载前核对；完整扩展anchor继承本run E20 EMA，无其他run权重。该源契约已验证，无需重建数据。

本评估ID=phase1_adv3b02_xuc_full6_eval_s392005_20260914_r1。输出在N607项目runs/<本ID>/<行>/predictions.json及score.json，日志logs/<本ID>；只读来源checkpoint，不写原run。唯一owner由publish_full6_evaluation.py提交，启动后独立核实PID/CWD/argv/日志。

使用原冻结predict_phase1_truth_last.py及score_phase1_truth_last.py；同原source/target契约、seed392005、clean与三LEO场景，全部六组4032000条预测固定后独立scorer读取truth。不向未完成实验传回指标或选模，原九行最终统一评估与本次用户要求的提前测试分开保留。六行评估是既有benchmark描述性测试，不是新盲确认集。

准入：每GPU最多两个进程且空闲>=12000MiB，最多四个并发predictor，继承冻结FULL调度器的active预占；每次启动新预测前确认原三个未完成行RUNNING/对应PID/尚无final且最新完成epoch<=185。发布时原owner仍TRAINING，GPU1/2/3/5/7空闲。若窗口关闭或出现技术错误，保留已启动推理及产物，不重复启动、不停止原训练。prediction不足不能评分。

验证：本地负测覆盖E199、9800步、target接触、外部checkpoint、源角色不匹配均拒绝；独立P0/P1审查通过。落地单文件传输一次SHA比对和编译；首次使用真实F-M12最终权重做无query零IQ forward，随后立即正式推理。

当前状态：VERIFIED / ARTIFACTS_COMPLETE，六组4032000条预测已固定并独立评分，2026-09-14T10:12:37.283833+08:00完成读回。详细结果见[detailed_results.md](detailed_results.md)。
