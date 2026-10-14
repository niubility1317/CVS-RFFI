# 可信门控内有界权重：冻结源代理预登记

依据用户附件报告路线四，当前只做源代理准备，不全面展开新GPU训练、不改8个健康层级模型。固定原moment-area SIG_TIME最低seed2026092701/final50/11100，执行512c10a267b5e654cfd2c983ab1debdeebc7f626；加载前source_proof核对原roles/physicalIDs/scratch来源/ownEMA、默认头额外fitdisabled及无target训练/选模。复用与前两个源诊断一致的固定source anchor，不按目标分数选父权重；无新模型训练初始化。

完整L_s6300只观测一次，固定shuffle2026101411、Dropoutseed2026101412、25个batch（24×256＋156）。每包teacherweak、studentclean、studentstrong三次forward，共18900。student.train仅回放原Dropout/strongnoise.015，identity无BN；teacher.eval读取同checkpoint的ownEMA。所有权重/buffers不更新，无optimizer或headfit；只暂开t1/t2/t3.pw.weight导数对象3456/6912/9216，autograd.grad计算监督LCE、原UniformU CE、WeightedU CE，结束恢复eval/requires_gradFalse，检查原tensors逐值相同和.grad未累积。不是实际训练阶段梯度历史。

原教师confidence、physical batch_neighbor window2、studentstrong argmax一致门控按原helper执行；不放松门控/变采样/改阶段/信道增强/EMA/头。h_i=1-p_student,strong(teacher伪类别)，w=stopgrad(.75+.5×clip(h/.1,0,1))，raw界[.75,1.25]、scale.1于源观测前固定，不扫描。lossU=.16×sum(a*w*CE)/sum(a*w)，空accept返回0；entropy路径不改，本代理只比较CE项。raw界不等于归一化effective系数同界，其理论界[.6,1.6667]。h只表示学生难度，不表示教师正确。

source labels允许用于clean LCE和mask/weights形成后的代理precision，也用于既有helper构造已批准physical metadata；不传入新门控或难度。15项overall/5RX/3day/6source-class-index统计保留count/accepted/hits/precision/weightedprecision/有效权重分母，source-class索引映射见原contract。25batch×3层=75项参数梯度范数、weighted−uniform差/cos及同层U/L比；不同层参数量不同，不把跨层norm排名解释为重要性。保存私有source_weight.pt后只重读arrays/gradientvectors复算，不新增IQ forward。

L6300每physical cell70包，U56700每cell630包，随机邻包可见率不同；同一个Lproxy同时用于监督/伪标签观察，与真实训练的独立L128/U256批不同。源precision/coverage和冻结梯度比不能冒充隐藏U真值、训练33/34/50梯度或性能收益。无U IQ/真值、V forward或目标访问；既有Vcache可materialize后discard。NoExplicitCFO及内部补偿、split392005/sourceRX1/3/4/6/8/day1/2/3/旧6TX与物理roles保持，复用VALIDATED_ONCE。

实现/config 051de1a177819c0f8a9cfe5aba0fd09e7cefef6d；local native真实模型2zeroIQ、stopgrad/原mask/归一化CE梯度精确/空accept/11负例/source统计与grad数学检查PASS，未读源/目标IQ。一次独立直接P0/P1 PASS，无发现；不重复父框架或报告审查。

入口python -u -m experiments.cvs_source_weight.probe；普通N607现有CVS-RFFI CPU、CUDA_VISIBLE_DEVICES空、Torch/OMP/MKL/OpenBLAS2threads。CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_weight_20261014_r01，immutable output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-trusted-weight-manysig-s01-m1-r01，owner codex/root/source-weight-20261014。一次archive本地/remote SHA比较及remotecompile；真实sourceCP2zeroIQ无查询smoke PASS直接继续，核对PID/CWD/argv/loggrowth/完成及保存数据复算。不占GPU、不停/改/重启其他owner。

Stop：角色/provenance/非有限/shape/原gate/state/grad/savedrecount失败保留输出停止，无自动retry、换source模型、改scale/bounds或选择性重采样。预期results/resolved/source_weight/completion及release preflight/submit/log。0新增epochs/optimizer/headupdates。只上传代码/config/聚合结果与报告，原IQ/weights/大日志/梯度数组不入Git。源诊断无target测试或84%达标结论；下一真实训练若发布仍须按报告固定33/34/50位置记录实际L/U贡献，不能用本代理替代。
