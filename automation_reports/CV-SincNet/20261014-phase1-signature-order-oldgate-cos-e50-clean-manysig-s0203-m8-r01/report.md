# 签名内训练机制：固定剩余种子条件扩展

首定义seed运行 20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01（执行 2658815a2fe5ef931f24558a6c6c61146433fd9c）仍健康训练，当前只准备包装与预登记，未启动扩展。仅原首seed四row完成且两个候选均通过原source-only资格才可发布本run。先读sourcequalification和sourcefeatures，不读新target数字决定扩种，不改原m.1/coeff.1/5warmup10ramp/33label17pseudo/oldgateownEMAcos/scratch50/defaulthead联合训练。首seed不重训，六新均scratch无继承；两候选一起扩，不按目标排行。

8row：seed2026092702/03各历史旧gatecos＋SIG_TIME ZERO/RAW/DYNAMIC三新。其方法与首seed相同，动态108仅辅助距离去4常量，部署网络固定SIG112/joint160，不增加结构或分类头。当前用户明确要求固定SIG_TIME，后续主攻泛化及半监督训练机制；本候选符合该范围。

剩余6新300model-epochs66600updates＋2历史100model-epochs22200updates；额外16500L90/1485000包。合首seed为12row/450新＋150历史model-epochs、99900新＋33300历史updates、600完整源epoch、2016000defaultdecisions/216组指标、24750extraL90/2227500额外源包、5940实际层cohort梯度。全部按各row50终轮冻结，自动defaultclean168000predict→独立truthlast评分→完整3seed配对统计。原始IQ/CP/预测/日志/梯度留远端，Git仅实现和必要聚合。

原样复用唯一已审核心，不新增假设或重复核心审查。10source资格负例拒绝singlecandidate/缺梯度/缺实际feature差/target反馈/改seed/换头等；六native上下文与same-seed同初态、原signature_order配置逐项一致、50/33+17/oldEMA/gate以及archive关键依赖/REMOTEcompile均PASS。只读audit增加可选design/build参数供本包装复用，不改已发布首seedarchive、训练或健康任务。

完整data/角色/physicalIDs/seed/CWD/GPU/owner/命令/output/log/stoprule见experiment.json。发布入口实验首步强制复核首seedsource资格、既定执行commit与原矩阵，然后真实SIGcheckpoint无query smoke，PASS后自动继续；没有额外审批、seal或数据重验。每GPU总进程≤3且free>=12000MiB，其他健康任务保持。技术失败保留产物，不自动重试；暂无新目标准确率或84结论。

包装代码提交 5eda4e0862e8e13cc3a2c79c3c31ef634a2fe72d，本地六native/10source负例PASS；首SOURCE资格尚待完成，当前未发布扩种。

首定义seed四row闭环完成，已独立只读SOURCE资格PASS且尚未读取新target数字：三finalLCE0.247153857/0.248473601/0.248924990；RAW/DYNAMIC−ZERO L720zL2=1.310938478/1.795316935；t1/t2/SIG投影17/33/50有效非零、11100steps/2750extra/220gradientbatches/0headfit均PASS。按原固定02/03六新scratch50扩展，不改参数或按目标排行。
