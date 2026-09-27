# ADV3B02 XUC15执行包

固定设计：`configs/matrix15.json`，M00—M14，seed392005，E200，FP32，从零训练。M00为本次ADV3B02 CORE90基线，M12为X＋Ux_normalized＋C*及暴露匹配课程。原C2为显式比较分支。原生A1两行M09/M10保持FastTrust/DAOT/MUSE路径。

数据：ManySig/equalized1，source RX1/3/4/6/8、day1/2/3；target RX0/2/5/7/9/10/11、day0/1/2/3。L/U/V=6300/56700/27000，容器内按sig_i连续划分。13行CORE90每epoch49次主更新，原生A1每epoch222次；原生行只与原生行做直接X效应比较。

所有CORE90行采用相同独立ticket随机流，基线目标项与原冻结CORE90在E1/E80/E131验证一致，但不声称与历史run逐步随机轨迹相同。M12/M14只重排窗口内票据，原始阶段权重、mask、通道seed随票据；优化器与控制时钟随已接受更新推进。M05仅被动审计。C*无可信恢复/几何确认时正常训练，不强制触发。

执行：`code/scripts/dispatch_xuc15.py --project <N607项目> --run-id <新ID> --commit <已发布Git提交>`。发布工具为`tools/publish_xuc_release.py`。每GPU最多两个进程，预留尚未初始化CUDA的子进程名额。健康行不因其他行失败而终止。所有15行E200结束后预测clean及三个leo_*_weak；10,080,000项预测全部固定后才由独立scorer连接truth。统一评估batch256、通道seed392005及CORE90通道配置。

恢复：先确认旧owner及健康行均结束，再用新run ID运行同入口，附`--recovery-from <旧run ID> --retry-rows Mxx,Myy`；训练只重试列出的技术故障行并从零开始，其他合规E200checkpoint仅用于推理。预测/评分故障可使用空retry-rows从冻结产物恢复。已固定预测保留引用，评分输出写新run。不因低分、机制未触发或合法排队重跑，不修改seed/方法/预算。同故障指纹修复后再失败则停止盲目重启。

验证：`python -m pytest tests/test_acceptance.py -q --confcutdir=tests`；`python code/scripts/check_xuc_execution.py --output <新目录> --device cpu`。后者覆盖M00/M08/M12真实入口、一步训练与自生成checkpoint严格重建，不访问query。正式启动前发布工具在N607对M12执行一次同类CUDA检查。

来源：X及native代码来自本机a1_fast_v2_20260909_wt快照；GAME/C2及CORE90隔离实现来自core90_game_20260911_wt；V2/U核与采样来自core90_cross_response_v2_20260911_wt。当前Git提交固定实际执行源码，不把快照目录名当成checkpoint来源证明。
