# 单视图完整 PAMF：监督与普通 PL 三种子匹配基线

按用户最新“补充单视图的监督，普通 PL 对比基线”要求，新建六模型，不重启已停止的两行源控制。现有三个 REPL 新学生保持运行。本次仅补缺失的同架构完整监督骨架对照，已完成的历史 X3/单视图其他配方全部保留。

完整 PAMF 保留时频、PA、参考响应和原 6×160 默认分类头，SIG_TIME 为一视图、18976 参数。两组原监督机制完全相同：ordinary/raw RDrop、As 两次真标签 CE 且 As KL=0、LS0.01、浅中层 cross-RX，默认头与骨干联合训练。监督版 U loader/所有伪标签/EMA 关闭；普通 PL 仅当前学生置信度≥0.95、单 U 学生视图、权重 1，无 EMA/REPL/新机制。PL 保留原第14轮启用U的日程。

模型 seed 2026092701/02/03。监督从零 20+10+20=E50/11100；PL 各自复用已核实且来源一致的 m201 单 SIG_TIME ordinaryPL E20/4440，再原生+10+20至E50/11100，累计13L+37LU。不得让监督继承U前缀或跨seed借权重。新训练合计240模型轮/53280步，继承三份E20/13320步和对应耗时另计。所有模型固定final，不按V或target选checkpoint。

沿用原F0 VALIDATED_ONCE L6300/U56700/V27000、源RX[1,3,4,6,8]及day[1,2,3]和原六TX。U真实TX始终隐藏，V仅原有遥测。训练只收到源context；目标句柄只进入独立预测/评分配置。正式测试为cross_rx126000+cross_both42000=168000包/seed，两个读出raw和同一原F0 DURA G-LU，clean及全部六practical。复用已完成固定G恢复IQ，不训练G，不重新生成目标IQ。G-LU用过源U的IQ，因此“监督F+G-LU”不称整个系统只使用L。

每行final后有资源立即自动测试。该行2任务×2读出×7视图预测全部固定，独立CPU scorer先核对完整ID/视图覆盖，之后才连接truth。保留逐TX/RX/day/scene准确率、Macro-F1、混淆矩阵和恢复净纠错；自动输出每seed、三seed均值±样本SD及配对差。168000口径按正确数合并，不能把126000跨RX和42000跨RX+跨日简单等权。单独跨日30000不加入本矩阵。

真实smoke为第一步，六个配置各执行实际L128联合backward；PL分别严格加载自己的真实单视图E20，U256隐藏标签；Sup U实际0。检验L768/AsKL0、单SIG_TIME和原head梯度非零有限，0optimizer；首个真实E20/defaulthead接同G还做sourceL2前向。PASS后立即训练。

唯一owner为codex-pamf-sig1-matched-baselines-20261026。代码/config由执行Git commit固定，N607既有Phase1 Python/Torch2.1、FP32/TF32off/CPU2，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-SIG1-matched-baselines-release-r01，命令和不可覆盖独立输出见experiment.json。全部GPU0..7可用，每次派发全局CUDA+pending<2且free≥10000MiB，其他owner计入，不停止/改动其健康任务。评分单独CPU最多1。

一次archive SHA传输比较和远端compile；独立核实PID/CWD/argv/GPU及日志增长。首个技术/输入/来源/shape/非有限/smoke/输出失败停止新派发，健康在跑行完成，保留产物，不自动重复提交或覆盖。科学低分不当技术失败。新基线不热改现有REPL源选择阈值；原clean+1pp、raw星地>1.5pp及同G提升目标保持，源/训练完成不当目标成功。
