# 纯 F E20：raw＋As 与机制匹配 NoAs

状态：ANALYZED / VERIFIED。两行E20、源资格、七视图预测与独立truth-last评分及完整日志分析均完成；whole-raw配方源规则FAIL，关闭不续E50。完整结果见 [final_analysis.md](final_analysis.md)。总体目标未达成。

唯一假设：ordinary与完整As共同训练可能存在监督任务竞争；以raw替代第一ordinary监督槽，检验是否改善严重受损身份提取并保持clean。当前纯F固定L诊断的raw180均正确，而low训练命名流重建45/69、新实现89/138、V2237/3600；翻倍low权重没有修复seen拟合。旧CROSS_RX/RDrop LRaw尾段曾source PASS，但旧无As/RX1留出raw-anchor降低clean且关闭。因此不预判ordinary整体有害，也不把当前As描述成重复损伤——As一直从raw独立生成。

该候选在本聊天首次读取新sceneRisk正式分数前已按上述源证据固定；候选、参数及source门槛不读取或使用正式分数。纯F从第1轮raw+As不同于旧架构tail和已关闭raw-anchor；不重新扫描温度、平滑、权重或比例。

固定F=三SIG_TIME＋统一Sinc坐标/optimizer修正，NoExplicitCFO、joint160、内部融合和默认6×160/cos30头联合训练；无RDrop、CROSS_RX、外部教师/KD/riskq、新参数或部署分支。两新行seed2026092701同初态scratchE20=13L7U、每epoch222步共4440更新。L6300/U56700隐TX/V27000、RX1/3/4/6/8、day1/2/3、split392005、原U/ownEMA规则完全保持。

逻辑矩阵：B0、B1完整复用 20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01 的NoAs/普通As状态、源读出和全部七视图既有预测/评分，不重训练、重测或重评。RAW_NOAS：第一raw L128＋第二raw L128，各0.5真实CE；RAW_AS：第一raw L128＋同包完整practical(raw)L128，各0.5真实CE。两行均原smoothing0.01/temperature1，1136640个学生L前向包。RAW_AS新增As568320包；RAW_NOAS As0。原U学生396900包/ownEMA4440更新及原EMA教师前向保持，外部教师包0。raw/raw不能替代原B0，更不能靠降低其分数制造clean净收益。

为只改变监督输入且匹配原随机消费，原ordinary生成保留为shadow，每步一次；其输出不进入模型或损失，开销单列，不声称删除此成本或提速。真实encoder第一前向及原生L hook逐项核对输入等于原raw；第二槽恰好一次。As配置、场景概率、采样、receiver session及命名流不变；三个训练场景low suburban/mid urban/low urban。所有合法有标签As保留真实监督，学生无需预先答对或高置信。

固定source决定：FINAL20 fullV27000 clean、既有V1800×2draw三场景；RAW_AS−B1的mid/low均值>=+0.5个百分点，且RAW_AS相对B0及RAW_NOAS的fullVclean均>=−0.2个百分点。三条件全满足才有自身E20+30源资格；不是直接延长，需先固定全部衔接及同E50参照。失败关闭whole-raw固定配方，不换raw比例、seed或日程救援，不新增几十个分层门槛。V不拟合参数/信道/阈值，不选epoch。

所有新行固定FINAL20后逐row自动源读出、原168000物理样本×clean及六practical单包默认头预测；全部视图固定后，独立exec scorer等待source_matrix_decision已保存才读truth。报告各视图accuracy/macroF1/worstRX、修正/新增错误、相对B0/B1/RAW_NOAS增量、各suburban相对自身clean差距、实际损失/NN/更新及完整轨迹。n=1模型seed，SD不适用；已暴露测试基准不称新盲测。

Git代码/config身份提交后记入experiment.json。命令 CVS-RFFI/python -u experiments/cvs_F_rawAs20/runner.py；CWD PROJECT/releases/cvs_F_rawAs_E20_20261019_r01，依赖既有不可改sceneRisk发布运行库。不可覆盖输出PROJECT/runs/20261019-phase1-F-rawAs-e20-s2026092701-m2-r01；日志logs/<row>.log，唯一owner codex/root/F-rawAs20。只用GPU4/5，所有owner合计每GPU最多2实验进程、至少12000MB空闲，CUDA初始化在既有共享锁内；FP32/TF32off/CPU2。

仅role/provenance/budget/nonfinite/实际raw槽或接口/输出技术错误停止并保留产物，不自动重启，不动健康旧任务。首步真实checkpoint无query smoke PASS立即训练。一次窄协议负测、一次P0/P1审查、一次归档比较、远端编译和启动后PID/CWD/argv/GPU/日志增长核验；不增加设计审批、签名或文件哈希。

本地11文件compile、真实ordinary shadow/RNG与raw消费、U标签负例、两槽CE/梯度、6项协议负测及native make_args比较均PASS；一次独立P0/P1审查PASS。完整发布依赖边界在N607 --checks执行；首步真实checkpoint smoke由launcher立即串联训练，不增加token/审批。尚未发布或启动。

实际发布代码/config commit `3d4e188890dce6df2c20c9df5576450d06a2eac4`（GitHub独立比对identical），单归档比较/11文件远端编译/真实依赖checks PASS。dispatcher PID2139499；RAW_NOAS GPU4 PID2139570、RAW_AS GPU5 PID2139653，实际PID/CWD/argv/NVML及增长日志均已核实。首步own scratch真实checkpoint无query smoke各PASS一次。四行初态206个state项逐项相等、参数900722；已观察原生444/222步第一槽raw实核全部通过，外部teacher/RDrop/CROSS_RX为0。当前还在L阶段，原U规则配置保持但尚未发生U阶段，不能写成全程核验或全部实验完成。

完训后只读完整分析入口：analysis/cvs_pure_F_rawAs20_analysis_20261019.py。仅complete/source决定已固定后读取全部4440步、20轮及冻结七视图；不会新增模型前向、预测或评分。完整分析尚未执行，不把准备好的工具写成性能证据。
