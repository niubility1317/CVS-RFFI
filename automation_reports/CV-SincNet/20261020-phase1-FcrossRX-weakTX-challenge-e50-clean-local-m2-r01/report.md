# F＋跨 RX 同 TX：弱发射机特征挑战

状态 LOCAL_VERIFIED，尚无本轮目标域结果。历史85.1139%是原F+CROSS_RX三seed方法锚点，实际新机制配对使用已完成的本机BASE三seed，不重复训练或预测。

原50轮＝33L＋17U、每轮222更新、RDrop alpha1、LS.01、普通增强、旧门控＋ownEMA＋cos和统一物理Sinc全部保留。关闭星地信道训练，只测试clean168000。默认分类头随原主目标正常联合训练，禁止另训头；新辅助只更新部署特征。

## 假设与固定矩阵

此前源V显示14-7与20-19类间方向接近。加权和joint160推远没有同时改善两弱类，简单条件对齐、MixStyle、Fishr和U特征一致性也无稳定历史收益。本轮检验不同假设：遮去弱TX当下最依赖的部分通道，能否迫使浅、中层学习替代身份线索。参考[RSC原实现](https://github.com/DeLightCMU/RSC)，不是RSC原配方复现或首创声明，也不预设有效。

首种子2026092701两臂RANDOM_WEAK／SALIENCY_WEAK并行。均仅选择源V预先固定weak indices1/3（14-7、20-19），不以新目标误差挑样本。step>1110且step%4==0时，在实际time_down48通道与t2.pool96通道交替，分别遮6／12通道，不重缩放；共2498调用、每层1249次。RANDOM使用私有CPU generator，每个实际weakL样本精确同样遮蔽预算；SALIENCY按第一实际L前向的sum(activation×d true-class-logit/d activation)选择最高通道，mask和选择梯度均detach。

原两次随机RDrop和两层CROSS_RX SupCon保持。额外前向只复用同次实际增强后的当前weakL IQ，运行既有identity backbone的functional eval路径、clone所有buffers、清空再恢复参数缓存、detach原默认W。额外平滑CE系数按原warm5/ramp10渐入至.1，只对fθ反传。没有新增参数、分类器、U监督、教师、原型、真实optimizer步骤或星地视图。实际额外包曝光和计算成本如实报告；不是相同算力比较。额外eval无主随机流消耗，已检查主状态和RNG保持。

初始两行source final50全部完成后，在读取本轮任何目标汇总前按各自源V决定扩02/03：固定弱2平均recall≥BASE+.2个百分点、整体≥BASE−.2、最差源RX≥BASE−.5、任意TX≥BASE−1。规则按臂独立，无目标排序或选择性重跑；失败臂保留n1，首种子仍自动clean测试。匹配参考BASE均为同seed、同数据角色、同原50配方的固定final50；source_proof先核对，绝不加载其权重进入新训练。

## 执行与判断

唯一代码/config身份为release_commit.txt中的Git提交。入口experiments.cvs_f_weaktx_challenge50.driver，普通本机Python3.10.19／Torch2.10+cu128／FULL_FP32_NO_TF32。CWD为LOCAL_RELEASE/cvs_F_weakTX_challenge50_local_20261020_r01，不可覆盖输出LOCAL_RUN/20261020-phase1-FcrossRX-weakTX-challenge-e50-clean-local-m2-r01。完整命令及六个条件行在experiment.json，具体本机路径由已有profile及private submit记录解析，不公开本机私有环境资料。

GPU0最多3个总训练／预测／诊断worker，计入其他owner；派发前空闲系统RAM≥4.5GiB；独立CPU scorer最多1。健康任务保留。数据复用既有VALIDATED_ONCE／EXACT_MATCH：L6300／U56700隐藏TX／V27000，原source RX1/3/4/6/8、day1/2/3、split392005。模型均scratch，ownEMA来自自己。source训练入口关闭原生另拟合RX/day诊断头。

每行固定final50后自动默认头clean预测，168000唯一ID和全预测固定后，独立CPU scorer才连接truth。随后自动聚合，解析全部50epoch／11100step及完整stdout，报告各臂每seed准确率／Macro-F1／逐TX／最差RX和day、配对均值／样本SD／正差数、SALIENCY−RANDOM、U实际曝光和接纳率、额外L包前向及时间显存。所有负结果保留，n1无SD；当前已暴露基准不称新独立盲测、Phase2或卫星部署。科学失败仍保留预测评分，不因差分数停健康训练。

技术停止条件：source角色／checkpoint来源／配置或输出冲突、非有限损失梯度、真实更新或scope计数错误、worker异常或不能产生合法预测评分时停止新派发，保留所有现有输出和健康worker，不自动重启或覆盖。没有额外审查、seal或权限阻塞。

聚焦检查PASS：8个协议负例、已知top6通道oracle、私有随机mask精确预算、2498／1249分层日程、实际F两臂两层梯度、辅助W梯度None而Sinc非零、主W正常、原RDrop/SupCon／RNG／state不变、warmup完全一致、EMA／实际feature导出保持900722参数、真实匹配final50 checkpoint的零输入no-query smoke。检查读取IQ和target均0，没有optimizer或head拟合。唯一独立P0/P1审查PASS，未发现会使下一真实run错误或越权的缺陷。

发布状态 **VERIFIED／RUNNING**。执行提交 `294e79315fd628c97dca7e312d7e9b0bb3c5a0b6` 已推送并独立对齐远端OID。两source worker的实际PID／CWD／cmdline／物理GPU0驻留／resolved33L＋17U／900722参数和日志增长已回读。CUDA合成输入完整main＋aux联合反传PASS，零IQ或target读取，零optimizer或head拟合。正在前5轮原配方预热，尚无本轮clean分数；源扩展在新目标访问前固定。
