# Ground A三阶段配对已完成

ANALYZED/VERIFIED：8行、320parent（Affine/Conditional各160）全部完成；唯一PID716759/start8502953已退出，runtime9b761828a。原输入不变，无训练、source/query访问；固定A先于truth连接。

实际程序wall224.820874s、CPU峰值RSS1243791360B；非星载资源或链路传输测量。真实A/B−A已测；原始完整行报告与配对表见 `results/raw_support_pairing/<row_id>`；[完整组合三阶段与K×新增类数报告](../../../docs/D92_GROUND_A_PAIRED_SUPPORT_RESULT_20261001.md)已完成，并经35项两reporter合成检查。完整完成证据见 [evidence/supervisor_readback_1790837667387096200.json](evidence/supervisor_readback_1790837667387096200.json) 和 [evidence/support_download_readback_20261001.json](evidence/support_download_readback_20261001.json)。不要重复publisher/评分。

历史预登记：

# D92 Ground A 三阶段配对补充

状态：PLANNED，未启动，A/B−A 尚未测量。

固定 Affine 与 Conditional 各4行；每种方法160个parent/1800条已冻结路径。先核对完整独立 summary，再在同一物理旧held support上以原地面6类float32分类头补测A。B/C只读取同源固定预测，不训练、不重新拟合；source/query样本均不读取。

矩阵K=1、5、10、20，旧类6，新增类0、2、5、10、20。分别报告A_old、B_old、C_old、C_new、适应提升、旧类下降、绝对差和H；全部K×新增类数与receiver/scenario/model分层保留。真实K1没有独立held样本时为N/A；不能以R0/B0填A。

三个理想目标分别为适应提升≥10个百分点、注册旧类下降≤1个百分点、新旧绝对差≤3个百分点，当前逐步提升，不作为本轮新增硬门槛。配对结果是support OOF/proxy诊断，不是query准确率或全新独立验证。结果不回流调参或选模。

Root为唯一launch owner，CPU1lane/BLAS2线程；Conditional健康独立分析完成后再唯一启动此独占run。任何直接技术失败保留已有输出、终止本队列，不自动重试或干预其他进程。

原地面packet r02实际文件8737B/8735B、原头各3840B；checkpoint来源匹配source-only scratch final200，packet runtime0145234。星地实际新增传输、GPU峰值和能耗未测量，为N/A。此分析保存实际CPU时间/RSS与固定prediction文件，文件字节不替代传输或星载资源测量。

配置：configs/d92_ground_a_support_20261001.json；入口：tools/run_d92_ground_a_support.py；评分：tools/score_d92_ground_a_support.py。
