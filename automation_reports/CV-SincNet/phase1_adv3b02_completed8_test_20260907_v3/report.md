# 已完成8行的冻结诊断测试

用户于2026-09-07明确要求对已完成行测试并提供详细数据。本次冻结07:34UTC已完成E200的8行：A_POINT的392005/392006，POINT_MEMORY的392006/392007，MATCHED_ZERO三个seed，以及TANGENT_ROUTE的392007。其他15行训练继续，B_SAFE392005的E109技术失败不重启。

本次用户请求覆盖原矩阵的target评估延期，仅进行这8个最终checkpoint的诊断测试；不做候选重排、调参、重训或默认晋升。NO_PROMOTION。r3与旧代码混合版本比较边界仍保留。

测试使用各行原始release严格重建学生模型，final_only/E200，FP32、eval模式、无适配、无标签前向。固定ManySig、equalized=1、长度256、目标接收机0/2/5/7/9/10/11、日期0/1/2/3；覆盖固定test_all_day_unseen_rx全量，不设batch上限。输出clean诊断对照和三种正式LEO弱场景。每个物理样本通过opaque ID散列固定分配一个LEO场景，只生成一次received IQ，8个模型共享同一份received；三场景样本互斥。sat_seed=2027，生成与预测batch=256。

数据builder准备输入与单独truth文件；predictor只读公开输入、opaque ID及冻结checkpoint，禁止读取truth。全部8行prediction固定后，独立scorer进程通过opaque ID连接truth。报告总体、每场景、每接收机、每天、每类和混淆矩阵。无真实新类/unknown输入，因此不能声明注册或拒识性能。

脚本tools/pair_completed_test.py；配置configs/phase1_adv3b02_completed8_test_manifest.json。每个真实checkpoint无query smoke检查严格加载和批次/单样本一致性。系统错误立即结束本测试队列并保留全部partial产物；不影响训练进程。输出root不可覆盖。

环境：普通账户N607；Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python；项目=/home/szu2070436088/2510044040/CV-SincNet；新测试release和runs目录均为phase1_adv3b02_completed8_test_20260907_v1；使用空闲GPU1，一个测试进程，顺序预测8行。运行命令：python -u pair_completed_test.py --manifest manifest.json --root <project>/runs/phase1_adv3b02_completed8_test_20260907_v1 --mode queue。

本地7项聚焦验证通过：opaque ID乱序连接、重复/缺失ID拒绝、场景子集计数、prediction不完整时禁止truth评分、输出不可覆盖。独立P0/P1审查通过；queue在prepare之前先检查全部8个真实checkpoint。

v1运行提交bc2dc7af，远端OID读回一致。两文件同步SHA及编译通过，队列PID3298999。首个无query smoke在prepare前失败：默认TF32时batch/single logits最大差0.00218582；独立无query对照关闭TF32后为0.00000524521，argmax一致。v1未读取target输入、未生成预测；保留队列日志。修复仅关闭matmul/cudnn TF32，保持FP32声明与1e-4容差。新v2目录恢复相同8行，不覆盖v1。

实际执行root和release为phase1_adv3b02_completed8_test_20260907_v2。

运行e0af64e6完成全部8个checkpoint无query smoke；prepare首批mask转换失败。远端PyTorch2.1.0+cu121/NumPy2.2.5的torch.from_numpy报expected np.ndarray (got numpy.ndarray)。无query矩阵对照证实仓库numpy_to_tensor_compat缓冲接口数值完全一致。复用已有接口处理预测输入，bool mask使用显式列表转换，不修改模型/数据/配置。v2没有prediction或评分，保留partial输入。新v3继续相同8行。

实际执行root和release为phase1_adv3b02_completed8_test_20260907_v3。
