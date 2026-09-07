# 已完成19行的增量测试

用户于2026-09-07再次请求测试已完成行。22:31现场确认19行E200完成、4行运行、B_SAFE392005历史失败1行。本轮冻结19行；其中8行复用已固定prediction，新增11行运行无query smoke和预测。全部19行prediction完整后独立scorer评分。

严格复用phase1_adv3b02_completed8_test_20260907_v3的168000个physical、clean对照及单次LEO received、场景分配、seed2027、batch256与truth表；不生成新观测、不覆盖原结果。完整FP32、TF32关闭、冻结学生最终E200模型、模型state不更新。每行使用其原始release严格重建，新增B_SAFE的r3版本和旧21行比较只能诊断，不晋升、不调参、不重训。

普通N607、GPU1、单进程顺序；release/runs根为phase1_adv3b02_completed19_test_20260907_v1。脚本tools/pair_completed_test.py --manifest manifest.json --root <runs>/phase1_adv3b02_completed19_test_20260907_v1 --mode queue。系统错误只结束本测试队列、保留产物；不修改训练。预期产物results.json、done.json及11份新prediction，8份引用原prediction。
