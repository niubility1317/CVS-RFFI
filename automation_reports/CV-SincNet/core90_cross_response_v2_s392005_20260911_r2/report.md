# V2 head_only单行兼容修复重发

状态：RUNNING_VERIFIED。所属矩阵为[原r1预登记](../core90_cross_response_v2_s392005_20260911_r1/report.md)。r1其余7行保持健康运行，数据、seed392005、E200、scratch、预算与选择规则完全沿用；仅重发head_only，不追加其他行。

r1 head_only在IndependentAuxiliaryTransaction初始化退出，没有进入训练：N607 PyTorch2.1无torch.amp.GradScaler。失败PID978200已消失，原日志/输出保留，不覆盖或原地重启。修复在本地按API可用性回退torch.cuda.amp.GradScaler，仍是独立辅助scaler，主训练不变。

新增回归删除统一GradScaler属性，修复前复现同一AttributeError；修复后transactions+integration共14项通过。独立原问题定点复审通过，无新增P0/P1；同样enabled=amp，不更改optimizer、梯度、scale/unscale/clip、步进或状态保存语义。

新run ID=core90_cross_response_v2_s392005_20260911_r2，新不可变release及精确命令在publication.json记录。仅--variants head_only，最多2总compute/GPU，6500MiB预留，源码归档一次SHA/compile，source scratch strict-load smoke后立即启动。停止规则和合法四场景prediction闭合要求沿用r1；相同兼容性指纹复发不再盲重启。

实机恢复VERIFIED：PID982997、GPU2，init=scratch、首次实际更新、日志增长及完整/proc/GPU绑定通过，见原矩阵matrix_startup_verified.json及本目录publication.json。
