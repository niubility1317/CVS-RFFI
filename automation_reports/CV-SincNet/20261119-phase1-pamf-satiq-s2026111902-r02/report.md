# SatIQ跨站实验第二次发布

运行编号 `20261119-phase1-pamf-satiq-s2026111902-r02`。数据、划分、配置、训练预算及全部对照沿用[本组设计](../20261119-phase1-pamf-satiq-s2026111902-r01/report.md)。本次修复服务器上PyTorch与NumPy的数组交换接口。实际代码与命令见[实验登记](experiment.json)。

N607独立读回确认进程1770759已退出。DLPack已成功载入英国训练数组，生成了本次随机初始化的权重文件。其后在GPU无瑞士测试检查中，旧Sinc层在 `torch.inference_mode()` 访问临时张量的版本计数，触发 `RuntimeError: Inference tensors do not track version counter`。本次没有发生训练更新或瑞士预测。原始运行根、权重和日志已保留。

修复将验证及预测切换为 `torch.no_grad()`。冻结Sinc所用的版本计数继续工作。两组在N607实测了源域前向、反向、一次内存内优化更新和只读数组预测。正式重发为 [r03](../20261119-phase1-pamf-satiq-s2026111902-r03/report.md)，使用相同的数据、设置及预算。
