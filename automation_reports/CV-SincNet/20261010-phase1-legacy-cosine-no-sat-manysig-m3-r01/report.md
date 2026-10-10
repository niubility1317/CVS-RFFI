# 旧门控＋cos、关闭星地增强：三seed基线

保留reference_response身份网络、EMA教师0.999、旧batch_neighbor门控、130+70半监督、label_smoothing0.01及cosine2e-4至1e-6。只关闭星地训练增强及其损失。模型seed为2026092701/02/03，固定数据契约，从零训练E200/44400更新。

每个row训练完成后冻结ownE200，直接执行clean+六种practical预测，七视图全部固定后独立scorer连接truth。其余seed训练不读取评分。已有视图VALIDATED_ONCE复用，K/适应/新类为N/A。当前PLANNED，启动和完成以独立读回为准。

本地验证：CPU与CUDA原生冒烟均通过，覆盖E1/80/131/132/200及checkpoint读取；独立P0/P1审查与登记校验通过。每GPU最多2个实验进程，本轮最多3个并行row。正式启动尚待远端核实。
