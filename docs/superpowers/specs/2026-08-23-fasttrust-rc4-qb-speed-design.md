# FastTrust-RC4-QB稳定性、伪标签质量预算与训练加速设计

## 目标

在不改变Phase1数据权限、Core90初始化、seed392002、U batch256和正式E200预算的前提下：

1. 修复共享U domain/adversarial分支放大和非有限跳步期间日志张量保留导致的显存雪崩；
2. 把P路由从固定低阈值和独立增量预算改为source V_cal校准阈值与H优先的总有效身份预算；
3. 保留严格H和现有Core90 LEO_WEAK拼接增强，关闭没有独立安全事件证据的N路由；
4. 通过稀疏重型source-val评估、单GPU单进程和现有快速数据路径缩短E200墙钟时间。

## 根因与约束

- RC4闭集路径直接组合`base + muse_total`，因此RC4内部domain/adversarial权重必须显式受控。
- 旧worker把`rc4_lambda_domain`固定为1.0；失败行先出现共享U domain/adversarial损失和梯度放大，OOM是次生终点。
- RC4部分集合、条件和负集合日志中存在未detach张量；非有限批次跳过反向传播时计算图可能保留至epoch结束。
- P/N各自0.10的增量有效预算提高了覆盖率，但没有提高clean、LEO均值或最差接收机单元。
- U_s真实TX标签禁止用于训练、路由、校准和选模；新阈值只使用source V_cal交叉拟合证据。

## 设计

### 稳定性

- worker和训练参数显式使用`rc4_lambda_domain=0.16`；所有实验行保持相同。
- epoch遥测写入前统一detach，防止日志保存计算图。
- RC4训练启用epoch内非有限批次保护：累计至少8批且比例达到5%时以明确系统技术失败终止该行并保留产物；准确率和低性能不得触发停止。

### RC4-QB路由

- `RC4Calibration`保存source V_cal交叉拟合得到的P集合安全阈值、精度和覆盖率。
- P集合安全阈值要求目标包含精度0.98并满足最小覆盖；无法达到时`partial_ready=false`，样本进入R。
- Q候选使用严格H＋P，关闭N。
- 总有效身份权重上限为`0.15×B_U`。H优先占用；若H超过上限，按H质量顺序截断并对边界样本裁权；P只填剩余预算。
- P权重使用`(p_set_safe-threshold)/(1-threshold)`的平滑质量余量，而不是固定0.5阈值。
- 保留class×receiver平衡；记录`p_correct`、`estimated_error`、阈值和H/P/R有效覆盖率。

### 加速

- epoch内重型source-val评估：E1–E180每5轮一次，E181–E200逐轮一次，最终评估仍强制执行；epoch内调用由200次降至56次。
- 训练source-val batch使用512；真实checkpoint smoke验证可运行。外部clean＋三LEO评估本身已使用512。
- 正式矩阵使用GPU0–2，每GPU一个训练进程，`RESOURCE_SLOT_LIMIT=1`。
- AMP、TF32、pinned memory、persistent workers、prefetch以及teacher/student视图融合保持不变。
- `torch.compile`和fused AdamW只作为后续非阻断吞吐基准，不进入本次正式矩阵。

## E200最小矩阵

|候选|U身份路由|共享U分支|用途|
|---|---|---|---|
|`E200_QB0_NO_U_ID_SAFE`|无H/P/N|domain scale0.16|无U身份反事实|
|`E200_QB1_STRICT_H_SAFE`|严格H，总有效身份预算0.15|domain scale0.16|与QB2同预算的严格H控制|
|`E200_QB2_H_PRESID_B15`|严格H＋P剩余预算0.15，N关闭|domain scale0.16|优化候选|

所有行固定同split、Core90 checkpoint、seed392002、E200、U batch256和训练步数。科学晋级要求QB2相对QB1：LEO均值至少+0.30pp、接收机单元floor至少+0.30pp、clean下降不超过0.50pp。低性能只形成结论，不中止训练。

## 验证顺序

1. 聚焦RED测试；
2. GREEN及相邻RC4/launcher/protocol/speed回归；
3. 真实Core90 checkpoint无query smoke；
4. 一次独立P0/P1正确性审查；
5. Git提交、自动push和远端OID读回；
6. N607只读preflight、单release归档一次SHA比对、远端编译、启动和一次健康读回。
