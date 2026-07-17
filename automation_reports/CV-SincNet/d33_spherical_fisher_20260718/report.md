# D33球面同尺度注册与Fisher快速适应实验

## 登记

- 实验ID：`d33_spherical_fisher_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`LOCAL_VERIFIED_PRELAUNCH`。
- 前置回顾：D30-D32三轮回顾已完成并提交。D30静态envelope旁路、D31事后bias存在训练/部署面不一致、D32约-9内生cap在fit support安全但held泛化失败；D33停止继续扫描bias/CVaR/DALI权重。
- 目标：用old/new统一球面centroid、robust radius和`-d/r-log(r)`评分消除跨组标尺失配，同时以对角Fisher近闭式Stage2-B降低适配MAC；域适应与新类注册保持同run、同等优先。
- 比较：Z0、历史B3诊断、C0、D33-A/B/C、D33-B3-FAST；7候选×3场景×5折=105行。

## 方法锁

所有D33候选使用同一个288维拼接身份空间：同一已接收LEO_weak IQ的z160、FFT96、RF32经块归一化后拼接；不增加物理样本、信道overlay或support view。

对每个注册类`c`，在冻结Stage2-B对角阵`D`后计算：

`u_i=norm(exp(D)⊙x_i)`，`p_c=norm(mean_{i:y_i=c}u_i)`，`d_c(x)=1-u(x)^T p_c`。

K>1时只用support LOSO自类角距离，在固定36点网格`q∈{0.5,0.75,0.9}`、`rho∈{0.25,0.5,0.75,1}`、`cap∈{1.15,1.25,1.5}`中估计类半径：原始分位数向全类median收缩并限制半径比。推理逐样本对所有注册类计算：

`s_c(x)=-d_c(x)/max(r_c,1e-4)-log(max(r_c,1e-4))`。

|候选|Stage2-B|Stage2-C/注册|选择目标|优化步|
|---|---|---|---|---:|
|D33-A|D26 AdamW 15步compact diagonal|统一球面centroid+robust radius|LOSO overall优先|15|
|D33-B|同上|同上|LOSO harmonic balance优先|15|
|D33-C|同上|同上|LOSO逐类floor优先|15|
|D33-B3-FAST|对角Fisher近闭式+固定5点收缩LOSO|统一球面centroid+balanced radius|联合balance|0|

K=1不存在独立类内半径证据，所有类统一`r=1`，评分严格退化为常数平移的cosine；不构造伪LOO、不做梯度更新。D33不使用DALI进行prediction，授权int8组件仅保持sealed bundle可用状态，不计入active predictor。

## 本地实现与验证

- 新增`code/cvsrffi/stage2_b3_fisher_closed_form.py`：对角Fisher类间/类内方差比、严格分块零均值与`log(1.5)`盒投影；固定5点强度只用old support LOSO选择。
- 新增`code/cvsrffi/stage2_d33_spherical_registration.py`：36点class-symmetric LOSO radius、A/B/C策略、K1旁路、逐样本all-registered scoring。
- 新增两组核心测试与共享runner集成测试；D33、Fisher、runner和原compact相邻测试54/54通过，`py_compile`、launcher `bash -n`和`git diff --check`通过。
- 6旧类K10 Fisher：活动2,016标量，估算865,728 adaptation MAC，相对Adam15参考5,443,200降低84.10%。
- 6旧+20新球面状态：活动7,828参数、实际常驻8,848B、K10适配估算2,564,640MAC；0个Stage2-C优化步，无dense query图。所有old/new centroid统一存为per-class symmetric int8+FP32 scale，不常驻FP32 centroid；相对原FP32 centroid版31,208B降低71.65%。
- 共享runner已完成candidate lock v11、105行fold、K10 full state、完整trace、MAC/延迟/状态、selection/receipt统一positive helper和真实old-score alias语义。2-new合成fold/full audit四个候选全部跑通；FAST为0步，Adam支线为15步。
- 本地SHA：runner `930a565a...5b50a`；D33 spherical `af4da352...50423f`；Fisher `2cc05c0f...5d8ef`；launcher `e5f30c76...ed536`。

## 协议与门禁

- receiver `20-1`、seed `713101`、K10、5个新类、3个LEO_weak场景；沿用现有密封support包，不新增数据准备。
- 每个physical support只有一个已经叠加LEO_weak的IQ观测；z160/FFT96/RF32只是同一IQ的确定性数学描述，不计入K。
- query为测试集且本轮保持未打开；外层held support只做开发泛化评估，不进入训练、半径选择或checkpoint选择。
- 无query标签、query角色Oracle、真实batch类数、类别配额、全局分配；预测是逐样本all-registered argmax。
- clean/source/cache/control-flow不可达；int8 Phase1组件只读且不更新。当前仍是`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`，不能作为正式query或多receiver确认性能声明。
- 晋级前必须同时改善注册后old、new、H、forgetting和逐类floor，重点核查14-7、09f8、f608；未超过B3同行联合指标不得扩展正式5 receiver×5 seed×3场景矩阵。

## N607计划

- 本地runner和核心测试已全部通过；下一步执行直接N607只读preflight与live GPU/process inventory。
- 仅同步共享runner、D33 spherical core、B3 Fisher core和D33 launcher；不修改或上传`stage2_diag_cosine_exploration.py`，远端固定SHA必须保持`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。
- 计划远端cwd：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU由preflight后记录。
- 计划输出：`runs/d33_spherical_fisher_20260718/output/support_screen_v1`；日志：`logs/d33_spherical_fisher_20260718/support_screen_v1.log`。

## 完成后回填

待回填完整105行训练日志、逐候选/场景/类结果、独立held矩阵、support合法清单与哈希、资源审计、selection、RECEIPT、远端命令/PID/GPU、artifact SHA、异常与下一轮判断。
