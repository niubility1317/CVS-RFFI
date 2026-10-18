# P59 共有普通 IQ 增强的实际运行审计

这是对已运行配方的静态核对，不是新候选，也不改变健康的 P59。No-As 指没有完整 practical 增强，不等于完全无增强。P59 仅改变 E21–30 的三场景分配，普通增强与两条匹配对照相同。

## 实际开关与顺序

原 P43 B00／B01 保存的 actual_native_args 均为 use_aug=true、aug_enable_pa_normal=true、aug_p_dac=0、aug_p_pa=.14、aug_p_cfo=.35、aug_enable_class_signature=false、strong_noise_std=.015。普通路径实际传入 no_pa=false，因此 PA 没有被默认 no_pa 参数关闭。use_concat_sat_channel_aug=false 也不表示 As 关闭：As 由专用训练 hook 添加。NoExplicitCFO 仅屏蔽参考特征第 27 坐标，不关闭 CFO 增强或既有内部补偿。

原 E20 配方在 E13 冻结 ordinary 日程；E21–30 继续使用该日程，而非按续训十轮重新计算。顺序为 DC／频带边缘 taper → 循环移位／增益／相位旋转／CFO／相位噪声／多径／AWGN → 可选 PA。PA 逐包抽样；通用 channel 算子按 batch 触发、逐包取参数，多径滤波器共享。defect-only 分支的开关不能作为普通路径关闭这些算子的证据。

| 算子 | E13 冻结后的实际概率 | 参数范围或上限 |
|---|---:|---|
| PA | .03508405 | mp_sigma .01253002；mem_sigma .01002401；ampm_max .05012007；iq_img_max .00501201 |
| DAC／class signature | 0／关闭 | 没有实际注入 |
| 循环移位 | .08771012 | 整数 −8 至 8，循环边界 |
| 增益 | .11277015 | .9 至 1.1 |
| 相位旋转 | .11277015 | 每包角度 [0, 2π)，角度范围不随日程缩小 |
| CFO | .08771012 | ±.00010024013 cycles/sample |
| 相位噪声 | .07518010 | sigma_max .001503602 |
| AWGN | .10024013 | 20 至 36 dB |
| 多径 | .04510806 | 2 至 4 taps，max_delay=1，reflect |
| DC | .07518010 | max .00501201 |
| 边缘 taper | .06265008 | alpha .00501201 至 .02506003 |

这些是运行参数与概率，不是实际触发次数。当前日志未保存逐算子事件数，不把期望次数当成观测值。普通 PA 确实开启，但尚无受控证据证明其已造成标签破坏；不能据此把下一轮机制直接定为删除 PA。

## 输入与归一化

原数据集先做 complex RMS。ordinary 输出仅 nan_to_num，不恢复 RMS。实际模型入口没有所有分支共享的 whole-IQ RMS：相同 IQ 分别进入骨干与 reference，256 点输入的 pad_crop_iq 原样返回。分支内 GN／LN、reference 局部能量归一化、分类前 embedding L2 归一化都不能证明 .9 至 1.1 增益对整网精确抵消。

完整 As64 从 raw IQ 经原 practical normalize_input 后生成；没有先叠加 ordinary PA／AWGN／多径，也没有在 As 后追加 RMS。RDrop 两次前向复用同一个已生成 IQ。U 教师使用数据集 RMS 后的原视图；U 学生仅加标准差 .015 的噪声，再 nan_to_num，不调用上述 RFFIAugmentor，也不添加 U-As。不能将 L 的算子概率套用到 U。

循环移位可能影响固定参考窗口，reflect 多径也有边界区域；本次只确认其真实实现，没有量化该影响的独立诊断结果。完整 practical 的固定正式 IQ、接收状态、EQ=false/G=1、SNR 和边界设置保持原样。

## 证据与限制

- [P43 B00 实际参数](<E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B00-s2026092701/source/actual_native_args.json>)
- [P43 B01 实际参数](<E:/type10-7/local_artifacts/cvs_FcrossRX_impulse_factorial_E30_20261018/runtime_metadata_r01/B01-s2026092701/source/actual_native_args.json>)
- [冻结日程重建](<E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/ordinary_augmentation_audit_r01.json>) 与 [实际属性表](<E:/type10-7/local_artifacts/cvs_FcrossRX_cleanurban_U_E30_20261018/ordinary_augmentation_actual_r01.csv>)：该旧审计记录包含历史 P39 部分行，不冒充 P59 的完整训练日志；这里只复用相同 E13 算子日程，并结合 P43 实际参数和 P59 训练遥测核对。
- [实际普通相位抽样](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/DataAugmentation.py:875>) 与 [原骨干入口](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/model.py:1672>)。

本次核对新增 IQ／checkpoint／网络前向／训练更新均为 0；没有停止、重启或热修改健康任务。既有的两个 checkpoint 静态接口读取属于历史审计，不能计为本次新增，也不能据此声称做过波形控制实验。
