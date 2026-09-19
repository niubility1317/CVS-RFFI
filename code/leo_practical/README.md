# LEO_practical v3：完整链路与直接残差

独立原始IQ模块，位于E:/type10-7/code/leo_practical，与旧sat_channel.py共享上级目录。没有STAR时频配对依赖，没有修改旧LEO/LEO_WEAK配置。

当前行为及验收以[V3_FULL_RESIDUAL.md](V3_FULL_RESIDUAL.md)为准。V2_RFF_SAFE.md、IMPLEMENTATION.md、ACCEPTANCE_20260918.md与STATIC_REVIEW.json保留历史；v2的均衡禁用策略已改为默认关闭、显式开启。

## 默认接收链

三状态传播、相关阴影/散射、多径与路径衰减→固定参考接收噪声→新增本振/相位/IQ影响→虚拟RX校准→质量相关的频偏及相位补偿→AGC。

默认quality_adaptation_enabled=true：15dB以上locked、5–15dB degraded、5dB以下unlocked。失锁保留粗Doppler，不再假装精细同步成功。阈值及残差是工程代理；真实总SNR、PLL和捕获器未实现。equalization_enabled默认false；开启后使用MMSE或正则化ZF传播均衡，失锁时跳过。TX失真校正、额外频谱白化及任意外部receiver_processor仍禁用。

## API

Python3.9+、NumPy；Torch可选。把code目录加入模块搜索路径：

```python
from leo_practical import Config, apply_leo_practical_channel_batch
cfg = Config(fs_hz=verified_fs_hz, scenario="practical_mid")
# 可选：processing_route="residual"；equalization_enabled=True；equalizer_method="mmse"或"zf"
y, metadata, states = apply_leo_practical_channel_batch(
    raw_iq, cfg, seed=299, sample_ids=physical_ids,
    session_ids=["virtual_rx4_session0"] * len(physical_ids),
    realization_namespace="source_augmentation_v3", receiver_seed=299,
)
```

输入real(B,2,T)或complex(B,T)，不限定T=256。输出保持布局；NumPy为float32/complex64；Torch恢复原device/dtype，但会detach往返CPU，是参考实现而非GPU高性能内核。

seed+namespace+scenario+sample_id决定独立信道，receiver_seed+session_id决定稳定硬件。不用TX标签作为session，不以batch内临时序号替代物理ID。训练动态增强可把epoch/view加入namespace；固定测试另设namespace及seed。

默认对各输入记录做单位RMS，原RMS保存在metadata；有统一功率参照可设normalize_input_rms=False。已有原始噪声不移除，signal_to_added_noise_db不是最终真实总SNR。

完整链路连续记录使用ChannelStream；仅在物理连续证据充分时复用，输入功率标尺保持一致。process支持preceding_context及quality_snr_db。阴影/散射/状态/相位/滤波历史持续；质量与AGC按块处理。默认反射延拓是合成边界近似；require_context要求真实前序IQ。完整链路均衡首块尚无接收端滤波历史入口，不支持与require_context组合。

processing_route="residual"使用ResidualChannel，直接合成practical信道与补偿/均衡的等效残差，不调用完整传播波形过程。仅支持post_sync、每条≤1ms的独立快照，不支持连续流；均衡关闭时保留未均衡的传播损伤。

## 配置与转换

2026-09-19补齐六场景矩阵；场景只改变地面环境及仰角抽样区间，状态内功率、时延和接收机参数仍共用。已有场景名称及默认场景practical_mid保持不变。

|地面环境|高仰角45°—80°|中仰角20°—45°|低仰角10°—30°|
|---|---|---|---|
|郊区|practical_high|practical_mid|practical_low_suburban（新增）|
|城市|practical_high_urban（新增）|practical_mid_urban（新增）|practical_low_urban|

每个名称在configs下有同名JSON；新增三场景支持full/residual及均衡关闭、MMSE、正则化ZF。它们是可显式选择的场景，不自动扩展训练或测试列表，不修改旧LEO_WEAK协议。三组仰角区间与已有配置匹配，并非互不重叠的统计分箱。合成验收见acceptance_results/20260919_six_scenarios_r01.json。

configs含practical_high、practical_mid、practical_low_urban场景以及ground/satellite post_sync和pre_sync配置。场景模板fs_hz=null，必须明确采样率；不能把25MHz测试采样率当成数据元信息。默认载频2.45GHz、高度600km均属工程代理，可显式覆盖。

从code运行python -m leo_practical --help查看原始npy转换接口。需指定input、output-dir（必须新目录）、config、fs-hz、fs-source、fc-source、ids、session-id或sessions、seed、namespace；receiver-profile可叠加地面/星载配置。--route full/residual选择路径；--equalization显式开启均衡，--no-equalization关闭；--equalizer-method mmse/zf选择方法。输出iq.npy、metadata.jsonl、manifest.json，失败保留partial状态。CLI不读取标签、不划分数据、不训练模型。

## 物理及数值边界

状态先验源于聂欣2012论文，按物理语义修正编号歧义。其Ka天气表未直接迁移。相关尺度、状态内功率、时延、RX误差与补偿阈值仍需测量标定。大气仅提供无额外天气简化及外部已计算衰减接口；几何为局部线性近似，不是完整轨道传播。

窗sinc传播滤波器有24点公共因果延迟，元数据记录numerical_filter_latency_samples。它不是均衡器，不删除信号的多径损伤。保护RFF的策略不保证任意强信道下RFF仍可辨，仅避免把TX特征当作待校正误差消除。

acceptance.py及acceptance_results保存合成验收证据；不构成真实在轨或分类性能验证。
