# v3：可选均衡与直接残差

代码：channel.py完整链路，equalization.py均衡器，residual.py直接残差，batch.py统一入口。采样率必须来自实际数据。保持原始IQ输入输出，不接入STAR的(2,2048)配对流程。

## 开关与默认值

|配置|默认|含义|
|---|---|---|
|processing_route|full|完整传播、接收补偿，再按开关均衡|
|equalization_enabled|false|两条路径均默认关闭|
|equalizer_method|mmse|可选zf；带正则化和增益限制|
|equalizer_length|129|因果FIR抽头数|
|equalizer_delay_samples|64|期望输出总延迟，不是额外物理传播时延|
|equalizer_max_gain_db|20|抑制深衰落处的噪声增强|
|channel_estimation_nmse_db|-25|工程估计误差先验，低质量时增大|
|zf_regularization|1e-6|避免直接除以接近零的响应|

两条路径都保留频偏、相位、RX IQ校准及AGC配置；失锁时跳过精细频偏/相位补偿和均衡。均衡仅针对新注入的传播响应，不做TX PA逆失真或TX IQ校正。开启均衡仍可能改变对RFF识别有用的频谱结构，因此默认关闭，并记录实际启用状态。

```python
from leo_practical import Config, apply_leo_practical_channel_batch
cfg = Config(fs_hz=verified_fs_hz, processing_route="residual",
             equalization_enabled=True, equalizer_method="mmse")
# full/residual可用同一seed、sample_ids、namespace匹配潜在信道参数。
# 批接口用法见README。
```

## 完整链路

生成三状态传播H与时变阴影/散射、多径时延、几何增益，然后注入参考噪声和接收机影响，执行补偿C，均衡开关开启时再应用G，最后AGC：y=AGC[G C(Hx+n)]。

MMSE频响为conj(Hhat)/(|Hhat|²+Pn/Px)，ZF使用conj(Hhat)/(|Hhat|²+epsilon)。二者均经过增益限制、因果延迟和有限FIR截断。Hhat来自practical真实参数加估计误差的仿真代理，不是从原始IQ导频实际估计；不使用TX标签。完整链路每块以首时刻信道构建估计，块内信道变化仍保留，因而存在跟踪误差。

## 直接残差链路

复用同一practical的几何、状态先验、阴影、散射、多径、噪声底、接收机和补偿参数，仅构建H及G的参数，不生成完整星地接收波形。先合成等效核R=GH，再直接作用于输入。可写成y=x_delayed+(R-I_delayed)x+n_residual，实际实现为一次等效滤波，避免两次滤波的中间波形。

实现保留两条广义线性分支：主分支Rx与残余IQ镜像分支R_image·conj(x)，叠加残余频偏/相位和均衡后有色噪声，再执行AGC。不是只给IQ加独立高斯噪声；残差可能与输入相关，也不保证总是很弱。

关闭均衡时G=1，残差仍包含未被同步补偿去掉的多径、衰落及镜像影响，不能凭空删除H。均衡开启时才形成补偿、均衡后的GH残留。

## 简化边界

- residual仅支持post_sync独立快照，每条最长1ms；状态、阴影、散射及几何在记录内冻结。完整链路继续保留时间演变。1ms是实现限制，不是所有场景下的物理相干时间保证。
- residual近似认为相位在均衡器记忆长度内缓变；载波的恒定频偏通过等效核调制处理。长记录、快速状态切换或相位变化应使用full。
- full默认用生成信号与新增噪声功率判断质量；residual使用白输入假设的期望功率代理，避免先生成Hx。彩色输入可能得到不同锁定状态；直接process可显式传入quality_snr_db统一接收质量假设。
- 默认reflect生成记录前边界，串级滤波与等效滤波在开头的延拓不保证相同。完整链路开启均衡时首块不支持require_context；残差原始IQ上下文可提供，但新增噪声历史仍用合成延拓并记录。
- 原始地面IQ已有的噪声、原RX响应和此前处理不被消除；报告的signal_to_added_noise_db不是真实总SNR。没有实现真实导频估计、捕获器或PLL。

## 验收证据

acceptance_results/20260918_v3_r01.json：原生ssr-gpu环境、CPU合成验收20项全部通过。包含默认关闭、禁止任意外部变换、补偿连续性、已注入TX特征保留控制，以及以下新增检查：

- 将ChannelStream.process替换为报错函数，residual仍可运行，证明未调用完整波形路径。
- 冻结信道、精确RX校准、固定质量且无相位演变的控制中，两条路径在去除首部边界后均衡开/关均得到零最大误差。这不是一般时变情形完全等价的证明。
- 已知多径控制验证MMSE和正则化ZF的恢复及增益限制。
- 残差开启均衡后的样本重排复现及元数据JSON序列化通过。

未运行真实数据生成或模型训练；不声称分类性能提升或实测在轨有效性。v1/v2验收文件作为历史证据保留。
