# LEO_practical

独立原始IQ信道模块，和`code/sat_channel.py`处于同一上级目录。没有修改旧LEO/LEO_WEAK入口、默认配置、训练代码或数据协议；没有STAR时频构造或配对数据依赖。

实现状态：代码已编写，静态检查另见`IMPLEMENTATION.md`。按本次用户要求没有执行运行测试、仿真、数据生成或训练。工程参数未经过实际LEO测量标定。

## 内容

- `channel.py`：NumPy信道核心、几何、会话硬件、连续时间过程、IQ失衡。
- `batch.py`：独立快照批处理，支持NumPy和可选Torch张量。
- `__main__.py`：原始`.npy`离线转换入口，仅在明确运行时写数据。
- `configs/`：三个场景模板；`fs_hz=null`刻意要求指定采样率。
- `configs/receiver_*.json`：地面/星载接收后的补偿配置及补偿前诊断配置。
- `IMPLEMENTATION.md`：设计对应、实现边界和未执行的验证。

需要Python3.9+、NumPy。Torch只在传入Torch张量时导入。当前是CPU参考实现，Torch适配会detach并往返CPU，不是GPU训练热路径的高性能实现。

## Python接口

将`E:/type10-7/code`加入Python模块搜索路径，或从该目录运行。以下示例中的`verified_fs_hz`、`raw_iq`和`physical_ids`由调用者提供，示例未执行：

```python
from leo_practical import Config, apply_leo_practical_channel_batch

cfg = Config(fs_hz=verified_fs_hz, scenario="practical_mid")
received, rows, terminal_states = apply_leo_practical_channel_batch(
    raw_iq, cfg,                   # real (B,2,T) or complex (B,T)
    seed=299,
    sample_ids=physical_ids,       # B个稳定物理记录ID，不用batch内序号替代
    session_ids=["virtual_rx4_session0"] * len(physical_ids),
    realization_namespace="diagnostic_fixed_v1",
    receiver_seed=299,
)
```

输出保持输入布局，NumPy为float32/complex64；Torch返回原device/dtype。第三项为每条记录结束时状态编号0/1/2，分别对应LOS/PARTIAL/BLOCKED。返回的metadata是每条记录一个普通字典。

`seed+namespace+scenario+sample_id`确定随机信道，与样本处理顺序无关。训练动态增强可显式把epoch/view加入namespace。固定评估使用不同namespace和seed，并保存配置。`receiver_seed+session_id`确定会话硬件，同一个虚拟RX跨样本、场景保持；不要把TX标签作为session_id，也不要从标签推导硬件参数。接口不读取标签、split真值、checkpoint或模型预测。

默认对每条输入IQ归一化至单位RMS，再施加信道。原始RMS保存在元数据。这是增量扰动的相对功率参照，不是绝对接收功率校准。若输入已在统一参考功率单位，可设`normalize_input_rms=False`。输入已有的地面信道、原始噪声及RX误差不会被移除；`signal_to_added_noise_db`不是最终真实总SNR。

## 连续记录

```python
from leo_practical import ChannelStream, receiver_for_session

hardware = receiver_for_session(cfg, seed=299, session_id="virtual_rx4_session0")
stream = ChannelStream(cfg, seed=1234, receiver=hardware)
y1, m1 = stream.process(first_contiguous_block)
y2, m2 = stream.process(next_contiguous_block)
```

只在物理上连续的记录上复用`stream`。输入块必须使用相同功率标尺，不逐块重新做单位RMS。阴影、散射、状态、相位、样本时钟与多径历史连续；AGC按每次调用的块统计，因此AGC结果依赖块划分。

几何支持`geometry_from_vectors`提供同一坐标系、同一历元的相对位置/速度，或者默认简化圆轨道过境。当前局部线性几何最多处理10秒（可配置），不传播完整轨道、不支持未知时间间隔跳跃、不把独立文件当连续过境。长轨迹需要外部轨道状态更新逻辑，当前未实现。

## 地面接收与星载接收后的补偿

`receiver_location=ground`对应星到地；`satellite`对应地到星。地面附近的遮挡/散射仍由同一个三状态模块描述。两套post_sync启动配置使用相同误差强度，没有测量证据时不假定星载接收机一定更差或地面接收机一定完美。

默认post_sync处理顺序：传播→固定参考接收噪声→本振频偏/相位噪声→接收机IQ失衡→估计参数的IQ补偿→数字频偏补偿→相位跟踪误差代理→可选外部定时/均衡→AGC。

| 补偿 | 当前实现 | 默认补偿后残差 |
|---|---|---|
| 公共轨道Doppler＋本振CFO | 用模拟估计值减去总新增频偏 | 100Hz标准差，工程假设 |
| RX IQ失衡 | 估计widely-linear响应的解析逆 | 幅度估计误差0.05dB、相位估计误差0.2°标准差 |
| 公共相位/相位噪声 | 使用真相位减去有相关性的跟踪误差，构造补偿器代理 | 1°标准差、1ms相关时间，工程假设 |
| 公共传播时延 | 输入被视作已截取记录，公共d/c只记元数据 | 没有模拟实际定时捕获 |
| 多径均衡 | 默认关闭，保留残余多径 | 未假设理想均衡器 |

前三项是参数化补偿效果模拟，利用仿真器内部已知损伤及误差分布，不是从IQ运行了真实盲估计、导频估计或PLL。误差不是估计器性能实测。IQ补偿先于数字频偏/相位校正，保留镜像和频移的处理顺序。pre_sync不做这些补偿，但默认仍做AGC；需要纯前端幅度诊断时设置agc_enabled=False。

传入`receiver_processor(received_iq, public_context)`可连接实际定时同步、导频信道估计和均衡代码；返回相同长度的complex IQ和JSON可序列化处理说明。context只含采样率、载频、接收位置和样本时钟，不提供clean参考、真实h、TX标签或随机seed。回调需要在不改变总长度的条件下自行管理保护区；独立快照batch应使用无跨记录状态的处理器，连续流可使用有状态处理器。外部处理后的SNR不能沿用内部统计，metadata明确标注SNR测量位置在外部处理之前。

所有补偿只作用于本次额外注入的虚拟传播/RX误差。WiSig原记录已有的地面接收机、均衡和同步影响不会被逆推出去；input_processing_state须按输入实际来源填写，未知时保留unknown。无需为了启用“地面补偿”对已经均衡的原始数据再次假装执行一次完美均衡。

使用地面或星载配置时，在CLI中增加`--receiver-profile leo_practical/configs/receiver_ground_post_sync.json`或`receiver_satellite_post_sync.json`。补偿前诊断使用`receiver_pre_sync.json`。配置合并后完整字段保存到manifest；切换pre/post不改变样本seed，便于使用同一次信道实现作接收链对照。

## 命令行

从`E:/type10-7/code`运行以下模板；`<...>`必须换成已核实值，不能原样执行。当前不会自动执行这些命令。

```text
python -m leo_practical --input <raw_iq.npy> --output-dir <new_output_directory> --config leo_practical/configs/practical_mid.json --fs-hz <verified_rate_hz> --fs-source <rate_provenance> --fc-source <carrier_provenance> --ids <physical_ids.json> --session-id virtual_rx4_session0 --seed 299 --receiver-seed 299 --namespace diagnostic_fixed_v1
```

输入允许complex `(N,T)`或real `(N,2,T)`，没有256点限制。ID文件为与记录一一对应且唯一的JSON列表。多个RX/session的文件改用`--sessions sessions.json`。仅接受新的输出目录，避免覆盖历史数据。

产物：`iq.npy`、逐记录`metadata.jsonl`、带源代码/config/input散列的`manifest.json`。运行中保留partial文件；失败写FAILED状态，不能当作完整产物。CLI不会重划分训练/验证/测试，不会创建新标签，不会启动模型。

## 主要默认值及来源

| 项目 | 配置 | 来源 |
|---|---|---|
| 三状态语义/仰角可见先验 | LOS、PARTIAL、BLOCKED；城市/郊区 | 聂欣2012论文，按物理语义修正编号歧义 |
| 三个场景 | 郊区45–80°、郊区20–45°、城市10–30° | 工程分层采样，不是过境时间分布 |
| 高度/载频 | 600km、2.45GHz | 工程代理，载频可覆盖 |
| LOS对数幅度 | 均值0/-8dB，标准差1/3dB，遮挡无LOS | 工程启动值，不是论文表2 |
| 散射总功率 | -18/-18/-20dB | 工程启动值，I/Q分量各占一半 |
| 总PDP目标RMS时延 | 20/50/100ns | 工程启动值，三条散射径共享总功率 |
| 阴影相关 | 3m/1m·s⁻¹=3s | 工程启动值；静止终端需显式时间尺度 |
| 状态过程 | CTMC，mix时间1s | 工程启动值，不是平均驻留时间 |
| 噪声 | 参考600km/60°/无遮挡，30dB | 工程相对标定，非绝对链路预算 |
| CFO | RX本振标准差200Hz；补偿误差标准差100Hz | 工程启动值 |
| Wiener相位噪声 | q=2π rad²/s | 工程1Hz等效线宽代理 |
| IQ误差 | 幅度±0.2dB，正交相位±1° | 每session固定的工程硬件先验 |
| AGC | 加噪之后，共同增益上限±30dB | 工程启动值 |

大气默认为`clear_simplified_no_weather_injection`。可传入外部计算的固定`atmosphere_loss_db`及其来源，但本模块没有内置P.618/P.676/P.838预测器，不宣称已经实现物理降雨时间序列。没有PA非线性、采样率偏差、ADC削顶、实际同步捕获器或实际星历传播器。

边界默认为反射延拓，标记`synthetic_reflect`；可选edge或require_context。连续输入自动使用真实历史。分数时延使用因果三次Lagrange插值，它有带宽相关误差，尚未做边界/频响验证，不应把这些数值近似解释成真实信道测量。

不读取目标标签调参，不改CVS正式LEO_WEAK场景定义，不因新增模块自动授权实验。
