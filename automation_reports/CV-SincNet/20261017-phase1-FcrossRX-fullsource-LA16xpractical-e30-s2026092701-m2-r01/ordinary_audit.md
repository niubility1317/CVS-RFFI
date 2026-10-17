# 本轮 E30 四格的共享普通 IQ 增强审计

终局补充：四组完整30轮的ordinary调用数、包数、累计MSE、最大幅度及实际概率摘要完全一致，0差异；四组终局均6660次/852480包。

实际证据来自本轮 B10/B11 的 `actual_native_args.json`、`ordinary_base_config.json`、`mechanism_progress.json`、`module_files.json`，以及同预算 B00/B01 的相同文件。四组基准配置完全相同；初次核验时，B00/B01均E30、B10到E20、B11到E9；四组共同完成的前9个epoch，逐epoch ordinary摘要（调用数、包数、累计MSE、最大幅度、实际概率）完全相同。尚不代表未完成epoch已验证。训练保持原配方，此审计不修改健康进程、不产生新IQ/NN/优化更新。

实际 `DataAugmentation`、`SSDG.train_ssdg`、`leo_practical` 均来自已固定release的 `experiments/adv3b02_xuc/code/`，不是顶层 `code/`。模块绝对路径见原run的 `source/module_files.json`；本轮Ag来自自身release的 `attenuation.py`。

实际顺序：原equalized1的256点IQ → 数据集逐包RMS（不去均值）→ 共享ordinary正常视图 → 保留一半ordinary主视图；另一半按B00复制、B10 LA16、B01 practical、B11 practical(LA16)处理。普通增强正常调用先DC/Tukey，再循环移位/增益/相位/CFO/相噪/轻多径/AWGN，随后可选DAC/PA。新LA16在ordinary之后；practical在LA16之后，仅一次。增强后到concat输入没有再次执行逐包RMS；原参考响应支路内部的能量/复标量归一化、原LayerNorm/GroupNorm与cosine归一化仍保留，不能据此断言全模型完全保留或完全消除平坦增益。As还包含自身原生AGC。

实际强度日程为 `s(e)=0.1+0.25*clip((min(e,20)-3)/15,0,1)^1.25`：E1至E3为0.1，E18起为0.35，E21至E30沿用E20的ordinary强度。下表概率是实际调用时的概率；general/anti-shortcut多数算子先按整批掷硬币，再为包采参数；PA按包mask。不是所有损伤同时作用于所有样本。

| 算子 | 实际概率（初期→后期） | 实际范围与作用 |
|---|---:|---|
| DC offset | 3%→10.5%/批 | 各I/Q独立均匀偏置，最大绝对值0.002→0.007 |
| 命名band-edge taper | 2.5%→8.75%/批 | 实现为256点时间Tukey窗；alpha初期0.002至0.01，后期0.007至0.035；不是频域滤波掩码 |
| 循环移位 | 3.5%→12.25%/批 | 包内独立整数移位，最大3→11点；模256重排，无真实前史 |
| 幅度缩放 | 4.5%→15.75%/批 | 每包共同I/Q因子0.9至1.1，范围不随s缩小；到concat前未被另一次包RMS取消 |
| 整体相位 | 4.5%→15.75%/批 | 每包0至2pi，范围不随s缩小 |
| CFO | 3.5%→12.25%/批 | 最大4e-5→1.4e-4 cycles/sample，对应25MHz下1000→3500Hz |
| Wiener相噪 | 3%→10.5%/批 | 每包逐点增量sigma上限0.0006→0.0021rad，累加相位 |
| 轻多径 | 1.8%→6.3%/批 | 2至4个复抽头，同批共享一个核；整数最大延迟0→1点，左侧reflect补边 |
| AWGN | 4%→14%/批 | 每包新增噪声比20至36dB，参照当时波形功率；该范围不随s变化 |
| DAC | 0 | 正常视图未调用DAC链，不因存在DAC参数而算启用 |
| PA | 1.4%→4.9%/包 | `aug_enable_pa_normal=true`使no_pa=false，正常视图确实可进入旧PA链；内含随机记忆多项式/FIR/Rapp/AM-PM/IQ镜像等分支 |
| class signature | 关闭 | 即使函数接收标签，IQ镜像类稳定偏置受enable_class_signature控制，本轮false；未按TX建立增强签名 |

循环移位会将切片末端接到前端，产生人工接缝。固定参考窗口80至160和原网络前向保持，接缝与窗口内容位移是否损害身份要以识别证据判断。轻多径的reflect也是合成边界，不是真实前史。LA16自身不移动索引、不跨边界、不新增噪声或RMS。

ordinary的CFO开关与NoExplicitCFO是不同层面的设置：本轮NoExplicitCFO屏蔽既有参考响应29维中的坐标27，实际可用28维，内部逐包CFO补偿保留；这不表示ordinary的CFO注入已关闭。旧PA链也不能被描述成新的、已证明身份保持的泛化增强，本轮只把它作为四组共有配方继承。

ordinary AWGN与轻多径可以先于As发生；因此完整训练视图可能叠加两层传播/噪声变化。As记录的 `signal_to_added_noise_db` 只衡量信号与该次新增噪声，已包含原录音未知噪声和可能的ordinary噪声，不能当成总SNR。质量白输入代理与实际波形功率的差异、native AGC限幅和残余核仍需按原physics记录分析，不能把配置reference30dB当实际urban SNR。

相同旧普通配方的B00/B01完整E30各6660次调用、852480包，累计逐元素波形MSE均0.2369534169965248、最大绝对采样值均385.13958740234375。这证明两组共有同样的普通增强实现及稀有大幅值现象；没有按组件定位大值来源，也没有证明其造成识别错误，不能直接归因给PA或宣称必须裁剪。新组尚未完成E30，终局继续保留实际统计。

随机流：ordinary使用私有seed20261016051与递增调用序号，恢复外部Python/NumPy/Torch RNG；Ag用独立seed20261017111与step，位置不读取TX/RX/质量；As用既定view seed202610172201，硬件session按RX/day且不按TX定义。两次RDrop共享同一个已经生成的concat IQ，仅dropout重抽。U保留原弱/强和ownEMA/旧门控，没有新增LA16/As；原HiddenU包装删除tx/tx_i/true_tx_i/tx_label/y_tx，并返回y=-1。

四格因果解释限于同一共有ordinary与固定F配方上的新增差异。B00不称完全无增强；B01/B11也不称只受一次纯净星地噪声。新增LA16是否有独立clean价值、是否帮助As，仍由冻结后的四格识别结果决定；本审计不构成其有效性证据。
