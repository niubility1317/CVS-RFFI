window.COURSE_CHAPTERS = window.COURSE_CHAPTERS || [];
window.COURSE_CHAPTERS.push(
{
 id:'f-start',group:'起点与基础',kicker:'START HERE / 学习路线',title:'从一段 IQ，到设备身份',
 lead:'把两张网络放回同一条信号链：DURA 修复观测，PAMF 提取身份。先知道每一步在解决什么，再逐层打开里面的计算。',
 html:String.raw`
 <div class="intro-badges"><span>20 个章节</span><span>19 组交互实验</span><span>从基础到推导</span><span>离线单文件</span></div>
 <div class="system-flow"><div class="flow-step"><em>y</em><div><b>收到一段 IQ</b><small>256 个复数采样点，含发射机、信道、接收机和噪声的混合影响</small></div></div><div class="flow-arrow">↓</div><div class="flow-step"><em>G</em><div><b>DURA：恢复波形</b><small>时频粗恢复 → 估计接收作用 → 两次学习先验与软一致性求解</small></div></div><div class="flow-arrow">↓</div><div class="flow-step"><em>F</em><div><b>PAMF：识别身份</b><small>时域、频域、PA、参考响应、路径特征 → 160 维表示 → 原余弦分类头</small></div></div></div>
 <h3>先用一个容易记住的类比</h3><p>设想从不同麦克风、不同房间的录音里识别同一个说话者。声音里同时有说话者特点、房间回声、麦克风音色与噪声。DURA 的工作类似修复录音；PAMF 的工作类似综合音高、发声方式和变化规律来识别人。类比只帮助理解分工，网络真正处理的是无线复数 IQ。</p>
 <p>无线模型写成 \(y=R_r(H*T_c(s))+n\)：\(T_c\) 是发射机作用，\(H\) 是信道，\(R_r\) 是接收机作用。识别目标是类别 \(c\)，输入却只有混合观测 \(y\)。换接收机以后，原先最显眼的差异不一定仍然可靠，这就是跨接收机泛化的难点。</p>
 <div class="learning-route"><div class="route-item"><span>01</span><b>建立数学工具</b><p>复数 IQ、张量、卷积、归一化、分类和梯度。</p></div><div class="route-item"><span>02</span><b>沿 PAMF 前向走一遍</b><p>每个分支看什么，怎样融合，训练怎样塑造这些特征。</p></div><div class="route-item"><span>03</span><b>推导 DURA 的恢复</b><p>从接收算子到 MAP，再到有限步 CG 与展开网络。</p></div></div>
 <h3>这份教程如何阅读</h3><p>每章先给直觉和真实尺寸；“逐步推导”可以展开。滑块会计算简化例子的数值，帮助观察因果关系。示意输入不是实验样本，页面也不会加载权重、访问数据集或运行真实模型。</p><p>如果先用于展示，切换“演示模式”；如果要系统学习，按左侧顺序阅读，并用末章自测。章节内的依据行指向项目源码或冻结定义，区分已经实现的机制与解释性的数学模型。</p>
 <div class="note"><b>版本先分清：</b>PAMF 的共同部署主干与训练算法是两个层面。原 T1 使用自身 EMA 与选择性门控；PAMF-REPL 是后来选定的另一套半监督配方。教程并列讲解，保留各自身份。DURA 固定为 v1.0 的两阶段、四次 CG、27 次 FFT/IFFT 路径。</div>
 <h3>两条边界贯穿全文</h3><p>“物理启发”表示网络结构利用了已知的信号规律，不等于它已经把 TX、RX、信道严格分解出来。“恢复到 clean”表示恢复到模拟扰动前的参考 IQ，不等于恢复成没有设备特点的理想符号。恢复器必须尽可能保留可用于识别的细节。</p>
 <p class="source">依据：<code>docs/PROJECT_PROTOCOL.md</code>；<code>docs/PAMF_FINAL_T1_E50.md</code>；<code>docs/PAMF_REPL.md</code>；<code>docs/DURA_V1.md</code>。教学整理日期：2026-10-25。</p>`
},
{
 id:'f-iq',group:'起点与基础',kicker:'FOUNDATIONS / 01',title:'IQ、复数与张量：网络究竟吃什么',
 lead:'先把 I、Q、幅度、相位和 [B,C,T] 对上号。后面每个卷积、FFT 和恢复算子，都在重新组织这些数。',
 html:String.raw`
 <h3>一个采样点是平面上的一个点</h3><p>复数 \(z[n]=I[n]+jQ[n]\)，其中 \(j^2=-1\)。I 是横坐标，Q 是纵坐标。用复数表示信号，可以同时保留幅度和相位：</p><div class="equation">\[a[n]=\sqrt{I[n]^2+Q[n]^2},\qquad \phi[n]=\operatorname{atan2}(Q[n],I[n]).\]</div><p>例如 \(z=3+4j\)，幅度是 5，相位约 \(53.13^\circ\)。只有幅度 5，无法知道这个点位于哪个方向；只看 I 或只看 Q 也会丢失信息。</p>
 <div class="lab" data-lab="iq"></div>
 <details class="derivation" data-depth="deep"><summary>逐步推导：复数乘法怎样改变幅度与相位</summary><div><p>欧拉公式给出 \(e^{j\theta}=\cos\theta+j\sin\theta\)。把 \(z=I+jQ\) 乘以 \(e^{j\theta}\)，展开并合并实虚部：</p><div class="equation">\[z'= (I\cos\theta-Q\sin\theta)+j(I\sin\theta+Q\cos\theta).\]</div><p>这正是二维旋转矩阵。若再乘实数 \(a\)，就是同时旋转与缩放。设 \(z=1+j\)、\(\theta=90^\circ\)，得到 \(-1+j\)：幅度不变，方向转了四分之一圈。</p><p>共轭为 \(\bar z=I-jQ\)，对应关于实轴镜像。复数内积要使用共轭：\(\langle u,v\rangle=\sum_n\bar u[n]v[n]\)。DURA 包含共轭项，所以需要它的实部来定义实线性空间中的内积：\(\langle u,v\rangle_\mathbb R=\Re\sum_n\bar u[n]v[n]\)。</p></div></details>
 <h3>[B,2,256] 不是一张 2×256 图片</h3><table><thead><tr><th>维度</th><th>本项目含义</th><th>容易混淆的地方</th></tr></thead><tbody><tr><td>B</td><td>一次处理几条 IQ 窗口</td><td>不同样本，不能把时间滤波跨到另一条样本</td></tr><tr><td>2</td><td>I、Q 两个实数通道</td><td>合起来对应一条复数序列</td></tr><tr><td>256</td><td>时间方向的 256 个采样点</td><td>每个点都有 I 和 Q</td></tr></tbody></table><p>采样率 \(F_s=25\,\mathrm{MHz}\)，相邻点间隔 \(1/F_s=40\,\mathrm{ns}\)。一个 256 点窗口按 \(256/F_s\) 计为 \(10.24\,\mu\mathrm{s}\)；首末采样时刻的距离是 \(255/F_s\)，两种口径不要混淆。</p>
 <h3>unitRMS 做了什么</h3><p>把 \(z\) 除以窗口 RMS：\(z_{\rm norm}=z/\sqrt{\frac1N\sum_n|z[n]|^2+\epsilon}\)。这样不同包的大尺度功率更可比，但相位关系、相对包络、频谱形状、非线性细节仍可能不同。注意，复数 RMS 与把 2N 个实数直接取均方根相差 \(\sqrt2\) 因子；必须按具体代码的定义解释数值。</p><div class="warning">实数通道数从 2 变成 96，不表示真的收到了 96 路天线。它表示网络生成了 96 种随时间变化的特征。</div>
 <p class="source">依据：<code>configs/pamf_final_t1_e50.json:data</code>；复数计算采用数学定义，交互波形为合成教学信号。</p>`
},
{
 id:'f-conv',group:'起点与基础',kicker:'FOUNDATIONS / 02',title:'卷积、滤波与下采样：一层到底算了什么',
 lead:'卷积不是“把图变小”的黑箱。它是在局部窗口内反复计算加权和，再把有用的局部模式传给下一层。',
 html:String.raw`
 <h3>先手算一条一维滤波</h3><p>离散因果 FIR 的定义是 \(u[n]=\sum_{k=0}^{K-1}h[k]x[n-k]\)。若 \(x=[1,2,0,1]\)、\(h=[1,0.5]\)，左侧补零，则 \(u=[1,2.5,1,1]\)。例如第三项 \(u[2]=1\times0+0.5\times2=1\)。</p><div class="lab" data-lab="conv"></div>
 <p>深度学习库的 Conv1d 通常实现互相关，即在滑动窗口内按存储顺序乘权重，而不是按数学卷积翻转核。可学习卷积核会自行适应这个约定。DURA 的物理 FIR 则有明确的因果与边界定义，不能把两种约定混着推伴随。</p>
 <details class="derivation" data-depth="deep"><summary>逐步推导：从单通道到多通道与可分离卷积</summary><div><p>若输入有 \(C_{\rm in}\) 个通道，每个输出通道都要整合所有输入通道：</p><div class="equation">\[u_o[n]=b_o+\sum_{c=1}^{C_{\rm in}}\sum_{k=0}^{K-1}W_{o,c,k}\,x_c[n+k-p].\]</div><p>一共有 \(C_{\rm out}C_{\rm in}K\) 个核参数（此处暂不计 bias）。深度可分离卷积先每通道各自做长度 K 的局部卷积，再用 1×1 卷积做通道混合，参数变成 \(C_{\rm in}K+C_{\rm in}C_{\rm out}\)。</p><p>例如 48→72、K=5：普通卷积需要 17,280 个核参数；可分离卷积为 240+3,456=3,696。计算减少，但表达结构也受到限制，不能直接推出识别性能更高。</p><p>膨胀卷积把核采样位置间隔设为 d，有效跨度是 \(1+(K-1)d\)。K=5、d=4 时覆盖 17 个点，但仍只有 5 个位置的核参数。</p></div></details>
 <h3>为什么时间长度会变</h3><div class="equation">\[T_{\rm out}=\left\lfloor\frac{T+2p-d(K-1)-1}{s}\right\rfloor+1.\]</div><p>这里 p 是 padding，s 是 stride，d 是 dilation。保持长度的卷积后再做 pool2，通常把时间位置减半。PAMF 时域的 256→128→64→32 来自一次均值降采样和两次最大池化。DURA 的四尺度编码器则是 256→128→64→32，之后用上采样与跳跃连接返回 256 点。</p>
 <h3>卷积、池化、激活各有职责</h3><p>卷积学习局部加权组合；池化压缩时间分辨率并汇总邻域；ReLU/SiLU 引入非线性，让多层网络能够表达超出线性滤波的映射。1×1 卷积主要混合通道，不扩大时间感受范围。</p><p>残差连接把输入与网络修正相加，使网络能够从“保留已有结果，再补一些变化”开始学习。它与 DURA 中的观测残差 \(y-Ax\) 含义不同：前者是连接结构，后者是当前预测无法解释的误差。</p><p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:DSConvBlock1d</code>；<code>experiments/pamf_reception_restore/models.py:Core</code>；<code>experiments/pamf_restoration/models.py:Block</code>。</p>`
},
{
 id:'f-norm',group:'起点与基础',kicker:'FOUNDATIONS / 03',title:'归一化、池化与 160 维特征',
 lead:'特征向量不是 160 个可直接命名的硬件参数。它是网络为了识别而学习的一组坐标，归一化决定这些坐标怎样比较和融合。',
 html:String.raw`
 <h3>均值和标准差，是一段特征的两种摘要</h3><p>设某通道沿时间为 \(a=[1,2,3,4]\)。均值为 2.5，总体方差为 1.25，标准差约 1.118。均值描述总体水平，标准差描述起伏大小。PAMF 的浅中层跨 RX 对比会读取每个通道的时间均值与总体标准差；这与最后的身份向量不是同一对象。</p><div class="lab" data-lab="norm"></div>
 <details class="derivation" data-depth="deep"><summary>逐步推导：标准化与 L2 归一化为什么不同</summary><div><p>标准化先减去平均值、再除以标准差：</p><div class="equation">\[\mu=\frac1d\sum_i a_i,\quad\sigma^2=\frac1d\sum_i(a_i-\mu)^2,\quad \tilde a_i=\frac{a_i-\mu}{\sqrt{\sigma^2+\epsilon}}.\]</div><p>减均值移除整体平移，除标准差让尺度可比。LayerNorm 按规定的特征维度在每个样本内部算统计，通常还有可学习仿射参数 \(\gamma,\beta\)：\(\gamma\odot\tilde a+\beta\)。GroupNorm 则把通道分组，并在各组通道与时间位置上计算统计。它们不是把所有样本混到一起的同一种操作。</p><p>L2 归一化是 \(\hat a=a/\sqrt{\sum_i a_i^2}\)：不减均值，只把向量放到单位球面。余弦分类比较的是单位向量之间的方向。例 \([3,4]\) 与 \([6,8]\) L2 归一化后都是 \([0.6,0.8]\)，方向完全相同。</p><p>实际实现会使用小的 epsilon 或下限，避免零方差、零范数带来除零。是否把均值方差的梯度断开，也是训练算法的一部分；不能仅凭一个“norm”名字判断。</p></div></details>
 <h3>为什么压缩成 160 维</h3><p>序列可能是 \([B,96,32]\)，池化后得到每通道一个摘要，再线性投影到 \([B,160]\)。不同分支投影到相同维度，便于归一化和逐通道残差相加。160 是当前结构选择，不是物理定律，也没有“一维必定对应某种设备缺陷”的保证。</p><p>最终特征 \(h\) 送入分类头，类别权重 \(w_c\) 也有 160 维。它们共同学习如何在空间里区分 TX；冻结后直接使用，不需要再训练一个读出头。</p><div class="note"><b>看尺寸的小习惯：</b>看到 \([B,C,T]\) 先问时间还在不在；看到 \([B,160]\) 就知道时间信息已经被汇总成每个样本的一条向量。三个 160 维分支相加后仍是 160 维，拼接则是 480 维。</div><p class="source">依据：<code>experiments/cvs_residual_identity/model.py:ResidualPhysicalHead</code>；<code>docs/PAMF_FINAL_T1_E50.md:浅中层跨RX对比</code>。交互统计使用总体方差。</p>`
},
{
 id:'f-learning',group:'起点与基础',kicker:'FOUNDATIONS / 04',title:'从分数到训练：Softmax、CE 与梯度',
 lead:'先理解一次预测怎样变成“哪里需要调整”的信号，后面 R-Drop、对比、伪标签和蒸馏就都有了共同语言。',
 html:String.raw`
 <h3>分类头先输出分数，不直接输出概率</h3><p>设三类分数为 \(\ell=[2,1,0]\)。Softmax 把它们变成和为 1 的非负数：</p><div class="equation">\[p_c=\frac{e^{\ell_c}}{\sum_j e^{\ell_j}}.\]</div><p>这个例子约为 \([0.6652,0.2447,0.0900]\)。若真实类别是第一类，交叉熵 \(L_{\rm CE}=-\log p_1\approx0.4076\)。如果同样概率对应的真实类别是第三类，损失则约 2.4076，网络就会受到更强的纠正。</p><div class="lab" data-lab="softmax"></div>
 <details class="derivation" data-depth="deep"><summary>逐步推导：为什么 CE 对分数的梯度是 p−q</summary><div><p>把真实目标写成概率向量 q，硬标签时正确类为 1、其余为 0。交叉熵为 \(L=-\sum_c q_c\log p_c\)。代入 Softmax 的对数：</p><div class="equation">\[\log p_c=\ell_c-\log\sum_j e^{\ell_j},\qquad L=-\sum_cq_c\ell_c+\log\sum_j e^{\ell_j}.\]</div><p>因为 \(\sum_c q_c=1\)，后一项只保留一次。对 \(\ell_k\) 求导，第一项得到 \(-q_k\)，第二项得到 \(p_k\)，所以：</p><div class="equation">\[\frac{\partial L}{\partial\ell_k}=p_k-q_k.\]</div><p>正确类概率不足时，梯度是负数，沿负梯度更新会提高它的分数。错误类概率过高时，梯度为正，会压低它的分数。链式法则再把这个信号传回分类头、融合层与各分支参数。</p><p>标签平滑把 q 换成 \((1-\varepsilon)\operatorname{onehot}(y)+\varepsilon/C\)。当前监督平滑系数为 .01，六类正确类目标约 .99167，其余各约 .001667。它减少对绝对尖锐目标的追逐，不能保证概率已校准。</p></div></details>
 <h3>训练与推理究竟差在哪</h3><table><thead><tr><th>训练</th><th>推理</th></tr></thead><tbody><tr><td>前向计算损失，反向传播梯度，优化器更新参数</td><td>只做前向，读取冻结参数，输出当前样本的结果</td></tr><tr><td>可能有 Dropout、多次前向、样本配对、EMA</td><td>Dropout 关闭，训练专用分支不参与判决</td></tr><tr><td>合法 L 可以使用 TX 标签；U 标签隐藏</td><td>不读取待预测样本的真值来更新或筛选模型</td></tr></tbody></table>
 <p>一次优化的概念写法为 \(\theta\leftarrow\theta-\eta\nabla_\theta L\)。实际 PAMF 使用 AdamW 与分段学习率日程，不是这条最简单的 SGD 公式。梯度方向的教学例子帮助理解作用，但不要把它当成实际 AdamW 数值轨迹。</p><p>U 是没有可见 TX 标签的源数据；伪标签是模型自己的判断，可能会错。恢复器则可以用模拟生成的成对 IQ 误差训练：没有 TX 标签，不等于没有任何训练目标。</p><p class="source">依据：<code>configs/pamf_final_t1_e50.json:training</code>；<code>configs/pamf_ssl_repl.json</code>。Softmax、CE 与链式法则推导为标准数学恒等式。</p>`
}
);
