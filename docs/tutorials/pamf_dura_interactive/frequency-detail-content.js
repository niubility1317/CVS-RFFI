/* Source-checked expansion; load after pamf-content.js. */
(function () {
  'use strict';
  const chapters = window.COURSE_CHAPTERS;
  if (!Array.isArray(chapters)) throw new Error('PAMF chapters must load before frequency details');
  const index = chapters.findIndex(chapter => chapter.id === 'p-frequency');
  if (index < 0) throw new Error('Existing p-frequency chapter was not found');
  const details = [
    {
      id: 'p-frequency', group: 'PAMF', kicker: '04A / 从复数波形到四条频率曲线',
      title: '频域特征：DFT、正负频率与精确的池化边界',
      lead: String.raw`这一章先不进入神经网络。把一包 \(I/Q\) 一步步变成 \([B,4,32]\)，辨认每个数怎样算出来、保留了什么，以及丢失了什么。`,
      html: String.raw`
<h3>为什么另看一次频谱</h3>
<p>时域关注变化的先后，频域关注能量分布的位置。平均幅度相同的两段 IQ，频带边缘或镜像侧能量仍可能不同。PAMF 默认把相同频率绝对值的正、负两侧放在一起比较，用四条曲线保留强弱和不对称性；它们是接收信号的描述，不直接等于某个 TX 或 RX 的硬件参数。</p>
<p class="note">本章对应实际 <code>lite_d/no_dac</code> 的 <code>raw_fft</code> 路径。频域前处理没有新增可学习参数；默认也没有 FFT 相位差通道。可选 <code>sinc_phase_asym</code> 是另一条来源分支，不能把它的名字或相位公式写进默认结构。</p>
<h3>先固定符号和实际输入</h3>
<table class="mini-table"><thead><tr><th>量或开关</th><th>本版实际值</th><th>含义</th></tr></thead><tbody>
<tr><td>\(x[n]=I[n]+jQ[n]\)</td><td>输入 \([B,2,256]\)</td><td>\(B\) 为批大小；每条有 \(N=256\) 个复数采样</td></tr>
<tr><td>\(F_s\)</td><td>\(25\,\mathrm{MHz}\)</td><td>采样间隔为 \(40\,\mathrm{ns}\)</td></tr>
<tr><td><code>freq_feature_source</code></td><td><code>raw_fft</code></td><td>直接使用当前输入的复数频谱</td></tr>
<tr><td><code>use_rfft_pair</code></td><td><code>True</code></td><td>偶数长度时由两条实数 FFT 重建复数 FFT</td></tr>
<tr><td>\(\epsilon\)</td><td>\(10^{-8}\)</td><td>功率比等公式的稳定项</td></tr>
<tr><td><code>freq_bands</code></td><td>\(32\)</td><td>池化输出 \(K=\max(4,\text{freq\_bands})=32\)</td></tr>
</tbody></table>
<p>输入先经过原模型的长度适配；当前已经是规定长度时不改变它。频域函数内部没有额外去均值、乘窗、RMS 归一化或 fftshift，也没有给 FFT 传入额外的归一化模式。上游数据的 unitRMS 约定和这里的算法操作要分开描述。</p>
<h3>第一步：DFT 是对不同旋转速度的投影</h3>
<div class="equation">\[
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\qquad k=0,\ldots,N-1.
\]</div>
<p>对每个 \(k\)，把输入与一条以该速度旋转的复数基底相乘再累加。相位变化匹配时，各项方向对齐、结果较大；不匹配时容易互相抵消。默认前向 DFT 不除以 \(N\)，因此幅值和后面的功率尺度都依赖这个约定。[<a href="https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html" target="_blank" rel="noreferrer">PyTorch FFT 默认归一化说明</a>]</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：实部、虚部，以及为何可以用两次 rfft</summary>
<p>令 \(\theta=2\pi kn/N\)。将 \((I+jQ)(\cos\theta-j\sin\theta)\) 展开，实部是 \(I\cos\theta+Q\sin\theta\)，虚部是 \(Q\cos\theta-I\sin\theta\)。所以 \(X=\mathcal F(I)+j\mathcal F(Q)\)。</p>
<p>由于 \(I,Q\) 分别为实数，它们各自的频谱满足共轭对称。长度为 \(256\) 时，每次 rfft 返回 \(129\) 项，包含 DC 与 Nyquist。源码取内部的 \([1:-1]\) 共 \(127\) 项，先共轭再反转，补到各自的正半谱后面，恢复长度 \(256\)，最后组合为 \(\mathcal F(I)+j\mathcal F(Q)\)。</p>
<p>两条实信号各自可共轭补全，不表示合成的复数 IQ 频谱也必须正负对称。完整 \(X[k]\) 一般不满足 \(X[N-k]=\overline{X[k]}\)。</p>
</details>
<p>手算一个仅说明 DFT 的缩小例子：\(N=4\)，\(x=(1,j,-1,-j)\)。它恰是正向一个周期的复指数，DFT 在 \(k=1\) 为 \(4\)，其余为 \(0\)。将输入取共轭，峰值转到 \(k=3\)，也就是负频率。真实模型仍使用 \(N=256\)。</p>
<div class="lab" data-lab="fft-detail"></div>
<h3>第二步：功率后，怎样准确配对正负频率</h3>
<div class="equation">\[
P[k]=(\Re X[k])^2+(\Im X[k])^2,\qquad
P_k^+=P[k],\quad P_k^-=P[N-k],\quad k=1,\ldots,127.
\]</div>
<p>功率是每个复数频谱值的模平方。实现将 NaN 置零、正无穷置 \(10^6\)、负无穷置零，再限制最小值为零。正常有限功率此时没有统一截到 \(10^6\)；不同后续公式有各自的截断位置。</p>
<table class="mini-table"><thead><tr><th>位置</th><th>源码索引</th><th>处理</th></tr></thead><tbody>
<tr><td>DC</td><td>\(k=0\)</td><td>不进入镜像配对</td></tr>
<tr><td>正侧</td><td>\(k=1,\ldots,127\)</td><td><code>power[:,1:128]</code></td></tr>
<tr><td>Nyquist</td><td>\(k=128\)</td><td>不进入镜像配对</td></tr>
<tr><td>负侧</td><td>先取 \(129,\ldots,255\)，再反转</td><td>配对顺序为 \(255,254,\ldots,129\)</td></tr>
</tbody></table>
<p>第一对为 \((+1,-1)\)，最后一对为 \((+127,-127)\)，共 \(127\) 对。频率间隔为 \(F_s/N=97.65625\,\mathrm{kHz}\)，最后一对绝对频率为 \(12.40234375\,\mathrm{MHz}\)；Nyquist 的 \(12.5\,\mathrm{MHz}\) 不被重复当成两侧。</p>
<p>通用辅助函数还为极短长度准备了回退：当 \(\lfloor L/2\rfloor\le1\) 时，两侧都取第一项；否则按两侧较短长度截齐。奇数长度用直接复数 FFT。当前入口固定适配到 \(256\)，所以这些不是实际默认路线中的额外步骤。</p>
<h3>第三步：为什么保留四条，而不是一条总功率</h3>
<div class="equation">\[
\begin{aligned}
v_{0,k}&=\log(1+\min(P_k^+,10^6)),\\
v_{1,k}&=\log(1+\min(P_k^-,10^6)),\\
v_{2,k}&=\operatorname{clip}\!\left(\log(P_k^++\epsilon)-\log(P_k^-+\epsilon),-20,20\right),\\
v_{3,k}&=\frac{|P_k^+-P_k^-|}{P_k^++P_k^-+\epsilon}.
\end{aligned}
\]</div>
<p>所有 log 都是自然对数。前两条保留两侧强度，并压缩动态范围；第三条保留谁更强的符号；第四条只描述不对称的幅度。log1p 是 \(\log(1+u)\)，不是 \(\log u\)，也不是 dB 变换。比值通道使用加稳定项后的两个 log 之差，不是先用已经取过 log1p 的前两通道相减。</p>
<p>若 \(P^+=9,P^-=1\)，忽略极小稳定项，四个数为 \((\log10,\log2,\log9,0.8)\approx(2.3026,0.6931,2.1972,0.8)\)。交换两侧，前两项交换、第三项变号、第四项不变；两侧都为零时四项均为零。</p>
<p>上一段 DFT 玩具例子会得到一对 \(P^+=16,P^-=0\)。原始 log 比约为 \(\log[(16+10^{-8})/10^{-8}]\approx21.1933\)，会截为 \(20\)；这展示了稳定项和截断各自的作用。</p>
<div class="lab" data-lab="frequency-channels"></div>
<h3>第四步：从 \(127\) 个位置精确压到 \(32\) 个</h3>
<p>源码同时堆叠四条描述和两条线性功率，形成 \([B,6,127]\)，一次自适应平均池化得到 \([B,6,32]\)。前四通道成为送入网络的 \([B,4,32]\)；后两通道用于计算谱统计。没有把线性功率再作为第五、第六个神经输入通道。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：自适应平均池化的边界为何会重叠</summary>
<p>用从零开始的输出索引 \(j=0,\ldots,31\)，输入曲线记为 \(v[r]\)，\(r=0,\ldots,126\)。PyTorch 使用</p>
<div class="equation">\[
s_j=\left\lfloor\frac{127j}{32}\right\rfloor,\qquad
e_j=\left\lceil\frac{127(j+1)}{32}\right\rceil,\qquad
\bar v[j]=\frac1{e_j-s_j}\sum_{r=s_j}^{e_j-1}v[r].
\]</div>
<p>第一段为 \([0,4)\)，第二段为 \([3,8)\)，最后一段为 \([123,127)\)。第一段和第二段都包含输入索引 \(3\)；边界重叠是自适应池化的定义，不是误重复数据。首尾各平均 \(4\) 点，中间各平均 \(5\) 点，不是把 \(127\) 点切成互不重叠的等长组。[<a href="https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/AdaptivePooling.h" target="_blank" rel="noreferrer">PyTorch 自适应池化边界实现</a>]</p>
<p>若玩具输入曲线 \(v[r]=r\)，前两项为 \((0+1+2+3)/4=1.5\)、\((3+4+5+6+7)/5=5\)，最后一项为 \(124.5\)。这里演示的是曲线索引值，原频谱配对索引还要加一。</p>
</details>
<p>先做非线性再平均很重要。功率 \(1,9\) 的 log1p 平均为 \((\log2+\log10)/2\approx1.4979\)，而先平均功率再取 log1p 得 \(\log6\approx1.7918\)。后者不是当前实现。自适应平均池化也不同于 D92 的 FFT96 线性插值。</p>
<div class="lab" data-lab="frequency-pooling"></div>
<details class="derivation" data-depth="deep"><summary>相位边界核实：默认没有相位差，可选分支究竟计算什么</summary>
<p>默认 raw FFT 先产生复数 \(X[k]\)，然后立即取功率，没有调用 angle、atan2、相位解缠或相邻频点相位差。不能凭“FFT 是复数”就给默认输入添加相位通道。</p>
<p>可选 <code>sinc_phase_asym</code> 读取的是 Sinc 滤后每个通道的<strong>时间相邻样本</strong>，不是 FFT 相邻频点。记滤后样本为 \(z[n]=i[n]+jq[n]\)，它计算</p>
<div class="equation">\[
c_n=q[n]i[n-1]-i[n]q[n-1],\quad
d_n=i[n]i[n-1]+q[n]q[n-1],\quad
\Delta\phi_n=\operatorname{atan2}(c_n,d_n+\epsilon),\quad n=1,\ldots,L-1.
\]</div>
<p>没有最后一点到第一点的环绕，没有给这个频域可选函数左补一个零；增量长度为 \(L-1\)，输入长度为 \(256\) 时有 \(255\) 项。当 \(L\le1\) 时，该条件块不执行。atan2 返回 \([-\pi,\pi]\) 内的主值角，源码没有 unwrap；之后对增量求总体标准差（<code>unbiased=False</code>），将该标准差的最小值限制为 \(\epsilon\)，再取 \(\log[1+\min(\sigma_\phi,10^6)]\) 替换第二条能量曲线。</p>
<p>因此相位从 \(170^\circ\) 到 \(-170^\circ\) 时，复数乘积给出的主值增量约为 \(+20^\circ\)，而不是直接相减得到的 \(-340^\circ\)。但若增量在 \(+179^\circ,-179^\circ\) 间跳动，普通标准差仍很大；它不是圆周统计标准差。加在 dot 上的 \(\epsilon\) 还会影响近零信号的角度。另一条可选时域 <code>phase_delta</code> stem 才会左补零，它也不属于本版默认路径。</p>
</details>
<p class="warning">常见误区：默认不是“只用半个复数频谱”，也没有把负频率当冗余删掉。功率会丢失频谱相位；对整包乘公共相位不会改变功率，但一般接收机变换、截窗或信道变化并不因此消失。这里的 \(32\) 个位置是描述分辨率，不是 \(32\) 条独立物理观测。</p>
<p class="source">项目依据：<code>experiments/cvs_identity_ce/model.py:18</code>、<code>experiments/adv3b02_xuc/code/model.py:1057</code>、<code>experiments/adv3b02_xuc/code/model.py:1484</code>、<code>experiments/adv3b02_xuc/code/model.py:1564</code>、<code>experiments/adv3b02_xuc/code/model.py:2003</code>；可选相位路径仅作边界辨析：<code>experiments/adv3b02_xuc/code/model.py:1525</code>、<code>experiments/adv3b02_xuc/code/model.py:126</code>。</p>`
    },
    {
      id: 'p-frequency-network', group: 'PAMF', kicker: '04B / 从频率曲线到可训练表示',
      title: '频域网络：门控、可分离卷积、归一化与谱统计',
      lead: String.raw`四条固定曲线只是描述。接下来学习哪些频率位置和通道组合有用，得到一个 \(160\) 维向量。每层形状、固定超参数和可学习参数都可以逐项算清。`,
      html: String.raw`
<h3>为什么固定描述之后还需要学习</h3>
<p>某一处镜像不对称可能来自身份，也可能来自接收机、信道或噪声。固定公式只把现象表达出来，不能替网络决定哪些组合有用。门控先温和调整频率位置，卷积学习局部形状和跨通道组合，最后汇总为供联合分类使用的向量。</p>
<p>直觉上，固定特征是把频谱整理成四张标尺一致的图；可学习网络才是读图的人。下面的“固定”指超参数不由梯度学习，不表示所有样本的中间输出都相同。</p>
<h3>当前默认值：不要把通用默认当作本版尺寸</h3>
<table class="mini-table"><thead><tr><th>配置</th><th>实际值</th><th>固定／可学习</th></tr></thead><tbody>
<tr><td>通道与 embedding</td><td>输入 \(4\)，三块输出 \(16,32,32\)，embedding 为 \(160\)</td><td>固定结构</td></tr>
<tr><td>频带门控</td><td>核长 \(5\)，\(\alpha_f=0.6\)，开启</td><td>核尺寸／倍率固定；卷积权重和偏置可学习</td></tr>
<tr><td>频率稳定性／CRRA</td><td><code>freq_stability_mode='off'</code>，<code>use_crra=False</code></td><td>默认不执行可选支路</td></tr>
<tr><td>统计路径</td><td><code>use_freq_stats=True</code>，<code>use_aux_spectral_stats=True</code>，<code>use_circularity=True</code></td><td>统计公式固定；统计投影可学习</td></tr>
<tr><td>独立 DAC 支路</td><td><code>no_dac</code>，无 subband aggregator</td><td>旧名 <code>dac_stats</code> 仍可存在</td></tr>
<tr><td>Dropout</td><td>三块各 \(p=0.05\)；统计投影 \(p=0.1125\)</td><td>固定训练设置，推理时关闭</td></tr>
</tbody></table>
<p>统计投影的 dropout 来自本版整体 <code>drop</code>：M 配置起点为 \(0.40\)，<code>lite_d</code> 加 \(0.05\) 得 \(0.45\)，再乘 \(0.25\) 得 \(0.1125\)。它不是三块卷积的 \(0.05\)。</p>
<h3>第一步：频带门控究竟在学习什么</h3>
<p>记上一章输出为 \(V\in\mathbb R^{B\times4\times32}\)。门控卷积沿频率轴移动，读取四个通道及邻近位置，输出一条长度 \(32\) 的曲线：</p>
<div class="equation">\[
u_k=b_g+\sum_{c=0}^{3}\sum_{r=0}^{4}W_{c,r}V_{c,k+r-2},
\quad g_k=\sigma(u_k),\quad
\widetilde V_{c,k}=V_{c,k}\,[1+0.6(2g_k-1)].
\]</div>
<p>越过两端的位置用零填充，既不环绕负频率，也不复制边界值。同一位置四个通道共同使用一个倍率。sigmoid 使倍率在 \(0.4\) 与 \(1.6\) 之间变化；它不是 softmax，不要求各位置权重和为 \(1\)，也不是把弱位置直接置零。</p>
<p>若某次计算 \(u_k=0\)，则 \(g_k=0.5\)，倍率为 \(1\)；若 \(g_k=0.75\)，倍率为 \(1.3\)。这是手算示例，不表示初始化一定处处为 \(1\)：卷积权重初始化为随机 Kaiming 权重，只有偏置初始化为零。门控参数为 \(1\times4\times5+1=21\)。</p>
<h3>第二步：一个卷积块的真实先后顺序</h3>
<div class="equation">\[
\text{Depthwise Conv}\ \longrightarrow\
\text{Pointwise Conv}\ \longrightarrow\
\text{GroupNorm}\ \longrightarrow\
\text{ReLU}\ \longrightarrow\
\text{MaxPool/Identity}\ \longrightarrow\
\text{Dropout}.
\]</div>
<details class="derivation" data-depth="deep"><summary>逐步推导：可分离卷积的公式与参数量</summary>
<p>深度卷积给每个输入通道一条长度 \(k\) 的核：\(d_c[t]=\sum_{r=0}^{k-1}h_c[r]\,v_c[t+r-p]\)，其中 \(p=\lfloor k/2\rfloor\)。它先沿频率轴寻找每个通道自己的局部形状，不混合通道。</p>
<p>逐点卷积再计算 \(u_o[t]=\sum_{c=1}^{C_{\mathrm{in}}}M_{oc}d_c[t]\)，在同一频率位置混合通道。两层都不带偏置。因此卷积参数为 \(kC_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}\)，不是普通卷积的 \(kC_{\mathrm{in}}C_{\mathrm{out}}\)。实现采用通常深度学习库的互相关式索引，不需手动再翻转核。</p>
<p>三块卷积步长均为 \(1\)、dilation 为 \(1\)，奇数核配两侧零 padding，先保持频率长度。降采样来自随后的最大池化，不是步长卷积。</p>
</details>
<table class="mini-table"><thead><tr><th>阶段</th><th>卷积后／归一化时形状</th><th>核、padding 与池化</th><th>块末输出</th></tr></thead><tbody>
<tr><td><code>f1</code></td><td>\([B,16,32]\)</td><td>\(k=5,p=2\)；MaxPool \(k=2,s=2,p=0\)</td><td>\([B,16,16]\)</td></tr>
<tr><td><code>f2</code></td><td>\([B,32,16]\)</td><td>\(k=5,p=2\)；MaxPool \(k=2,s=2,p=0\)</td><td>\([B,32,8]\)</td></tr>
<tr><td><code>f3</code></td><td>\([B,32,8]\)</td><td>\(k=3,p=1\)；pool=1 为 Identity</td><td>\([B,32,8]\)</td></tr>
</tbody></table>
<p>MaxPool 每两个相邻位置取较大值，默认不向上取整；例如激活后的 \((1,3,2,4)\) 变成 \((3,4)\)，不是平均池化的 \((2,3)\)。Dropout 是普通逐元素 Dropout；训练以概率 \(0.05\) 随机置零，保留项除以 \(0.95\)，推理则是恒等映射。[<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html" target="_blank" rel="noreferrer">最大池化默认步长与边界</a>]</p>
<h3>GroupNorm：按哪个集合求均值和方差</h3>
<p>分组函数依次尝试 \(16,8,4,2,1\)，取能整除输出通道数的第一个。因此三块都用 \(16\) 组：第一块每组一个通道，后两块每组两个通道。统计在<strong>单个样本、该组通道及全部当前位置</strong>上计算，不跨其他样本。</p>
<div class="equation">\[
\mu_g=\frac1M\sum_{(c,t)\in g}u_{c,t},\quad
\sigma_g^2=\frac1M\sum_{(c,t)\in g}(u_{c,t}-\mu_g)^2,\quad
y_{c,t}=\gamma_c\frac{u_{c,t}-\mu_g}{\sqrt{\sigma_g^2+10^{-5}}}+\beta_c.
\]</div>
<p>第一块归一化时每组有 \(1\times32=32\) 个数，第二块为 \(2\times16=32\)，第三块为 \(2\times8=16\)。采用总体方差；\(\gamma_c,\beta_c\) 是每通道可学习仿射参数，初值为 \(1,0\)。这里的 \(10^{-5}\) 是 GroupNorm 默认稳定项，和前处理 \(10^{-8}\) 不同。GroupNorm 在训练、推理都读取当前输入统计，没有 BatchNorm 式运行均值库。[<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html" target="_blank" rel="noreferrer">GroupNorm 公式与默认参数</a>]</p>
<p>把一个组缩成 \((1,3)\) 手算，均值为 \(2\)、总体方差为 \(1\)，忽略稳定项且取默认仿射参数，得到 \((-1,1)\)，ReLU 再变成 \((0,1)\)。真实组大小如上，不是只归一化两个数，也不保证最后 embedding 是单位向量。</p>
<h3>第三步：平均池化、线性投影与旁路统计相加</h3>
<div class="equation">\[
\bar f_c=\frac18\sum_{t=0}^{7}f_{c,t},\qquad
f_0=W_f\bar f+b_f\in\mathbb R^{160},\qquad
f_{\mathrm{emb}}=f_0+\Delta f.
\]</div>
<p>全局自适应平均池化将 \([B,32,8]\) 压到 \([B,32,1]\)，去掉最后一维后才做 <code>Linear(32→160)</code>。这个投影带偏置，后面没有额外 ReLU 或 LayerNorm；旁路 \(\Delta f\) 才是 <code>Linear(3→160) → ReLU → Dropout(0.1125)</code>。两者按坐标相加，保持 \([B,160]\)。</p>
<h3>三项谱统计：来自门控之前的固定描述</h3>
<p>令 \(\bar P_k^+,\bar P_k^-\) 是上一章六通道池化中保留的线性功率，令 \(E_k=\operatorname{clip}(\bar P_k^++\bar P_k^-,\epsilon,10^6)\)，并执行源码的非有限值清理。它们在门控和卷积之前就已算好，不是从门控输出或 embedding 反推。</p>
<div class="equation">\[
\begin{aligned}
s_{\mathrm{HF}}&=\frac{\sum_{k=24}^{31}E_k}{\sum_{k=0}^{31}E_k+\epsilon},\\
s_{\mathrm{asym}}&=\frac18\sum_{k=24}^{31}\bar v_{3,k},\\
s_{\mathrm{flat}}&=
\frac{\exp\{\operatorname{clip}(\frac1{32}\sum_k\log E_k,-20,20)\}}
{\frac1{32}\sum_k E_k+\epsilon}.
\end{aligned}
\]</div>
<p>末尾四分之一即索引 \(24\) 至 \(31\) 的高频位置。第一项是能量占比，第二项是未按能量加权的不对称均值，第三项是几何均值／算术均值形式的谱平坦度，并带明确截断。旧变量名为 <code>dac_stats</code>，不表示独立 DAC 支路仍开启。</p>
<p>若每对频率始终是 \(P^+=9,P^-=1\)，则池化后各 \(E_k=10\)，三项近似为 \((0.25,0.8,1)\)。它们仍不能告诉我们是哪一个真实硬件参数造成该模式。前处理另算的 PA 统计供 PA 分支使用，不是这里再追加的三个神经输入通道。</p>
<h3>圆度 \(\rho\)：单独进入时频融合的标量</h3>
<div class="equation">\[
\rho=\frac{|\operatorname{mean}_n x[n]^2|}
{\operatorname{mean}_n|x[n]|^2+\epsilon}
=\frac{|\operatorname{mean}(I^2-Q^2)+j\,2\operatorname{mean}(IQ)|}
{\operatorname{mean}(I^2+Q^2)+\epsilon}.
\]</div>
<p>它直接从当前原 IQ 计算，没有额外去复均值。若 \(x=(1,j,-1,-j)\)，则 \(\operatorname{mean}x^2=0\)、\(\rho=0\)；若全为非零实数 IQ，则近似为 \(1\)。这是固定零参数统计，单独与时域、频域向量拼接成 \(160+160+1=321\) 维再融合；没有进入 <code>f_proj</code> 的输入。</p>
<h3>参数怎样一项项数：频域专属共 \(7897\) 个</h3>
<table class="mini-table"><thead><tr><th>模块</th><th>计算</th><th>参数数</th></tr></thead><tbody>
<tr><td>频带门控</td><td>\(4\times1\times5+1\)</td><td>\(21\)</td></tr>
<tr><td><code>f1</code></td><td>DW \(4\times5\) ＋ PW \(4\times16\) ＋ GN \(2\times16\)</td><td>\(116\)</td></tr>
<tr><td><code>f2</code></td><td>DW \(16\times5\) ＋ PW \(16\times32\) ＋ GN \(2\times32\)</td><td>\(656\)</td></tr>
<tr><td><code>f3</code></td><td>DW \(32\times3\) ＋ PW \(32\times32\) ＋ GN \(2\times32\)</td><td>\(1184\)</td></tr>
<tr><td><code>f_proj</code></td><td>\(32\times160+160\)</td><td>\(5280\)</td></tr>
<tr><td><code>freq_stats_proj</code></td><td>\(3\times160+160\)</td><td>\(640\)</td></tr>
<tr><td>合计</td><td>\(21+116+656+1184+5280+640\)</td><td>\(7897\)</td></tr>
</tbody></table>
<p>FFT、四条固定曲线、谱统计、\(\rho\)、池化、激活与 Dropout 没有可训练参数。若只数门控＋三块＋主投影，不含统计旁路，是 \(7257\) 个。上表不重复计入共享时频融合层、PA 统计投影、全局分类头或参考／路径分支；默认频域直接读 raw FFT，也不把共享 Sinc 滤波器算成频域专属参数。</p>
<div class="lab" data-lab="spectrum"></div>
<p class="warning">常见误区：卷积在这里沿频率位置移动，不能把输出长度当成时间采样数。GroupNorm 不是把每条曲线总能量归一为 \(1\)；最大池化也不是平均池化。固定超参数、可学习权重、样本相关的中间统计是三个概念。增大网络或把频谱画得更好看，都不能代替冻结后的跨 RX 身份评测。</p>
<p class="source">项目依据：<code>experiments/adv3b02_xuc/code/model.py:64</code>、<code>experiments/adv3b02_xuc/code/model.py:344</code>、<code>experiments/adv3b02_xuc/code/model.py:387</code>、<code>experiments/adv3b02_xuc/code/model.py:1277</code>、<code>experiments/adv3b02_xuc/code/model.py:1310</code>、<code>experiments/adv3b02_xuc/code/model.py:1450</code>、<code>experiments/adv3b02_xuc/code/model.py:1622</code>、<code>experiments/adv3b02_xuc/code/model.py:1644</code>、<code>experiments/adv3b02_xuc/code/model.py:1731</code>、<code>experiments/adv3b02_xuc/code/model.py:2003</code>；默认装配：<code>experiments/cvs_identity_ce/model.py:18</code>。</p>`
    }
  ];
  const replaceCount = chapters[index + 1]?.id === 'p-frequency-network' ? 2 : 1;
  chapters.splice(index, replaceCount, ...details);
})();
