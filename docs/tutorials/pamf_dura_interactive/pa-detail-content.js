/* PA detail follows the executed lite_d residual-fusion path. */
(function () {
  const chapters = window.COURSE_CHAPTERS;
  if (!Array.isArray(chapters)) throw new Error('Load the course chapters before PA detail');
  const existingStatistics = chapters.findIndex(chapter => chapter.id === 'p-pa-statistics');
  if (existingStatistics >= 0) chapters.splice(existingStatistics, 1);
  const position = chapters.findIndex(chapter => chapter.id === 'p-pa');
  if (position < 0) throw new Error('The PA chapter is required before PA detail');
  chapters.splice(position, 1,
    {
      id: 'p-pa', group: 'PAMF', kicker: '05 / 观察非线性与记忆',
      title: 'PA 局部路：把幅度次方和短延迟交给卷积',
      lead: '先把一段 IQ 展开成 24 个可解释通道，再用包络门控与普通膨胀卷积学习组合。PA 是物理启发的分支名称，输出不是已辨识的真实功放系数。',
      html: String.raw`
<h3>为什么把同一波形展开多次？</h3>
<p>线性滤波对输入幅度的响应是成比例的，功放等非线性系统却可能在大幅度时压缩或产生谱再生；带记忆的作用还会依赖前几个采样点。PA 局部路先显式提供一次、三次、五次的幅度相关基，并加入四个短延迟，让网络更容易比较这些模式。RX 非线性、信道与调制同样可能改变这些模式，所以分支名称不构成已分离真实 PA 的证明。</p>
<p>本版使用原 IQ 输入 \(x\in\mathbb R^{B\times2\times256}\)，写成复数 \(z[n]=I[n]+\mathrm iQ[n]\)。PA 局部路没有先做 CFO 估计、相位展开或瞬时频率计算，也没有在分支内部先做逐窗 RMS 归一化。它与时域分支中的 Sinc 后三阶基是两条不同的计算路径。</p>
<h3>精确基函数：先裁 I/Q，再延迟，最后取幂</h3>
<div class="equation">\[
\widetilde I[n]=\operatorname{clip}(I[n],-2,2),\quad
\widetilde Q[n]=\operatorname{clip}(Q[n],-2,2),\quad
\widetilde z[n]=\widetilde I[n]+\mathrm i\widetilde Q[n].
\]</div>
<p>对 \(m\in\{0,1,2,3\}\)，先构造 \(\widetilde z_m[n]=\widetilde z[n-m]\)；越过左边界的采样点补零，右侧丢去被移出的点，不使用循环移位。对阶数 \(p\in\{1,3,5\}\)，实际带数值保护的公式为：</p>
<div class="equation">\[
s_m[n]=\max\!\left(|\widetilde z_m[n]|^2,10^{-8}\right),\qquad
b_{m,p}[n]=\widetilde z_m[n]\,s_m[n]^{(p-1)/2}.
\]</div>
<p>一次项的尺度直接取 1；三次项乘平方幅度，五次项乘平方幅度的平方。边界补零处仍为零，因为零乘任何有限尺度都是零。最小值保护只作用于平方幅度；不要写成给 I/Q 加噪声，也不要漏掉 \((p-1)/2\) 中的除以 2。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么恰好是 24 通道？</summary>
<ol class="steps"><li>固定一个延迟，依次产生 \(p=1,3,5\) 三个复数基。</li><li>每个复数基展开实部、虚部，得到六个实通道。</li><li>对四个延迟重复，得到 \(4\times3\times2=24\) 个通道，形状为 \([B,24,256]\)。</li><li>通道顺序是“延迟、阶数、实虚部”：延迟 0 占通道 0 至 5，延迟 1 占 6 至 11，依此类推。</li></ol>
<p>传统记忆多项式可以直接线性组合这些基。当前网络随后还有门控、归一化、激活与卷积，因此并不等价于只估计一组传统多项式系数。乘正实尺度保留的是裁剪后样本的相位；I/Q 分别裁剪可能改变原样本的相位，例如 \(3+\mathrm i\) 会变为 \(2+\mathrm i\)。</p></details>
<p>手算 \(z=0.3+0.4\mathrm i\)，平方幅度为 0.25。三个基依次为 \(0.3+0.4\mathrm i\)、\(0.075+0.1\mathrm i\)、\(0.01875+0.025\mathrm i\)。若幅度恰为 1，三个阶数在该点相同；恒模单点不能展示高阶基的区别。记忆深度为 4，最大显式延迟是 3 点，不能把 4 误读成四阶多项式。</p>
<div class="lab" data-lab="pa"></div>
<div class="lab" data-lab="pa-basis-detail"></div>
<h3>包络门控读取的是原输入</h3>
<div class="equation">\[
e[n]=\sqrt{\max(I[n]^2+Q[n]^2,10^{-8})},\quad
g=\sigma(\operatorname{Conv}_{1\to24,k=5}(e)),\quad
v=b\odot[1+0.5(2g-1)].
\]</div>
<p>门控卷积两侧补 2 个零，保留长度 256，生成每个通道、每个位置的权重。门控读取原 IQ 包络，只有多项式基内部裁 I/Q；不能把两者混为统一幅度裁剪。包络最小为 \(10^{-4}\)，倍率有界于 0.5 至 1.5。例如 \(g=0.8\) 时倍率为 1.3，\(g=0.5\) 时为 1。固定常数 0.5 不是另一个可训练参数，卷积权重与偏置才参与学习。</p>
<h3>普通膨胀卷积、完整尺寸和参数量</h3>
<p>三个 PA Block 都是普通实数一维卷积，分组数为 1，没有 depthwise 分解，也没有残差相加。顺序为：无偏置卷积、GroupNorm、SiLU、平均池化、Dropout。SiLU 定义为 \(u\sigma(u)\)。当前通道数 48、64 都选择 16 组 GroupNorm，默认 \(\epsilon=10^{-5}\)，每个输出通道各有一个缩放和偏置。</p>
<table class="mini-table"><thead><tr><th>模块</th><th>输出形状</th><th>实际配置</th><th>可训练参数</th></tr></thead><tbody><tr><td>基函数展开</td><td>\([B,24,256]\)</td><td>固定延迟、阶数与裁剪</td><td>0</td></tr><tr><td>包络门控</td><td>\([B,24,256]\)</td><td>卷积 \(1\to24,k=5\)，有偏置</td><td>144</td></tr><tr><td><code>pa_b1</code></td><td>\([B,48,128]\)</td><td>\(24\to48,k=7,d=1\)，padding 3，池化 2</td><td>8064＋96＝8160</td></tr><tr><td><code>pa_b2</code></td><td>\([B,64,64]\)</td><td>\(48\to64,k=7,d=2\)，padding 6，池化 2</td><td>21504＋128＝21632</td></tr><tr><td><code>pa_b3</code></td><td>\([B,64,64]\)</td><td>\(64\to64,k=5,d=4\)，padding 8，不池化</td><td>20480＋128＝20608</td></tr><tr><td>全长平均池化</td><td>\([B,64,1]\to[B,64]\)</td><td><code>AdaptiveAvgPool1d(1)</code></td><td>0</td></tr><tr><td>局部投影</td><td>\([B,160]\)</td><td>Linear \(64\to160\)、ReLU、Dropout</td><td>10400</td></tr></tbody></table>
<p>三个 Block 的 Dropout 均为 0.08。当前 <code>lite_d</code> 的整体 dropout 为 0.45，所以局部投影使用 \(0.45/4=0.1125\)。推理时关闭 Dropout；GroupNorm 仍按当前窗口计算，不使用跨样本的运行均值。上述局部路径合计 60944 个参数，包括门控、三个 Block 和局部投影；下一章的统计投影及融合增益另计。</p>
<p>原生构造用 Kaiming 正态初始化卷积和线性权重，卷积采用 ReLU 增益约定，线性层采用 linear 增益约定，存在的偏置初始化为零；这不改变 Block 实际使用 SiLU 的事实。随后这些参数随默认头联合训练并继承，初始化规则不能代替已经训练好的权重数值。实现见 <code>experiments/adv3b02_xuc/code/model.py:1450</code>。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：7、13、17 的卷积跨度怎样变成 98？</summary>
<p>膨胀卷积有效核长为 \(k_{\rm eff}=1+(k-1)d\)，所以三层分别为 7、13、17 个当前层位置。设 \(r\) 为几何覆盖长度、\(j\) 为相邻特征在 PA 第一层输入位置上的间距。初始 \(r=1,j=1\)，卷积更新 \(r\leftarrow r+(k_{\rm eff}-1)j\)；长度为 2、步长为 2 的池化更新 \(r\leftarrow r+j,\ j\leftarrow2j\)。</p>
<table class="mini-table"><thead><tr><th>操作后</th><th>\(r\)</th><th>\(j\)</th></tr></thead><tbody><tr><td>第一层卷积</td><td>7</td><td>1</td></tr><tr><td>第一次池化</td><td>8</td><td>2</td></tr><tr><td>第二层卷积</td><td>\(8+12\times2=32\)</td><td>2</td></tr><tr><td>第二次池化</td><td>34</td><td>4</td></tr><tr><td>第三层卷积</td><td>\(34+16\times4=98\)</td><td>4</td></tr></tbody></table>
<p>98 只描述三次卷积和两次池化相对于 <code>pa_b1</code> 输入特征位置的几何足迹。更早的基函数还读取左侧 0 至 3 点，包络门控还有左右各 2 点；不能据此直接宣称整个分支只依赖 98 个原始 IQ 点。GroupNorm 的组内统计还覆盖该组通道的完整时间轴，最终平均池化也聚合全长，因此严格依赖范围不能只由卷积核跨度概括。</p></details>
<div class="lab" data-lab="pa-dilation"></div>
<p class="warning">PA 基没有提供真实 PA 参数或独立标签。高阶幂强调幅度差异，也会强调异常值；裁剪、门控和归一化控制数值，却不保证 TX 身份细节保留。局部路径没有 CFO 补偿，其完整卷积网络也不因使用这些基而自动对相位或接收机变化不变。</p>
<p class="source">依据：<code>experiments/cvs_identity_ce/model.py:20</code>、<code>experiments/adv3b02_xuc/code/model.py:64</code>、<code>experiments/adv3b02_xuc/code/model.py:366</code>、<code>experiments/adv3b02_xuc/code/model.py:622</code>、<code>experiments/adv3b02_xuc/code/model.py:695</code>、<code>experiments/adv3b02_xuc/code/model.py:1326</code>、<code>experiments/adv3b02_xuc/code/model.py:1768</code>、<code>experiments/adv3b02_xuc/code/model.py:2003</code>。参数量和几何足迹由这些构造逐项计算。</p>`
    },
    {
      id: 'p-pa-statistics', group: 'PAMF', kicker: '05 · 进阶 / 查看全窗频谱',
      title: 'PA 统计路：三项谱形状怎样修正 160 维嵌入',
      lead: '局部路逐位置分析非线性模式，统计路则概括当前窗口的频谱边缘与中心。当前实际启用的是三维频谱统计，不是另一套 17 维非线性或瞬时频率模块。',
      html: String.raw`
<h3>为什么还需要一条统计路？</h3>
<p>局部卷积保留了变化的先后与邻域关系；全窗频谱统计提供另一种概括：频谱边缘占多少、边缘相对中心有多强、功率分布是否尖锐。它们不负责独立给出 TX 类别，而是经过学习投影，补充局部 PA 嵌入。三项都是物理启发的谱形状指标，不能直接等同于真实功放的增益、AM/AM 或 AM/PM 系数。</p>
<p>当前路径保留 <code>raw_fft</code>、<code>raw_iq</code> 与 PA 频谱统计，输入仍是一份当前 IQ。这里没有 17 维额外 PA 统计，也没有把 CFO、相位差或瞬时频率拼进这三个坐标；其他实验中出现过的可选模块不能并入本版的结构、公式或参数量。</p>
<h3>从 256 点 IQ 到 32 个镜像配对频带</h3>
<div class="equation">\[
Z_k=\sum_{n=0}^{255}z[n]e^{-\mathrm i2\pi kn/256},\qquad
P_k=(\Re Z_k)^2+(\Im Z_k)^2.
\]</div>
<p>FFT 使用默认归一化约定，正向不除以 \(\sqrt{256}\)。代码可以用 I、Q 的两次实数 FFT 重建同一个完整复数频谱。初始功率将 NaN 替换为 0、正无穷替换为 \(10^6\)、负无穷替换为 0，再限制为非负。有限功率并不是在此统一截到 \(10^6\)；后面构造统计量时才有另一层上界。</p>
<p>按从零开始的索引，正频功率为 \(P[1:128]\)，负频功率为反向排列的 \(P[129:256]\)，均有 127 项。两者分别对应 \(+1\) 至 \(+127\) 和 \(-1\) 至 \(-127\) 的配对距离，排除 DC 与 Nyquist。当前 \(K=32\)，六张图一起自适应平均池化；PA 统计只读取其中两张<strong>线性功率</strong>池化图，不是把平均对数功率再取指数。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：127 个输入位置怎样池化成 32 个位置？</summary>
<p>对输出位置 \(j=0,\ldots,31\)，自适应平均池化的起止边界为 \(a_j=\lfloor127j/32\rfloor\)、\(b_j=\lceil127(j+1)/32\rceil\)，取输入区间 \([a_j,b_j)\) 的均值。第一个区间是 \([0,4)\)，第二个是 \([3,8)\)。127 不能整除 32，边界可能重叠，不能替换成 32 个不重叠等宽求和框。</p>
<div class="equation">\[
T_j=\operatorname{clip}_{[10^{-8},10^6]}
\left(\operatorname{sanitize}\left[
\frac1{b_j-a_j}\sum_{\ell=a_j}^{b_j-1}(P^+_\ell+P^-_\ell)
\right]\right).
\]</div>
<p>这里 sanitize 同样将 NaN、正无穷、负无穷分别替换为 0、\(10^6\)、0。代码先分别池化两侧线性功率，再相加；上式是其数学等价写法。输入为 FP32 时，FFT 为 complex64，功率、池化与这三项统计为 float32，没有转为 float64 的统计计算。</p></details>
<h3>第一项：边缘能量占全部能量的比例</h3>
<p>定义 \(\epsilon=10^{-8}\)。边缘宽度取 \(\max(1,\lfloor0.20K\rfloor)=6\)，所以边缘为 \(T[26:32]\)，包含位置 26 至 31；不是正负频谱各随意再取六个原始 FFT 点。</p>
<div class="equation">\[
E_{\rm edge}=\sum_{j=26}^{31}T_j,\qquad
r_{\rm edge}=\frac{E_{\rm edge}}{\sum_{j=0}^{31}T_j+\epsilon}.
\]</div>
<p>这项衡量固定配对频带的高频端占比。它更像全窗谱形状线索，而不是已校准的带外辐射测量：频带划分固定于本表示，没有自动识别真实占用带宽或邻道边界。</p>
<h3>第二项：同一边缘相对中心有多强</h3>
<p>中心左界为 \(\lfloor0.30K\rfloor=9\)，右界为 \(\lfloor0.70K\rfloor=22\)，取半开区间 \(T[9:22]\)，即位置 9 至 21，共 13 项：</p>
<div class="equation">\[
E_{\rm center}=\sum_{j=9}^{21}T_j+\epsilon,\qquad
r_{\rm regrowth}=\frac{E_{\rm edge}}{E_{\rm center}}.
\]</div>
<p>分母不是全频带能量，因此第二项可以大于 1。代码没有额外把它截到 1 或 \(10^4\)。它的名称包含 regrowth，只表示谱再生启发，不能跳过 RX、信道与调制的影响，认定这个比例完全由发射机功放造成。</p>
<h3>第三项：带数值保护的谱峰度</h3>
<div class="equation">\[
\mu=\frac1{32}\sum_{j=0}^{31}T_j,\quad
d_j=T_j-\mu,\quad
v=\frac1{32}\sum_{j=0}^{31}d_j^2+\epsilon,\qquad
\kappa=\operatorname{clip}_{[0,10^4]}
\frac{\frac1{32}\sum_{j=0}^{31}\operatorname{clip}(d_j,-10^3,10^3)^4}
{\max(v^2,\epsilon)}.
\]</div>
<p>这是总体二阶矩加 \(\epsilon\)，分母用 \(v^2\) 再取最小值保护；不能改成无偏方差或单独除以 \(v\)。只有四次幂分子的偏差先截到 \([-10^3,10^3]\)，二阶矩中的偏差没有这层截断。最后才把峰度截到 \([0,10^4]\)。它没有减去 3，因而不是 excess kurtosis。</p>
<p>手算若全部 \(T_j=1\)，边缘为 6、中心为 13，前两项约为 0.1875、0.461538，峰度为 0。若整段 IQ 恰为零，则功率会被统计下界抬到 \(T_j=\epsilon\)，此时前两项是 \(6/33\) 和 \(6/14\)，并非零；峰度仍为零。这是 eps 参与分母后的定义，不应把零输入示例擅自修成三个零。</p>
<div class="lab" data-lab="pa-stat-detail"></div>
<h3>三维坐标怎样进入实际融合？</h3>
<div class="equation">\[
s_{\rm PA}=[r_{\rm edge},r_{\rm regrowth},\kappa]\in\mathbb R^3,\quad
\Delta p=\operatorname{Dropout}_{0.1125}\!\left(\operatorname{ReLU}(W_s s_{\rm PA}+b_s)\right)
\in\mathbb R^{160}.
\]</div>
<p>投影为有偏置 Linear \(3\to160\)，共有 \(3\times160+160=640\) 个参数。投影前没有另做三维 z-score、LayerNorm 或逐坐标标准化；谱峰度内部中心化不是对这三个输入坐标统一标准化。推理时 Dropout 关闭。</p>
<p>记上一章局部投影为 \(p_0\)。主干先返回 \(p_{\rm local}=p_0+0.25\Delta p\)，并把 \(\Delta p\) 单独传给残差头；残差头又加一次，实际 PA 归一化输入为：</p>
<div class="equation">\[
u=p_0+1.25\Delta p,\qquad
p=\operatorname{LayerNorm}_{160}(u),\qquad
h=\operatorname{LayerNorm}_{160}(\mathrm{base})+\tanh(g_p)\odot p.
\]</div>
<p>LayerNorm 在本窗口的 160 个嵌入坐标上使用总体方差、默认 \(\epsilon=10^{-5}\)，包含 160 个缩放和 160 个偏置。\(g_p\in\mathbb R^{160}\) 为可学习参数，初始化为 \(\operatorname{atanh}(0.25)\)，所以有效增益初始为 0.25，之后有界于 \((-1,1)\)，也可学为负值。初始化不是读取到的最终训练值，不能用它替代 checkpoint 中的实际增益。</p>
<p>PA 专属归一化和增益共 \(320+160=480\) 个参数。局部路 60944、统计投影 640、PA 归一化与增益 480，合计 62064；不把共享的 base 归一化、分类权重、参考响应和路径分支算作 PA 独占参数。参考与路径残差随后还会参与同一个最终嵌入，不能把此处 \(h\) 当成已经包含所有后续分支。</p>
<p class="warning">三项统计与局部卷积互补，但不能独立证明身份保真。整体恒定相位旋转在理想计算中不改变频谱功率；CFO 会移动谱能量，所以这些统计并不自动对 CFO 不变。它们也不能唯一分离 TX、RX 与信道；本教程只解释当前启用结构，不把历史 17 维模块或新实验接入现有版本。</p>
<p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:1059</code>、<code>experiments/adv3b02_xuc/code/model.py:1317</code>、<code>experiments/adv3b02_xuc/code/model.py:1484</code>、<code>experiments/adv3b02_xuc/code/model.py:1583</code>、<code>experiments/adv3b02_xuc/code/model.py:1608</code>、<code>experiments/adv3b02_xuc/code/model.py:1632</code>、<code>experiments/adv3b02_xuc/code/model.py:1777</code>、<code>experiments/cvs_residual_identity/model.py:23</code>。自适应池化区间按所调用算子的标准起止规则展开，未改变实际池化。</p>`
    }
  );
}());
