/* PAMF teaching content. Equations follow the executed architecture and named recipes. */
window.COURSE_CHAPTERS = window.COURSE_CHAPTERS || [];
window.COURSE_CHAPTERS.push(
{
  id: 'p-time', group: 'PAMF', kicker: '03 / 从波形到时域特征',
  title: '时域分支：先学会“听”不同频带，再理解局部变化',
  lead: '把一个复数序列看成一串二维坐标。Sinc 前端选择频带，卷积沿时间寻找局部规律，最终把一包 IQ 压成 160 个数。',
  html: String.raw`
<h3>为什么先从时域开始</h3>
<p>输入不是图片，而是 \(x[n]=I[n]+jQ[n]\)。\(I,Q\) 是同一信号的两个正交分量，\(j^2=-1\)；它们共同给出幅度 \(|x|=\sqrt{I^2+Q^2}\) 和相位。实现每包接收 <code>[B,2,256]</code>，\(B\) 为批大小，采样率 \(F_s=25\,\mathrm{MHz}\)。相邻点间隔 \(1/F_s=40\,\mathrm{ns}\)，256 点覆盖的采样时长为 \(10.24\,\mu\mathrm{s}\)。时域分支关心这些点如何连续变化，而不只关心平均能量。</p>
<p>直觉上，Sinc 滤波器像 24 个可调收音旋钮。每个旋钮保留一个频带；随后卷积查看各频带的局部起伏。滤波器由上下截止频率决定，学习的是频率坐标，不是 79 个互不相关的核系数。</p>
<h3>从理想低通逐步得到带通</h3>
<div class="equation">\[
\operatorname{sinc}(u)=\frac{\sin(\pi u)}{\pi u},\quad
h_{\mathrm{LP}}(t;f_c)=2f_c\operatorname{sinc}(2f_ct),\quad
h_{\mathrm{BP}}(t)=h_{\mathrm{LP}}(t;f_2)-h_{\mathrm{LP}}(t;f_1).
\]</div>
<p>\(f_1,f_2\) 是以 Hz 表示的下、上截止频率。先留下 \([-f_2,f_2]\)，再减掉 \([-f_1,f_1]\)，便得到对称的两个通带。这里采用归一化 sinc，零点处取连续极限 1。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么是两个 sinc 相减</summary>
<ol class="steps"><li>理想低通的频率响应在 \([-f_c,f_c]\) 为 1，其余为 0。逆傅里叶变换是 \(h(t)=\int_{-f_c}^{f_c}e^{j2\pi ft}\,df\)。</li><li>积分得到 \(\frac{\sin(2\pi f_ct)}{\pi t}\)，写成归一化 sinc 即 \(2f_c\operatorname{sinc}(2f_ct)\)。</li><li>实际核只有 79 点：令 \(t_n=(n-39)/F_s,\ n=0,\ldots,78\)。乘 Hamming 窗 \(w[n]=0.54-0.46\cos(2\pi n/78)\)，减轻有限截断造成的频率振铃。</li><li>源码再除以该核的最大绝对值。它是幅值归一化，不是“每个频带能量相等”，也不是理想砖墙滤波器。</li></ol>
</details>
<h3>真实的 100 通道从哪里来</h3>
<p>同一组 24 个实核分别滤 \(I,Q\)，得到 48 通道。对滤后复数 \(z_k\) 再计算 \(z_k|z_k|^2\)，增加 48 通道；这一计算先把 \(I,Q\) 各裁到 \([-1.2,1.2]\)。最后拼入原 IQ 的一阶、二阶差分共 4 通道。所以输入 <code>time_fuse</code> 的确是 \(48+48+4=100\)，不是 24 或 96 通道。</p>
<table class="mini-table"><thead><tr><th>操作</th><th>输出（省略 \(B\)）</th></tr></thead><tbody><tr><td>100 通道拼接 → \(1\times1\) 卷积、归一化、激活</td><td>\(48\times256\)</td></tr><tr><td>平均池化 <code>time_down</code></td><td>\(48\times128\)</td></tr><tr><td><code>t1</code>：深度可分离卷积、最大池化</td><td>\(72\times64\)</td></tr><tr><td><code>t2</code>：深度可分离卷积、最大池化</td><td>\(96\times32\)</td></tr><tr><td><code>t3</code> → 全局平均 → 线性投影</td><td>\(96\times32\to96\to160\)</td></tr></tbody></table>
<p>深度可分离卷积先逐通道卷积，再用 \(1\times1\) 卷积交换通道信息。忽略偏置时，核长 \(k\) 的参数量由 \(kC_{\mathrm{in}}C_{\mathrm{out}}\) 变成 \(kC_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}\)。它节省计算，但通道混合仍然可以学习。</p>
<h3>手算：频率坐标和优化单位</h3>
<p><code>CENTER_25K</code> 保存原点 \(p_0\)，用 \(p=p_0+25000q\) 还原低频／带宽原始参数。例如 \(p_0=100000\,\mathrm{Hz},q=0.4\)，还原为 \(110000\,\mathrm{Hz}\)。随后才施加原生的绝对值、最小 50 Hz 和 Nyquist 截断；所以这个 \(p\) 还不是无条件等于最终截止频率。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为何换坐标还要换学习率、\(\epsilon\) 和衰减</summary>
<p>令 \(s=25000\)，则链式法则给出 \(\frac{\partial L}{\partial q}=s\,\frac{\partial L}{\partial p}\)。只把变量名换成 \(q\)，梯度大小就已改变。若以归一化频率 \(r=p/F_s\) 为优化单位，则 \(q=(F_s/s)r-p_0/s\)；本例倍率 \(F_s/s=1000\)。</p>
<p><code>CENTER_UNITS</code> 因此令 Sinc 参数组的学习率为基础值的 1000 倍、Adam \(\epsilon\) 为基础值的 \(1/1000\)。基础学习率 \(2\times10^{-4}\) 对应坐标学习率 0.2。物理频率衰减采用 \(q\leftarrow q-\alpha\lambda(q+p_0/s)\)，同时裁剪前暂把 \(g_q\) 除以 \(s\)，与其他参数一起做原来的全局 clip 5，再还原。这是一组一致的单位映射，不能再叠乘旧倍率。</p>
</details>
<div class="lab" data-lab="sinc"></div>
<p class="warning">常见误区：Sinc 前端受采样率、有限核长和 Nyquist 频率约束；可学习频带不等于发现了真实硬件频带。这里的三阶基也不等于已把 PA 失真独立分离。推理使用同一套滤波合成与单次前向。</p>
<p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:182</code>、<code>experiments/adv3b02_xuc/code/model.py:1227</code>、<code>experiments/adv3b02_xuc/code/model.py:1468</code>、<code>experiments/adv3b02_xuc/code/model.py:2003</code>；坐标／优化继承实现：<code>experiments/cvs_rdrop_sinc_units/coordinate.py:5</code>、<code>experiments/cvs_rdrop_sinc_units/optimizer.py:12</code>。</p>`
},
{
  id: 'p-frequency', group: 'PAMF', kicker: '04 / 查看频谱的两侧',
  title: '频域分支：把正负频率配对，保留功率与不对称性',
  lead: String.raw`时域告诉我们变化发生的顺序，频域告诉我们能量分布在哪里。PAMF 特别把 \(+f\) 与 \(-f\) 放在一起观察。`,
  html: String.raw`
<h3>为什么另看一次频谱</h3>
<p>两段 IQ 可能有相近的平均幅度，却在频带边缘、镜像泄漏或频谱起伏上不同。频域分支把整包波形变成按频率排列的描述，让局部卷积沿频率轴寻找规律。它处理的是接收到的综合结果，包含 TX、RX、信道、均衡和噪声；不直接输出这些因素的真实参数。</p>
<p>直觉上，把频谱以零频为轴对折，同一距离的 \(+f\) 与 \(-f\) 并排比较。某侧更强时，只有总功率会漏掉方向信息；因此网络同时读取两侧功率、功率比和不对称程度。</p>
<h3>从复数 DFT 到四张频率曲线</h3>
<div class="equation">\[
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\qquad P[k]=|X[k]|^2,\qquad N=256.
\]</div>
<p>\(X[k]\) 是第 \(k\) 个离散频率的复数幅度，\(P[k]\) 是功率。频率间隔为 \(F_s/N=97.65625\,\mathrm{kHz}\)。源码可用 \(I,Q\) 各自的实数 FFT 重建同一复数 FFT；这不是只保留了正半谱。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：如何对折，为什么产生四个通道</summary>
<ol class="steps"><li>排除 DC 的 \(k=0\) 与 Nyquist 的 \(k=128\)，令 \(P_k^+=P[k]\)、\(P_k^-=P[N-k]\)，\(k=1,\ldots,127\)。负侧反转后，索引代表相同的频率绝对值。</li><li>保留 \(a_k=\log(1+P_k^+)\)、\(b_k=\log(1+P_k^-)\)。对数压缩动态范围，使大功率点不至于独占特征。</li><li>加入有符号比值 \(r_k=\log(P_k^++\epsilon)-\log(P_k^-+\epsilon)\)，源码把它裁到 \([-20,20]\)。\(r_k>0\) 表示正侧更强。</li><li>加入无符号不对称度 \(d_k=\frac{|P_k^+-P_k^-|}{P_k^++P_k^-+\epsilon}\)。忽略极小稳定项时，它在 0 至 1 之间，表达差异幅度。</li><li>对四条曲线自适应平均池化到 32 个位置，得到 <code>[B,4,32]</code>。实现是先求对数／比值再池化，不能随意改成“先平均功率再求对数”。</li></ol>
</details>
<h3>真实实现：32 个频带怎样变成 160 维</h3>
<p>频带门控先沿频率做 <code>Conv1d(4→1,k=5)</code>，经 sigmoid 得 \(g_k\)，再将四通道共同乘 \(1+0.6(2g_k-1)\)，倍率范围为 0.4 至 1.6。它能温和增强或减弱位置，不是删除某些真实频率。</p>
<table class="mini-table"><thead><tr><th>阶段</th><th>输出（省略 \(B\)）</th></tr></thead><tbody><tr><td>镜像特征与频带门控</td><td>\(4\times32\)</td></tr><tr><td><code>f1</code>，核长 5、池化 2</td><td>\(16\times16\)</td></tr><tr><td><code>f2</code>，核长 5、池化 2</td><td>\(32\times8\)</td></tr><tr><td><code>f3</code>，核长 3、无降采样</td><td>\(32\times8\)</td></tr><tr><td>全局平均、线性投影</td><td>\(32\to160\)</td></tr></tbody></table>
<p>网络还计算 3 个频谱统计：高频功率占比、高频不对称均值和谱平坦度。它们经 <code>3→160</code> 投影后加到频域 embedding。虽然旧变量叫 <code>dac_stats</code>，本版并没有开启独立 DAC 分支。另一个圆度指标 \(\rho=\frac{|\operatorname{mean}(x^2)|}{\operatorname{mean}|x|^2+\epsilon}\) 作为 1 个数进入后面的时频融合。</p>
<h3>手算：同一个频点的四种描述</h3>
<p>假设某对频点 \(P^+=9,P^-=1\)，忽略 \(\epsilon\)：正侧为 \(\log 10\approx2.3026\)，负侧为 \(\log 2\approx0.6931\)，比值为 \(\log 9\approx2.1972\)，不对称度为 \(8/10=0.8\)。交换两侧后，前两数交换、比值变为负数、不对称度仍为 0.8。因此四个量各保留了不同信息。</p>
<p>谱平坦度是 \(\frac{\exp(\operatorname{mean}\log E)}{\operatorname{mean}E}\)。例如两个频带功率为 1、9，几何均值是 3、算术均值是 5，平坦度为 0.6；若功率都是 5，则为 1。这个量描述是否尖锐，不能单独判断身份。</p>
<div class="lab" data-lab="spectrum"></div>
<p class="warning">常见误区：实信号的频谱有共轭对称性，复数 IQ 一般没有，因此负频率不是“重复的一半”。功率谱会丢失绝对相位；32 带池化也会丢失精细频率位置。频域与时域互补，任何单个不对称指标都不能直接当作已校准的 I/Q 硬件失衡。</p>
<p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:387</code>、<code>experiments/adv3b02_xuc/code/model.py:1277</code>、<code>experiments/adv3b02_xuc/code/model.py:1310</code>、<code>experiments/adv3b02_xuc/code/model.py:1485</code>、<code>experiments/adv3b02_xuc/code/model.py:1564</code>。</p>`
},
{
  id: 'p-pa', group: 'PAMF', kicker: '05 / 观察非线性与记忆',
  title: 'PA 分支：用延迟多项式把幅度相关模式显式展开',
  lead: '“PA”是分支的物理启发，不是一个已辨识的真实功放。这里先构造可解释的非线性基，再让卷积学习如何利用它们。',
  html: String.raw`
<h3>为什么线性滤波之外还要多项式</h3>
<p>线性系统满足叠加性：输入乘 2，输出也乘 2。功放等非线性系统可能在大幅度时压缩、产生谱再生，并且当前输出受前几个输入影响。直接让普通卷积发现全部规律也可以，但显式提供“幅度的不同次方”和“短延迟”能为学习提供一种结构。</p>
<p>直觉上，同一段波形同时用几副放大镜看：一次项保留原样，三次项更强调高幅度，五次项进一步强调高幅度；再比较当前、前 1 点、前 2 点、前 3 点的结果。这里的 4 是记忆深度，\((1,3,5)\) 是阶数，不能把它们误读成三个延迟。</p>
<h3>公式中的每个符号</h3>
<div class="equation">\[
b_{m,p}[n]=\tilde x[n-m]\,|\tilde x[n-m]|^{p-1},\quad
m\in\{0,1,2,3\},\ p\in\{1,3,5\}.
\]</div>
<p>\(\tilde x\) 表示构造基之前、\(I\) 与 \(Q\) 分别裁到 \([-2,2]\) 的 IQ；越过序列左边界的点补 0。每个 \(b_{m,p}\) 都是复数。展开实部和虚部后，\(4\times3\times2=24\) 个实通道仍保持原来的 256 个时间位置。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：从复数样本得到 24 通道</summary>
<ol class="steps"><li>写 \(\tilde x=a+jb\)，先求 \(r^2=a^2+b^2\)。源码给这个平方幅度加最小值保护，避免数值退化。</li><li>一次项 \(p=1\) 的缩放是 1，输出 \((a,b)\)；三次项 \(p=3\) 的缩放是 \(r^2\)，输出 \((ar^2,br^2)\)。</li><li>五次项 \(p=5\) 的缩放是 \(r^4=(r^2)^2\)，输出 \((ar^4,br^4)\)。乘正实数改变幅度而保留该点的相位。</li><li>对四个延迟重复上述计算，按“延迟、阶数、实虚部”排列。它是在构造基函数，尚未估计传统记忆多项式里的物理系数 \(a_{m,p}\)。</li></ol>
<p>传统模型可写为 \(y[n]\approx\sum_{m,p}a_{m,p}b_{m,p}[n]\)。PAMF 没有把学习限制为这一个线性求和，而是在 24 通道后继续接门控和非线性卷积，因此表达能力更强，解释也不再等同于物理参数辨识。</p>
</details>
<h3>包络门控与真实网络尺寸</h3>
<p>门控读取原输入的包络 \(e[n]=\sqrt{I[n]^2+Q[n]^2}\)，经核长 5 的 <code>1→24</code> 卷积和 sigmoid，生成逐通道权重 \(g\)。基函数乘上 \(1+0.5(2g-1)\)，倍率介于 0.5 至 1.5。注意，门控读的是原 IQ 包络，基函数内部的裁剪并不把整个网络输入都裁掉。</p>
<table class="mini-table"><thead><tr><th>阶段</th><th>输出（省略 \(B\)）</th><th>卷积配置</th></tr></thead><tbody><tr><td>多项式展开、包络门控</td><td>\(24\times256\)</td><td>4 个延迟、3 个奇数阶</td></tr><tr><td><code>pa_b1</code></td><td>\(48\times128\)</td><td>\(k=7\)，<code>dilation=1</code>，平均池化 2</td></tr><tr><td><code>pa_b2</code></td><td>\(64\times64\)</td><td>\(k=7\)，<code>dilation=2</code>，平均池化 2</td></tr><tr><td><code>pa_b3</code></td><td>\(64\times64\)</td><td>\(k=5\)，<code>dilation=4</code></td></tr><tr><td>平均池化、投影与激活</td><td>\(64\to160\)</td><td>单个 PA embedding</td></tr></tbody></table>
<p>膨胀卷积的有效核跨度为 \(1+(k-1)d\)，三层分别为 7、13、17 个当前层位置。考虑两次池化，最后一层单点在原序列上的理论感受野约为 98 点；这比把 17 直接当成原始采样点数更准确。</p>
<h3>手算：幅度改变，阶数才产生差异</h3>
<p>取一个样本 \(x=0.3+0.4j\)，则 \(|x|=0.5,|x|^2=0.25\)。三个基依次为 \(0.3+0.4j\)、\(0.075+0.1j\)、\(0.01875+0.025j\)。若前一采样点为 0，则延迟 1 的三个基在此位置都为 0。反过来，若 \(|x|=1\)，三个阶数在该点相同；单个恒模点无法展示高阶基的差异。</p>
<p>本版还从频谱计算边缘能量占比、边缘／中心能量比和谱峰度，经 <code>3→160</code> 投影得到 \(\Delta p\)。它不是另一个 24 通道展开；它随后参与 PA embedding 的残差融合，精确加法顺序见融合章。</p>
<div class="lab" data-lab="pa"></div>
<p class="warning">常见误区：奇数阶基可以承载幅度相关信息，但 RX 非线性、传播、均衡与调制也会改变这些基。输出 160 维没有被监督成真实 PA 系数。更大的阶数不自动更好，高次幂会放大异常值；裁剪与有界门控只是控制数值，不保证身份信息完全保留。</p>
<p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:622</code>、<code>experiments/adv3b02_xuc/code/model.py:695</code>、<code>experiments/adv3b02_xuc/code/model.py:1317</code>、<code>experiments/adv3b02_xuc/code/model.py:1326</code>、<code>experiments/adv3b02_xuc/code/model.py:1760</code>。</p>`
},
{
  id: 'p-reference', group: 'PAMF', kicker: '06 / 用已知激励作参照',
  title: '参考响应：先对齐重复训练序列，再比较相对形状',
  lead: '如果知道发出的理想训练符号长什么样，便可以问：这一包收到的版本偏离它多少？这里得到的是相对接收响应，仍然包含接收机与信道。',
  html: String.raw`
<h3>为什么已知激励有帮助</h3>
<p>不同消息会改变波形，直接比较两个包容易把内容差异当成设备差异。Wi-Fi L-STF 是公开的重复训练结构；使用同一理想激励作参照，可以减少内容变化带来的歧义。实现只在当前包内前向计算，没有跨包记忆、拟合或 query 状态更新。</p>
<p>直觉上，先将四次重复的“同一句话”相位对齐并平均，再和公开模板比较。整体放大和整体相位旋转会被商掉，剩下各占用频点的相对形状。未知频率选择性信道并不会因此消失。</p>
<h3>先补偿相对频偏，再平均四个周期</h3>
<p>取包内 Python 切片 <code>x[...,80:160]</code>，共 80 点，即 \(3.2\,\mu\mathrm{s}\)。记它为 \(w[n]\)，周期 \(P=20\)，有 4 个周期。下式的 \(\omega\) 是弧度／采样点，\(\epsilon=10^{-6}\) 是稳定常数。</p>
<div class="equation">\[
C=\sum_{n=0}^{59}w[n+20]\overline{w[n]},\quad
\omega=\frac{\operatorname{atan2}(\Im C,\Re C+\epsilon^2)}{20},\quad
\tilde w[n]=w[n]e^{-j\omega n}.
\]</div>
<details class="derivation" data-depth="deep"><summary>逐步推导：从周期相关到相对传递响应</summary>
<ol class="steps"><li>若相邻重复周期主要相差 \(e^{j20\omega}\)，则乘上前一周期的共轭后，相位近似为 \(20\omega\)；将 60 个配对相加，再取相角可降低单点噪声影响。</li><li>补偿后令 \(m[r]=\frac14\sum_{q=0}^3\tilde w[20q+r]\)，\(r=0,\ldots,19\)。再用 \(\sqrt{\operatorname{mean}|m|^2+\epsilon^2}\) 归一化，得到 \(u[r]\)。</li><li>公开模板库有 320 个候选延迟，步长 \(1/16\) 个采样点。计算 \(c_d=\frac1{20}\sum_r u[r]\overline{s_d[r]}\)，选 \(|c_d|^2\) 最大的延迟，记对应相关为 \(a\)。这是有限候选的前向路由，不是在线训练。</li><li>在 12 个占用频点计算 \(H_k=U_k\overline{S_k}/|S_k|^2\)。再除去一个公共复增益：\(\bar H_k=H_k\bar a/(|a|^2+\epsilon)\)。</li><li>取偏离理想值的残差 \(r_k=\bar H_k-1\)，并软限制为 \(v_k=r_k/\sqrt{1+|r_k|^2/4^2}\)。小残差几乎不变，大残差的模逐渐接近 4，避免极端比值主导网络。</li></ol>
</details>
<h3>29 个槽位，当前实际启用 28 个</h3>
<p>12 个复数 \(v_k\) 拆成 12 个实部和 12 个虚部，再加 5 个标量，共 \(24+5=29\)。五个标量依次为模板相关质量 \(|a|^2\)、周期相干度、相干平均能量占比、归一化相对 CFO、窗口／整包功率比。</p>
<p>当前输入策略把索引 27（从 0 开始）的显式 CFO 标量置 0，因此保持 29 维接口而只有 28 个有效特征。<strong>前面的内部 CFO 补偿仍然执行。</strong>相对 CFO 的无混叠区间约为 \(\pm625\,\mathrm{kHz}\)，周期为 \(F_s/20=1.25\,\mathrm{MHz}\)；它不能独立辨认 TX 与 RX 各自的振荡器误差。</p>
<p>随后经过 <code>Linear(29→64) → SiLU → Linear(64→160) → LayerNorm</code>。所得向量通过逐通道 \(\tanh(g_r)\) 加到已有表示，增益初值为 0。没有再训练一个参考分类头，也不把这 29 个槽位称作 29 个硬件参数。</p>
<h3>手算：公共旋转被去除，频率形状留下</h3>
<p>忽略 \(\epsilon\)，设某频点 \(H_k=a(1+0.2j)\)，公共增益 \(a=e^{j\pi/3}\)。则 \(H_k\bar a/|a|^2=1+0.2j\)，整体的 60° 旋转被消去，残差仍为 \(0.2j\)。软限制后为 \(0.2j/\sqrt{1+0.04/16}\approx0.19975j\)。所以“去除公共相位”没有把这个频点自己的相对形状一起清掉。</p>
<p>再看频偏：若 \(C\) 的相角是 0.4 rad，则 \(\omega=0.02\) rad／点，折算频率约为 \(79.58\,\mathrm{kHz}\)。内部会按此补偿；供身份投影读取的显式 CFO 槽仍为 0。这两个操作发生在不同位置。</p>
<div class="lab" data-lab="reference"></div>
<p class="warning">常见误区：模板来自公开协议，不来自某个目标样本；但匹配上模板并不证明包内窗口总是理想 L-STF。频率选择性信道、RX、均衡和窗口偏差仍可混淆。商掉一个公共复数不等于消除未知 FIR，参考分支的性质也不能直接推广为整个 PAMF 对信道或 CFO 严格不变。</p>
<p class="source">依据：<code>experiments/cvs_rff_physics/known_excitation.py:17</code>、<code>experiments/cvs_reference_identity/model.py:25</code>、<code>experiments/cvs_reference_identity/model.py:51</code>、<code>experiments/cvs_reference_identity/model.py:98</code>；输入策略：<code>experiments/cvs_source_identity_readout/input_policy.py:5</code>；实际版本约定：<code>docs/PAMF_FINAL_T1_E50.md:51</code>。</p>`
},
{
  id: 'p-signature', group: 'PAMF', kicker: '07 / 保留变化的先后',
  title: '路径签名：终点相同，走过的方向可以不同',
  lead: '平均池化容易忘记先后。<code>SIG_TIME_X3</code> 把局部特征看成一条路径，用位移与有向面积补充“怎样走到这里”的信息。',
  html: String.raw`
<h3>为什么只有均值还不够</h3>
<p>一圈顺时针和一圈逆时针，可以经过相同的点、回到同一终点，却有不同的顺序。单纯平均或首尾相减无法区分它们。路径描述不要求先解释每个通道的物理含义，而是读取通道共同变化的方向与先后。</p>
<p>实现从真实 <code>t2</code> 读取 <code>[B,96,32]</code>。三组独立的无偏置 <code>1×1 Conv(96→6)</code> 加 tanh，各自学习六维局部坐标 \(u_t\)。每组分成 4 个连续窗口，每窗 8 个位置，再追加从 0 至 1 的相对时间，得到七维路径 \(z_t=(u_t,\tau_t)\)。三个视图属于一个网络，不是三个模型投票。</p>
<h3>先计算位移，再计算反对称面积</h3>
<div class="equation">\[
\Delta_k=z_{k+1}-z_k,\quad B_k=\sum_{\ell\lt k}\Delta_\ell,\quad
S^{(1)}=\sum_k\Delta_k=z_{\mathrm{last}}-z_{\mathrm{first}},
\]</div>
<div class="equation">\[
A_{ab}=\frac12\sum_k\big(B_k^a\Delta_k^b-\Delta_k^aB_k^b\big),\qquad 1\le a\lt b\le7.
\]</div>
<p>\(\Delta_k\) 是相邻位置的位移，\(B_k\) 是这一步之前已经积累的位移。\(S^{(1)}\) 有 7 个分量；任意两个坐标组成一张平面，共 \(\binom{7}{2}=21\) 个 \(A_{ab}\)。每窗因此得到 \(7+21=28\) 维。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：从分段直线路径到有向面积</summary>
<ol class="steps"><li>在第 \(k\) 条线段内，用 \(z=z_0+B_k+s\Delta_k\)，\(0\le s\le1\)，描述线性移动。</li><li>二阶迭代积分包含 \(\sum_k B_k^a\Delta_k^b+\frac12\sum_k\Delta_k^a\Delta_k^b\)。前一项记录“先在 \(a\) 方向移动，再在 \(b\) 方向移动”。</li><li>将 \(a,b\) 交换并作差。线段内的对称项互相抵消，剩下 \(\frac12\sum_k(B_k^a\Delta_k^b-B_k^b\Delta_k^a)\)。这就是实现保留的反对称面积。</li><li>代码只取上三角 \(a<b\)，避免把 \(A_{ab}\) 与 \(-A_{ab}=A_{ba}\) 重复存储。它没有存下完整的 \(7\times7\) 二阶张量。</li><li>追加时间后，\(A_{a,\tau}\) 还记录特征变化相对于窗口内时间的位置。例如较早上升和较晚上升即使首尾相同，也可能产生不同的时间面积。</li></ol>
</details>
<h3>手算：两条一级项相同、面积相反的路径</h3>
<p>取二维示例 \((0,0)\to(1,0)\to(1,1)\to(0,1)\to(0,0)\)。四步增量是 \((1,0),(0,1),(-1,0),(0,-1)\)，一级项都是 0。面积公式每步贡献依次为 \(0,\frac12,\frac12,0\)，所以 \(A_{12}=+1\)。</p>
<p>将路线反向，一级项仍为 0，但 \(A_{12}=-1\)。它区分了逆时针和顺时针，尽管终点、经过点的集合与一级项相同。真实网络用七维、8 点窗口；这个五点方形只是把同一个公式降到二维便于手算。追加单调时间后，时间一级项为 1，而空间闭合项仍为 0。</p>
<h3>真实输出和参数量</h3>
<p>一组视图是 \(4\times28=112\) 维；三组拼接为 336 维，经一个无偏置 <code>Linear(336→160)</code> 和 LayerNorm，再乘逐通道 \(0.2\tanh(g_s)\)，最后加到参考响应之后的融合特征。增益初值为 0.05，绝对上界为 0.2。</p>
<div class="equation">\[
\underbrace{3(96\times6)}_{\text{三组局部投影}}+
\underbrace{336\times160}_{\text{输出投影}}+
\underbrace{2\times160}_{\text{LayerNorm}}+
\underbrace{160}_{\text{残差增益}}=55968.
\]</div>
<p>这里没有局部卷积偏置，也没有输出线性偏置。固定的时间坐标和缓冲常量不算可训练参数。默认读取 <code>t2</code> 输出；训练时它与 <code>t2.pool</code> 虽然尺寸相同，但后者在该块 Dropout 之前，不能随意混称为相同数值。</p>
<div class="lab" data-lab="signature"></div>
<p class="warning">常见误区：本版是“一级增量＋二阶反对称面积”，不是完整二阶 signature，也没有附加 <code>MEAN_VAR</code> 或 <code>COV</code> 视图。窗口内时间不是真实时间戳、TX 或 RX 标签。相同一级／面积仍可能对应不同路径，低阶描述不会保留全部序列；局部路径的数学平移性质也不等于原 IQ 到整个网络严格平移不变。</p>
<p class="source">依据：<code>experiments/cvs_rdrop_sinc_units/descriptors.py:12</code>、<code>experiments/cvs_rdrop_sinc_units/x3_features.py:18</code>、<code>experiments/cvs_rdrop_sinc_units/x3_features.py:39</code>、<code>experiments/cvs_rdrop_sinc_units/x3_features.py:56</code>；固定版本：<code>configs/pamf_final_t1_e50.json:44</code>。</p>`
},
{
  id: 'p-fusion', group: 'PAMF', kicker: '08 / 从多个视角得到一个判决',
  title: '残差融合与余弦头：加的是特征，比较的是方向',
  lead: '先把不同分支组织成一个 160 维表示，再用同一个默认分类头识别 TX。顺序、归一化位置和残差倍率都属于真实模型。',
  html: String.raw`
<h3>为什么不能把所有分支直接相加</h3>
<p>不同分支的尺度和统计性质不同。某个向量数值大，不代表身份信息更多。PAMF 先学习时频混合，再用归一化和有界逐通道增益接入 PA、参考响应与路径信息。这样的融合允许网络逐渐使用补充信息，同时保留已有主干。</p>
<p>直觉上，每个分支先写一份笔记，融合模块将它们整理为一份 160 维笔记。六个类别各有一个可学习方向，最终看这份笔记更指向哪个方向。分支没有各自投票，也没有冻结骨干后另训新头。</p>
<h3>按代码顺序展开每一次加法</h3>
<p>记时域向量为 \(t\in\mathbb R^{160}\)，频域卷积向量为 \(f_0\)，三项频谱统计的投影为 \(\Delta f\)，圆度为 \(\rho\)。先做 \(f=f_0+\Delta f\)，再把 \([t;f;\rho]\in\mathbb R^{321}\) 送入 <code>Linear(321→160) → ReLU → Dropout</code>，得到基础向量 \(b\)。</p>
<div class="equation">\[
\begin{aligned}
p_{local}&=p_0+0.25\Delta p,\\
h_{core}&=\operatorname{LN}_b(b)+\tanh(g_p)\odot\operatorname{LN}_p(p_{local}+\Delta p),\\
h_{ref}&=h_{core}+\tanh(g_r)\odot r,\\
h&=h_{ref}+0.2\tanh(g_s)\odot s.
\end{aligned}
\]</div>
<p>\(p_0\) 是 PA 卷积投影，\(\Delta p\) 是三项 PA 频谱统计投影；\(r\) 是参考分支投影并 LayerNorm 后的向量；\(s\) 是 336 维路径描述投影并 LayerNorm 后的向量；\(\odot\) 表示逐通道相乘。</p>
<p>有个容易漏掉的实现细节：主干先加 \(0.25\Delta p\)，残差头内又加一次 \(\Delta p\)，因此进入 PA LayerNorm 的实际值是 \(p_0+1.25\Delta p\)。这里报告执行顺序，不把它改写成只有 0.25。PA 增益 \(\tanh(g_p)\) 初值为 0.25，参考增益初值为 0，路径增益初值为 0.05；前两种绝对值上界为 1，路径为 0.2，均可学习为负值。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：归一化为何影响融合与分类</summary>
<ol class="steps"><li>LayerNorm 在单个样本的 160 个坐标上求均值 \(\mu\) 和方差 \(\sigma^2\)，计算 \(\gamma\odot(v-\mu)/\sqrt{\sigma^2+\epsilon}+\beta\)。它还有可学习仿射参数，不等于固定单位长度。</li><li>完成所有残差相加后，分类头再计算 L2 单位方向 \(\hat h=h/\max(\|h\|_2,10^{-4})\)，每个类别权重同样归一化为 \(\hat w_c\)。</li><li>余弦相似度为 \(\hat w_c^\top\hat h=\cos\theta_c\)，于是 logit 为 \(z_c=30\cos\theta_c\)。再经 softmax，\(P(c\mid x)=e^{z_c}/\sum_{a=1}^{6}e^{z_a}\)。</li><li>乘 30 不改变最大 logit 的类别，但改变概率尖锐程度与训练梯度。所有类别共用它；T1 的 As 温度为 1，不另加部署温度。</li><li>先分别归一化再相加、先相加再归一化，通常不是同一个结果。也不能把逐通道增益解释为和为 1 的“分支置信度”。</li></ol>
</details>
<h3>真实的默认头</h3>
<p>最终只保留一个 \(h\in\mathbb R^{160}\)，类别权重为 <code>[6,160]</code>，共 960 个参数。当前残差头的判决没有额外类别偏置，也不使用依赖真实标签的 CosFace margin；默认头与全部部署分支共同训练。参考与路径特征是先加到 \(h\) 后再分类，不能只拿原始主干向量代表完整 PAMF。</p>
<h3>手算：补充特征怎样转动方向</h3>
<p>用二维缩小例子：设 \(h_{\mathrm{core}}=(1,0)\)，参考向量 \(r=(0,1)\)、当前增益 0.1，路径向量 \(s=(0,1)\)、当前实际增益 0.05，则 \(h=(1,0.15)\)。对两个示例类别方向 \((1,0)\) 与 \((0,1)\)，余弦约为 0.98894、0.14834，对应 logit 约为 29.668、4.450。补充分支改变方向，便可能改变分类边界；真实模型仍是 160 维、6 类。</p>
<div class="lab" data-lab="cosine"></div>
<p class="warning">常见误区：分支名称和归一化不构成“真实硬件完全解耦”的证明；有界增益也不保证没有 clean 损伤。推理关闭 Dropout，训练用 R-Drop、对比配对、EMA 和 U 门控不参与判决。DURA 是前面的恢复器 G，PAMF 是识别器 F，二者参数与性能证据应分别说明。</p>
<p class="source">依据：<code>experiments/adv3b02_xuc/code/model.py:1341</code>、<code>experiments/adv3b02_xuc/code/model.py:1760</code>、<code>experiments/adv3b02_xuc/code/model.py:1822</code>、<code>experiments/cvs_residual_identity/model.py:21</code>、<code>experiments/cvs_reference_identity/model.py:109</code>、<code>experiments/cvs_rdrop_sinc_units/x3_features.py:56</code>。</p>`
},
{
  id: 'p-supervised', group: 'PAMF', kicker: '09 / 用合法标签约束表示',
  title: '监督训练：正确分类、随机一致、跨接收机同类相近',
  lead: 'CE 教模型答对，R-Drop 要求同一输入的随机网络输出一致，<code>CROSS_RX</code> 要求来自不同接收机的同 TX 表示更接近。三者各管一件事。',
  html: String.raw`
<h3>为什么分类损失之外还需要约束</h3>
<p>只把训练样本答对，模型仍可能依赖接收机特征；随机 Dropout 也可能让同一包输出摇摆。PAMF 的监督部分因此同时约束预测与表示。所有真实 TX 标签都来自合法 \(L_s\)，不以 \(U_s\) 隐藏标签或目标测试标签配对。</p>
<p>直觉上，CE 是“答案要对”，R-Drop 是“同一题两次作答要稳”，跨 RX 对比是“同一 TX 换一个接收机也应相近”。这三句话不是互相替代的目标。</p>
<h3>从 CE 推到一对 R-Drop 损失</h3>
<div class="equation">\[
q_c=0.99\,\mathbf1[c=y]+\frac{0.01}{6},\quad
\mathrm{CE}_{0.01}(p,y)=-\sum_cq_c\log p_c,\quad
\mathrm{CE}_2=\frac{\mathrm{CE}(p_1,y)+\mathrm{CE}(p_2,y)}2.
\]</div>
<p>\(p_1,p_2\) 来自同一个已生成 IQ 的两次前向，随机性主要来自训练网络中的 Dropout。第二次不重新抽取另一份 IQ 增强。平滑系数为 0.01，六类的正确类别目标为 \(0.991666\ldots\)，其余各为 \(0.001666\ldots\)。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：对称 KL 怎样惩罚不一致</summary>
<ol class="steps"><li>单向 \(D_{\mathrm{KL}}(p_1\|p_2)=\sum_c p_{1c}\log(p_{1c}/p_{2c})\)，把第一份分布当权重。</li><li>交换两份输出，再取均值：\(K=\frac12[D_{\mathrm{KL}}(p_1\|p_2)+D_{\mathrm{KL}}(p_2\|p_1)]\)。合并得到 \(K=\frac12\sum_c(p_{1c}-p_{2c})(\log p_{1c}-\log p_{2c})\)。</li><li>一对 R-Drop 为 \(\mathrm{RD}=\mathrm{CE}_2+K\)，KL 系数为 1。这里两边输出都保留梯度，不是一个冻结教师。</li><li>例如用二类数演示，\(p_1=(0.8,0.2),p_2=(0.6,0.4)\)，则 \(K=0.1\log(8/3)\approx0.0981\)；两份分布相同时为 0。真实训练仍为六类。</li></ol>
</details>
<h3>T1 的六次 L 前向与 1.25 剂量</h3>
<table class="mini-table"><thead><tr><th>一对前向的输入</th><th>两次输出的损失</th><th>外部系数</th></tr></thead><tbody><tr><td>ordinary 128</td><td>\(\mathrm{RD}\)</td><td>0.5</td></tr><tr><td>mixed batch 的 ordinary 64</td><td>\(\mathrm{RD}\)</td><td>0.25</td></tr><tr><td>同一 mixed batch 的 As 64</td><td>两次真实标签 CE，无独立 KL</td><td>0.25</td></tr><tr><td>raw 128</td><td>\(\mathrm{RD}\)</td><td>0.25</td></tr></tbody></table>
<p>mixed 仍是一批 128，三对共六次、每步 768 个 L 样本前向视图。As 从原始合法 L IQ 生成 <code>practical residual/post_sync</code> 视图，使用 low suburban、mid urban、low urban；T1 固定 \(T=1\)、As 的 \(\mathrm{KL}=0\)。精确总损失为：</p>
<div class="equation">\[
L_L=0.5\mathrm{RD}_{ord128}+0.25\mathrm{RD}_{ord64}+0.25\mathrm{CE}_{2,As64}+0.25\mathrm{RD}_{raw128}+\lambda_{RX}L_{RX}.
\]</div>
<p>CE 系数相加为 1.25，本 T1 没有再除以 1.25。“去掉 As KL”仍保留 As 的两次真实标签 CE、ordinary/raw 的 KL，也没有额外 ordinary-As logits KL。两个 ordinary 项来自不同前向和槽位，不能数值合并为同一个损失。</p>
<h3>浅中层 <code>CROSS_RX</code>：从特征到合法正负对</h3>
<p>在 ordinary128 的 <code>time_down</code>（\(48\times128\)）与 <code>t2.pool</code>（\(96\times32\)）计算每通道时间均值与总体标准差，分别得到 96、192 维描述。然后用<strong>当前批、停止梯度的</strong>逐坐标均值和总体标准差标准化，标准差下限 \(10^{-4}\)，再做 L2 归一化。统计不参与反传，但特征仍参与。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：源码中的 <code>CROSS_RX</code> 对比目标</summary>
<p>对锚点 \(i\)，正对 \(P_i\) 是同 TX、不同物理 RX；负对 \(N_i\) 是不同 TX。自身和同 TX 同 RX 都不进入分母。令标准化后的单位描述为 \(z_i\)，相似度 \(s_{ia}=z_i^\top z_a/0.2\)。源码使用</p>
<div class="equation">\[
\ell_i=\log\sum_{a\in P_i\cup N_i}e^{s_{ia}}-\frac1{|P_i|}\sum_{p\in P_i}s_{ip}-\log|P_i|.
\]</div>
<p>第一项压低所有候选的总竞争，第二项提高每个合法正对的相似度；最后一项是源码保留的正对数量常数。只有同时有正、负对的锚点参与平均。若一正一负得分为 2、0，损失为 \(\log(1+e^{-2})\approx0.1269\)。</p>
<p>两个深度、两次 ordinary 前向再平均，复用已有特征。权重 \(\lambda_{\mathrm{RX}}=0.1\min(\max((e-5)/10,0),1)\)，前 5 轮为 0，第 15 轮达到 0.1；As 不作 ORD128 锚点，没有新分类头或额外骨干前向。</p>
</details>
<p>再手算一次 CE：六类输出为正确类 0.8、其余各 0.04，平滑 CE 约为 0.2481，硬标签 CE 为 \(-\log0.8\approx0.2231\)。平滑仍奖励正确类别，但不鼓励无限确信。</p>
<div class="lab" data-lab="rdrop"></div><div class="lab" data-lab="contrast"></div>
<p class="warning">常见误区：<code>CROSS_RX</code> 是训练约束，不保证真正消除了 RX；缺少跨 RX 正对的批次也不能制造有效锚点。标准化统计的 detached 与“冻结特征”不同。损失变小只能说明优化目标变化，跨接收机与信道收益仍需冻结预测后独立评分。</p>
<p class="source">依据：<code>docs/PAMF_FINAL_T1_E50.md:67</code>、<code>docs/PAMF_FINAL_T1_E50.md:100</code>；实际同形式对比实现：<code>experiments/cric_dura_comparison/f_model.py:12</code>；As KL0 继承实现：<code>experiments/cvs_as_temperature84/loss.py:6</code>、<code>experiments/cvs_as_temperature84/model.py:12</code>。</p>`
},
{
  id: 'p-ssl', group: 'PAMF', kicker: '10 / 在隐藏标签下学习',
  title: '半监督：先分清 T1，再推导 PAMF-REPL',
  lead: '无标签并不意味着没有监督信号：模型可以提出伪标签，但必须控制错误被自己放大的风险。固定 T1 与后续选定 REPL 使用不同机制。',
  html: String.raw`
<h3>为什么伪标签要选择，也要保护 L</h3>
<p>\(L_s\) 有合法 TX 标签，\(U_s\) 的 TX 真值始终隐藏。模型把当前概率最大类别记为 \(\hat y=\arg\max_c p_c\)，置信度为 \(c=\max_c p_c\)，再用 \(\hat y\) 训练自己。直觉上，这是把有把握的未标注例题变成练习；若一开始答错并确信，重复练习也可能强化错误。</p>
<p class="note"><strong>两个已命名版本分别解释。</strong><code>pamf_t1_e50</code> 是固定 T1 识别器，使用自身 EMA 与三条件门控。2026-10-24 选定的半监督方法 <code>pamf_repl</code> 使用当前学生、固定 0.95 门槛、接收机外推及完整 L 梯度保护。这里讲解 REPL 不会把 T1 的原配方改写成普通 PL。</p>
<h3>T1：自身 EMA 与三条件选择</h3>
<div class="equation">\[
\theta_T\leftarrow0.999\theta_T+0.001\theta_S,\quad
m_i=m_i^{conf}\land m_i^{neighbor}\land m_i^{strong},\quad
L=L_L+0.16L_{PL}+0.01L_{ent}.
\]</div>
<p>\(\theta_S,\theta_T\) 是学生和教师参数。只有学生成功更新后才更新自身 EMA，教师不反传；部署使用最终学生。置信门控在当前 U 批的 RX/day 分组内取 0.86 分位数，再截到 \([0.92,0.97]\)。相邻门控要求同批、同 RX/day/eq，<code>sig_i</code> 与 <code>base_index</code> 距离都不超过 2，双方置信度至少 0.8，并且另一个样本预测相同。强视图门控要求加标准差 0.015 高斯噪声后的学生 argmax 与 EMA 一致。</p>
<p>\(L_{\mathrm{PL}}\) 只对选中者求均值，空选时为 0；\(L_{\mathrm{ent}}=-\operatorname{mean}_i\sum_c p_{ic}\log p_{ic}\) 对全部 U 强视图求均值。相邻条件不是跨 epoch 记忆库，也没有额外域分类头正确性门控。</p>
<div class="lab" data-lab="gate"></div>
<h3>REPL：同伪类别、不同 RX，构造更难的练习</h3>
<p>REPL 在当前学生的 eval／no-grad 前向上生成 raw 伪标签，固定 \(m_i=\mathbf1[c_i\ge0.95]\)，不使用 EMA 教师。训练用六次 U 学生前向，伪标签平滑 0.01，总 U 权重 6。其带掩码 CE 按<strong>整个 U 批大小</strong>求均值，和 T1 的“选中者均值”不同。例如 4 个样本仅选中两项，损失为 0.2、0.1，单项 masked CE 为 \((0.2+0.1)/4=0.075\)，再乘 6 为 0.45。</p>
<p>在已接受、伪类别相同而源 RX 不同的样本间选伙伴 \(j\)。浅中层风格扰动作用在 <code>time_down</code>、<code>t2.pool</code>，每层概率 0.5，同一个配对计划供六次 U 前向共用。均值和标准差停止梯度，\(\lambda\sim\mathrm{Beta}(0.1,0.1)\)。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：风格扰动与完整特征外推</summary>
<p>先用每通道时间均值 \(\mu_i\) 和 \(\sigma_i=\sqrt{\mathrm{var}(h_i)+10^{-6}}\) 标准化。再混合统计 \(\mu'=\lambda\mu_i+(1-\lambda)\mu_j\)、\(\sigma'=\lambda\sigma_i+(1-\lambda)\sigma_j\)，重建 \(h_i'=(h_i-\mu_i)\sigma'/\sigma_i+\mu'\)。例如 \(h_i=(1,3)\)、\(\mu_i=2,\sigma_i=1\)，伙伴统计为 \(\mu_j=6,\sigma_j=2\)，取 \(\lambda=0.25\)，忽略 \(\epsilon\) 后得到 \((3.25,6.75)\)。</p>
<p>接着在真实部署使用的完整 160 维融合特征上计算权重 0.05 的同伪类跨 RX 对比，沿用温度 0.2。它包含参考与路径分支，不只取一个中间主干向量。</p>
<div class="equation">\[
h_i^{ext}=h_i+0.25(h_i-h_j)=1.25h_i-0.25h_j.
\]</div>
<p>这是向远离伙伴的方向外推，不是插值平均。若 \(h_i=(2,1),h_j=(1,1)\)，则 \(h_i^{\mathrm{ext}}=(2.25,1)\)。让原默认头仍判为 \(\hat y_i\)，尝试扩大对源 RX 差异的容忍。对于有效配对，ordinary CE 的 0.5 系数分为原特征 0.25 与外推特征 0.25，保持 1.25 的 CE 归一化；无合法配对者保留原 CE，不丢样本。复用前两次 ordinary U 特征，只增加两次既有分类头调用，不增加骨干前向或推理参数。</p>
</details>
<div class="lab" data-lab="repl"></div>
<h3>梯度保护：让强 U 信号服从完整 L 的方向</h3>
<p>记完整当前 \(L_L\) 的梯度为 \(g_L\)，已包含 U 权重 6、对比与外推的梯度为 \(g_U\)。实际按 <code>t2 / cls_head / other</code> 三组分别计算，保护锚点包含完整 L 目标，不只是 clean CE。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：移除冲突方向，再限制 U/L 范数</summary>
<p>用一阶近似，沿 \(-g_U\) 更新对 L 的影响是 \(-\eta g_L^\top g_U\)。若点积为负，它会倾向增加 L。求距离原 \(g_U\) 最近且满足 \(g_L^\top v\ge0\) 的向量，得到</p>
<div class="equation">\[
v=g_U-\frac{\min(g_L^\top g_U,0)}{\|g_L\|^2}g_L,\qquad
u=v\min\left(1,\frac{\|g_L\|}{\|v\|}\right),\qquad g=g_L+u.
\]</div>
<p>第一步只删除反向分量，第二步把 U/L 范数上限设为 1。若 \(\|g_L\|^2\le10^{-12}\)，该组 U 置 0；若 \(v=0\) 则不做除法。例如 \(g_L=(1,0),g_U=(-2,3)\)，先得 \(v=(0,3)\)，限幅为 \(u=(0,1)\)，最终为 \((1,1)\)。完整 L 梯度保留，随后交给原优化器。这个几何条件作用于参数梯度；曲率、有限步长和自适应优化器意味着它不是训练损失永不增加的保证。</p>
</details>
<div class="lab" data-lab="gradient"></div>
<h3>真实日程、权限与局限</h3>
<p>T1 与 REPL 都记录累计 E50／11100 步：13 轮 L，37 轮联合 L/U，每轮 222 步。T1 由自身 E20、E10、E20 续训并保留 EMA；REPL 从同契约、各自 seed 的原普通 PL E20 前段继续 10＋20 轮，新段 6660 步，无 EMA。“37U”不是只喂 U。</p>
<p class="warning">常见误区：高置信度不等于真实正确率，同伪类也未必同真实 TX；外推不是已知真实接收机变换。源 V 只诊断，不更新参数、统计或原型；目标先固定全部预测，再独立 truth-last。REPL 按已公开 clean 结果选定，不是独立盲选，也不表示所有 practical 场景都优于监督版。训练配对和梯度保护不进入单样本推理。</p>
<p class="source">依据：<code>docs/PAMF_FINAL_T1_E50.md:106</code>、<code>docs/PAMF_REPL.md:15</code>、<code>configs/pamf_ssl_repl.json:46</code>、<code>experiments/pamf_ssl_research/hard_pl_objective.py:54</code>、<code>experiments/pamf_ssl_research/u_mixstyle_objective.py:24</code>、<code>experiments/pamf_ssl_research/hard_pl_receiver_extrapolation_objective.py:25</code>、<code>experiments/pamf_ssl_research/u_gradprotect_objective.py:12</code>。</p>`
}
);
