/* A complete Conv1d lesson, inserted before the actual PAMF time branch. */
(function(){
'use strict';
const chapter={
 id:'p-convolution',group:'PAMF',kicker:'03A / 先看清每一次乘加',
 title:'普通卷积与深度可分离卷积：先分别处理，再交换通道信息',
 lead:String.raw`普通卷积一次完成“沿序列看邻居”和“混合通道”；深度可分离卷积把这两件事拆开。先用两条短序列手算，再代入 PAMF 的 \(48\to72\) 个通道。`,
 html:String.raw`
<h3>从一个数开始：卷积到底算了什么</h3>
<p>把一条特征序列想成一排传感器读数。卷积核是一个小窗口中的可学习权重。窗口向右滑动，每次把读数与权重逐项相乘、相加，生成一个输出。<strong>通道</strong>表示同一位置的不同特征，<strong>长度</strong>表示有多少个位置；二者不能交换。输入形状为 \([B,C_{\mathrm{in}},L]\)，其中 \(B\) 是样本数。</p>
<p>本教程讲 PAMF 实际使用的一维卷积。PyTorch <code>Conv1d</code> 按互相关约定计算，不把核反转。前面基础章的因果 FIR 写作历史点卷积；这里改用神经网络的精确索引。核宽为 1 的逐点卷积常写作“\(1\times1\)”，在一维代码中就是 <code>kernel_size=1</code>。</p>
<div class="equation">\[
y_o[t]=b_o+\sum_{c=0}^{C_{\mathrm{in}}-1}\sum_{r=0}^{k-1}
 W_{o,c,r}\,x_c[st-p+dr].
\]</div>
<table class="mini-table"><thead><tr><th>符号／配置</th><th>含义</th><th>PAMF 的 DS 卷积</th></tr></thead><tbody>
<tr><td>\(c,o\)</td><td>输入、输出通道索引</td><td>由各层宽度固定</td></tr>
<tr><td>\(k\)</td><td>核长，即每条输入通道看几个位置</td><td>\(5\) 或 \(3\)</td></tr>
<tr><td>\(s,p,d\)</td><td>步长、左右各补零点数、膨胀率</td><td>\(s=1,\ p=\lfloor k/2\rfloor,\ d=1\)</td></tr>
<tr><td>\(W,b\)</td><td>训练得到的权重和可选偏置</td><td>两个卷积都关闭偏置</td></tr>
<tr><td><code>groups</code></td><td>通道连接分组</td><td>逐通道层为 \(C_{\mathrm{in}}\)，逐点层为 \(1\)</td></tr>
</tbody></table>
<p>普通卷积中，每个输出通道都能读取全部输入通道，而且对不同输入、输出组合有独立的长度 \(k\) 的核。例如 \(C_{\mathrm{in}}=2,C_{\mathrm{out}}=2,k=3\)，共有 \(2\times2=4\) 个短核、12 个权重。</p>
<h3>拆成两步：Depthwise 与 Pointwise</h3>
<p><strong>逐通道卷积（Depthwise，DW）</strong>先让每个输入通道只看自己的邻居。这里每个输入通道产生一个中间通道，通道数不变。它不能独自把 \(48\) 个通道变成 \(72\) 个通道。<strong>逐点卷积（Pointwise，PW）</strong>再在每个位置混合中间通道；核长为 1，但它读取全部通道，并且可以改变通道数。</p>
<div class="equation">\[
u_c[t]=\sum_{r=0}^{k-1}D_{c,r}\,x_c[st-p+dr],
\qquad
y_o[t]=\sum_{c=0}^{C_{\mathrm{in}}-1}M_{o,c}\,u_c[t].
\]</div>
<p>\(D\) 是逐通道核，\(M\) 是逐点通道混合矩阵，二者都由反向传播学习。“深度”在这里指通道方向的拆分，不是网络层数更多，也不是简单地少读一半输入。PW 的权重跨时间位置共享；每个位置都用同一张矩阵。</p>
<div class="lab" data-lab="depthwise-step"></div>
<h3>完整手算：两条输入如何得到一个输出</h3>
<p>为看清乘加，下面暂时不补零、不做归一化与激活，步长为 1。输入、逐通道核和逐点矩阵分别取：</p>
<div class="equation">\[
x_1=(1,2,3,4,5,6),\quad x_2=(2,1,0,-1,-2,-3),\quad
D_1=(1,0,-1),\quad D_2=(1,2,1),\quad
M=\begin{bmatrix}1&2\\-1&1\end{bmatrix}.
\]</div>
<ol class="steps"><li>在输出位置 \(t=0\)，两条输入窗口是 \((1,2,3)\) 和 \((2,1,0)\)。DW 分别得到 \(u_1=1+0-3=-2\)、\(u_2=2+2+0=4\)。</li>
<li>PW 的第一行产生 \(y_1=1(-2)+2(4)=6\)；第二行产生 \(y_2=-1(-2)+1(4)=6\)。每个输出都用到了两条中间通道。</li>
<li>如果把权重合并，第一输出对应的普通卷积核是 \((1,0,-1)\)、\((2,4,2)\)。直接在原输入上相乘相加也得到 \(6\)。这个例子说明可以合并<strong>这组已分解的权重</strong>，不表示任意普通卷积都能这样精确分解。</li>
<li>输入长 6、核长 3、不补零，输出长 \(6-3+1=4\)。把滑块移到下一位置，核系数保持不变，输入窗口才向右移动。</li></ol>
<h3>参数和计算为什么会减少：逐项数出来</h3>
<details class="derivation" data-depth="deep"><summary>从每个核的长度，推导到一般节省比例</summary>
<p>普通卷积有 \(C_{\mathrm{out}}C_{\mathrm{in}}\) 个长度为 \(k\) 的核；DW 有 \(C_{\mathrm{in}}\) 个长度为 \(k\) 的核；PW 是一个 \(C_{\mathrm{out}}\times C_{\mathrm{in}}\) 矩阵。忽略偏置和后续归一化：</p>
<div class="equation">\[
P_{\mathrm{ordinary}}=kC_{\mathrm{in}}C_{\mathrm{out}},\qquad
P_{\mathrm{DS}}=kC_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}},
\qquad
\frac{P_{\mathrm{DS}}}{P_{\mathrm{ordinary}}}
=\frac1{C_{\mathrm{out}}}+\frac1k.
\]</div>
<p>在 DW 与 PW 的输出长度相同时，每个位置都用一次对应权重，乘加次数也具有同样的比例：</p>
<div class="equation">\[
\mathrm{MAC}_{\mathrm{ordinary}}=L'kC_{\mathrm{in}}C_{\mathrm{out}},
\qquad
\mathrm{MAC}_{\mathrm{DS}}=L'(kC_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}).
\]</div>
<p>这里一次 MAC 指一次乘法及其累加。若某报告把乘、加各计一次，主项 FLOPs 约为 \(2\,\mathrm{MAC}\)，必须声明口径。上式尚未计入归一化、激活、池化、访存和运行调度。</p>
</details>
<p>代入 PAMF 的 <code>t1</code>：\(C_{\mathrm{in}}=48,C_{\mathrm{out}}=72,k=5,L'=128\)。普通卷积需要 \(17280\) 个核权重；DW 为 \(240\)，PW 为 \(3456\)，合计 \(3696\)。核权重减少约 \(78.61\%\)，两者理论 MAC 分别为 \(2211840\)、\(473088\)。<strong>不能把这个比例直接写成整网实测加速倍数。</strong></p>
<div class="lab" data-lab="depthwise-cost"></div>
<h3>回到源码：一个 DSConvBlock 不只有两层卷积</h3>
<p>实际顺序是 <code>DW → PW → GroupNorm → ReLU → MaxPool → Dropout</code>。DW 与 PW 中间没有激活，也没有归一化。本版 DW 的通道乘数为 1。最大池化另做降采样；卷积自身的步长仍为 1。</p>
<div class="equation">\[
L'=\left\lfloor\frac{L+2p-d(k-1)-1}{s}+1\right\rfloor.
\]</div>
<p>奇数核、\(s=d=1,p=(k-1)/2\) 给出 \(L'=L\)。<code>t1</code> 的形状因而是 \(48\times128\to48\times128\to72\times128\to72\times64\)。最后一步来自核长与步长均为 2 的最大池化。池化之后才丢弃训练用的 Dropout 随机单元，推理关闭 Dropout。</p>
<table class="mini-table"><thead><tr><th>层</th><th>输入 → 池化后输出</th><th>核长／池化</th><th>DW＋PW 权重</th><th>含 GN 仿射的块参数</th></tr></thead><tbody>
<tr><td><code>t1</code></td><td>\(48\times128\to72\times64\)</td><td>\(5/2\)</td><td>\(3696\)</td><td>\(3840\)</td></tr>
<tr><td><code>t2</code></td><td>\(72\times64\to96\times32\)</td><td>\(5/2\)</td><td>\(7272\)</td><td>\(7464\)</td></tr>
<tr><td><code>t3</code></td><td>\(96\times32\to96\times32\)</td><td>\(3/1\)</td><td>\(9504\)</td><td>\(9696\)</td></tr>
<tr><td><code>f1</code></td><td>\(4\times32\to16\times16\)</td><td>\(5/2\)</td><td>\(84\)</td><td>\(116\)</td></tr>
<tr><td><code>f2</code></td><td>\(16\times16\to32\times8\)</td><td>\(5/2\)</td><td>\(592\)</td><td>\(656\)</td></tr>
<tr><td><code>f3</code></td><td>\(32\times8\to32\times8\)</td><td>\(3/1\)</td><td>\(1120\)</td><td>\(1184\)</td></tr>
</tbody></table>
<p>GroupNorm 每个输出通道有一个可学习缩放和一个偏移，所以增加 \(2C_{\mathrm{out}}\) 个参数；均值、方差是当前样本算出的统计，不是额外可训练参数。源码从 \(16,8,4,2,1\) 依次选择能整除通道数的组数，因此 <code>t1</code> 用 8 组，每组 9 通道；频域 16、32 通道均用 16 组。时域三个 DS 块的 Dropout 概率为 \(0.10\)，频域为 \(0.05\)。</p>
<h3>好处、代价和一个反例</h3>
<p>直接好处是减少核参数、理论乘加与权重存储；仍能通过 PW 学习跨通道组合。节约的预算可以支持多分支结构。代价是给卷积核施加了结构限制：对同一个输入通道，各输出所用的空间核只能是同一个 \(D_c\) 的不同倍数。</p>
<div class="equation">\[
W^{\mathrm{equiv}}_{o,c,r}=M_{o,c}D_{c,r},\qquad
\operatorname{rank}\!\left(W^{\mathrm{equiv}}_{:,c,:}\right)\le1.
\]</div>
<p>例如同一输入通道到第一个输出想用 \((1,0,-1)\)，到第二个输出想用 \((0,1,0)\)。两向量不成比例，单个通道乘数为 1 的 DW＋PW 无法同时精确实现。堆叠多层及非线性可扩大表达能力，但不能据此声称这一个块与任意普通卷积等价。</p>
<p class="warning">实际延迟取决于硬件、库、张量大小与访存；更少 MAC 不保证同等比例加速，也不保证识别更准。这里解释的是已固定 PAMF 架构的选择，没有替换网络或重新训练。PA 路径用的是普通膨胀卷积，不能把本章 DS 参数公式套到 PA 块。</p>
<p class="source">项目依据：<code>experiments/adv3b02_xuc/code/model.py:67</code>、<code>model.py:344</code>、<code>model.py:1251</code>、<code>model.py:1284</code>。互相关、分组与尺寸约定参见 <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html" target="_blank" rel="noreferrer">PyTorch Conv1d 官方文档</a>；DW＋PW 的原始方法说明见 <a href="https://arxiv.org/abs/1704.04861" target="_blank" rel="noreferrer">MobileNets 原论文</a>。图与数值推导按本版一维配置编写。</p>`
};
const at=window.COURSE_CHAPTERS.findIndex(c=>c.id==='p-time');
window.COURSE_CHAPTERS.splice(at,0,chapter);
})();
