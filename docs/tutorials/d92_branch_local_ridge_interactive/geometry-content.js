/* Geometry chapters for the frozen D92-BranchLocalRidge-v1 method. */
window.COURSE_CHAPTERS = window.COURSE_CHAPTERS || [];
window.COURSE_CHAPTERS.push(
{
  id: 'g-features', group: 'D92', kicker: '03 / 先认识输入坐标',
  title: '五块特征：先统一尺度，再拼成背景与辅助两组',
  lead: '方法接收每个物理样本的五块固定特征。归一化和拼接决定了后续“距离”的含义，不能只把 736 个数随意连接起来。',
  html: String.raw`
<h3>为什么先处理尺度</h3>
<p>不同特征块可能有不同数值范围。如果直接拼接，幅度最大的块容易主导距离，即使它未必最有用。这里先逐块归一化，再使用固定系数组织两组坐标；Phase2 不重新学习分支权重。该历史版本连接冻结的 DAOT A1FastTrustRC4 编码器，不自动换成后来的 PAMF-T1、REPL 或 DURA 组合。</p>
<p>直觉上，先把五种测量统一为方向描述，再规定如何摆放。每个物理 support 仍然只有一行；从同一 received 观测计算多个数学表征，不会把一条物理记录变成多个 shot。</p>
<p>“背景”和“辅助”只是固定的算法分组，不表示前者只有环境信息、后者只有身份信息。分组也不随旧类、新类或某个 TX 改变。我们要先理解坐标的构造，再讨论这些坐标是否有利于泛化。</p>
<h3>五块输入与两个归一化下限</h3>
<table class="mini-table"><thead><tr><th>输入</th><th>每样本宽度</th><th>作用</th></tr></thead><tbody><tr><td><code>z_id</code></td><td>160</td><td>冻结身份表示</td></tr><tr><td><code>fft</code></td><td>96</td><td>历史频谱对数幅度 sketch</td></tr><tr><td><code>t_emb</code>、<code>f_emb</code>、<code>pa_local</code></td><td>各 160</td><td>三个已有分支表示</td></tr></tbody></table>
<p>总输入宽度是 \(160+96+3\times160=736\)。本模块使用 \(\operatorname{unit}_\epsilon(v)=v/\max(\|v\|_2,\epsilon)\)，其中 \(\epsilon=10^{-12}\)。冻结定义中的 FFT sketch 还有自己的 \(10^{-8}\) 归一化下限；这是前一计算阶段的规则，不能拿它替换本模块的 \(10^{-12}\)，也不是额外可调超参数。</p>
<p>FFT96 的固定流程是：先把收到的 IQ 表示为 float32，再逐条转成 float64 复数序列 \(x\)，减去复均值；仅当 \(\operatorname{RMS}(x)>10^{-8}\) 时除以该 RMS。随后乘 Hann 窗 \(w[n]=0.5[1-\cos(2\pi n/255)]\)，\(n=0,\ldots,255\)，计算 FFT、fftshift，再取 \(\log(1+|X|)\)。把 \([0,1]\) 上 256 个等距源位置的曲线<strong>线性插值</strong>到 96 个等距目标位置，得到的 sketch 先转 float32，再减去自身 96 维均值，最后除以 \(\max(\|\mathrm{sketch}\|_2,10^{-8})\)。所以它不是简单截取 96 个 FFT bin，而是固定的 96 维数学特征，不更新骨干。这里用 Hann 窗；前一教程 Sinc 滤波核使用的是 Hamming 窗，二者不要混同。</p>
<div class="equation">\[
b=\operatorname{unit}_{10^{-12}}\!\left(
[\operatorname{unit}_{10^{-12}}(z),\,4\operatorname{unit}_{10^{-12}}(FFT)]
\right)\in\mathbb R^{256},
\]</div>
<div class="equation">\[
a=\frac{[\operatorname{unit}_{10^{-12}}(t),\operatorname{unit}_{10^{-12}}(f),
\operatorname{unit}_{10^{-12}}(pa)]}{\sqrt3}\in\mathbb R^{480}.
\]</div>
<details class="derivation" data-depth="deep"><summary>逐步推导：系数 4 为什么对应 16∶1 能量</summary>
<ol class="steps"><li>假设身份和 FFT 两块的原始范数均不低于归一化下限。各自归一化后，平方范数都是 1。</li><li>拼接时两块占不同坐标，平方范数相加，得到 \(1+4^2=17\)，没有两个块之间的交叉项。</li><li>外层归一化再除以 \(\sqrt{17}\)。身份部分能量为 \(1/17\)，FFT 部分为 \(16/17\)，因此比例为 16∶1。它不是分类概率，也不是适配时学出的可靠性。</li><li>辅助组的三个块若都达到正常单位范数，拼接能量为 3，除以 \(\sqrt3\) 后能量为 1。方法不再对 \([b;a]\) 做一次整体归一化。</li></ol>
</details>
<h3>手算：一个缩小的拼接例子</h3>
<p>把维度临时缩小，取 \(z=(3,4)\)、\(FFT=(0,2)\)。归一化后分别为 \((0.6,0.8)\)、\((0,1)\)，所以 \(b=(0.6,0.8,0,4)/\sqrt{17}\)。这保留方向，并按既定 16∶1 分配两块能量。真实宽度仍为 160 与 96，而不是这个四维玩具例子。</p>
<h3>零块不是普通单位向量</h3>
<p>若某块为零，归一化后仍为零，不会补一个随机方向。若三个辅助块只有两个非零且都达到正常单位范数，则 \(\|a\|^2=2/3\)；三块全零时 \(a=0\)。若一个非零向量的范数只有 \(5\times10^{-13}\)，除以下限后范数为 0.5，也不是 1。因此“背景、辅助恒为单位范数”的说法需要条件。</p>
<p>对 \(N\) 个 support，实际形成 <code>[N,256]</code> 与 <code>[N,480]</code> 两个 float64 数组。原五块 float32 缓存每行是 \(736\times4=2944\) bytes；两组 float64 数值每行是 \(736\times8=5888\) bytes。维度不增加物理样本数，方法也不要求重新运行编码器。</p>
<div class="lab" data-lab="blocks"></div>
<p class="warning">常见误区：先把全部 736 维一起归一化，会改变固定的块间比例；给零块补方向，会改变历史方法。特征块名称不是物理因素严格解耦的证明。本章说明既有输入处理，不为任何未知 checkpoint 或数据来源补作合规认定。</p>
<p class="source">依据：<code>configs/d92_branch_local_ridge_frozen_20260929.json:8</code>、<code>code/cvsrffi/d92_branch_support_probe.py:65</code>、<code>code/cvsrffi/d92_branch_support_probe.py:73</code>、<code>code/cvsrffi/d92_branch_interaction.py:45</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:32</code>。FFT96 的同配方参考实现：<code>code/cvsrffi/stage2_d92_mv_kme.py:116</code>，只引用其中的 <code>received_fft96</code>，不引入该文件的多视图或随机 Fourier 机制。</p>`
},
{
  id: 'g-interaction', group: 'D92', kicker: '04 / 从并列到交互',
  title: '交互映射：把“背景与分支共同出现”写进相似度',
  lead: '单纯拼接只能相加两组的相似度。外积再加入二者的乘积关系，而且不用真的保存十二万多个交互坐标。',
  html: String.raw`
<h3>为什么还需要交互项</h3>
<p>背景相似、分支相似，是两条信息；“两者同时相似”又是一条信息。若只使用 \([b;a]\) 做线性分类，模型把两组贡献相加。加入外积后，分类器还可以利用某个背景坐标与某个辅助坐标的共同变化。这里是特征坐标之间的代数关系，不是在估计真实硬件之间的因果作用。</p>
<p>直觉上，原来记的是两个读数，现在还记下两两乘积。后续 ridge 对这些坐标仍然是线性的；相对于原始输入，两两乘积已经提供了非线性。</p>
<h3>显式映射为什么有 123,616 维</h3>
<div class="equation">\[
\phi(x)=[\,b(x),\,a(x),\,\operatorname{vec}(b(x)a(x)^\top)\,],
\quad
256+480+256\times480=123616.
\]</div>
<p>外积 \(ba^\top\) 是 <code>256×480</code> 矩阵，第 \((r,s)\) 项是 \(b_ra_s\)。\(\operatorname{vec}\) 只是把矩阵按固定顺序排成向量；两边使用相同顺序，内积便不受排法影响。这个完整映射没有再次整体归一化。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：高维内积如何化成三个标量项</summary>
<ol class="steps"><li>记两个样本为 \(i,j\)，背景内积 \(B_{ij}=b_i^\top b_j\)，辅助内积 \(A_{ij}=a_i^\top a_j\)。</li><li>拼接向量的内积等于各块内积之和，因此前两块贡献 \(B_{ij}+A_{ij}\)。</li><li>外积块的贡献为 \(\sum_{r,s}(b_{ir}a_{is})(b_{jr}a_{js})\)。把两个求和拆开，得到 \((\sum_r b_{ir}b_{jr})(\sum_s a_{is}a_{js})=B_{ij}A_{ij}\)。</li><li>所以 \(k_0(i,j)=\langle\phi_i,\phi_j\rangle=B_{ij}+A_{ij}+B_{ij}A_{ij}\)。矩阵写法是 \(K_0=B+A+B\odot A\)，其中 \(\odot\) 为逐元素乘法，绝不是矩阵乘法 \(BA\)。</li></ol>
<p>这也解释了核的合法性：\(K_0\) 本来就是显式向量的 Gram 矩阵，因此对任意系数 \(v\)，都有 \(v^\top K_0v=\|\sum_i v_i\phi_i\|^2\ge0\)。不必先生成完整交互向量才能利用这一性质。</p>
</details>
<h3>手算：相似度乘积来自哪里</h3>
<p>用二维块演示，取 \(b_i=(1,0),b_j=(0.6,0.8)\)，以及 \(a_i=(1,0),a_j=(0.8,0.6)\)。则 \(B_{ij}=0.6,A_{ij}=0.8\)，交互项为 0.48，总核值为 1.88；单纯拼接只有 1.4。</p>
<p>这个例子两组都为单位向量，自核值各为 \(1+1+1=3\)，所以对应完整平方距离为 \(3+3-2\times1.88=2.24\)。若存在零块或范数下限，自核不一定等于 3，不能无条件沿用数字 6。</p>
<p>还可把交互映射上的一个类别权重拆开，写成 \(s_c=w_{bc}^\top b+w_{ac}^\top a+b^\top M_ca\)。最后一项说明：同一背景坐标的作用可以随辅助坐标变化，而不只是固定加分。矩阵 \(M_c\) 只是高维线性权重的另一种排法，不是单独估计的物理作用矩阵。实际采用核式求解，无须显式保存它；冻结编码器也不会因这个代数展开增加神经训练步骤。</p>
<h3>三个名称分别改了哪一层</h3>
<table class="mini-table"><thead><tr><th>固定对照／候选</th><th>表示或核</th><th>关系</th></tr></thead><tbody><tr><td>BranchRidge</td><td>\([b;a]\)，核 \(B+A\)</td><td>736 维并列特征</td></tr><tr><td>BranchInteraction</td><td>\(\phi\)，核 \(B+A+B\odot A\)</td><td>加入跨块外积</td></tr><tr><td>BranchLocalRidge</td><td>\(\exp(-\|\phi_i-\phi_j\|^2/\tau)\)</td><td>在完整交互几何上再做径向变换</td></tr></tbody></table>
<p>LocalRidge 的三个臂预先固定，实际函数直接调用原 BranchRidge 与 Interaction 实现作对照；没有运行时挑最好的头。对 \(N\) 行训练数据，背景、辅助 Gram 都是 <code>N×N</code>，完整高维 \(\phi\) 不必物化。单个 query 的核行只有 \(N\) 项，最后仍面对全部注册类。</p>
<div class="lab" data-lab="interaction"></div>
<p class="warning">常见误区：123,616 是有限交互映射的维度，不是后面高斯核空间的维度。核值不是概率，可以为负；“交互项”也不等于增加一个神经网络分支。对完整映射再做归一化、把逐元素乘法换成矩阵乘法，都会得到另一个方法。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_ridge.py:45</code>、<code>code/cvsrffi/d92_branch_interaction.py:15</code>、<code>code/cvsrffi/d92_branch_interaction.py:74</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:279</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:36</code>。</p>`
},
{
  id: 'g-distance', group: 'D92', kicker: '05 / 算对接近零的距离',
  title: '稳定距离：用秩至多为 2 的差，避开大数相减',
  lead: '数学上等价的公式，在计算机里未必同样可靠。这里直接累加平方，保留近重复特征之间的小距离。',
  html: String.raw`
<h3>为什么不直接算“6 减去两倍相似度”</h3>
<p>一般恒等式是 \(D_{ij}=\|\phi_i\|^2+\|\phi_j\|^2-2\langle\phi_i,\phi_j\rangle\)。只有两组块都达到正常单位范数时，才可简写成 \(6-2k_0(i,j)\)。当两点很近、核值接近 3，后者要让几个接近的大数互相抵消，微小距离可能被浮点舍入淹没。</p>
<p>直觉上，要量两条几乎相同的路线，先比较各自的差，比先量两次很长的总长度再相减更稳。实现沿用同一个完整交互距离，并没有换一种近似度量。</p>
<h3>先把完整距离拆成三个平方和</h3>
<div class="equation">\[
d_b=b_i-b_j,\quad d_a=a_i-a_j,\qquad
D_{ij}=\|d_b\|^2+\|d_a\|^2+
\|b_i a_i^\top-b_j a_j^\top\|_F^2.
\]</div>
<p>\(\|\cdot\|_F\) 是矩阵所有元素平方和再开方的 Frobenius 范数。前两项很容易算；困难只在最后的 <code>256×480</code> 外积差。这个差恰好可以用两个秩 1 项表示。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：外积差、两列矩阵与 reduced QR</summary>
<ol class="steps"><li>加入并减去 \(b_j a_i^\top\)，得到 \(b_i a_i^\top-b_j a_j^\top=(b_i-b_j)a_i^\top+b_j(a_i-a_j)^\top\)。</li><li>令 \(U=[d_b,b_j]\in\mathbb R^{256\times2}\)，\(V=[a_i^\top;d_a^\top]\in\mathbb R^{2\times480}\)，那么外积差就是 \(UV\)。这里 \(V\) 已按行排列，不再额外转置一次。</li><li>对 \(U\) 作 reduced QR：\(U=QR_{\mathrm{qr}}\)，其中 \(Q^\top Q=I_2\)、\(R_{\mathrm{qr}}\) 是 <code>2×2</code> 上三角矩阵。</li><li>于是 \(\|UV\|_F^2=\operatorname{tr}(V^\top R_{\mathrm{qr}}^\top Q^\top Q R_{\mathrm{qr}}V)=\|R_{\mathrm{qr}}V\|_F^2\)。只需计算 <code>2×480</code> 的乘积并求平方和。</li><li>整个推导没有用到单位范数假设；零块、一般非单位块、两列线性相关都适用。秩亏时相应三角分量为零，不需要人为给 \(U\) 添加扰动。</li></ol>
</details>
<h3>手算：用非单位向量核对恒等式</h3>
<p>把两块都缩成二维，取 \(b_i=(2,1),b_j=(1,1)\)，\(a_i=(1,2),a_j=(1,1)\)。则 \(d_b=(1,0),d_a=(0,1)\)，两项平方范数都是 1。外积差为</p>
<div class="equation">\[
\begin{pmatrix}2&4\\1&2\end{pmatrix}
-\begin{pmatrix}1&1\\1&1\end{pmatrix}
=\begin{pmatrix}1&3\\0&1\end{pmatrix},
\quad D_{ij}=1+1+(1+9+1)=13.
\]</div>
<p>对应 \(U=\begin{pmatrix}1&1\\0&1\end{pmatrix}\)、\(V=\begin{pmatrix}1&2\\0&1\end{pmatrix}\)，乘积正是上述差矩阵。这个例子还可用一般核公式核对：自核为 35、8，互核为 15，因此 \(35+8-2\times15=13\)；把自核硬写为 3 则会出错。</p>
<h3>实际分块与精确零规则</h3>
<p>每个点对只处理两列 QR，最多以 256 个点对为一块；不是对一个巨大的交互矩阵做 QR。训练距离矩阵为 <code>N×N</code>，每个无序点对只计算一次，再写入对称位置。query 到 support 使用同一计算式。</p>
<p>这里减少计算不是把 480 个辅助坐标压成两个，也不是删去小奇异值。两行乘积仍完整覆盖外积差的全部能量，只因这个“两个外积之差”的秩至多为 2，才可先换一组正交基再求范数。核方法需要的是该范数，不需要外积差的每个坐标。</p>
<p>例如 \(N\) 个训练点有 \(N(N-1)/2\) 个无序点对；分块只限制一次处理多少对，不改变哪些样本参与。配置中的 256 是点对块大小，不是 support 数量上限、近邻数量或维度截断。</p>
<p>若 \(b_i,b_j\) 与 \(a_i,a_j\) 逐元素完全相同，源码明确把距离设为精确 0。除此之外，不用经验阈值把“很近”合并成相同。如果非同一特征的三个平方和全部下溢为 0，抛出 <code>DISTANCE_UNDERFLOW_NONIDENTICAL_FEATURES</code>。没有相减后截成 0，也没有静默改度量。</p>
<div class="lab" data-lab="distance"></div>
<p class="warning">常见误区：非负平方和减少了消减误差，仍然受 float64 范围限制，不是无限精度。距离下溢失败与后面高斯相似度自然下溢为 0 是两件事。精确相同规则不能换成“距离小于某个 epsilon”，否则会改变零带宽等价类核。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:66</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:83</code>、<code>configs/d92_branch_local_ridge_frozen_20260929.json:18</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:87</code>。</p>`
},
{
  id: 'g-bandwidth', group: 'D92', kicker: '06 / 给距离确定单位',
  title: '带宽：用训练 support 的最近异类间距定义尺度',
  lead: '距离为 1 到底算近还是远，需要一个参照。该版本预先固定“最近异类平方距离的中位数”，不扫描带宽，也不查看 query。',
  html: String.raw`
<h3>为什么距离还要除以一个尺度</h3>
<p>同一个数值距离，在紧密的 support 集合中可能很远，在分散的集合中却很近。高斯核需要尺度 \(\tau\) 来表达相对远近。这里把尺度锚在“每个训练样本离最近竞争类别有多远”，以当前训练任务的类间竞争定义单位。</p>
<p>直觉上，先给每个 support 找一个最近的异类对手，再用所有人的典型对手距离作尺子。它不估计每个类别自己的方差，所以即使每类只有一个物理样本，也能在至少两个注册类时定义。</p>
<h3>符号与固定计算顺序</h3>
<div class="equation">\[
N=CK,\qquad
\delta_i=\min_{j:\,y_j\ne y_i}D_{ij},\qquad
\tau=\operatorname{median}(\delta_1,\ldots,\delta_N).
\]</div>
<p>\(C\) 是注册类别数，\(K\) 是每类物理 support 数，\(y_i\) 是当前训练 support 的可见标签，\(D_{ij}\) 已经是完整交互<strong>平方距离</strong>。每个物理样本贡献一个 \(\delta_i\)，不按类别难度或 old/new 身份加权。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：带标签的最近距离如何成为一个标量</summary>
<ol class="steps"><li>先构造全部训练点之间的距离矩阵。对第 \(i\) 行，把所有同标签位置设为不参与最小值，相当于用 \(+\infty\) 遮住它们。</li><li>对剩余异类位置取最小值。不是寻找最近的任意样本，也不是只排除自身；整个同类集合都被排除。</li><li>将得到的 \(N\) 个 \(\delta_i\) 排序，奇数个取中间项，偶数个取中间两项的算术平均。所有零值保留，不能删除冲突标签的重复特征。</li><li>核定义 \(R=\exp(-D/\tau)\) 后，距离恰为 \(\tau\) 的相似度是 \(e^{-1}\)。这给出了可解释的参照，不表示每个数据点都恰好具有这个异类距离。</li></ol>
</details>
<h3>手算：四个 support 的最近异类距离</h3>
<p>把完整坐标缩成一维演示：A 类位于 0、1，B 类位于 3、5。四个最近异类平方距离依次为 \(9,4,4,16\)，排序为 \(4,4,9,16\)，于是 \(\tau=(4+9)/2=6.5\)。对距离 6.5，核相似度为约 0.3679；距离 4 时约为 0.5404。尺度来自既定规则，没有尝试多个数再挑结果最好的那个。</p>
<p>再看零值：A 类在 0、1，B 类也在 0、1。每个样本都有一个不同标签、坐标完全相同的对手，所以四个 \(\delta_i\) 都是 0，\(\tau=0\)，但整个集合仍有两个不同坐标。零值揭示标签与特征的冲突，不能通过删掉零距离把它掩盖。</p>
<h3>零带宽与单类的实际分支</h3>
<p>当 \(\tau=0\) 且完整特征不全相同，固定使用极限核 \(R_{ij}=\mathbf1[D_{ij}=0]\)。同一完整特征的样本构成一个等价类，不同特征之间相似度为 0。若只有一个注册类，根本没有异类对手，源码令 \(\tau=\mathrm{null}\)，走单类零分数路径；不会凭空设一个带宽。</p>
<p>实际 \(\delta\) 是长度为 \(N\) 的向量，\(\tau\) 只是一个全训练任务共用的标量。每个诊断折都只用该折训练 support 重新计算；持出样本和 query 不进入尺度估计。query 使用固定 \(\tau\)，不根据它预测成哪一类再换一把尺子。</p>
<div class="lab" data-lab="bandwidth"></div>
<p class="warning">常见误区：标签参与尺度定义，不等于使用 query 标签；这是带监督的 support 几何。它不是类内方差、交叉验证选参或逐 query 局部带宽。新增类别会同时改变竞争候选和样本集合，\(\tau\) 可能变化，但不能简单宣称全任务中位数必然随类数单调下降。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:113</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:99</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:126</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:49</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:77</code>。</p>`
},
{
  id: 'g-radial', group: 'D92', kicker: '07 / 从距离得到局部相似',
  title: '径向核：“局部”是距离衰减，不是只保留几个邻居',
  lead: '把交互距离送入指数函数，近点保持较强联系，远点贡献迅速变小。所有 support 仍保留，最终决策依然覆盖全部注册类。',
  html: String.raw`
<h3>为什么再加一层距离非线性</h3>
<p>交互映射已经包含两块特征的乘积，但它仍是一个固定有限维空间中的全局线性模型。LocalRidge 想检验另一种假设：两点很近时的关系，比很远时的关系更值得细分。把距离变成衰减相似度，便可在同一组固定特征上形成更灵活的边界。</p>
<p>直觉上，每个 support 周围有一团逐渐变淡的影响，而不是一条延伸到无穷远的直线。这里没有只取最近几个 support；所有样本都保存在状态中，核行也没有做 top-k 截断或按行归一化。</p>
<h3>平方距离与带宽的准确位置</h3>
<div class="equation">\[
R_{ij}=\exp\!\left(-\frac{D_{ij}}{\tau}\right),\quad \tau>0,
\qquad D_{ij}=\|\phi_i-\phi_j\|^2.
\]</div>
<p>\(D\) 已是平方距离，不再平方一次。若习惯写 \(\exp[-d^2/(2\sigma^2)]\)，这里只是 \(2\sigma^2=\tau\)；不能再往分母额外加一个 2。自相似度为 1，非对角项在 0 至 1 之间。</p>
<h3>手算：同一尺度下怎样衰减</h3>
<p>设 \(\tau=2\)。当平方距离为 0、2、8 时，相似度分别为 \(1,e^{-1}\approx0.3679,e^{-4}\approx0.0183\)。距离翻成原欧氏距离的 2 倍，平方距离会变成 4 倍，因此衰减并不是线性的。相似度不是类别概率；训练后的类别系数可以有正有负，最后分数也不是“附近正权重投票”的简单平均。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么高斯核对应一般的无限维空间</summary>
<p>先展开平方距离：</p>
<div class="equation">\[
R(x,y)=e^{-\|\phi_x\|^2/\tau}e^{-\|\phi_y\|^2/\tau}
e^{\,2\langle\phi_x,\phi_y\rangle/\tau}.
\]</div>
<p>再使用指数级数 \(\exp(u)=\sum_{m=0}^{\infty}u^m/m!\)。内积的 \(m\) 次幂等于 \(m\) 阶张量积的内积，因此可以定义</p>
<div class="equation">\[
\psi_m(x)=e^{-\|\phi_x\|^2/\tau}
\sqrt{\frac{(2/\tau)^m}{m!}}\ \phi_x^{\otimes m},
\qquad
R(x,y)=\sum_{m=0}^{\infty}\langle\psi_m(x),\psi_m(y)\rangle.
\]</div>
<p>零阶项是一个标量，一阶、二阶乃至任意更高阶项继续出现。于是有限的 123,616 维 \(\phi\) 是高斯核的输入，而不是高斯 RKHS 的最终维度；在一般连续输入域上，后者是无限维。固定 \(N\) 个训练样本只产生一个 <code>N×N</code> Gram 矩阵，其秩至多为 \(N\)，两种“维度”不要混淆。</p>
<p>这个展开还证明半正定性：对任意系数 \(v_i\)，\(\sum_{ij}v_iR_{ij}v_j=\|\sum_i v_i\psi(x_i)\|^2\ge0\)。无需真的存下无限个坐标；实现只计算距离和指数。</p>
</details>
<h3>实际矩阵与零带宽边界</h3>
<p>若 \(C=3,K=2\)，则 \(N=6\)，训练相似度是 <code>6×6</code>；单个 query 到全部 support 是长度 6 的核行。方法常驻保留背景与辅助数组，而不保存显式外积特征或无限维 \(\psi\)。它不会为每一个 query 重新拟合模型。</p>
<p>当 \(\tau\downarrow0\)，对任何严格正的距离，指数极限为 0；对精确零距离始终为 1。所以零带宽采用 \(R(x,y)=\mathbf1[D(x,y)=0]\)。这相当于同一完整特征的等价类核，仍然半正定。没有精确匹配的 query 原始核行全为零，随后中心化会使这类 query 得到相同的核行与分数；不能暗中增大带宽来挽救它。</p>
<div class="lab" data-lab="radial"></div>
<p class="warning">常见误区：窄核可能记忆 support，宽核可能接近常量；增加非线性并不保证泛化改善。极小正带宽使指数在 float64 中自然趋零是允许路径，和非同特征距离整体下溢为零的技术失败不同。本教程不据核的数学性质宣告实际 query 性能通过。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:99</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:106</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:232</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:61</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:75</code>。</p>`
},
{
  id: 'g-centering', group: 'D92', kicker: '08 / 把训练均值作为原点',
  title: '核中心化：在看不见的特征空间里减去同一个均值',
  lead: '不必显式展开高斯特征，也能完成“每个坐标减去训练均值”。关键是 query 使用训练时固定的参照，而不是参与更新这个参照。',
  html: String.raw`
<h3>为什么要换一个原点</h3>
<p>ridge 要拟合类别差异，不需要把所有样本共同拥有的常量部分当作判别方向。设高斯特征为 \(\psi_i\)，先减去训练均值 \(\bar\psi=N^{-1}\sum_i\psi_i\)。直觉上，把 support 云团的中心移到原点，再判断各点朝哪个方向偏离。</p>
<p>中心化并不重新训练编码器。它只改变当前 support 核的坐标原点，训练后便固定；query 只和这个固定均值比较。</p>
<h3>从减均值一步步得到双中心化</h3>
<div class="equation">\[
H=I_N-\frac1N\mathbf1\mathbf1^\top,\qquad
G_R=HRH.
\]</div>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么要减行均值、列均值，再加总均值</summary>
<ol class="steps"><li>展开 \(\langle\psi_i-\bar\psi,\psi_j-\bar\psi\rangle\)，得到 \(R_{ij}-\langle\psi_i,\bar\psi\rangle-\langle\bar\psi,\psi_j\rangle+\|\bar\psi\|^2\)。</li><li>其中 \(\langle\psi_i,\bar\psi\rangle=N^{-1}\sum_lR_{il}\) 是第 \(i\) 行均值；另一项是第 \(j\) 列均值；最后一项是全矩阵均值。</li><li>令 \(J=\mathbf1\mathbf1^\top/N\)，展开 \((I-J)R(I-J)=R-JR-RJ+JRJ\)，正好对应上述四项。由于 \(H\mathbf1=0\)，结果每行、每列之和均为 0。</li><li>单个 query 记 \(r_j=R(q,x_j)\)，固定训练列均值为 \(m_j=N^{-1}\sum_lR_{lj}\)，总均值为 \(m_\star\)。其中心化核行是 \(r_j^c=r_j-\bar r-m_j+m_\star\)，其中 \(\bar r=N^{-1}\sum_jr_j\)。</li></ol>
<p>这里 \(\bar r\) 是<strong>这一个 query 对全部 support</strong>的平均相似度，不是多个 query 的批均值。它是计算当前行所需的代数项，不会回写训练状态；改变其他 query 的数量、顺序或内容不应改变此行分数。</p>
</details>
<h3>手算：两点训练集与一个 query</h3>
<p>设 \(R=\begin{pmatrix}1&0.2\\0.2&1\end{pmatrix}\)。两列均值与总均值都是 0.6，所以 \(G_R=\begin{pmatrix}0.4&-0.4\\-0.4&0.4\end{pmatrix}\)。负的非对角项表示相对均值的反向关系，并不破坏半正定性。</p>
<p>若 query 原始核行是 \((0.8,0.3)\)，其自身行均值为 0.55，得到 \(r^c=(0.25,-0.25)\)。加进第二个 query 时，前者仍使用 0.55，而不会改用两个 query 的平均。</p>
<h3>源码为何先计算 expm1 和参考差分</h3>
<p>当距离很小时，直接计算 \(\exp(-D/\tau)-1\) 可能把小差值舍入掉。源码用 \(E_{ij}=\operatorname{expm1}(-D_{ij}/\tau)=R_{ij}-1\)。这里减的是<strong>全 1 矩阵</strong>，不是单位矩阵；由于 \(H\mathbf1=0\)，有 \(HEH=HRH\)。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：参考差分为什么不改变中心化结果</summary>
<p>以第 0 个训练 support 为参照，先计算 \(F_{ij}=E_{ij}-E_{i0}-E_{0j}+E_{00}\)。减去的项分别只随行、只随列变化，以及一个公共常数；左右再乘 \(H\) 时它们全被消掉，所以 \(HFH=HEH\)。再对 \(F\) 做普通双中心化，即得到同一个 \(G_R\)。这种先差分的写法还使完全相同的特征行直接得到精确零差。</p>
<p>query 同样先算 \(F_{qj}=e_{qj}-e_{q0}-E_{0j}+E_{00}\)，再减去本行均值和固定训练 \(F\) 列均值，加回固定总均值。参照行、参照自核、训练均值都在拟合时保存，不使用 query 更新。</p>
</details>
<p>实际代码保存长度 \(N\) 的参考核行、中心化均值和几个标量，而不显式构造 \(H\)。local score 最后由中心化核行、固定倍率与系数矩阵得到，没有额外拟合一个 learned bias。若换回未中心化坐标，平移修正由训练均值和已有系数决定，并不是另开一个分类头。</p>
<div class="lab" data-lab="centering"></div>
<p class="warning">常见误区：负核元素不等于负特征范数；中心化后的对角也不再都为 1。用 query 批均值替换 support 均值会改变方法与独立推理性质。这里的均值是当前训练 support 的统计，不是读取地面 source 统计。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_interaction.py:95</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:148</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:218</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:232</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:98</code>。</p>`
},
{
  id: 'g-trace', group: 'D92', kicker: '09 / 对齐总能量再比较',
  title: 'Trace 匹配：固定总尺度，保留径向核真正改变的几何',
  lead: '核整体放大，会在 ridge 系数不变时减弱相对正则。该版本用训练中心化 trace 对齐尺度，让比较更集中在几何变化。',
  html: String.raw`
<h3>为什么核的倍率也会改变分类器</h3>
<p>固定 ridge 系数 \(\lambda=1\) 并不意味着核的数值尺度无关。在核的某个特征方向上，训练响应的收缩因子是 \(\mu/(\mu+1)\)，\(\mu\) 为中心化 Gram 的特征值。把核乘 \(c\) 后变成 \(c\mu/(c\mu+1)=\mu/(\mu+1/c)\)，相当于改变相对正则强度。</p>
<p>直觉上，先用同一份总能量作比较，避免“声音更响”被误读为“结构更好”。尺度只由训练特征确定，不用 held 或 query 选择倍率。</p>
<h3>两个 trace 与唯一的倍率</h3>
<div class="equation">\[
G_0=HK_0H,\quad G_R=HRH,\quad
s_0=\operatorname{tr}(G_0),\quad s_R=\operatorname{tr}(G_R),\quad
\gamma=\frac{s_0}{s_R},\quad G=\gamma G_R.
\]</div>
<p>非退化时，\(\operatorname{tr}(G)=s_0\)。\(\gamma\) 不是与旧核混合的权重，也不是交叉验证挑选的超参数；它把当前径向核的中心化总能量换算到原 interaction 的总能量。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：trace 为什么能只用成对距离计算</summary>
<p>对一般有限特征，不需要单位范数：</p>
<div class="equation">\[
s_0=\sum_i\|\phi_i-\bar\phi\|^2
=\sum_i\|\phi_i\|^2-\frac1N\left\|\sum_i\phi_i\right\|^2.
\]</div>
<p>展开全部无序点对的距离，得到 \(\sum_{i\lt j}D_{ij}=N\sum_i\|\phi_i\|^2-\|\sum_i\phi_i\|^2\)。所以</p>
<div class="equation">\[
s_0=\frac1N\sum_{i\lt j}D_{ij},\qquad
s_R=N-\frac1N\sum_{ij}R_{ij}
=\frac2N\sum_{i\lt j}(1-R_{ij}).
\]</div>
<p>第二式使用了高斯／等价类核的自核 \(R_{ii}=1\)。源码因此用非负距离求和计算 \(s_0\)，用 \(-\operatorname{expm1}(-D/\tau)\) 的正项求和计算 \(s_R\)，减少近常量核的消减。</p>
</details>
<h3>手算：两个 support 的能量对齐</h3>
<p>设两个 support 的完整平方距离为 4，\(\tau=4\)，则 \(R_{12}=e^{-1}\)。有 \(s_0=4/2=2\)、\(s_R=1-e^{-1}\approx0.63212\)，所以 \(\gamma\approx3.16395\)。中心化径向矩阵原为 \(\frac{1-e^{-1}}2\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\)，缩放后成为 \(\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\)，trace 正好为 2。</p>
<p>两点训练集中心化后只有一个非零方向，同 trace 会固定该训练 Gram 的尺度。但这不说明未见 query 的核延伸也相同；相同训练矩阵不等于所有输入上的预测函数相同。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：一阶径向展开为何恰好回到 interaction</summary>
<p>当 \(\tau>0\)，仅作代数对照，把指数替换成一阶式 \(R_{\mathrm{lin}}=\mathbf1\mathbf1^\top-D/\tau\)。距离矩阵满足 \(D=d\mathbf1^\top+\mathbf1d^\top-2K_0\)，其中 \(d_i=K_{0,ii}\)。左右乘 \(H\)，与常量方向有关的项消失，因此 \(HR_{\mathrm{lin}}H=2G_0/\tau\)。</p>
<p>此时对应 trace 为 \(2s_0/\tau\)，匹配倍率为 \(\tau/2\)，最终恰好得到 \(G_0\)。所以完整指数相对于原 interaction 的差异来自更高阶的径向几何，不只是整体能量放大。这个推导不是新增候选；零带宽边界也不能套用它。</p>
</details>
<h3>相同 trace 仍然可能很不一样</h3>
<p>例如两个中心化 PSD 矩阵的非零方向特征值分别为 \((2,0)\) 与 \((1,1)\)，trace 都是 2。前者 ridge 的有效自由度为 \(2/3\)，后者为 \(1/2+1/2=1\)。总能量相同，能量分配、收缩和边界仍然不同，不能从 trace 相同推出性能相同或一定更好。</p>
<p>实际 \(G\) 是 <code>N×N</code>，回归使用 \(G+I_N\)，每个物理样本权重为 1，损失按物理样本求和而非取均值。每个折只用自己的训练 support 计算 \(\tau,s_0,s_R,\gamma\)。全同特征使 \(s_0=0\) 时走精确零分类器；若非退化却出现 \(s_R\le0\) 或非有限倍率，则报数值失败，不加经验下限或另选方法。</p>
<div class="lab" data-lab="trace"></div>
<p class="warning">常见误区：trace 匹配不匹配全谱、每个对角、有效自由度或 query 范数，也不消除带宽选择本身的影响。它是既有 v1 的固定单位换算；本教程没有修改配置、启动实验，或据此宣告协议合规、科学假设成立与性能晋级。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:113</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:148</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:161</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:63</code>、<code>docs/D92_NEXT_AFTER_ORBIT_20260929.md:125</code>。</p>`
}
);
