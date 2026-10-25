window.COURSE_CHAPTERS = window.COURSE_CHAPTERS || [];
window.COURSE_CHAPTERS.push(
  {
    id: 'd-coarse', group: 'DURA', kicker: 'DURA · 01 / 先找一个合理起点',
    title: '时频粗恢复：先看频谱，再回到波形',
    lead: String.raw`粗恢复 \(c\) 给两阶段求解提供初值和共同参照。它融合当前观测的频域关系与时域局部结构，输出仍是完整的 256 点 IQ。`,
    html: String.raw`
<h3>为什么先做粗恢复？</h3>
<p>把接收 IQ 想成一段同时带有模糊、旋转和镜像干扰的录音。直接要求求解器从任意波形开始，有限的两次 CG 很难走到合适位置。粗恢复先给出一个较合理的草稿；后续阶段再检查草稿经接收作用后能否解释观测。草稿是网络对当前输入的判断，并不等于已经找到真实 clean。</p>
<p>一个窗口写作 \(y_n=I_n+\mathrm{i}Q_n\)，\(n=0,\ldots,255\)。存储时实部与虚部分开，所以输入形状是 <code>[B,2,256]</code>。\(B\) 是批量窗口数。DURA 从输入到输出始终保留这 256 个时间点；编码器中的下采样只改变中间特征。</p>
<h3>第一条路：让频谱及其共轭镜像一起说话</h3>
<div class="equation">\[
Y_k=\frac{1}{\sqrt{256}}\sum_{n=0}^{255}y_n\exp\!\left(-\mathrm{i}\frac{2\pi kn}{256}\right),\qquad
\widetilde Y_k=\overline{Y_{(-k\bmod256)}}.
\]</div>
<p>代码采用正交归一化 FFT。\(\widetilde Y\) 是“先按负频率取索引，再共轭”，不能省成同一索引的 \(\overline{Y_k}\)。它恰好是共轭波形 \(\overline y\) 的频谱。若 I/Q 不平衡让原信号与其共轭发生耦合，同时看 \(Y\) 与 \(\widetilde Y\) 就能提供有用线索。</p>
<p>可以直接核对这个关系：对 \(Y_{-k}\) 的求和式取共轭，指数的正号变回负号，样本则变成 \(\overline{y_n}\)，于是得到 \(\operatorname{FFT}(\overline y)_k\)。索引按 256 取模，零频点仍映射到自身；例如索引 1 的镜像来自索引 255。对于一般复数 IQ，不能套用“实信号频谱天然共轭对称”的结论。</p>
<p>频谱网络收到四个实数通道：\([\Re Y,\Im Y,\Re\widetilde Y,\Im\widetilde Y]\)，输出四张长度为 256 的实数参数图 \(u_0,u_1,u_2,u_3\)，再构造主系数 \(g\) 与镜像系数 \(b\)：</p>
<div class="equation">\[
g_k=e^{\log4\,\tanh u_{0,k}}e^{\mathrm{i}\pi\tanh u_{1,k}},\qquad
b_k=\tfrac14\bigl(\tanh u_{2,k}+\mathrm{i}\tanh u_{3,k}\bigr),\qquad
y_F=\operatorname{IFFT}(g\odot Y+b\odot\widetilde Y).
\]</div>
<p>因此主系数幅度被界定在 \([1/4,4]\)，相位在 \([-\pi,\pi]\)；镜像系数的实部、虚部分别在 \([-1/4,1/4]\)，其幅度上界是 \(\sqrt2/4\)。这些是系数的范围约束，不是对恢复 IQ 做幅度裁剪。它们也不适用于下一章作用估计器中的 \(m,q\)。</p>
<h3>第二条路：时域网络融合，残差形式输出</h3>
<div class="equation">\[c=y+T\bigl([y,y_F]\bigr),\qquad x_0=c.\]</div>
<p>时域网络输入原始 IQ 与频域修正 IQ，共四通道，输出两通道修正量，再加回 \(y\)。因此 \(y_F\) 是供融合的中间视图，最终粗恢复是 \(c\)。保留原观测的残差连接有利于学习小修正，但不能从连接形式推出 clean 永远无损。</p>
<p>频域中，一个频点综合整个时间窗口；时域卷积则从邻近采样之间的关系开始建模。把两条路合起来，是让同一输入接受两种组织方式的分析。FFT 本身是可逆变换，不会自动去噪；真正改变波形的是网络预测的系数和时域修正。网络也没有凭空获得第二次独立观测。</p>
<table class="mini-table"><thead><tr><th>编码层</th><th>特征长度</th><th>频谱 Core 通道</th><th>时域 Core 通道</th></tr></thead><tbody><tr><td>1</td><td>256</td><td>8</td><td>12</td></tr><tr><td>2</td><td>128</td><td>12</td><td>16</td></tr><tr><td>3</td><td>64</td><td>16</td><td>28</td></tr><tr><td>4</td><td>32</td><td>28</td><td>40</td></tr></tbody></table>
<p>每层有两个残差 Block；第 2 层起先用长度为 2 的平均池化。解码器依次线性插值到 64、128、256 点，与对应编码特征拼接，再经两个 Block。最后的 \(1\times1\) 卷积产生输出。跳跃连接让细节不必全挤进最短的 32 点表示。</p>
<p>例如时域第一段解码把最深的 40 通道插值到 64 点，再拼接编码器的 28 通道，得到 68 通道输入，解码成 28 通道。随后分别形成 44 通道、28 通道的拼接输入，输出 16 通道、12 通道。最终从 12 通道映射回 \(I,Q\) 两通道。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：一个残差 Block 到底计算什么？</summary>
<p>设 \(\phi(t)=t\sigma(t)\) 为 SiLU，\(\operatorname{GN}_4\) 为四组 GroupNorm。一个 Block 的主分支和输出为：</p>
<div class="equation">\[
F(z)=\phi\!\left(\operatorname{GN}_4\!\left(C_3\!\left[\phi\!\left(\operatorname{GN}_4(C_5z)\right)\right]\right)\right),\qquad
\mathcal B(z)=\frac{F(z)+S(z)}{\sqrt2}.
\]</div>
<p>\(C_5,C_3\) 分别是核长 5、3 的一维卷积；通道相同时 \(S\) 是恒等映射，否则是 \(1\times1\) 卷积。卷积两侧补零以保留当前特征长度。四组归一化解释了各层宽度为何保持为 4 的倍数。除以 \(\sqrt2\) 是残差相加的固定缩放，并不保证特征能量严格不变。</p></details>
<h3>手算一个频点，再算一个时间点</h3>
<p>取某频点 \(Y=1\)、\(\widetilde Y=0.4\mathrm{i}\)、\(g=0.5\mathrm{i}\)、\(b=0.25\)，修正频谱为 \(0.5\mathrm{i}+0.1\mathrm{i}=0.6\mathrm{i}\)。所有频点一起 IFFT 后才得到 \(y_F\)。再假设某时间点 \(y=0.3+0.1\mathrm{i}\)，时域网络输出 \(0.02-0.03\mathrm{i}\)，则 \(c=0.32+0.07\mathrm{i}\)。这两步的网络不能混成一个逐频点逆滤波器。</p>
<div class="lab" data-lab="coarse"></div>
<p class="warning">常见误区：频谱更平滑、均方误差更小，均不能单独证明 TX 身份细节得到保留。粗恢复也不是唯一的物理逆解；它只是后续学习先验和一致性求解共同使用的起点。</p>
<p class="source">依据：<code>experiments/pamf_reception_restore/models.py:13</code>、<code>experiments/pamf_reception_restore/models.py:18</code>、<code>experiments/pamf_reception_restore/models.py:80</code>、<code>experiments/pamf_restoration/models.py:29</code>；固定宽度见 <code>experiments/pamf_r46_delayed_prune/design.py:14</code> 与 <code>experiments/pamf_r46_delayed_prune/models.py:18</code>。</p>`
  },
  {
    id: 'd-action', group: 'DURA', kicker: 'DURA · 02 / 学会解释接收过程',
    title: '接收作用 A：实线性、边界与真正的伴随',
    lead: String.raw`作用估计器根据当前 \(y\) 与 \(c\) 建立可计算的接收近似。伴随把观测误差送回波形空间；伴随不是逆运算。`,
    html: String.raw`
<h3>为什么恢复器还要学“怎样接收到它”？</h3>
<p>若只输出一段看起来干净的 IQ，网络可能删去微弱但重要的结构。DURA 还建立一个接收作用 \(A\)：把候选恢复 \(x\) 送入 \(A\)，检查能否重新解释当前观测 \(y\)。这类似修复照片后，再按估计的模糊过程生成一次模糊图，检查两者是否相容。这里的 \(A\) 是学得的近似，不是模拟器提供的真实参数。</p>
<h3>四通道输入产生三类作用参数和一个方差</h3>
<p>估计器把 \([y,c]\) 拼为 <code>[B,4,256]</code>。其 Core 宽度为 12、16、28、40，编码长度为 256、128、64、32。最深特征沿时间取平均，得到 <code>[B,40]</code>，经 129 维线性头输出 64 个实部、64 个虚部和一个方差参数。另一条全长解码路径输出 <code>[B,4,256]</code>，生成逐点复数 \(m,q\)。</p>
<div class="equation">\[
\widehat h=\delta_{24}+a+\mathrm{i}b,\quad h=\frac{\widehat h}{\max(\|\widehat h\|_2,10^{-6})},\quad
m=1+(c_0+\mathrm{i}c_1),\quad q=c_2+\mathrm{i}c_3.
\]</div>
<p>\(h\) 是每窗口 64 抽头复数 FIR；\(\delta_{24}\) 表示索引 24 处的初始脉冲。常规非退化输出归一化到单位核能量，极小范数时由分母保护数值。\(m,q\) 在每个时间点都可变化，且没有粗恢复频谱分支的 \(\tanh\) 边界。</p>
<div class="equation">\[
\log v=\log10^{-8}+(\log1-\log10^{-8})\,\sigma(s+0.351),\qquad v=e^{\log v}.
\]</div>
<p>\(v\) 是一个窗口一个正实数，范围被限定在 \([10^{-8},1]\)，计算时才广播为 <code>[B,256]</code>。它不是逐点噪声图，也不是分类置信度；它控制稍后一致性项的权重。</p>
<p>输入一次后，\(h,m,q,v\) 都只估计一次，两阶段共用。长度 64 的 \(h\) 描述跨时间的混合；长度 256 的 \(m,q\) 描述逐点主作用和镜像作用，两种能力并不重复。\(v\) 可吸收噪声与模型不匹配的程度，但正值和范围限制本身不能证明它等于真实噪声方差，或已经得到概率校准。</p>
<h3>反射 FIR 与共轭耦合</h3>
<div class="equation">\[A(x)=m\odot H_h(x)+q\odot\overline{H_h(x)}.\]</div>
<p>\(H_h\) 先在左边反射扩展 63 点，索引为 \([63,62,\ldots,1,0,1,\ldots,255]\)，扩展长度 319；与长度 64 的核做线性卷积需容纳 382 点，实际用 512 点 FFT。最后取卷积输出索引 63 至 318，共 256 点。这里不能把边界默认为周期环绕。</p>
<p>镜像项让 \(A\) 对实数系数满足线性，却通常不满足复线性。例如令 \(H=I,m=1,q=0.2\)：\(A(1)=1.2\)，而 \(A(\mathrm{i})=0.8\mathrm{i}\ne\mathrm{i}A(1)\)。把 IQ 拆开，它始终可写成一个作用于 512 个实数的实矩阵。实线性正是后面采用实内积 CG 的原因。</p>
<p>若暂时只看一个点，令 \(m=a+\mathrm{i}b,q=c+\mathrm{i}d,z=u+\mathrm{i}w\)，展开乘法便能看清这种实矩阵：</p>
<div class="equation">\[
\begin{bmatrix}\Re(mz+q\overline z)\\\Im(mz+q\overline z)\end{bmatrix}
=\begin{bmatrix}a+c&-b+d\\b+d&a-c\end{bmatrix}
\begin{bmatrix}u\\w\end{bmatrix}.
\]</div>
<p>例如 \(m=1,q=0.2\) 对应对角线为 1.2、0.8 的实矩阵，分别放大 \(I\)、缩小 \(Q\)。对这个矩阵转置才是相应实内积下的伴随规则；一般不能把复杂作用当作一个复数增益后直接相除。加上 FIR 后，各时间点还会相互耦合。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么伴随中是 \(q\)，而不是 \(q\) 的共轭？</summary>
<p>定义 \(\langle u,r\rangle_{\mathbb R}=\Re\sum_n\overline{u_n}r_n\)。伴随必须满足 \(\langle Ax,r\rangle_{\mathbb R}=\langle x,A^*r\rangle_{\mathbb R}\)。暂记 \(z=H_hx\)。主项给出：</p>
<div class="equation">\[\Re\sum\overline{mz}\,r=\Re\sum\overline z\,(\overline m r).\]</div>
<p>镜像项利用“一个数与其共轭的实部相同”：</p>
<div class="equation">\[
\Re\sum\overline{q\overline z}\,r
=\Re\sum\overline q\,z\,r
=\Re\sum q\,\overline z\,\overline r
=\langle z,q\overline r\rangle_{\mathbb R}.
\]</div>
<p>再把 \(H_h\) 移到另一边，得到：</p>
<div class="equation">\[A^*r=H_h^*\bigl(\overline m\odot r+q\odot\overline r\bigr).\]</div>
<p>若 \(H_h=C F_h E\)，其中 \(E\) 为反射扩展、\(F_h\) 为卷积、\(C\) 为裁剪，则 \(H_h^*=E^*F_h^*C^*\)。\(C^*\) 把残差放回零数组对应区间；\(F_h^*\) 用共轭频率响应相关；\(E^*\) 把反射产生的重复索引累加回去，代码使用 <code>scatter_add</code>。只把 FIR 共轭后再正向过滤并不等价。</p></details>
<h3>手算边界的伴随</h3>
<p>缩成三点、左边反射一点：\(E[x_0,x_1,x_2]=[x_1,x_0,x_1,x_2]\)。因此 \(E^*[r_0,r_1,r_2,r_3]=[r_1,r_0+r_2,r_3]\)。两份 \(x_1\) 的贡献必须相加，不能随意丢掉一份。再看无 FIR 的小例子：\(A(1+0.5\mathrm{i})=1.2+0.4\mathrm{i}\)，主分量与镜像对 \(I,Q\) 的作用并不相同。</p>
<p>这个边界例子还能检查内积：左边 \(\langle Ex,r\rangle\) 中 \(x_1\) 出现两次，右边必须对应 \(r_0+r_2\)。反射是处理窗口外未知采样的一种固定计算约定，不意味着窗口外真实波形必然对称；作用模型的适用性仍要用合法证据检验。</p>
<p>检查伴随时，比较的应是两侧内积是否相等，而不是观察输出波形是否更像输入。</p>
<div class="lab" data-lab="action"></div>
<p class="warning">伴随负责正确传递误差，不负责直接还原输入。\(A^*A\) 通常不等于 \(I\)。从 \(y,c\) 学出的参数也不能被解释为唯一分离的真实传播信道、接收机响应和发射机失真；核归一化并没有证明这种可辨识性。</p>
<p class="source">依据：<code>experiments/pamf_action_restore/models.py:24</code>、<code>experiments/pamf_action_restore/operators.py:15</code>、<code>experiments/pamf_action_restore/operators.py:27</code>、<code>experiments/pamf_action_restore/operators.py:37</code>；固定轻量头见 <code>experiments/pamf_r46_delayed_prune/models.py:22</code>。</p>`
  },
  {
    id: 'd-prior', group: 'DURA', kicker: 'DURA · 03 / 提出下一份候选波形',
    title: '九通道学习先验：读懂当前状态与反投影误差',
    lead: String.raw`每阶段的 prior 网络依据当前窗口提出参考波形 \(p\)。它既看原观测，也看当前估计还解释不了的部分；两次 prior 之后各接两步 CG。`,
    html: String.raw`
<h3>为什么不能只靠观测一致性？</h3>
<p>有噪声或不可逆失真时，许多候选波形都能近似解释同一观测。仅让 \(Ax\) 贴近 \(y\)，可能把噪声也拟合进去。学习先验利用合法源域配对 IQ 学到的结构，提出“在这些线索下，恢复波形大概应该是什么样”。它提供参考，而下一步的软一致性求解保留对当前观测的约束。</p>
<p>DURA 初始化 \(x_0=c\)。第 \(k\) 阶段读取 \(x_{k-1}\)，先算接收空间的残差 \(y-Ax_{k-1}\)，再经伴随送回恢复空间：</p>
<div class="equation">\[
e_k=A^*(y-Ax_{k-1}),\qquad
\widehat e_k=\frac{\operatorname{pair}(e_k)}{\sqrt{\operatorname{mean}_{2\times256}\!\bigl(\operatorname{pair}(e_k)^2\bigr)+10^{-6}}}.
\]</div>
<p>\(\operatorname{pair}\) 把复数拆成实、虚两个通道。\(e_k\) 是反投影误差，和原接收残差不在同一种坐标中。正负方向也有意义：这里使用 \(y-Ax\)，其反方向会给出不同的网络输入。</p>
<h3>九个通道分别带来什么？</h3>
<table class="mini-table"><thead><tr><th>输入</th><th>实数通道</th><th>告诉 prior 的线索</th></tr></thead><tbody><tr><td>接收 IQ \(y\)</td><td>2</td><td>当前实际看到了什么</td></tr><tr><td>粗恢复 \(c\)</td><td>2</td><td>始终保留的共同参照</td></tr><tr><td>当前恢复 \(x_{k-1}\)</td><td>2</td><td>上一阶段已经走到哪里</td></tr><tr><td>归一化反投影 \(\widehat e_k\)</td><td>2</td><td>当前解释还缺少什么结构</td></tr><tr><td>方差特征 \(\ell_v\)</td><td>1</td><td>本窗口的一致性可靠程度线索</td></tr></tbody></table>
<div class="equation">\[
\ell_v=\frac{\operatorname{clip}(\log v,-16,4)}8,\qquad
p_k=c+\tau D_k\bigl([y,c,x_{k-1},\widehat e_k,\ell_v]\bigr),\quad
\tau=0.1169459182403555.
\]</div>
<p>\(\ell_v\) 沿时间复制成一张常数图，全部拼接后为 <code>[B,9,256]</code>。\(D_k\) 输出 <code>[B,2,256]</code>。注意残差基准是 \(c\)，不是 \(x_{k-1}\)；上一阶段状态通过输入影响 \(D_k\)。\(\tau\) 是固定尺度，\(\tau^2\) 还会作为求解中的先验方差，它不是 CG 自动选择的步长。</p>
<p>例如 \(v=10^{-4}\) 时，\(\ell_v=\log(10^{-4})/8\approx-1.1513\)；\(v=10^{-8}\) 时对数小于 -16，输入特征截为 -2。这个截断只用于 prior 的方差特征。后续求解仍使用原始正方差 \(v\)，不能把截断后的对数当作新的求解方差。</p>
<h3>轻量网络怎样看到不同时间尺度？</h3>
<p>每个 prior Core 的四层宽度为 12、16、28、40，对应长度 256、128、64、32；解码回 64、128、256 点并融合跳跃连接。与粗恢复一样，每层包含两个 Conv5/GN4/SiLU、Conv3/GN4/SiLU 残差 Block。短尺度帮助辨认局部变化，下采样特征扩大可综合的信息范围；这说明结构如何提供能力，并不证明网络一定学会了某种物理分解。</p>
<p>两阶段 prior 共享核长大于 1 的卷积<strong>权重</strong>，即 Conv5、Conv3 的权重张量；卷积偏置、GroupNorm 参数、\(1\times1\) 投影与输出等仍按阶段各自保留。共享的是部分参数，不代表两个 prior 是同一个函数。整个 DURA 的 266951 个参数按唯一张量统计，不能把别名共享权重重复相加。</p>
<p>一次推理的顺序是：先得到 \(c,A,v\)，再由 \(x_0=c\) 计算 \(e_1,p_1\)，两步 CG 得到 \(x_1\)；随后由 \(x_1\) 计算 \(e_2,p_2\)，再得到 \(x_2\)。改变的是当前窗口的中间张量，权重没有在线更新。共享参数也没有消掉第二次 prior 的计算，两个阶段仍各执行一次完整前向。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：归一化保留什么，又丢掉什么？</summary>
<p>先看只有两个实数分量的教学例子 \(e=(3,4)\)，暂略 \(10^{-6}\)。其 RMS 为 \(\sqrt{(9+16)/2}=\sqrt{12.5}\)，归一化约为 \((0.849,1.131)\)。若误差整体变成 \((6,8)\)，结果相同。因此归一化突出相对形状和方向，削弱整体幅度差异；它不是把每个时间点变成单位复数。</p>
<p>实际 RMS 在本窗口全部 512 个实数分量上计算，\(10^{-6}\) 防止接近零时除零。\(\ell_v\)、\(y\)、\(c\) 等仍提供额外尺度线索，但不能由此声称全部误差幅度信息都被完整保留。求解器复用的是未归一化 \(e_k\)，不能拿 \(\widehat e_k\) 代入下一章的正规方程。</p></details>
<h3>手算一个 prior 输出</h3>
<p>假设某点 \(c=0.6+0.2\mathrm{i}\)，当前 prior 网络的该点输出为 \(D_k=0.5-0.25\mathrm{i}\)，则 \(p_k\approx0.658473+0.170764\mathrm{i}\)。这只是候选参考；最终 \(x_k\) 还要经过两步 CG。第二阶段重新计算误差与 \(p_2\)，因此可能向不同方向修正，而不是机械重复第一次输出。</p>
<div class="lab" data-lab="prior"></div>
<p class="warning">这里的“先验”是依赖 \(y,c,x_{k-1}\) 的学习参考，不是已证明归一化的无条件概率分布，也不是分类类别模板。两阶段会改变 \(p_k\)，不能宣称整个网络在下降同一个全局目标。波形更平滑或平均谱更像 clean，也不能替代 TX 身份保真验证。</p>
<p class="source">依据：<code>experiments/pamf_r65_four_route/stage_polyphase.py:60</code>、<code>experiments/pamf_r65_four_route/stage_polyphase.py:84</code>、<code>experiments/pamf_r17_fidelity/models.py:9</code>、<code>experiments/pamf_stage2_inference/program.py:18</code>；轻量 Core 见 <code>experiments/pamf_r46_delayed_prune/models.py:24</code>。</p>`
  },
  {
    id: 'd-map', group: 'DURA', kicker: 'DURA · 04 / 把两种证据写进同一问题',
    title: '软一致性 MAP：从高斯直觉推到正规方程',
    lead: String.raw`固定某阶段的 \(A,v,p\) 后，求解器平衡“能够解释观测”和“不要过度偏离 prior”。这个局部二次问题有唯一解，实际只做两步近似求解。`,
    html: String.raw`
<h3>为什么是软约束？</h3>
<p>若强迫 \(Ax=y\)，就等于要求候选恢复连噪声与作用模型误差都解释掉。DURA 允许接收残差存在，同时对偏离 prior 的程度收费。这像两位提供线索的人：观测报告实际收到什么，prior 提供合理恢复的参考；两者的相对权重由方差尺度决定。</p>
<p>本章先固定当前阶段网络算出的 \(A,v,p\)。把复数 256 点 IQ 展开成 512 个实数，考虑条件化的高斯解释：\(y=Ax+n\)，\(n\sim\mathcal N(0,vI)\)；参考恢复 \(x\) 围绕 \(p\) 波动，尺度为 \(\tau^2I\)。这里 \(v\) 指每个实数分量的方差；复数模平方恰好是 \(I,Q\) 平方和。</p>
<p>“固定”指本次局部求解中的条件：CG 改变候选 \(x\) 时，不重新调用作用估计器，也不重新生成当前的 \(p\)。若边走 CG 边随意改变这些条件，下面的梯度与正定系统就不再对应同一个问题。训练时可以通过展开的运算反向传播，和部署时在一个局部问题内保持条件固定，是两个不同层次。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：概率最大为什么变成误差最小？</summary>
<ol class="steps"><li>观测的条件密度具有形式 \(P(y\mid x,A,v)\propto\exp[-\|y-Ax\|^2/(2v)]\)。</li><li>局部先验具有形式 \(P(x\mid p,\tau)\propto\exp[-\|x-p\|^2/(2\tau^2)]\)。</li><li>最大化两者乘积，等价于最小化负对数。固定的归一化项与 \(x\) 无关，可以去掉。</li><li>再乘以正数 2 不改变极小点，得到代码对应的目标。</li></ol>
<div class="equation">\[
x^\star=\arg\min_x J(x),\qquad
J(x)=v^{-1}\|y-Ax\|_2^2+\tau^{-2}\|x-p\|_2^2.
\]</div>
<p>这是给固定阶段二次求解建立的 MAP（最大后验估计）解释。实际 \(A,v,p\) 都由输入及中间网络状态产生，特别是 \(p\) 已依赖 \(y\)。上面的局部解释不证明整个 DURA 是一个归一化、全局一致的概率生成模型，也不能用它推导网络输出具有校准后的后验置信度。</p></details>
<h3>从目标到正规方程</h3>
<p>取任意实方向 \(d\)，让 \(x\) 变为 \(x+td\)。实线性保证 \(A(x+td)=Ax+tAd\)。对实数 \(t\) 求导：</p>
<div class="equation">\[
\left.\frac{dJ(x+td)}{dt}\right|_{t=0}
=2v^{-1}\langle Ax-y,Ad\rangle_{\mathbb R}
+2\tau^{-2}\langle x-p,d\rangle_{\mathbb R}
=2\langle v^{-1}A^*(Ax-y)+\tau^{-2}(x-p),d\rangle_{\mathbb R}.
\]</div>
<p>在极小点，这个表达式对所有 \(d\) 都为零，故得到：</p>
<div class="equation">\[
\underbrace{(v^{-1}A^*A+\tau^{-2}I)}_{M}x
=\underbrace{v^{-1}A^*y+\tau^{-2}p}_{b}.
\]</div>
<p>因为本版 \(v\) 是每窗口实数常量，可与 \(A^*\) 交换；若以后改为逐点方差，必须保留 \(A^*V^{-1}A\) 的运算顺序。代码不会显式建立或求逆一个 \(512\times512\) 矩阵，只定义“给定向量 \(d\)，怎样计算 \(Md\)”这套运算。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：\(A\) 不能求逆，为什么本问题仍有唯一解？</summary>
<p>对任意非零实向量 \(d\)，伴随关系给出：</p>
<div class="equation">\[\langle d,Md\rangle_{\mathbb R}=v^{-1}\|Ad\|^2+\tau^{-2}\|d\|^2\gt0.\]</div>
<p>因此 \(M\) 在 512 维实空间中对称正定。即使某些方向被 \(A\) 完全抹去，第二项仍为正，固定阶段的问题具有唯一极小点。唯一的是给定 \(A,v,p\) 的正则化解，不是“真实原始信号已经可辨识”。把变量改成 \(x=p+\delta\)，还可得到 CG 使用的偏移方程：</p>
<div class="equation">\[M\delta=v^{-1}A^*(y-Ap),\qquad x=p+\delta.\]</div></details>
<h3>手算：方差如何改变信任？</h3>
<p>先缩成实数 \(A(x)=ax\)，其中 \(a\) 是固定实数。对目标求导，得到：</p>
<div class="equation">\[x^\star=\frac{ay/v+p/\tau^2}{a^2/v+1/\tau^2}.\]</div>
<p>取 \(a=1,y=2,p=0,\tau^2=1\)：当 \(v=0.25\) 时 \(x^\star=1.6\)；当 \(v=1\) 时为 \(1\)。观测方差越大，一致性收费越轻，结果越靠近 prior。这里为便于手算使用 \(\tau^2=1\)，实际版本的固定 \(\tau\) 约为 0.116946。</p>
<p>再取 \(a=2,v=1\)，其余不变，则 \(x^\star=4/5\)。不能把它说成直接对 \(y=2\) 和 \(p=0\) 做平均，因为 \(y\) 属于接收空间；当 \(a\ne0\) 时，才可改写成对 \(y/a\) 与 \(p\) 的加权平均。\(a=0\) 时观测完全不约束 \(x\)，解就是 \(p\)。</p>
<p>具体地，\(a\ne0\) 时设 \(\lambda=(a^2/v)/(a^2/v+1/\tau^2)\)，便有 \(x^\star=\lambda(y/a)+(1-\lambda)p\)。增大 \(v\) 让 \(\lambda\) 变小；增大先验方差 \(\tau^2\) 则让先验牵引变弱。这两个方差含义相反，不能只看到“方差更大”便断言更信哪一项。</p>
<p>如果同时把两个方差都乘同一个正数，整个目标只缩放，精确极小点不变；真正决定平衡的是相对尺度。实际固定版本只让估计器输出 \(v\)，并固定 \(\tau\)。多维情况下还存在各方向不同的作用增益，不能把标量平均公式逐点套到带卷积、镜像和边界的 \(A\) 上。</p>
<div class="lab" data-lab="map"></div>
<p class="warning">本章的唯一解和正定性针对固定阶段。实际每阶段只执行两步 CG，未必达到该极小点；第二阶段还会重新生成 \(p_2\)。因此不能据此声称两阶段全局收敛，或把 \(A^*y\) 当作已经恢复的 IQ。</p>
<p class="source">依据：<code>experiments/pamf_uncertainty_restore/models.py:11</code>、<code>experiments/pamf_action_restore/operators.py:42</code>、<code>experiments/pamf_r65_four_route/stage_polyphase.py:73</code>。高斯与方向导数部分是对该固定二次目标的数学推导。</p>`
  },
  {
    id: 'd-cg', group: 'DURA', kicker: 'DURA · 05 / 用固定计算预算解方程',
    title: '两步 CG：共轭方向、热启动与残差复用',
    lead: 'CG 用矩阵作用计算替代显式求逆。DURA 每阶段只走两步，继承当前波形作为热启动；残差复用只省重复运算。',
    html: String.raw`
<h3>为什么不直接求逆？</h3>
<p>上一章得到 \(Mx=b\)，其中 \(M\) 对称正定。显式建立再求逆既耗存储，又忽略了 \(A\) 可以快速计算的结构。共轭梯度法（CG）只需要反复计算 \(Md=v^{-1}A^*(Ad)+\tau^{-2}d\)。它在一系列互不重复的曲率方向上修正答案，适合这种“知道算子怎样作用，却不建立完整矩阵”的问题。</p>
<p>可以把二次型想成一个倾斜的碗。普通梯度每次只沿当前最陡方向走，狭长的碗中容易来回摆动；CG 利用已经走过方向的曲率信息安排下一方向。这是解释计算安排的直觉，不是说每一步都能恢复某一段频率，也不是把 CG 的方向等同于波形的物理传播方向。</p>
<p>先在 512 维实空间中写二次型 \(Q(x)=\tfrac12\langle x,Mx\rangle-\langle b,x\rangle\)。梯度为 \(Mx-b\)，定义方程残差 \(r=b-Mx\)，因此 \(-\nabla Q=r\)。CG 的“残差”在本章始终指方程尚未满足的量。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：步长 \(\alpha\) 和共轭系数 \(\beta\) 从哪里来？</summary>
<p>沿搜索方向 \(d_j\) 更新 \(x_{j+1}=x_j+\alpha_jd_j\)。精确线搜索令 \(Q(x_j+\alpha d_j)\) 对 \(\alpha\) 的导数为零：</p>
<div class="equation">\[
0=-\langle r_j,d_j\rangle+\alpha_j\langle d_j,Md_j\rangle,
\qquad \alpha_j=\frac{\langle r_j,r_j\rangle}{\langle d_j,Md_j\rangle}.
\]</div>
<p>第二个等号用到 CG 的残差正交关系：\(d_j\) 等于当前残差加旧方向的组合，而当前残差与旧方向正交，所以 \(\langle r_j,d_j\rangle=\|r_j\|^2\)。更新后 \(r_{j+1}=r_j-\alpha_jMd_j\)。</p>
<p>下一方向设为 \(d_{j+1}=r_{j+1}+\beta_jd_j\)。要求它与旧方向在 \(M\) 内积下共轭，即 \(\langle d_{j+1},Md_j\rangle=0\)。由 \(Md_j=(r_j-r_{j+1})/\alpha_j\) 和相邻残差正交可得：</p>
<div class="equation">\[
\beta_j=-\frac{\langle r_{j+1},Md_j\rangle}{\langle d_j,Md_j\rangle}
=\frac{\|r_{j+1}\|^2}{\|r_j\|^2}.
\]</div>
<p>“共轭”不是复数共轭。这里指不同方向满足 \(\langle d_i,Md_j\rangle=0\)，让下一次修正避免重新破坏前面已处理的二次曲率方向。</p></details>
<h3>热启动必须体现在 \(\delta\) 的初值里</h3>
<p>DURA 实际求偏移量 \(\delta=x-p\)。本阶段已有当前波形 \(x_0=x_{k-1}\)，因此应设 \(\delta_0=x_0-p\)，而不是零。若设 \(\delta_0=0\)，对应的波形初值就是 \(p\)，不能同时声称从 \(x_{k-1}\) 热启动。</p>
<div class="equation">\[
t=v^{-1}A^*(y-Ap),\qquad
r_0=t-M\delta_0
=v^{-1}\underbrace{A^*(y-Ax_0)}_{e_k}-\tau^{-2}(x_0-p).
\]</div>
<p>这一步利用 \(A\) 的实线性，以及每窗口实数 \(v^{-1}\) 可移到伴随外面。prior 刚刚已经计算了未归一化的 \(e_k\)，所以可以直接复用。\(e_k\) 是观测误差的反投影；\(r_0\) 还包含先验牵引项，二者不能互换。</p>
<p>把原式完整展开便能核对：\(y-Ap-A(x_0-p)=y-Ax_0\)，所以两次正向项合并；伴随再按实线性合并。这个整理只复用了同一 \(x_0\) 对应的误差。如果先更新了 \(x\)，再使用更新前的 \(e_k\) 冒充新误差，就不再等价。归一化 prior 输入也不能代替这里带原始幅度的反投影。</p>
<ol class="steps"><li>令 \(d_0=r_0\)，计算 \(Md_0\)、\(\alpha_0\)，更新 \(\delta_1=\delta_0+\alpha_0d_0\)。</li><li>计算 \(r_1=r_0-\alpha_0Md_0\)、\(\beta_0=\|r_1\|^2/\|r_0\|^2\)、\(d_1=r_1+\beta_0d_0\)。</li><li>计算 \(Md_1\)、\(\alpha_1\)，更新 \(\delta_2\)，返回 \(x_k=p+\delta_2\)。最终答案已得到，无须再更新不会被读取的 \(r_2,d_2\)。</li></ol>
<h3>手算二维正定系统</h3>
<p>取 \(M=\begin{bmatrix}4&1\\1&3\end{bmatrix}\)、\(b=(1,2)^\mathsf T\)，从 \(x_0=(0,0)^\mathsf T\) 开始。这里直接演示 \(x\) 的 CG，等价于取 \(p=0\) 的偏移形式。</p>
<table class="mini-table"><thead><tr><th>量</th><th>第一步</th><th>第二步</th></tr></thead><tbody><tr><td>方向 \(d\)</td><td>\((1,2)\)</td><td>\((-7/16,3/8)\)</td></tr><tr><td>\(Md\)</td><td>\((6,7)\)</td><td>\((-11/8,11/16)\)</td></tr><tr><td>步长 \(\alpha\)</td><td>\(1/4\)</td><td>\(4/11\)</td></tr><tr><td>更新后 \(x\)</td><td>\((1/4,1/2)\)</td><td>\((1/11,7/11)\)</td></tr></tbody></table>
<p>第一步残差为 \((-1/2,1/4)\)，于是 \(\beta_0=(5/16)/5=1/16\)。第二步得到精确解；代回可验 \(Mx=(1,2)^\mathsf T\)。二维正定系统在精确算术下最多两步即可求解，但实际 DURA 有 512 个实维度，每阶段也只给两步，不能把这个二维结论照搬过去。</p>
<p>第一步的数字也可以逐项核对：\(\|r_0\|^2=1^2+2^2=5\)，\(d_0^\mathsf TMd_0=1\times6+2\times7=20\)，故 \(\alpha_0=5/20\)。注意如果把步长固定成 1，答案会直接跳到 \((1,2)\)，反而离解更远。CG 的步长由当前方向上的二次曲率决定，并非网络学出的常数。</p>
<p>实际程序以 <code>[B,256]</code> 复数形式保存向量，但点积取复数内积的实部，并仅沿本窗口的 256 点求和。于是 \(\alpha,\beta\) 为 <code>[B]</code>，再广播到各窗口；不同批量样本不会共用一个点积或搜索方向。若某窗口残差已经严格为零，数值保护使剩余更新保持为零；程序仍执行固定预算。</p>
<div class="lab" data-lab="cg"></div>
<p class="warning">固定两步意味着固定计算预算，不意味着已经收敛。代码给分母设置 \(10^{-20}\) 数值下界；FP32 舍入也会影响理论正交关系。残差复用整理的是同一阶段同一窗口的代数式，没有使用上一个样本的输出或历史缓存。</p>
<p class="source">依据：<code>experiments/pamf_uncertainty_restore/models.py:19</code>、<code>experiments/pamf_r65_inference/program.py:9</code>、<code>experiments/pamf_stage2_inference/program.py:18</code>。步长、共轭系数及二维例子为对实现公式的逐步推导。</p>`
  },
  {
    id: 'd-training', group: 'DURA', kicker: 'DURA · 06 / 明白权重怎样来，部署怎样跑',
    title: '训练、蒸馏与执行：同一个 DURA 的两种状态',
    lead: '两阶段学生继承三阶段父版，再用合法源域 IQ 联合训练。部署时只需要当前接收 IQ 与冻结权重，Graph 与残差复用负责减少执行开销。',
    html: String.raw`
<h3>为什么删掉一个阶段后还要训练？</h3>
<p>父版 R65 有三个 prior、六步 CG。直接删除前一阶段，会改变后续 prior 看到的当前状态。DURA 的两阶段学生继承父版的粗恢复与作用估计器，保留原第 2、第 3 个 prior 及其共享关系，然后让学生第 1 次输出学习教师第 2 次输出，学生第 2 次输出学习教师第 3 次输出。教师是冻结的原 R65，推理时不随学生运行。</p>
<p>保留组件全部联合优化，包含粗恢复、作用估计器和两个 prior。某个成员在源码中名为 <code>frozen</code>，不代表学生训练期间该成员不更新；应看实际梯度设置与调用。新增训练为 20 轮、4940 次更新。配置保留了组件已有的 100 个源域训练轮次祖先，不能把新增 20 轮称为从零训练总成本，也不能把各轮不同的稠密、压缩计算简单视为相同耗时。</p>
<h3>源域配对监督不等于 TX 标签监督</h3>
<p>训练读取合法源域 63000 个 IQ 窗口：6300 个来自 \(L_s\)，56700 个来自 \(U_s\)。恢复任务使用源 clean 与其固定模拟接收版本构成配对。\(U_s\) 的 TX 标签始终隐藏；重建目标是 IQ 数值，不是发射机类别。V 和 target 不参与更新，也不读取已训练 F 的权重、输出或 TX 真值来优化恢复器。</p>
<table class="mini-table"><thead><tr><th>项目</th><th>本版学生训练</th><th>冻结部署</th></tr></thead><tbody><tr><td>输入网络的内容</td><td>当前模拟接收 IQ</td><td>当前接收 IQ</td></tr><tr><td>配对 clean</td><td>仅用于计算训练损失</td><td>不需要</td></tr><tr><td>教师输出</td><td>冻结教师提供阶段 IQ 目标</td><td>不需要教师</td></tr><tr><td>TX 标签与 F</td><td>不读取标签，不访问 F 训练恢复器</td><td>G 输出 IQ；需要识别时再交给冻结 F</td></tr><tr><td>学生参数</td><td>全部保留组件联合更新</td><td>eval，梯度关闭，参数固定</td></tr></tbody></table>
<h3>三个训练目标分别约束什么？</h3>
<p>令配对 clean 为 \(s\)，学生两阶段输出为 \(x_1,x_2\)，教师三个输出为 \(t_1,t_2,t_3\)。以每窗口真实、虚部均方能量 \(E=\max(\operatorname{mean}(s^2),10^{-4})\) 归一化：</p>
<div class="equation">\[
L_{\rm IQ}=\frac{\operatorname{mean}[(x_2-s)^2]}{E},\quad
L_{\rm KD}=\frac12\sum_{k=1}^2\frac{\operatorname{mean}[(x_k-t_{k+1})^2]}{E},\quad
L=L_{\rm IQ}+0.05L_{\rm spec}+\lambda_eL_{\rm KD},\quad
\lambda_e=1-0.9\frac{e-1}{19}.
\]</div>
<p>这里先逐分量平方，再在该窗口的 \(2\times256\) 分量上平均。\(e=1,\ldots,20\) 是训练轮次。最终配对误差直接约束答案；阶段蒸馏让缩短后的轨迹接近父版后两阶段；蒸馏权重从 1 线性降至 0.1，后期仍保留教师约束。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：多尺度谱误差具体比较什么？</summary>
<p>对 \(m\in\{64,128,256\}\)，以周期 Hann 窗、步长 \(m/2\) 切帧，计算正交 FFT 的功率谱 \(P_m(z)=|\operatorname{FFT}(\operatorname{frame}_m(z)\odot w_m)|^2\)。这三种窗长在 256 点信号上分别产生 7、3、1 帧。</p>
<div class="equation">\[
a_m=\max\!\left(\operatorname{mean}P_m(s),10^{-4}\right),\qquad
L_{\rm spec}=\frac13\sum_m\operatorname{mean}\!\left(\left[\log\!\left(1+\frac{P_m(x_2)}{a_m}\right)-\log\!\left(1+\frac{P_m(s)}{a_m}\right)\right]^2\right).
\]</div>
<p>归一化尺度来自配对 clean 的平均功率，\(\log(1+\cdot)\) 压缩大功率差异；误差仍逐帧、逐频点计算后取均值，并非只比较一条平均频谱。谱功率不包含完整相位，所以它只能补充 IQ 损失。教师和 clean 都停止梯度，优化器只更新学生；训练采用重置的 AdamW。</p></details>
<h3>手算损失，再看执行预算</h3>
<p>若一个窗口 \(L_{\rm IQ}=0.02,L_{\rm spec}=0.10,L_{\rm KD}=0.03\)，第 1 轮总损失为 \(0.02+0.005+0.03=0.055\)，第 20 轮为 \(0.02+0.005+0.003=0.028\)。这展示权重日程，不表示模型误差真的按此下降。网络对全部训练窗口取平均后反向传播。</p>
<p>冻结 DURA v1.0 含 266951 个唯一参数；每窗口一次粗恢复、一次 \(A,v\) 估计、两次 prior、四次 CG。执行入口固定为 <code>InferenceEngine(reuse=True, graph=True, native=False)</code>，使用 FP32、关闭 TF32，没有混合精度或量化替代。</p>
<p>残差复用沿用上一章的等价式，并省掉最终不会再读取的残差和方向更新，使同一双阶段程序的 FFT/IFFT 调用由 43 次降到 27 次。CUDA Graph 则按设备和输入形状捕获运算结构，每次复制<strong>当前输入</strong>后重放，并克隆独立输出；它缓存执行结构，不缓存历史预测。有限性与正方差检查在设备内汇总，末尾仍检查异常。执行优化和版本固定没有新增训练更新。</p>
<p class="warning">F-free 指本恢复训练没有访问 F，不等于网络没有父版训练经历。低 IQ 误差、谱误差或蒸馏误差也不能证明每个 TX 细节都恢复。DURA 既有正式识别证据对应冻结的原 F50 配对；不能把这些结果自动改写成与 PAMF-T1-E50、PAMF-REPL、其他识别器或其他数据契约的组合结果，更不能据此宣称 Phase2 有效。</p>
<p class="source">依据：<code>experiments/pamf_r65_four_route/stage_polyphase.py:84</code>、<code>experiments/pamf_r65_four_route/training.py:15</code>、<code>experiments/pamf_r65_four_route/training.py:43</code>、<code>experiments/pamf_r65_four_route/design.py:36</code>、<code>experiments/pamf_r44_pruned_refine/models.py:46</code>、<code>experiments/pamf_r46_multipoint_fast/design.py:26</code>；执行与版本见 <code>experiments/pamf_r46_cuda_graph/graph.py:67</code>、<code>experiments/pamf_stage2_inference/engine.py:28</code>、<code>experiments/pamf_stage2_inference/frozen_version.json</code>。</p>`
  }
);
