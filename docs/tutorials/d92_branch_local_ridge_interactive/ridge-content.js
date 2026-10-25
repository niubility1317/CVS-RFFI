window.COURSE_CHAPTERS = window.COURSE_CHAPTERS || [];
window.COURSE_CHAPTERS.push(
  {
    id: 'r-ridge', group: '求解与判决', kicker: '10 / 从目标函数到解析解',
    title: '核岭回归：没有优化器，也在学习分类规则',
    lead: '每个物理 support 的标签都进入平方误差目标。冻结版使用一次 float64 Cholesky 求解，得到依赖当前 support 的分类头。',
    html: String.raw`
<h3>先区分原始交互特征与径向核特征</h3>
<p>设注册类别数为 \(C\)，每类有 \(K\) 个物理 support，总数 \(N=CK\)。背景块 \(b_i\in\mathbb R^{256}\)、辅助块 \(a_i\in\mathbb R^{480}\) 构成交互表示 \(\phi_i=[b_i,a_i,\operatorname{vec}(b_i a_i^\mathsf T)]\)。其维度为 123616，但实现不显式生成这个长向量。</p>
<p>本章真正用于回归的是径向核对应的特征。设径向核矩阵为 \(R\)，support 中心化矩阵为 \(H=I-\mathbf1\mathbf1^\mathsf T/N\)，trace 比例为 \(\gamma\)，最终 Gram 矩阵为 \(G=\gamma HRH\)。把其隐式、缩放并中心化后的特征记为 \(\psi_i\)，于是 \(G_{ij}=\langle\psi_i,\psi_j\rangle\)。不要把 \(\psi\) 与原始的 \(\phi\) 混用；径向核的特征空间一般并不是 123616 维。</p>
<h3>物理求和、标签中心化与特征中心化</h3>
<div class="equation">\[
(Y_c)_{ij}=\mathbf1[y_i=j]-\frac1C,\qquad
\mathcal J(W)=\frac12\sum_{i=1}^{N}\|\psi_i^\mathsf T W-(Y_c)_i\|_2^2
+\frac12\|W\|_{\mathcal H,F}^2.
\]</div>
<p>\(Y_c\in\mathbb R^{N\times C}\) 是中心化 one-hot 目标；\(\mathcal H\) 是核对应的再生核希尔伯特空间（RKHS）。每个物理样本权重为 1，正则系数固定为 \(\lambda=1\)。平衡的每类 \(K\) 使目标每列均值恰为零，而特征中心化把几何参考点移到 support 均值。二者做的事情不同，不能互相替代。</p>
<p>这里用求和而非均值。若把第一项除以 \(N\)，却仍保留同一个正则系数，乘回 \(N\) 后就相当于把正则强度改成 \(N\)，已经改变冻结方法。平方误差比较的是分数与目标，分数不是概率；没有交叉熵训练，也没有在此搜索温度。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：从原始正规方程到核系数</summary>
<p>把训练特征按行组成 \(\Psi\)。与这些行张成空间正交的权重分量不会改变任何训练预测，却会增加正则范数，所以最优权重位于 support 特征张成的空间，可写成 \(W=\Psi^\mathsf T\alpha\)。原始变量的梯度为零给出：</p>
<div class="equation">\[
(\Psi^\mathsf T\Psi+I)W=\Psi^\mathsf T Y_c,\qquad
(\Psi^\mathsf T\Psi+I)\Psi^\mathsf T=\Psi^\mathsf T(G+I).
\]</div>
<p>因此选择 \(\alpha=(G+I)^{-1}Y_c\) 后，\(W=\Psi^\mathsf T\alpha\) 满足原始方程。原始目标的正则项保证最优函数唯一；\(G+I\) 对称正定，因而这个系数方程也有唯一解。</p>
<p>若直接对系数目标求导，只得到 \(G[(G+I)\alpha-Y_c]=0\)。当 \(G\) 奇异时，不能约去 \(G\) 来证明系数表示唯一：给 \(\alpha\) 加上 \(\ker G\) 中的向量，不改变对应函数。代码选择上面的正规化系数表示，并非证明所有表示都相同。全同特征退化时代码直接保存零系数，仍表示同一个零函数。</p></details>
<h3>Cholesky 为什么只需要两个三角求解？</h3>
<div class="equation">\[
M=G+I=LL^\mathsf T,\qquad LU=Y_c,\qquad L^\mathsf T\alpha=U.
\]</div>
<p>\(M,L\) 为 \(N\times N\)，\(U,\alpha\) 为 \(N\times C\)。下三角求解从前往后代入，上三角求解从后往前代入。实现使用 float64，不添加 jitter，不用显式求逆。给对角线额外加一个数就会改变正则，不能当作无影响的实现细节。</p>
<p>中心化使 \(G\mathbf1=0\)，所以本版的 Gram 本来就可能奇异，不需要把奇异误认为失败。对任意非零向量 \(z\)，有 \(z^\mathsf T(G+I)z=z^\mathsf TGz+\|z\|^2>0\)，加单位矩阵后仍严格正定。由 \(\operatorname{tr}G=s_0\) 还可得到条件数上界 \(1+s_0\)；这是数学上界，不是实测条件数。</p>
<p>同一个 Cholesky 因子同时服务全部 \(C\) 列目标，不需要每个类别重新分解一次。更换 support、注册类别或标签，会改变几何或目标；即使学习率与 epoch 都不存在，分类规则仍随当前有标签数据而变化。</p>
<p>手算取 \(G=\tfrac12\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix}\)，两个类别的 \(Y_c=\tfrac12\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix}\)。此时 \(M\) 对角线为 1.5、非对角线为 -0.5，解为 \(\alpha=Y_c/2\)，训练分数 \(G\alpha=Y_c/2\)。分数的绝对值为 0.25，仍能正确选类；不需要把它硬解释成 25% 的概率。</p>
<div class="lab" data-lab="ridge"></div>
<div class="lab" data-lab="cholesky"></div>
<p class="warning">优化器步数为零，只说明没有迭代梯度优化。标签已经改变了 \(\alpha\) 和预测函数，因此这是监督学习。冻结版也没有把类别角色、旧新标签或 query 分布写进正则系数。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:138</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:155</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:161</code>、<code>configs/d92_branch_local_ridge_frozen_20260929.json:25</code>。原始变量与系数表示的关系为对冻结目标的数学推导。</p>`
  },
  {
    id: 'r-inference', group: '求解与判决', kicker: '11 / 当前 query 怎样作出决定',
    title: '逐样本判决：只与固定 support 比较',
    lead: '拟合结束后，support 参考、带宽、trace 比例和系数全部固定。每个 query 独立面对全部注册类别，批次中的其他 query 不参与它的决定。',
    html: String.raw`
<h3>从相似度行到类别分数</h3>
<p>对当前 query \(q\)，冻结编码器产生同样的五块特征，再构成 \(b_q,a_q,\phi_q\)。设 support 的交互特征为 \(\phi_1,\ldots,\phi_N\)，距离为 \(d_j^2=\|\phi_q-\phi_j\|^2\)。当冻结带宽 \(\tau>0\) 时，径向相似度向量为 \(r(q)\in\mathbb R^N\)，其第 \(j\) 项为 \(\exp(-d_j^2/\tau)\)。这里 \(N\) 是 support 数，不是 query 批量大小。</p>
<p>核特征必须使用训练 support 的参考中心。令 \(\bar R=R\mathbf1/N\)，\(\bar{\bar R}=\mathbf1^\mathsf T R\mathbf1/N^2\)，则中心化并缩放后的交叉核向量为：</p>
<div class="equation">\[
g(q)=\gamma\left[r(q)-\frac{\mathbf1^\mathsf T r(q)}N\mathbf1-\bar R+\bar{\bar R}\mathbf1\right],
\qquad f(q)=g(q)^\mathsf T\alpha\in\mathbb R^C.
\]</div>
<p>唯一与当前 query 有关的均值，是它对这 \(N\) 个 support 相似度的均值。训练的 \(\bar R,\bar{\bar R},\gamma,\tau,\alpha\) 都不更新；更没有对同批 query 求均值。代码用参考差分及 \(R-1\) 的形式实现等价中心化，减少接近常数时的相消误差，而不是增加新的统计来源。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：为什么交叉核也要减去两种均值？</summary>
<p>设径向核的原始特征映射为 \(\eta\)，训练均值为 \(\bar\eta=N^{-1}\sum_j\eta_j\)。query 与第 \(j\) 个 support 都减去同一训练均值：</p>
<div class="equation">\[
\langle\eta_q-\bar\eta,\eta_j-\bar\eta\rangle
=R(q,j)-\frac1N\sum_\ell R(q,\ell)
-\frac1N\sum_\ell R(\ell,j)
+\frac1{N^2}\sum_{\ell,m}R(\ell,m).
\]</div>
<p>乘上 \(\gamma\) 就是 \(g_j(q)\)。query 不需要计算其他 query 的特征或相似度。把 query 均值换成整批 query 的均值，会引入跨 query 依赖并改变方法；即使没有读取真值，也不再是冻结版的逐样本推理。</p></details>
<h3>全注册类别竞争与确定的平局规则</h3>
<div class="equation">\[
\widehat c(q)=\operatorname*{arg\,max}_{c\in\mathcal C_{\rm registered}}f_c(q).
\]</div>
<p>所有 \(C\) 个注册类都参加竞争。旧类、新类只用于分组报告，没有专属系数、独立候选集合或 query 的旧新角色提示。若两个分数精确相等，按物理 class ID 的字典序选最前者；这不是按当前数组中的类别编号猜测。训练先规范排序物理 support ID 与类别 ID，最后才恢复请求的输出列顺序。</p>
<p>例如 \(\alpha=\begin{bmatrix}0.25&amp;-0.25\\-0.25&amp;0.25\end{bmatrix}\)，某 query 的 \(g=(0.3,-0.3)^\mathsf T\)，则 \(f=(0.15,-0.15)\)，选择第一类。若两类分数同为零，就由 class ID 字典序决定。正负分数和不为 1 的总和都是正常情况；可视化颜色若显示相对强弱，也不能冒充经过校准的概率。</p>
<p>实际上目标每行的类别和为零，故 \(Y_c\mathbf1_C=0\)，非退化系数解满足 \(\alpha\mathbf1_C=0\)，理论分数也满足 \(f(q)\mathbf1_C=0\)。这和概率总和为 1 明显不同。给分数额外做 softmax 虽不改变最大项，却不能凭空得到概率校准，也不是本版拟合目标。</p>
<div class="lab" data-lab="decision"></div>
<h3>什么在变，什么保持冻结？</h3>
<p>query 改变时，只产生它自己的距离行、核行和 \(C\) 维分数。把同一批 query 拆成两批、改变批量大小或交换 query 顺序，都不会改变单个 query 所用的 support 状态；实现按行计算也避免把批次规模带入矩阵聚合。没有用其他 query 做再分配，也没有按真实类别数量施加配额。</p>
<p>本版编码器不做监督微调（SFT），也不独立拟合一个加性 bias 参数。中心化本身仍会形成相对 support 参考的函数偏移，所以“没有 bias 参数”不等于“没有参考中心”。常驻的 \(b_i,a_i\) 来自当前合法目标 support，不能称为地面 source 记忆。历史冻结编码器仍是 DAOT A1/RC4；接上 T1、REPL 或 DURA 都会改变版本边界。</p>
<p>一般任务中，\(\alpha\) 允许正负系数，分数聚合全部 support 的证据，并不总等价于找最近的一个锚点。同一观测产生的 FFT、身份与分支表征只是计算视图，不增加物理样本数。编码器已有历史训练经历，与本轮不使用 source 数据拟合头，也是需要分别说明的两件事。</p>
<p class="warning">本章复述的是既有版本对当前 support 的使用方式，不为新数据合同新增权限。分类头保持冻结，并不自动证明任意 checkpoint 来源或新的 Phase2 数据使用合法；也不能把待识别样本加入 support 后继续叫作独立 query。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:218</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:265</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:302</code>、<code>code/cvsrffi/d92_branch_interaction.py:95</code>、<code>configs/d92_branch_local_ridge_frozen_20260929.json:38</code>。</p>`
  },
  {
    id: 'r-validation', group: '验证与证据', kicker: '12 / 用已有 support 检查泛化',
    title: '物理分折与单锚点 proxy：验证不能偷偷参与拟合',
    lead: 'support 内诊断把物理样本分成训练与持出两部分。每次只用训练部分重估全部状态；真实每类一个样本时，没有独立持出样本。',
    html: String.raw`
<h3>训练能分对，不等于新样本能分对</h3>
<p>如果头已经看过某个 support 的特征与标签，再用它检验分类，主要检查的是拟合能力。持出诊断则暂时隐藏一部分物理样本，用其余样本拟合后再预测。它模拟同一 support 来源内的未见样本，但不等价于新的独立 query 基准，更不能把相同物理 IQ 的多个数学视图拆到两侧冒充独立样本。</p>
<p>注册类别数为 \(C\)，parent episode 每类 \(K\) 个物理样本，总数 \(N=CK\)。当 \(K>1\) 时，折数为 \(F=\min(K,3)\)。每类先按物理 ID 字典序排序，位置从 0 开始，第 \(j\) 个样本分到 \(j\bmod F\) 折。每次持出一折，训练使用其余折；各类采用相同位置规则，所以训练中的每类样本数仍相等。</p>
<table class="mini-table"><thead><tr><th>parent 每类样本数</th><th>各折持出数／类</th><th>各折训练数／类</th></tr></thead><tbody><tr><td>\(K=5\)</td><td>2、2、1</td><td>3、3、4</td></tr><tr><td>\(K=10\)</td><td>4、3、3</td><td>6、7、7</td></tr><tr><td>\(K=20\)</td><td>7、7、6</td><td>13、13、14</td></tr></tbody></table>
<h3>每折都必须从训练部分重新开始</h3>
<p>例如 \(C=8,K=5\) 时共有 40 个 parent support。第一折每类持出位置 0、3，训练只有每类 3 个，共 24 个；该折距离、Gram 矩阵为 \(24\times24\)，系数为 \(24\times8\)。第三折每类只持出位置 2，训练为 32 个，不能把所有折都误记成同一种训练尺寸。</p>
<ol class="steps"><li>只取本折训练样本构造距离，并估计最近异类距离中位数 \(\tau\)。</li><li>只用训练部分计算中心化统计量、\(s_0,s_R,\gamma\)，求出 \(\alpha\)。</li><li>冻结本折状态，对持出物理样本逐个评分。</li><li>把持出预测拼成 OOF 结果；持出标签只用于诊断指标，不回流改变带宽、权重或系数。</li></ol>
<p>OOF 指 out-of-fold，即每个样本都由没有用它拟合的那一折模型预测。若先用完整 parent 估好带宽或中心，再分折求系数，持出特征已经进入了拟合条件，会夸大验证意义。三个预定诊断臂使用相同拆分，冻结版并不根据持出分数在臂之间选择部署方法。</p>
<p>这里的交叉验证用于诊断固定方法，不是搜索超参数。正式完整头仍按同一冻结规则，用该任务全部合法 support 重新拟合；它不继承某个“最好折”的系数，也不对各折系数简单平均，因为各折的 support 基、带宽和中心都不同。持出结果可以揭示问题，但不应暗中变成本版的候选筛选反馈。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：全锚点的 1-shot proxy 怎样计数？</summary>
<p>对 parent 的全部位置 \(r=0,\ldots,K-1\)，每类只选位置 \(r\) 的一个样本训练，其余 \(K-1\) 个样本持出。每次训练数为 \(C\)，持出数为 \(C(K-1)\)；总共 \(K\) 次锚点试验。</p>
<div class="equation">\[
\text{训练出现次数}=CK=N,\qquad
\text{持出出现次数}=CK(K-1)=N(K-1).
\]</div>
<p>同一个物理样本恰好训练一次、持出 \(K-1\) 次。因此持出“出现次数”不是独立样本量。parent \(K=5,C=8\) 时，五次试验共 160 次持出出现，只有 40 个不同物理样本。每次仍要单独重估 \(\tau,\gamma,\alpha\)，不能共享完整 parent 的状态。</p></details>
<h3>两层平均的意义</h3>
<div class="equation">\[
m_{\rm parent}=\frac1K\sum_{r=0}^{K-1}m_{{\rm parent},r},\qquad
m_{\rm proxy}=\frac1P\sum_{{\rm parent}=1}^{P}m_{\rm parent}.
\]</div>
<p>先在 parent 内平均锚点指标，再让每个 parent 等权。假设一个 parent 的两个锚点准确率为 40%、60%，它贡献 50%；另一个 parent 有五个锚点且均为 80%，总体是 65%，不能把七个锚点直接混成 71.43%。不同 parent 的 \(K\) 或覆盖权重改变时，这个区别尤其重要。</p>
<div class="lab" data-lab="folds"></div>
<p class="warning">真实 \(K=1\) 每类没有可独立持出的物理样本，probe 只做数值诊断，不伪造 OOF；正式头仍可用完整 support 拟合。高 \(K\) parent 切出的单锚点 proxy 保留了自己的 parent 来源，不能宣称它就是正式真实 \(K=1\)。训练全对也不能证明泛化：已有报告的 proxy 训练准确率为 100%，同时仍存在明显持出误差。</p>
<p>把真实单锚点任务中的某一类整个持出，会让训练时缺少该注册类，并不等价于“所有注册类都有合法 support”的原问题。复制原样本或切换数学视图也不能修补这一缺项；缺少独立物理证据时，应保留没有持出结论这一边界。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:367</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:418</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:447</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:463</code>；结果口径见 <code>docs/D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md:103</code>。</p>`
  },
  {
    id: 'r-metrics', group: '验证与证据', kicker: '13 / 正确读懂适应、注册与均值',
    title: 'A、B、C 与 H：指标回答的是不同问题',
    lead: '历史报告把冻结原生分类器、旧类 support 拟合和旧新联合拟合分开比较。准确率的差值需要配对解释，调和均值必须先在每个任务单元内计算。',
    html: String.raw`
<h3>三个状态不是一条连续训练轨迹</h3>
<ul><li><strong>A：</strong>历史冻结 Phase1 原生分类器，没有用目标 support 拟合，报告旧类识别作为参考。</li><li><strong>B：</strong>只用六个旧类的当前 support，独立拟合 LocalRidge 头，面对六个注册类判决。</li><li><strong>C：</strong>用六个旧类与全部新增类的当前 support，从头重新估计带宽、中心、trace 比例与系数，面对全部注册类判决。</li></ul>
<p>C 并不接收 B 的已拟合系数作为初始化，也不沿用 B 的优化器状态。底层历史编码器冻结，但当前 support 头重新拟合。因此 \(B-A\) 同时包含利用目标 support 和替换分类头的影响，不能单独称为编码器适应收益；\(B-C\) 包含新增类别竞争和重新拟合的影响，不能单独解释成参数遗忘。</p>
<p>本版矩阵的旧类数为 6，新增类数 \(N_{\rm new}\in\{0,2,5,10,20\}\)，每类物理样本数 \(K\in\{1,5,10,20\}\)。总注册类数 \(C=6+N_{\rm new}\)，总 support 数 \(N=CK\)，最大为 \(C=26,N=520\)。每类一个样本不表示整个任务只有一个 support。</p>
<h3>先算每个任务的 H，再平均</h3>
<p>设第 \(e\) 个配对任务单元的旧类、新类准确率分别为 \(A_{{\rm old},e}\)、\(A_{{\rm new},e}\)，则：</p>
<div class="equation">\[
H_e=\frac{2A_{{\rm old},e}A_{{\rm new},e}}{A_{{\rm old},e}+A_{{\rm new},e}},\qquad
\overline H=\frac1E\sum_{e=1}^{E}H_e.
\]</div>
<p>分母为零时取零；没有新增类别时新类准确率和 \(H\) 都是 N/A，而不是零。这个指标对旧新两边不均衡更敏感，但它仍只是准确率的汇总，不能证明每个类别或每个单元都达到目标。</p>
<p>手算两个单元：旧新准确率分别为 90%／10% 与 10%／90%，两个 \(H\) 都是 18%，平均仍为 18%。若先平均准确率，两边都变成 50%，再算得到 50%，完全改变了问题。类似地，逐单元绝对新旧差的均值，也不能用两项平均准确率之差代替。</p>
<div class="lab" data-lab="metrics"></div>
<h3>2026 年 9 月 30 日的既有配对结果</h3>
<p>下表单位均为 %，按新增 2、5、10、20 类的联合任务等权汇总；每个 \(K\) 有 960 个配对单元。两个 cohort 按实际单元数 3∶1 合并，不是各给一半权重。</p>
<table class="mini-table"><thead><tr><th>每类 \(K\)</th><th>A 旧类</th><th>B 旧类</th><th>C 旧类</th><th>C 新类</th><th>逐单元 \(H\) 均值</th></tr></thead><tbody><tr><td>1</td><td>61.146</td><td>50.949</td><td>42.064</td><td>29.536</td><td>33.139</td></tr><tr><td>5</td><td>61.146</td><td>68.704</td><td>60.038</td><td>46.126</td><td>51.524</td></tr><tr><td>10</td><td>61.146</td><td>72.637</td><td>64.749</td><td>53.051</td><td>57.868</td></tr><tr><td>20</td><td>61.146</td><td>75.914</td><td>68.833</td><td>58.445</td><td>62.840</td></tr></tbody></table>
<p>例如 \(K=1\) 的 \(B-A=-10.197\) 个百分点，\(B-C=8.885\) 个百分点；这说明用单个旧类锚点替换原生头并不自动带来正收益。\(K=20\) 的 B 旧类均值提高到 75.914%，但注册后的旧类仍有下降，新类准确率也低于旧类。不能只挑一个较高数字概括整条适应与注册过程。</p>
<p>差值的符号要与名称一致：\(B-C>0\) 表示注册后旧类降低，若为负则表示注册后旧类提高；它不是绝对差。准确率从 50% 到 60% 是增加 10 个百分点，不是增加 10%。表中先按同一物理任务配对计算，再汇总；不能把不同场景或 support 抽样的两项均值随意相减，冒充同条件比较。</p>
<p>一个任务单元包含固定的模型、接收机与场景、support 抽样和类别注册设置。多个任务复用模型或相同 query 时，它们并非完全独立的重复。均值随 \(K\) 提高也只是这张冻结矩阵的描述，不能由此推出每一个接收机、类别或样本都单调改善。表中四舍五入后的旧新均值，也不足以重新还原逐单元 \(H\)。</p>
<details class="derivation" data-depth="deep"><summary>逐步读表：新增零类的行为什么单列？</summary>
<p>当 \(N_{\rm new}=0\) 时，C 的类别和 support 与 B 相同，故 \(C=B\)。例如 \(K=5\) 时两者旧类均为 68.704%，旧类下降为零，新类与 \(H\) 不定义。把这一行纳入“旧新联合任务 H 均值”，会制造一个并不存在的新类问题。表中 A 是同一冻结参考，其数值不随 K 变化；这也不意味着 A 具有新类注册能力。</p></details>
<p class="warning">这些数字来自已暴露、复用既有模型与 support/query 的重复基准，实际场景标识为 <code>practical_high</code>、<code>practical_mid</code>、<code>practical_low_urban</code>。它们不是 fresh blind 验证，不证明统计显著性、每个任务共同改善或真实在轨有效；也不自动扩展到 T1、REPL、DURA 或新的 Phase2 数据合同。</p>
<p class="source">依据：<code>docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md:3</code>、<code>docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md:7</code>、<code>docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md:16</code>、<code>docs/D92_BRANCH_LOCAL_RIDGE_ABC_RESULT_20260930.md:43</code>、<code>docs/D92_BRANCH_LOCAL_RIDGE_RESULT_20260929.md:7</code>。</p>`
  },
  {
    id: 'r-limits', group: '边界与演算', kicker: '14 / 退化规则与真实成本',
    title: '零带宽、全同特征与常驻状态：边界也是方法定义',
    lead: '冻结版没有用带宽下界、jitter 或临时换头隐藏退化。不同边界有明确的数学含义，资源口径也必须与完整部署成本分开。',
    html: String.raw`
<h3>只有一个注册类时，不需要虚构二类问题</h3>
<p>当 \(C=1\) 时，目标 \(\mathbf1[y_i=1]-1/C=0\)，最近异类距离没有定义。代码把带宽保留为空，不做 Cholesky，输出唯一注册类对应的零分数。分类当然选择这唯一一类；这不能用来证明有区分能力，也不等于已经支持未知拒识。真实报告的最小类数为 6，单类路径来自预先的合成数值检查。</p>
<h3>完整特征全相同时，零函数就是正确的退化</h3>
<p>定义 \(D_{ij}=\|\phi_i-\phi_j\|^2\)，\(s_0=N^{-1}\sum_{i\lt j}D_{ij}\)。若全部 \(b_i,a_i\) 完全相同，则 \(s_0=0\)。所有 support 的核特征都相同，中心化后 \(G=0\)，无法从特征中区分标签。冻结实现直接保存零系数、输出零分数，按类别字典序处理平局，不让数值舍入偷偷决定类别。</p>
<p>例如两个平衡类别各有两个 support，但四份完整特征全相同。一个确定性函数只能给同一输入相同输出，最优平方误差也无法同时匹配两种不同目标。换求解器或增加优化器迭代都不会补上缺失的信息。这里相同指处理后的完整特征，不是要求原始 IQ 字节相同。</p>
<h3>零带宽与极小正带宽有不同规则</h3>
<div class="equation">\[
R_{ij}=
\begin{cases}
\exp(-D_{ij}/\tau),&amp;\tau>0,\\
\mathbf1[D_{ij}=0],&amp;\tau=0.
\end{cases}
\]</div>
<p>当 \(\tau=0\) 但 \(s_0>0\) 时，使用精确特征等价类核：同特征相似度为 1，不同特征为 0。若两个标签冲突的 support 完整特征相同，它们仍属于同一个等价组，不会因为标签不同被强行拆开。标签冲突正是不可辨识的一部分，不能伪造一个很小的正距离躲开它。</p>
<p>手算四个 support：第一组特征为 \(u\)，第二组为 \(v\)，\(\|u-v\|^2=2\)；每组各有一个 A 类、一个 B 类样本。所有最近异类距离为零，所以 \(\tau=0\)，但 \(s_0=2>0\)。核矩阵有两个 \(2\times2\) 的全 1 块。组内标签对称冲突，使分类分数不能区分 A 与 B。</p>
<p>在零带宽路径上，所有不与任何 support 精确匹配的 query，其原始相似度行都为零；经同一 support 中心化后，仍得到相同分数行，因此预测相同。这个共同分数未必为零。极小的正带宽则继续按指数公式计算，没有人为 bandwidth floor；远距离相似度下溢成零与“把正带宽改成零”也不是同一个定义。</p>
<div class="lab" data-lab="degeneracy"></div>
<details class="derivation" data-depth="deep"><summary>逐步计算：为什么常驻数值状态是这个大小？</summary>
<p>一般非退化情况下，五个 float64 数组分别是当前 support 背景 \(N\times256\)、辅助 \(N\times480\)、系数 \(N\times C\)、参考核行 \(N\) 与中心均值 \(N\)，再加四个 float64 标量：</p>
<div class="equation">\[
B_{\rm state}=8[N(256+480+C+1+1)]+4\times8
=8N(738+C)+32.
\]</div>
<p>最大形状 \(N=520,C=26\) 得到 3178272 bytes。这里保存的是当前目标 support 的特征，不是 source 样本记忆。注册表、审计字典、Python 对象和临时距离、核、Cholesky 矩阵不在这个数字内；退化路径的可用标量数量也不同，不能无条件套一般公式。</p></details>
<h3>不展开长特征，仍然有拟合和评分成本</h3>
<p>距离用秩 2 QR 与非负平方和计算，不物化 123616 维交互向量。成对距离计算约为 \(O(N^2\cdot736)\)，核存储为 \(O(N^2)\)，Cholesky 为 \(O(N^3)\)。单 query 评分约为 \(O(N\cdot736+NC)\)。最大形状下，仅一个 \(520\times520\) 的 float64 矩阵就占 2163200 bytes，不能把它漏算进拟合峰值。</p>
<p class="warning">无优化器、无 source 新载荷或复用已有特征，不等于零部署成本。冻结编码器前向、FFT 表征、support 存储、头拟合及 query 评分都有成本。遇到非同特征距离下溢、非有限数或数值残差超限，固定路径报告技术失败；不添加 jitter、截谱或回退其他头来冒充同一版本。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:66</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:99</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:113</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:314</code>；资源及退化证据边界见 <code>docs/D92_BRANCH_LOCAL_RIDGE_SUPPORT_RESULT_20260929.md:143</code>。</p>`
  },
  {
    id: 'r-worked', group: '边界与演算', kicker: '15 / 把整条流程接起来',
    title: '二类单锚点手算与完整玩具流水线',
    lead: '用两个非同特征锚点，可以把带宽、trace 匹配、岭解和判决全部推到底。交互再扩展到每类两个样本，观察类内变化怎样进入几何。',
    html: String.raw`
<h3>明确这个小例子在计算什么</h3>
<p>设两类各有一个 support，故 \(C=2,K=1,N=2\)。两个交互特征不相同，记平方距离 \(D=\|\phi_1-\phi_2\|^2>0\)。这里 \(D\) 已经是平方距离，不能在指数中再平方。教学中的人工坐标只用于核代数演算，不替换历史冻结编码器，也不冒充真实 IQ 或新实验结果。</p>
<h3>第一步：带宽与径向核</h3>
<p>每个锚点唯一的异类锚点就是另一个，因此最近异类平方距离都为 \(D\)，其中位数 \(\tau=D\)。记 \(r=e^{-1}\)，于是：</p>
<div class="equation">\[
R=\begin{bmatrix}1&amp;r\\r&amp;1\end{bmatrix},\qquad
H=\frac12\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix},\qquad
HRH=\frac{1-r}{2}\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix}.
\]</div>
<h3>第二步：匹配 trace，而不是随意放大核</h3>
<div class="equation">\[
s_0=\frac D2,\qquad s_R=\operatorname{tr}(HRH)=1-e^{-1},\qquad
\gamma=\frac{D}{2(1-e^{-1})},\qquad
G=\frac D4\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix}.
\]</div>
<p>原始交互几何的中心化 trace 是两点离均值平方距离之和，两点各为 \(D/4\)，所以总和为 \(D/2\)。径向核的中心化 trace 为 \(1-r\)。二者的比值固定了 \(\gamma\)，没有用 query 误差选择缩放。</p>
<details class="derivation" data-depth="deep"><summary>逐步推导：二类岭解为什么只剩一个缩放？</summary>
<p>记 \(S=\begin{bmatrix}1&amp;-1\\-1&amp;1\end{bmatrix}\)，有 \(S^2=2S\)。中心化目标 \(Y_c=S/2\)，而 \(G=DS/4\)，因此 \(GY_c=(D/2)Y_c\)。把系数方程作用在目标所在方向上：</p>
<div class="equation">\[
\alpha=(G+I)^{-1}Y_c=\frac{Y_c}{1+D/2},\qquad
G\alpha=\frac{D/2}{1+D/2}Y_c.
\]</div>
<p>取 \(D=2\)，则 \(\tau=2\)，\(r\approx0.367879\)，\(\gamma\approx1.581977\)，\(\alpha=Y_c/2\)。两个训练锚点的分数分别为 \((0.25,-0.25)\)、\((-0.25,0.25)\)。训练能分对，不代表已经知道每类附近所有样本的分布。</p></details>
<p>这个特例还暴露了中心化的秩：常数方向 \((1,1)\) 被 \(G\) 映射成零，类间方向 \((1,-1)\) 的特征值为 \(D/2\)。因此 \(G\) 只有秩 1，仍不妨碍 \(G+I\) 求解。可逐项核对 \(\operatorname{tr}G=D/2\)、\(G\mathbf1=0\)、\((G+I)\alpha=Y_c\)；这些检查比只看最后类别更能定位计算错误。</p>
<h3>第三步：一个新点的判决</h3>
<p>令 query 到两锚点的相似度为 \(r_1,r_2\)，则 \(g(q)=\gamma(r_1-r_2)(1,-1)^\mathsf T/2\)，从而：</p>
<div class="equation">\[
f(q)=\frac{\gamma(r_1-r_2)}{2(1+D/2)}(1,-1).
\]</div>
<p>例如仍取 \(D=2\)，query 到两点的平方距离为 0.5、2，则 \(r_1=e^{-0.25}\)、\(r_2=e^{-1}\)，第一类分数约为 0.1625，第二类为其负数。此特例的判决等价于比较哪个锚点更近；两者距离相同则按类别字典序。这个结论依赖二类、单锚点与本例的对称结构，不能推广成所有 LocalRidge 都是最近邻。</p>
<h3>两锚点没有提供类内方差</h3>
<p>每类只有一个点，类中心就是该点，类内散布的观测和为零。零散布只说明没有看到偏离中心的第二个样本，不是证明真实类内方差为零。许多具有完全不同类内变化的分布，都可能给出这同一对锚点；仅凭它们无法识别真实方差。径向带宽来自异类距离，也不能补成类内方差估计。</p>
<div class="lab" data-lab="pipeline"></div>
<h3>扩成每类两个样本：交互显示完整中间量</h3>
<p>当 \(K=2\) 时，输入变为四个 support，距离、核和 Gram 为 \(4\times4\)，目标及系数为 \(4\times2\)。交互会由人工坐标重新计算全部量，展示 \(\tau\)、\(s_0\)、\(s_R\)、\(\gamma\)、Cholesky 与最终分数。不能把上面二锚点的 \(\tau=D\) 或单个缩放公式直接套入。两折持出与全锚点 proxy 此时各用每类一个样本训练，最后完整头仍只用四个 support 拟合。</p>
<p>移动一个同类样本后，可以先观察类内距离和最近异类距离的变化，再追踪核行与最终判决。分数变化来自这些明确的计算依赖，不代表交互已经估计出真实信号分布；人工例子的结果也不会进入冻结模型。</p>
<details class="derivation" data-depth="deep"><summary>可检查的预测伪代码：逐行对应数学对象</summary>
<pre><code>support = sort_by_physical_id(current_support)
b, a = frozen_feature_blocks(support)
D = stable_interaction_squared_distances(b, a)
s0 = sum_upper_triangle(D) / N
if C == 1 or s0 == 0:
    return zero_score_head_with_lexicographic_ties
tau = median(nearest_other_class_distances(D))
R = radial_or_exact_equivalence(D, tau)
G, center_state, gamma = train_center_and_trace_match(R, s0)
Yc = one_hot(support_labels, C) - 1 / C
L = cholesky(G + identity(N))
alpha = upper_solve(transpose(L), lower_solve(L, Yc))
for q in queries:
    r = radial_to_fixed_support(q, b, a, tau)
    g = apply_fixed_train_center(r, center_state, gamma)
    scores = transpose(g) @ alpha
    prediction = argmax_with_class_id_lexicographic_ties(scores)</code></pre>
<p>这是讲解计算依赖的伪代码，不是实验 launcher。它不读取 query 真值，不拟合 query，不搜索超参数，也不包含任何发布或数据授权步骤。实际 v1 的稳定中心化使用参考差分与 \(R-1\)，而不是要求直接减去接近的浮点大数。</p></details>
<p class="warning">本教程的全部 support 方法都属于同一个冻结 v1。后续 margin、adapter 或新编码器方案没有混入此流水线；可调人工例子也不会修改冻结配置。分数始终是回归分数，不伪造模型概率。</p>
<p class="source">依据：<code>code/cvsrffi/d92_branch_local_ridge.py:113</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:147</code>、<code>code/cvsrffi/d92_branch_local_ridge.py:218</code>、<code>configs/d92_branch_local_ridge_frozen_20260929.json:20</code>。二类特例为冻结公式的直接代数推导。</p>`
  }
);
