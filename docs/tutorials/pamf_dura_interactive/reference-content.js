window.COURSE_CHAPTERS.push({
 id:'r-check',group:'串联与自测',kicker:'CONNECT THE DOTS / 19',title:'把全网串起来，再检查自己是否掌握',
 lead:'能讲清一个输入经过哪些计算、损失怎样影响参数、哪些结论不能从结构直接推出，就已经掌握了两张网络的骨架。',
 html:String.raw`
 <h3>用一条样本串起部署流程</h3><ol><li>收到 \(y\in\mathbb R^{2\times256}\)，不需要知道真实 TX 或场景。</li><li>DURA 计算一次粗恢复 c，并从 y/c 估计当前窗口的 A 与 v。</li><li>令 x₀=c。第一阶段读九通道输入，生成 p₁，以 x₀ 热启动做两次 CG 得到 x₁。</li><li>第二阶段以 x₁ 重算反投影残差，生成 p₂，再做两次 CG 得到恢复 IQ x₂。</li><li>PAMF 从同一个 x₂ 同时计算 Sinc 时域、FFT 频域、PA、参考响应；路径描述读取当前时域中层序列。</li><li>按真实顺序归一化与残差融合，得到一个 160 维 h。</li><li>原默认余弦头计算所有注册类分数，argmax 输出身份。当前窗口结束，不更新模型。</li></ol>
 <div class="note">DURA 不会先判断“现在是 low urban”再选一套参数。PAMF 推理也不需要同 TX 跨 RX 的配对样本。配对、伪标签、梯度保护属于训练机制。</div>
 <h3>三种训练目标，不能混成一种“物理损失”</h3><table><thead><tr><th>目标</th><th>谁使用</th><th>需要什么监督</th><th>希望改变什么</th></tr></thead><tbody><tr><td>身份 CE／R-Drop／CRIC</td><td>PAMF 的 L 部分</td><td>合法 L 标签及源 RX 元信息</td><td>身份可分性、随机输出一致性、跨 RX 特征关系</td></tr><tr><td>T1 或 REPL 的 U 目标</td><td>PAMF 的半监督训练</td><td>源 U IQ，TX 标签隐藏</td><td>利用额外源样本，按各自配方控制伪标签影响</td></tr><tr><td>IQ／谱／阶段蒸馏误差</td><td>DURA 源训练</td><td>合法源模拟配对 IQ 与冻结 G 教师</td><td>波形恢复与有限步学生轨迹；无 F/TX 监督</td></tr></tbody></table>
 <h3>想判断“这个模块真有用”，需要怎样的证据</h3><p>结构给出假设，消融给出该设置下的条件差异。删去模块可能同时减少参数、改变优化和前向曝光；比较时需要披露这些差别。更低 NMSE 不自动代表更高身份准确率，源域几何更好也不自动代表目标接收机更好。</p><p>已有结果支持一些结构在具体合同中有用，但不支持“每个物理先验都稳定正收益”。例如有些时间重排与协方差控制并不比原签名差；REPL 也不能宣称在每个合同和受损场景都优于监督或普通 PL。T1、REPL、旧 F50 的分数与时延必须保留各自模型、数据与执行路径，不互相改标。</p>
 <h3>十道自测：答案后面都有解释</h3><div id="courseQuiz"></div>
 <h3>源码阅读顺序</h3><ol><li><code>configs/pamf_final_t1_e50.json</code> 与 <code>configs/pamf_ssl_repl.json</code>：先区分结构和训练版本。</li><li><code>experiments/adv3b02_xuc/code/model.py</code>：看真实时频／PA 主干和 forward。</li><li><code>experiments/cvs_residual_identity/model.py</code>、<code>experiments/cvs_reference_identity/model.py</code>：看真实融合和已知激励特征。</li><li><code>experiments/cvs_rdrop_sinc_units/{x3_features,descriptors,coordinate,optimizer}.py</code>：看签名及 Sinc 单位映射。</li><li><code>experiments/pamf_ssl_research/</code>：看 REPL 伪标签、风格、外推与分组梯度保护。</li><li><code>experiments/pamf_reception_restore/models.py</code>、<code>pamf_action_restore/{models,operators}.py</code>：看粗恢复、A 与 A*。</li><li><code>experiments/pamf_r65_four_route/{stage_polyphase,training}.py</code>：看两阶段和学生训练。</li><li><code>experiments/pamf_stage2_inference/program.py</code>、<code>pamf_r65_inference/program.py</code>：看固定推理与 CG 残差复用。</li></ol>
 <p class="source">版本与边界：<code>docs/PROJECT_PROTOCOL.md</code>、<code>docs/PAMF_FINAL_T1_E50.md</code>、<code>docs/PAMF_REPL.md</code>、<code>docs/DURA_V1.md</code>。部分继承模块从项目主目录只读核对，未假设每个工作树都包含全部历史源码。</p>
 <p class="source">公式显示：内嵌 KaTeX 0.16.22（MIT）。<a href="https://katex.org/docs/browser.html" target="_blank" rel="noopener">KaTeX 官方使用说明</a>。字体与渲染脚本均包含在单文件中；所有交互只在浏览器本地计算，不请求数据、权重或远程 API。</p>`
});
window.COURSE_QUIZ=[
 {q:'输入 [B,2,256] 中的“2”代表什么？',options:['两台发射机','I 与 Q 两个实数通道','网络的两个阶段'],correct:1,why:'I+jQ 共同表示一条复数序列；B 是样本数，256 是时间采样数。'},
 {q:'为什么 PAMF 同时保留多个分支？',options:['把多个分类器投票变成必然正确','从同一 IQ 的不同表征中学习互补线索','保证每条分支只含一种真实硬件因素'],correct:1,why:'多分支提供不同的归纳偏置，但 RX、信道和 TX 仍可能混合。最终只有一条 160 维表示和原分类头。'},
 {q:'一个七维窗口的当前签名描述有多少维？',options:['7+7²=56','7+21=28','6×4=24'],correct:1,why:'一级增量 7 维，二阶反对称上三角有 C(7,2)=21 维。它不是完整的 7×7 二阶张量。'},
 {q:'原 T1 的 As KL=0 意味着什么？',options:['As 不再训练','整个网络的 KL 全部关闭','As 保留两次真实标签 CE，普通／raw 的 R-Drop 继续'],correct:2,why:'取消的是 As 槽独立 KL。六次 L 前向、普通与 raw 的对称 KL、As CE 及原总剂量都保留。'},
 {q:'REPL 的 hᵢ+.25(hᵢ−hⱼ) 有什么边界？',options:['差向量已经是严格解耦的接收机向量','同伪类、不同 RX 的特征外推，分类保持由训练约束','在推理时每条 query 必须找一个同 TX 配对'],correct:1,why:'它使用完整融合特征，差向量也包含样本差异。外推与配对属于训练，推理不需要同类配对。'},
 {q:'对 DURA 来说，A* 是什么？',options:['A 的精确逆','复数实内积下与 A 匹配的伴随','恢复教师'],correct:1,why:'A* 反投影接收残差，是正规方程和 CG 的组成。伴随不等于逆，反射边界也必须匹配。'},
 {q:'DURA 的 pₖ 以哪个量为残差基点？',options:['共同粗恢复 c','必定是当前 xₖ₋₁','真实 clean IQ'],correct:0,why:'pₖ=c+τDₖ(九通道输入)。当前恢复仍作为网络输入影响修正，clean 在部署时不可见。'},
 {q:'对固定 a、p 的标量 MAP，增大 v 通常怎样？',options:['让观测一致性权重更大','让解更依赖先验 p','保证真实整网准确率提高'],correct:1,why:'目标中的观测权重为 1/v，所以局部解更靠近 p。但整网中 v 还参与 prior 输入，不能把局部结论直接扩成整网性能规律。'},
 {q:'为什么热启动 CG 的 δ₀ 通常不是零？',options:['因为 δ=x−p，所以 δ₀=x₀−p','因为加入了随机噪声','因为必须多跑两个阶段'],correct:0,why:'从现有 x₀ 开始，与从 p 开始不同。其初始残差可整理为 v⁻¹A*(y−Ax₀)−τ⁻²(x₀−p)。'},
 {q:'两阶段四次 CG 加 CUDA Graph，是否意味着缓存旧预测？',options:['是，后续输入不再计算','不是；当前输入更新，重放运算结构','只有 clean 缓存，urban 重新计算'],correct:1,why:'图缓存固定执行结构，残差复用只利用同一当前样本已计算的量。每个输入仍产生自己的恢复与分类。'}
];
window.COURSE_GLOSSARY=String.raw`
<table><thead><tr><th>符号</th><th>这里的含义</th></tr></thead><tbody>
<tr><td>\(I,Q,j\)</td><td>实部、虚部、虚数单位；\(z=I+jQ\)</td></tr>
<tr><td>\(B,C,T\)</td><td>批量样本数、特征通道数、时间位置数</td></tr>
<tr><td>\(\bar z,\ |z|\)</td><td>复共轭、复数幅度</td></tr>
<tr><td>\(\odot,\ *\)</td><td>逐元素乘法、卷积（具体边界按实现）</td></tr>
<tr><td>\(\mathcal F,\mathcal F^{-1}\)</td><td>傅里叶变换及逆变换</td></tr>
<tr><td>\(y,c,x_k,p_k\)</td><td>DURA 的观测、共同粗恢复、第 k 阶段结果、第 k 阶段学习先验</td></tr>
<tr><td>\(A,A^*\)</td><td>当前窗口的接收作用近似及其实内积伴随；不是已知真信道和精确逆</td></tr>
<tr><td>\(h,m,q\)</td><td>A 内的 FIR 核、逐点复主系数、逐点复镜像系数；h 在 PAMF 章节也可表示最终身份特征，按上下文区分</td></tr>
<tr><td>\(v,\tau\)</td><td>正实数窗口方差尺度、固定先验尺度（约 .116946）</td></tr>
<tr><td>\(e,r,\delta\)</td><td>反投影残差、CG 的线性系统残差、相对先验偏移 x−p；三者不同</td></tr>
<tr><td>\(\ell,p,q\)</td><td>分类分数、预测分布、监督目标分布；与 DURA 镜像系数 q 按上下文区分</td></tr>
<tr><td>CE、KL、EMA、CG</td><td>交叉熵、分布相对熵、指数滑动平均、共轭梯度</td></tr>
<tr><td>TX、RX、L、U、V</td><td>发射机、接收机、有标签源训练、TX 标签隐藏的源训练、源验证</td></tr>
</tbody></table><p class="lab-caption">字母在不同领域可能复用。每章先读符号定义，尤其不要把“分类概率 p”与“prior 波形 pₖ”混成同一个对象。</p>`;
