/* Tensor and arithmetic diagrams for the expanded PAMF branch lessons. */
(function(){
'use strict';
const{register,figure,box,edge}=window.CourseFigures;
register('p-convolution','相同输入，两种连接方式：普通卷积 vs DW＋PW','上方每个输出拥有全部通道的独立短核；下方先各自滑动，再在同一位置用小矩阵混合。箭头表示连接，数值由下方交互实验计算。',figure('ds-compare',510,
`<text x="20" y="27" class="figure-emphasis">普通卷积：空间／时间与通道一起组合</text>
${box(20,55,140,57,'输入通道 1','一条序列')+box(20,146,140,57,'输入通道 2','另一条序列')}
${box(275,48,265,73,'输出 1 的两条独立短核','W₁₁[r] 与 W₁₂[r]','accent')+box(275,143,265,73,'输出 2 的另两条独立短核','W₂₁[r] 与 W₂₂[r]','accent')}
${box(690,57,160,57,'输出通道 1','长度由步长/补零决定','pamf')+box(690,151,160,57,'输出通道 2','每条输出都读两个输入','pamf')}
${edge('ds-compare','160,83 270,83')+edge('ds-compare','160,174 220,174 220,103 270,103')+edge('ds-compare','160,83 240,83 240,163 270,163')+edge('ds-compare','160,174 270,183')+edge('ds-compare','540,84 685,84')+edge('ds-compare','540,180 685,180')}
<path d="M20,240H850" class="figure-dashed"/><text x="20" y="275" class="figure-emphasis">深度可分离卷积：DW 各算各的 → PW 交换通道信息</text>
${box(20,307,130,60,'输入通道 1','x₁[t]')+box(20,407,130,60,'输入通道 2','x₂[t]')}
${box(205,307,135,60,'DW 核 D₁','只读第 1 通道','accent')+box(205,407,135,60,'DW 核 D₂','只读第 2 通道','accent')}
${box(385,307,110,60,'中间 u₁','通道数保持')+box(385,407,110,60,'中间 u₂','尚未通道混合')}
${box(557,327,135,121,'PW 矩阵 M','核长 1，读取 u₁/u₂','pamf')}
${box(745,307,110,60,'输出 y₁','M₁₁u₁+M₁₂u₂','accent')+box(745,407,110,60,'输出 y₂','M₂₁u₁+M₂₂u₂','accent')}
${edge('ds-compare','150,337 200,337')+edge('ds-compare','150,437 200,437')+edge('ds-compare','340,337 380,337')+edge('ds-compare','340,437 380,437')+edge('ds-compare','495,337 552,360')+edge('ds-compare','495,437 552,415')+edge('ds-compare','692,360 740,337')+edge('ds-compare','692,415 740,437')}
<text x="440" y="497" text-anchor="middle">DW 权重 kCᵢₙ ＋ PW 权重 CᵢₙCₒᵤₜ；通道之间仍通过 PW 学习组合。</text>`));
register('p-frequency','先把复数 FFT 排好，再取 ±k 的同距离频点','直流与 Nyquist 被排除；负侧按127…1的反向存储顺序重排，最终与正侧的1…127一一对齐。这里没有新增相位差通道。',figure('frequency-pair-detail',360,
`${box(20,25,150,64,'I＋jQ，256 点','25 MHz')+box(225,25,175,64,'FFT → |X|²','未加窗，未除 N','accent')+box(455,25,395,64,'原始索引顺序','0 │ 1…127 │ 128 │ 129…255','neutral')}
${edge('frequency-pair-detail','170,57 220,57')+edge('frequency-pair-detail','400,57 450,57')}
${box(40,145,190,65,'正侧 P⁺[k]','P[1],P[2],…,P[127]','accent')+box(300,145,235,65,'负侧 P⁻[k]','P[255],P[254],…,P[129]','pamf')+box(620,145,230,65,'同一 k 成对比较','f=±k×97.65625 kHz')}
${edge('frequency-pair-detail','650,90 650,120 135,120 135,140')+edge('frequency-pair-detail','730,90 730,120 418,120 418,140')}
${[['log(1+P⁺)','正侧强弱'],['log(1+P⁻)','负侧强弱'],['logR','哪一侧更强'],['asym','差异有多大']].map((a,i)=>box(30+i*214,273,190,65,a[0],a[1],i===2?'pamf':'accent')).join('')}
${edge('frequency-pair-detail','230,178 260,178 260,240 440,240')+edge('frequency-pair-detail','535,178 570,178 570,240 440,240')}
<text x="440" y="258" text-anchor="middle">每项先逐频点计算，再分别做127 → 32的自适应平均池化</text>`));
register('p-frequency-network','频域有两条汇合路线：局部卷积与三项统计','四通道负责局部形状，三项固定统计补充全局摘要；两者先投影到相同160维，再相加。圆度ρ另送时频融合。',figure('frequency-full-network',380,
`${box(20,30,145,60,'四曲线 4×32','不是32个可训练参数')+box(215,30,170,60,'频带门控','4→1,k5；倍率.4…1.6','accent')+box(435,30,170,60,'f1：4→16','k5，池化2','accent')+box(660,30,190,60,'16×16','GN→ReLU→MaxPool')}
${edge('frequency-full-network','165,60 210,60')+edge('frequency-full-network','385,60 430,60')+edge('frequency-full-network','605,60 655,60')}
${box(660,145,190,60,'f2：16→32','k5，池化2；32×8','accent')+box(435,145,170,60,'f3：32→32','k3，不池化','accent')+box(215,145,170,60,'全局平均 → Linear','32 → 160','pamf')}
${edge('frequency-full-network','755,90 755,140')+edge('frequency-full-network','660,175 610,175')+edge('frequency-full-network','435,175 390,175')}
${box(20,270,195,70,'原线性功率/不对称','高频占比、差异、平坦度')+box(265,270,170,70,'3 → 160 投影','ReLU＋Dropout','accent')+box(500,270,130,70,'相加','频域 fₑ','pamf')+box(695,270,155,70,'时频融合','与时域及ρ拼接')}
${edge('frequency-full-network','215,305 260,305')+edge('frequency-full-network','435,305 495,305')+edge('frequency-full-network','300,205 300,235 565,235 565,265')+edge('frequency-full-network','630,305 690,305')}
<text x="440" y="370" text-anchor="middle">固定谱统计没有训练权重；卷积核、GN仿射和投影矩阵才通过标签损失学习。</text>`));
register('p-pa-statistics','局部PA与频谱PA统计，在160维空间相遇','两条输入都来自同一包IQ。统计支路先做3→160投影；主干加0.25份、残差头再加1份，累计1.25份，然后做LayerNorm并乘有界增益。',figure('pa-statistics-routes',380,
`${box(20,115,135,70,'同一包 IQ','2×256')}
${box(215,25,195,75,'记忆多项式与门控','24×256','accent')+box(470,25,170,75,'普通膨胀卷积','24→48→64→64','accent')+box(695,25,155,75,'平均→投影','64→160','pamf')}
${box(215,180,195,75,'FFT正负线性功率','各池化→相加 T[32]')+box(470,180,170,75,'3个PA谱统计','外侧占比 / 比值 / 峰度','accent')+box(695,180,155,75,'线性投影','3→160','pamf')}
${edge('pa-statistics-routes','155,140 180,140 180,62 210,62')+edge('pa-statistics-routes','155,140 180,140 180,217 210,217')+edge('pa-statistics-routes','410,62 465,62')+edge('pa-statistics-routes','640,62 690,62')+edge('pa-statistics-routes','410,217 465,217')+edge('pa-statistics-routes','640,217 690,217')}
${box(385,305,300,60,'p = LayerNorm(p₀＋1.25Δp)','160维，随后乘tanh增益并与主干相加','accent')}
${edge('pa-statistics-routes','850,62 865,62 865,283 630,283 630,300')+edge('pa-statistics-routes','775,255 775,283 630,283')}
<text x="30" y="338">三项统计是输入描述，</text><text x="30" y="359">不是辨识出的功放系数。</text>`));
})();
