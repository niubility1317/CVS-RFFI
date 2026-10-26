/* Exact small algebra examples. These are synthetic inputs, never checkpoint inference. */
(function(){
'use strict';
const U=window.CourseLabUI,M=window.CourseMath,{slider,select,fmt,arr,linePlot,plane}=U;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),sum=a=>a.reduce((s,x)=>s+x,0),mean=a=>sum(a)/a.length;
const math=root=>{if(typeof renderMathInElement==='function')renderMathInElement(root,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],throwOnError:false,strict:'ignore',trust:false});};
const mount=(r,t,b,f)=>U.mount(r,t,b,(s,el)=>{f(s,el);math(el);},'可逐项核对的合成算例');
const result=(r,t)=>r.querySelector('[data-result]').innerHTML=t;
const table=(headers,rows)=>`<div class="detail-table"><table><thead><tr>${headers.map(x=>'<th>'+x+'</th>').join('')}</tr></thead><tbody>${rows.map(row=>'<tr>'+row.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')}</tbody></table></div>`;
const D={
 conv1d(x,w,p=0,d=1){const L=x[0].length,k=w[0][0].length,n=L+2*p-d*(k-1);return w.map(row=>Array.from({length:Math.max(0,n)},(_,t)=>row.reduce((a,kernel,c)=>a+kernel.reduce((v,h,i)=>v+h*(x[c][t-p+d*i]??0),0),0)));},
 separated(x,depth,point,p=0,d=1){const u=depth.map((h,c)=>D.conv1d([x[c]],[[h]],p,d)[0]);return{u,y:point.map(row=>u[0].map((_,t)=>sum(row.map((v,c)=>v*u[c][t]))))};},
 equivalent(depth,point){return point.map(row=>row.map((v,c)=>depth[c].map(h=>h*v)));},
 cost(cin,cout,k,L){const dense=k*cin*cout,dw=k*cin,pw=cin*cout;return{dense,dw,pw,separable:dw+pw,withNorm:dw+pw+2*cout,macDense:dense*L,macDS:(dw+pw)*L,ratio:(dw+pw)/dense};},
 poolRange(i,L,K){return[Math.floor(i*L/K),Math.ceil((i+1)*L/K)];},
 pool(a,K){return Array.from({length:K},(_,i)=>{const[start,end]=D.poolRange(i,a.length,K);return mean(a.slice(start,end));});},
 mirror(pos,neg,eps=1e-8){return pos.map((p,i)=>{const n=neg[i];return[Math.log1p(Math.min(p,1e6)),Math.log1p(Math.min(n,1e6)),clamp(Math.log(p+eps)-Math.log(n+eps),-20,20),Math.abs(p-n)/(p+n+eps)];});},
 lift(x,m,p,n){const z=n<m?[0,0]:x[n-m].map(v=>clamp(v,-2,2)),r2=Math.max(sum(z.map(v=>v*v)),1e-8),scale=p===1?1:r2**((p-1)/2);return{z,r2,scale,value:z.map(v=>v*scale)};},
 stats(energy,asym=energy.map(()=>0),eps=1e-8){const E=energy.map(v=>clamp(v,eps,1e6)),K=E.length,edgeBins=Math.max(1,Math.floor(.20*K)),l=Math.floor(.30*K),r=Math.max(l+1,Math.floor(.70*K)),hf=Math.floor(.75*K),edge=sum(E.slice(-edgeBins)),center=sum(E.slice(l,r)),total=sum(E),mu=mean(E),variance=mean(E.map(v=>(v-mu)**2))+eps,numerator=mean(E.map(v=>clamp(v-mu,-1e3,1e3)**4)),denominator=Math.max(variance*variance,eps);return{E,edgeBins,l,r,hf,edge,center,total,mu,variance,numerator,denominator,pa:[edge/(total+eps),edge/(center+eps),clamp(numerator/denominator,0,1e4)],frequency:[sum(E.slice(hf))/(total+eps),mean(asym.slice(hf)),Math.exp(clamp(mean(E.map(Math.log)),-20,20))/(mu+eps)]};},
 receptive(){let r=1,j=1;return[[7,1,2],[7,2,2],[5,4,1]].map(([k,d,p])=>{r+=(k-1)*d*j;const afterConv=r;if(p>1){r+=(p-1)*j;j*=p;}return{afterConv,r,j};});}
};
window.CourseDetailMath=D;
const register=(name,f)=>window.CourseLabs.register(name,f);
register('depthwise-step',root=>{
 mount(root,'滑动窗口，分两步得到同一个输出',`<div class="lab-controls">${slider('dwPosition','输出位置 t',0,3,0)}${select('dwOutput','选中输出通道',[['0','输出通道 1'],['1','输出通道 2']])}${slider('dwStep','展示步骤',0,3,2)}</div><div data-diagram class="detail-svg"></div><div class="pill-row"><button type="button" data-next-step>下一步</button><button type="button" data-restart>回到窗口</button></div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">两条长 6 的输入，核长 3、不补零。这里特意把普通卷积权重设为 DW×PW 的乘积，以核对算术；任意普通卷积没有这个等价保证。</p>`,(s,r)=>{
 const x=[[1,2,3,4,5,6],[2,1,0,-1,-2,-3]],depth=[[1,0,-1],[1,2,1]],point=[[1,2],[-1,1]],t=s.dwPosition,o=+s.dwOutput,out=D.separated(x,depth,point),dense=D.conv1d(x,D.equivalent(depth,point));
 const cell=(cx,cy,text,on=false)=>`<rect x="${cx}" y="${cy}" width="45" height="37" rx="5" class="${on?'detail-selected':'detail-cell'}"/><text x="${cx+22.5}" y="${cy+24}" text-anchor="middle">${text}</text>`;
 let svg=`<svg viewBox="0 0 760 330" role="img" aria-label="两通道逐通道卷积和逐点混合逐步计算"><text x="20" y="27">① 输入窗口（高亮三点，t=${t}）</text>`;
 x.forEach((v,c)=>{svg+=`<text x="15" y="${75+c*73}">x${c+1}</text>`+v.map((a,n)=>cell(52+n*49,51+c*73,a,n>=t&&n<t+3)).join('');});
 svg+=`<path class="detail-arrow" d="M357,91H392M357,166H392"/><text x="403" y="28">② DW：各算各的</text>`;
 depth.forEach((h,c)=>{svg+=`<text x="405" y="${73+c*73}">D${c+1}=${arr(h)}</text><text x="405" y="${99+c*73}" class="detail-strong">${s.dwStep>=1?'u'+(c+1)+'='+fmt(out.u[c][t],0):'先逐项相乘再求和'}</text>`;});
 svg+=`<path class="detail-arrow" d="M573,91L618,126M573,164L618,126"/><text x="612" y="56">③ PW 混合</text><rect x="610" y="106" width="133" height="70" rx="8" class="detail-selected"/><text x="676" y="134" text-anchor="middle">${s.dwStep>=2?'y'+(o+1)+'='+fmt(out.y[o][t],0):'权重 '+arr(point[o])}</text><text x="676" y="159" text-anchor="middle">${arr(point[o])}</text><text x="22" y="243">④ 合并后的普通卷积：两个通道的全部乘积相加</text><text x="22" y="281">${s.dwStep>=3?'本位置直接计算结果 '+fmt(dense[o][t],0)+'；完整输出 '+arr(dense[o]):'把步骤滑块移到 3，核对两种计算路径。'}</text></svg>`;
 r.querySelector('[data-diagram]').innerHTML=svg;
 const a=out.u.map(v=>v[t]),y=out.y[o][t],terms=point[o].map((w,c)=>`(${w})\\cdot(${a[c]})`).join('+');
 result(r,String.raw`\(\displaystyle y_{${o+1}}[${t}]=${terms}=${y}\)。完整输出 \(\mathbf y_{${o+1}}=${arr(out.y[o])}\)。当前步骤：<b>${['选输入窗口','逐通道乘加','逐点跨通道混合','与合并的普通卷积核对'][s.dwStep]}</b>。`);
 r.querySelector('[data-table]').innerHTML=s.dwStep>=1?table(['通道','输入窗口','DW 的三项乘积','DW 输出'],x.map((v,c)=>[c+1,arr(v.slice(t,t+3)),arr(depth[c].map((w,i)=>w*v[t+i])),fmt(a[c],0)])):'';
 r.querySelector('[data-next-step]').disabled=s.dwStep===3;
 });
 root.querySelector('[data-next-step]').onclick=()=>{const q=root.querySelector('[name=dwStep]');q.value=Math.min(3,+q.value+1);root._labUpdate();};
 root.querySelector('[data-restart]').onclick=()=>{root.querySelector('[name=dwStep]').value=0;root._labUpdate();};
});
register('depthwise-cost',root=>{
 const presets={t1:[48,72,5,128],t2:[72,96,5,64],t3:[96,96,3,32],f1:[4,16,5,32],f2:[16,32,5,16],f3:[32,32,3,8]};
 mount(root,'参数量与乘加计算器：把真实层代进去',`<div class="lab-controls">${select('costPreset','真实层预设',[['t1','t1：48→72'],['t2','t2：72→96'],['t3','t3：96→96'],['f1','f1：4→16'],['f2','f2：16→32'],['f3','f3：32→32'],['custom','自定义教学配置']])}${slider('costIn','自定义输入通道',1,128,48)}${slider('costOut','自定义输出通道',1,160,72)}${select('costKernel','自定义核长',[['3','3'],['5','5'],['7','7']])}${slider('costLength','自定义卷积输出长度（池化前）',8,256,128,8)}</div><div data-cost class="detail-svg"></div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">预设按实际网络；自定义控件只在“自定义教学配置”生效。比较仅计卷积核和主项 MAC，GN 另列；没有测量整网延迟。</p>`,(s,r)=>{
 const [ci,co,k,L]=presets[s.costPreset]||[s.costIn,s.costOut,+s.costKernel,s.costLength],q=D.cost(ci,co,k,L),px=v=>440*v/q.dense;
 r.querySelector('[data-cost]').innerHTML=`<svg viewBox="0 0 710 175" role="img" aria-label="普通卷积与深度可分离卷积核权重数量对比"><text x="5" y="43">普通卷积</text><rect x="125" y="21" width="440" height="30" rx="4" fill="var(--pamf)"/><text x="577" y="43">${q.dense}</text><text x="5" y="98">DW＋PW</text><rect x="125" y="76" width="${px(q.dw)}" height="30" fill="var(--accent)"/><rect x="${125+px(q.dw)}" y="76" width="${px(q.pw)}" height="30" fill="var(--blue)"/><text x="${137+px(q.separable)}" y="98">${q.separable}</text><text x="125" y="146">绿色 DW ${q.dw} ＋ 蓝色 PW ${q.pw}；同长度核权重占比 ${fmt(q.ratio*100,2)}%</text></svg>`;
 result(r,String.raw`当前 \(C_{\rm in}=${ci},C_{\rm out}=${co},k=${k},L'=${L}\)。<br>\(P_{\rm ordinary}=${k}\times${ci}\times${co}=${q.dense}\)；\(P_{\rm DS}=${q.dw}+${q.pw}=${q.separable}\)。核权重${q.ratio<=1?'减少':'增加'} <b>${fmt(Math.abs(1-q.ratio)*100,2)}%</b>。`);
 r.querySelector('[data-table]').innerHTML=table(['项目','普通卷积','DW＋PW'],[['卷积核权重',q.dense,q.separable],['加上同宽 GN 的仿射参数',q.dense+2*co,q.withNorm],['卷积 MAC（每条样本）',q.macDense,q.macDS],['MAC 比例',1,fmt(q.ratio,6)]]);
 });
});
register('fft-detail',root=>mount(root,'跟踪一个复数频点：相位旋转会改变功率吗',`<div class="lab-grid"><div><p class="lab-caption">时域：I、Q 两个分量</p><div data-iq></div></div><div><p class="lab-caption">完整频谱功率（已按负→正频率排列）</p><div data-fft></div></div></div><div class="lab-controls">${slider('fftPositive','正频复指数幅度',0,2,1,.05)}${slider('fftNegative','负频复指数幅度',0,2,.5,.05)}${slider('fftPhase','整体相位',-180,180,0,5,'°')}${slider('fftInspect','检查 FFT 索引 k',0,15,2)}</div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">缩小到 N=16、频点±2，采样率仍设25 MHz。直接用 DFT 逐项求和，不加窗、没有1/N归一化，与当前raw_fft约定一致；实际包长为256。</p>`,(s,r)=>{
 const N=16,phi=s.fftPhase*Math.PI/180,z=Array.from({length:N},(_,n)=>{const a=2*Math.PI*2*n/N;return[s.fftPositive*Math.cos(a+phi)+s.fftNegative*Math.cos(-a+phi),s.fftPositive*Math.sin(a+phi)+s.fftNegative*Math.sin(-a+phi)];}),f=M.dft(z.map(v=>v[0]),z.map(v=>v[1])),power=f.r.map((v,i)=>v*v+f.i[i]*f.i[i]),order=Array.from({length:N},(_,i)=>(i+N/2)%N),k=s.fftInspect;
 linePlot(r.querySelector('[data-iq]'),[{values:z.map(v=>v[0])},{values:z.map(v=>v[1])}],{min:-4,max:4,label:'16点合成IQ'});
 linePlot(r.querySelector('[data-fft]'),[{values:order.map(i=>power[i])}],{min:0,max:1024,xLabel:i=>i-8,label:'正负频率完整功率谱'});
 result(r,String.raw`\(X[${k}]=${fmt(f.r[k])}+j(${fmt(f.i[k])})\)，\(P[${k}]=${fmt(power[k])}\)。两条理想谱线：\(P[2]=(16a_+)^2=${fmt(power[2],2)}\)，\(P[14]=(16a_-)^2=${fmt(power[14],2)}\)。<br>改变整体相位，复数坐标旋转，功率保持；交换正负幅度，左右谱线交换。此例频率间隔 \(25/16=1.5625\,\mathrm{MHz}\)，真实256点间隔为 \(97.65625\,\mathrm{kHz}\)。`);
 r.querySelector('[data-table]').innerHTML=table(['n','输入实部／虚部','该点对 X[k] 的实部贡献','虚部贡献'],[0,1,2,3].map(n=>{const a=-2*Math.PI*k*n/N,c=Math.cos(a),d=Math.sin(a);return[n,arr(z[n]),fmt(z[n][0]*c-z[n][1]*d),fmt(z[n][0]*d+z[n][1]*c)];}));
}));
register('frequency-channels',root=>mount(root,'同一对正负功率，为什么要保留四种描述',`<div class="lab-controls">${select('mirrorCase','功率设置',[['manual','手动调整'],['silence','全零频点'],['extreme','大动态范围：10⁸ 与 10⁻⁸']])}${slider('mirrorPos','手动正侧功率',0,20,9,.1)}${slider('mirrorNeg','手动负侧功率',0,20,1,.1)}</div><div data-four class="four-values"></div><div class="lab-result" data-result></div><p class="lab-caption">自然对数，ε=10⁻⁸。log1p通道的输入上限10⁶；logR用原功率加ε后相减，再裁到±20。这里只演示确定性特征，不假装执行学到的门控。</p>`,(s,r)=>{
 const [p,n]=s.mirrorCase==='silence'?[0,0]:s.mirrorCase==='extreme'?[1e8,1e-8]:[s.mirrorPos,s.mirrorNeg],v=D.mirror([p],[n])[0],swap=D.mirror([n],[p])[0];
 r.querySelector('[data-four]').innerHTML=v.map((x,i)=>`<div><span>${['正侧 log1p','负侧 log1p','有符号 logR','无符号不对称'][i]}</span><b>${fmt(x,6)}</b></div>`).join('');
 result(r,String.raw`\(P^+=${p},\ P^-=${n}\)。交换两侧后，四通道变为 \((${swap.map(x=>fmt(x,5)).join(',')})\)。<br>前两项交换，第三项反号，第四项不变。${s.mirrorCase==='extreme'?'当前 log1p 与比值已经触发不同的数值保护，不能再直接代入未裁剪的简化等式。':''}`);
}));
register('frequency-pooling',root=>mount(root,'127 个频点怎样变成 32 个位置：看真实池化边界',`<div data-pool class="detail-svg"></div><div class="lab-controls">${slider('poolBand','选择输出位置 b（从 0 开始）',0,31,0)}${select('poolSignal','输入曲线',[['ramp','递增：u[j]=j+1'],['alternating','交替功率：1 与 9'],['peak','单峰：第64个频点为100']])}</div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">当前分支使用PyTorch AdaptiveAvgPool1d的分段平均，不是插值，也不是每次恰好取4点。相邻分段在非整数边界处可共享输入点。</p>`,(s,r)=>{
 const L=127,K=32,b=s.poolBand,values=Array.from({length:L},(_,j)=>s.poolSignal==='ramp'?j+1:s.poolSignal==='alternating'?(j%2?9:1):(j===63?100:1)),a=D.poolRange(b,L,K),next=D.poolRange(Math.min(b+1,31),L,K),v=D.pool(values,K),shared=b<31?Array.from({length:Math.max(0,a[1]-next[0])},(_,i)=>next[0]+i):[];
 const mx=Math.max(...values),px=i=>25+i*5.5;let svg=`<svg viewBox="0 0 760 230" role="img" aria-label="127点到32点平均池化的输入覆盖与重叠"><text x="24" y="24">输入频率点 j：高亮当前输出位置对应的区间</text>`;
 values.forEach((q,j)=>{const chosen=j>=a[0]&&j<a[1],over=shared.includes(j);svg+=`<rect x="${px(j)}" y="${160-q/mx*100}" width="4" height="${q/mx*100}" fill="${over?'var(--pamf)':chosen?'var(--accent)':'var(--line)'}"/>`;});
 svg+=`<path d="M25,161H724" class="axis"/><text x="25" y="187">j=0</text><text x="690" y="187">126</text><text x="25" y="218">本区间 [${a[0]},${a[1]})，${a[1]-a[0]} 点；橙色为与下一输出位置共用的输入点。</text></svg>`;
 r.querySelector('[data-pool]').innerHTML=svg;
 const avgLog=mean(values.slice(...a).map(Math.log1p)),logAvg=Math.log1p(mean(values.slice(...a)));
 result(r,String.raw`\(s_b=\lfloor${b}\cdot127/32\rfloor=${a[0]}\)，\(e_b=\lceil${b+1}\cdot127/32\rceil=${a[1]}\)。<br>\(\bar u_b=\frac1{${a[1]-a[0]}}\sum_{j=${a[0]}}^{${a[1]-1}}u_j=${fmt(v[b])}\)。与下一位置共享索引：<b>${b===31?'末段，没有下一段':shared.length?shared.join(', '):'无'}</b>。<br>本段“先log1p后平均”为 \(${fmt(avgLog,6)}\)；“先平均后log1p”为 \(${fmt(logAvg,6)}\)。${Math.abs(avgLog-logAvg)>1e-10?'两者不同，实际网络采用前者。':'该段恰好相等；非恒定输入一般不相等。'}`);
 r.querySelector('[data-table]').innerHTML=table(['输入索引j','实际±频点编号k=j+1','值','当前平均权重'],values.slice(...a).map((q,i)=>[a[0]+i,a[0]+i+1,fmt(q,2),String.raw`\(1/${a[1]-a[0]}\)`]));
}));
register('pa-basis-detail',root=>mount(root,'选一个采样点，看 4 个延迟 × 3 个阶次',`<div class="lab-controls">${slider('basisAmplitude','对整条教学序列的幅度缩放',0,2,1,.05)}${slider('basisPhase','整体旋转',-180,180,0,5,'°')}${slider('basisIndex','观察原序列位置 n',0,5,0)}${slider('basisDelay','高亮延迟 m',0,3,0)}${select('basisOrder','高亮阶次',[['1','1阶'],['3','3阶'],['5','5阶']])}</div><div data-basis-plane></div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">基础六点为(0.3+0.4j)、(0.6+0.8j)、(1.2+1.6j)、(−2.8+0.5j)、(0.2−0.7j)、0。先旋转/缩放，再按源码分别裁剪I、Q到±2；不加载PA卷积权重。</p>`,(s,r)=>{
 const base=[[.3,.4],[.6,.8],[1.2,1.6],[-2.8,.5],[.2,-.7],[0,0]],theta=s.basisPhase*Math.PI/180,x=base.map(([a,b])=>[(a*Math.cos(theta)-b*Math.sin(theta))*s.basisAmplitude,(a*Math.sin(theta)+b*Math.cos(theta))*s.basisAmplitude]),n=s.basisIndex,m=s.basisDelay,p=+s.basisOrder,q=D.lift(x,m,p,n),orders=[1,3,5],v=orders.map(o=>D.lift(x,m,o,n).value),lim=Math.max(1,...v.flat().map(Math.abs))*1.2;
 plane(r.querySelector('[data-basis-plane]'),v.map((a,i)=>({points:[[0,0],a],cls:['series-a','series-b','series-c'][i]})),v.map((a,i)=>({x:a[0],y:a[1],label:orders[i]+'阶'})),{min:-lim,max:lim,label:'不同阶数的复数基向量'});
 result(r,String.raw`选中 \(n=${n},m=${m},p=${p}\)，读的是 \(x[n-m]=x[${n-m}]\)${n<m?'（左越界补零）':''}。<br>裁剪后 \(\tilde x=${fmt(q.z[0])}+j(${fmt(q.z[1])})\)，保护后的平方幅度 \(r^2=${fmt(q.r2,8)}\)，倍率 \(s_p=${fmt(q.scale,6)}\)。<br>输出 \(b_{m,p}=${fmt(q.value[0],6)}+j(${fmt(q.value[1],6)})\)。此处基展开没有可训练系数；得到12个复数＝24个实数通道。`);
 r.querySelector('[data-table]').innerHTML=table(['延迟 m','阶 p','实通道编号','实部','虚部'],Array.from({length:4},(_,delay)=>orders.map((order,j)=>{const z=D.lift(x,delay,order,n).value;return[delay,order,2*(delay*3+j)+' / '+(2*(delay*3+j)+1),fmt(z[0],5),fmt(z[1],5)];})).flat());
}));
register('pa-dilation',root=>mount(root,'膨胀卷积：跳着取点，核权重个数仍然相同',`<div data-dilation class="detail-svg"></div><div class="lab-controls">${select('dilationKernel','单层核长',[['5','5'],['7','7']])}${slider('dilationRate','单层膨胀率 d',1,4,2)}${slider('dilationCenter','窗口中心 t',0,31,16)}</div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">下图只画单层采样几何。真实PA块还做GroupNorm、SiLU和池化；GroupNorm跨时间统计，因此局部卷积足迹不等于整个网络的严格依赖范围。</p>`,(s,r)=>{
 const k=+s.dilationKernel,d=s.dilationRate,t=s.dilationCenter,indices=Array.from({length:k},(_,i)=>t+(i-Math.floor(k/2))*d),span=1+(k-1)*d,px=i=>24+i*22;
 let svg=`<svg viewBox="0 0 760 195" role="img" aria-label="单层膨胀卷积的离散采样位置"><text x="24" y="27">只在绿色位置读取特征；虚线跨度 ${span} 个位置</text>`;
 for(let i=0;i<32;i++)svg+=`<rect x="${px(i)}" y="110" width="19" height="32" rx="3" class="${indices.includes(i)?'detail-selected':'detail-cell'}"/><text x="${px(i)+9}" y="161" text-anchor="middle">${i}</text>`;
 indices.forEach((n,i)=>{if(n>=0&&n<32)svg+=`<path class="detail-link" d="M${px(t)+9},56L${px(n)+9},106"/><text x="${px(n)+9}" y="95" text-anchor="middle">w${i}</text>`;});
 svg+=`<circle cx="${px(t)+9}" cy="52" r="8" fill="var(--pamf)"/></svg>`;r.querySelector('[data-dilation]').innerHTML=svg;
 result(r,String.raw`\(k_{\rm effective}=1+(k-1)d=1+(${k}-1)\times${d}=${span}\)，实际权重仍为每输入输出通道对 <b>${k} 个</b>。采样位置：\((${indices.join(',')})\)；越界位置补零。<br>改变膨胀率会扩大单层跨度，但不会自动增加核参数。三层及两次池化的累计几何足迹如下。`);
 const rf=D.receptive();r.querySelector('[data-table]').innerHTML=table(['阶段','卷积后跨度r','池化后跨度r','相邻输出的原层间隔j'],rf.map((a,i)=>[['pa_b1：k7,d1,pool2','pa_b2：k7,d2,pool2','pa_b3：k5,d4,pool1'][i],a.afterConv,a.r,a.j]));
}));
register('pa-stat-detail',root=>mount(root,'给32个频带分配能量，逐项算出PA的3个统计量',`<div data-energy class="detail-svg"></div><div class="lab-controls">${select('statsShape','能量分布',[['bands','分区调整'],['flat','所有频带都是1'],['spike','中心一个尖峰，其余为1']])}${slider('statsEdge','最外侧6带的能量（或尖峰高度）',0,12,4,.1)}${slider('statsCenter','中间13带的能量',0,6,1,.1)}</div><div class="lab-result" data-result></div><div data-table></div><p class="lab-caption">输入已是pos/neg线性功率分别池化后的和T，32维。索引26至31共6带作为外侧，9至21共13带作为中间；这里的“中心”不是包含DC的完整中心频谱。没有执行3→160的训练投影。</p>`,(s,r)=>{
 const E=Array.from({length:32},(_,b)=>s.statsShape==='flat'?1:s.statsShape==='spike'?(b===15?s.statsEdge:1):b>=26?s.statsEdge:b>=9&&b<22?s.statsCenter:1),q=D.stats(E),max=Math.max(...q.E,1),px=i=>25+i*21.6;
 let svg=`<svg viewBox="0 0 760 230" role="img" aria-label="PA频谱统计的32带能量分区"><text x="25" y="24">蓝色：9…21；橙色：26…31；灰色：其他位置</text>`;
 q.E.forEach((v,b)=>{const color=b>=26?'var(--pamf)':b>=9&&b<22?'var(--blue)':'var(--line)',h=v/max*140;svg+=`<rect x="${px(b)}" y="${175-h}" width="17" height="${h}" rx="2" fill="${color}"/><text x="${px(b)+8}" y="197" text-anchor="middle">${b}</text>`;});svg+='</svg>';r.querySelector('[data-energy]').innerHTML=svg;
 result(r,String.raw`总能量 \(S=${fmt(q.total,6)}\)，外侧能量 \(E=${fmt(q.edge,6)}\)，中间能量 \(C=${fmt(q.center,6)}\)。<br>\(r_{\rm edge}=E/(S+10^{-8})=${fmt(q.pa[0],6)}\)；\(r_{\rm regrowth}=E/(C+10^{-8})=${fmt(q.pa[1],6)}\)；谱峰度 \(=${fmt(q.pa[2],6)}\)。外侧与中间带数不同，第二项是能量和的比，没有按6/13校正。`);
 r.querySelector('[data-table]').innerHTML=table(['峰度计算的中间量','数值'],[['均值 μ',fmt(q.mu,8)],['总体方差＋ε',fmt(q.variance,8)],['裁剪偏差后的四次幂均值',fmt(q.numerator,8)],['分母 max(方差²,ε)',fmt(q.denominator,8)],['最终谱峰度（再限0至10⁴）',fmt(q.pa[2],8)]]);
}));
window.COURSE_QUIZ.push(
 {q:'DW 与 PW 在当前 PAMF 中分别负责什么？',options:['DW 混通道，PW 看时间邻居','DW 分别看邻居，PW 在同一位置混合通道','两者都只处理一个通道','PW 只是固定平均'],correct:1,why:'DW 的分组数等于输入通道数；PW 的核长为1，但读取全部中间通道，权重可以学习。'},
 {q:'在 t1 中，48 个输入通道、72 个输出通道、核长5，DW＋PW 的卷积核权重是多少？',options:['240','3456','3696','17280'],correct:2,why:'DW 为48×5＝240，PW 为48×72＝3456，合计3696。再算GroupNorm的144个仿射参数，块总数才是3840。'},
 {q:'频域从127点平均池化到32点，哪一种描述符合实际？',options:['每组一定4点，而且完全不重叠','先平均功率，再统一取对数','按floor/ceil确定区间，边界可重叠；四通道先算再池化','把FFT插值成32点，再求功率'],correct:2,why:'首段含输入索引0至3，次段含3至7。log1p、log比与不对称先逐点计算，不可随意交换非线性与平均的顺序。'},
 {q:'PA 路径的24个实数通道与3个频谱统计是什么关系？',options:['同一组数字的两种写法','24通道是学出的物理功放系数','3个统计替代了24通道','两条并行描述，分别投影到160维后相加'],correct:3,why:'24＝4个延迟×3个阶次×实虚两部分，保留时间轴；3项统计来自32带线性功率。它们都不是已辨识的真实功放参数。'}
);
})();
