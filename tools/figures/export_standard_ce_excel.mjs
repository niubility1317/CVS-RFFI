// Build one research workbook from the frozen standard-CE score intermediates.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [inputFile,outputFile,figureDir,efficiencyFile,deploymentFile]=process.argv.slice(2);
if(!inputFile||!outputFile)throw new Error('Usage: node export_standard_ce_excel.mjs INPUT_JSON NEW_OUTPUT_XLSX');
try {await fs.access(outputFile);throw new Error('Output exists; preserve it and choose a new name.');} catch(e){if(e.code!=='ENOENT')throw e;}
const data=JSON.parse(await fs.readFile(inputFile,'utf8'));
const efficiency=efficiencyFile?JSON.parse(await fs.readFile(efficiencyFile,'utf8')):null;
const deployment=deploymentFile?JSON.parse(await fs.readFile(deploymentFile,'utf8')):null;
const raw=data.raw,tables=data.tables;
const progressMode=Boolean(raw.snapshot_hkt);
if(!progressMode&&(raw.run_id!=='20261010-phase1-standard-ce-baselines-manysig-m56-r01'||raw.rows.length!==56))throw new Error('Unexpected benchmark');
const source=tables['network_summary.csv'];
if(progressMode&&(source.length!==13||raw.rows.some(r=>r.row.variant==='native')))throw new Error('Excluded architecture present');
const order=[...source].sort((a,b)=>b.accuracy_pct-a.accuracy_pct).map(r=>r.variant);
const names=Object.fromEntries(source.map(r=>[r.variant,r.variant==='residual_fusion'?'CVS':r.network]));
const short={residual_fusion:'CVS',native:'Original CVS',resnet50:'R50-1D',cvresnet18_1d:'CR18-1D',
 resnet34:'R34-1D',resnet18:'R18-1D',densenet121:'Dense121',vgg16_bn:'VGG16-BN',bilstm3:'BiLSTM',
 resnet18_2d:'R18-2D',convnext_tiny:'ConvNeXt',resnet50_2d:'R50-2D',cvresnet18_2d:'CR18-2D',transformer6:'Transformer'};
const rows=order.flatMap(v=>raw.rows.filter(r=>r.row.variant===v).sort((a,b)=>a.row.model_seed-b.row.model_seed));
const wb=Workbook.create();
const sheetNames=['总体汇总',...(efficiency?['效率汇总']:[]),...(figureDir?['论文图表']:[]),'种子结果',...(progressMode?['实验进度']:[]),'接收机分层','发射机分层','配对差','资源测量',...(deployment?['部署测量','请求时延']:[]),'混淆矩阵','分层原始数据','实验说明'];
const sheets=Object.fromEntries(sheetNames.map(n=>[n,wb.worksheets.add(n)]));
const font='Microsoft YaHei';
const previews=path.join(path.dirname(outputFile),'excel_preview');await fs.mkdir(previews,{recursive:true});
function col(n){let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function table(sheet,headers,values,{start=1,widths=[],formats={},name}={}){
 const end=start+values.length,last=col(headers.length-1);
 const block=sheet.getRange(`A${start}:${last}${end}`);block.values=[headers,...values];
 block.format.font={name:font,size:10,color:'#263445'};block.format.rowHeight=22;
 sheet.getRange(`A${start}:${last}${start}`).format={fill:'#36556F',font:{name:font,size:10,bold:true,color:'#FFFFFF'},rowHeight:30,verticalAlignment:'center',horizontalAlignment:'center'};
 for(let c=0;c<headers.length;c++)sheet.getRange(`${col(c)}${start}:${col(c)}${end}`).format.columnWidth=widths[c]||18;
 for(const [c,fmt] of Object.entries(formats))sheet.getRange(`${col(Number(c))}${start+1}:${col(Number(c))}${end}`).setNumberFormat(fmt);
 const t=sheet.tables.add(`A${start}:${last}${end}`,true,name);t.showBandedRows=true;
 sheet.showGridLines=false;
 if(values.length>25)sheet.freezePanes.freezeRows(start);
 return {end,last};
}
function title(sheet,text,note){sheet.getRange('A2').values=[[text]];sheet.getRange('A2').format.font={name:font,size:15,bold:true,color:'#263445'};sheet.getRange('A2').format.rowHeight=26;
 if(note){sheet.getRange('A3').values=[[note]];sheet.getRange('A3').format.font={name:font,size:10,italic:true,color:'#596B78'};}}
function mean(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function sd(a){const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1));}

const seedValues=rows.map(r=>{const s=r.scores.results.find(x=>x.dimension==='overall');const rx=r.scores.results.filter(x=>x.dimension==='receiver');
 const worst=rx.reduce((a,b)=>a.accuracy<b.accuracy?a:b);const cm=s.confusion;const nonzero=cm[0].map((_,j)=>cm.reduce((s,row)=>s+row[j],0)).filter(x=>x>0).length;
 return [r.row.variant,names[r.row.variant],String(r.row.model_seed),s.query_count,s.accuracy,s.macro_f1,worst.accuracy,worst.stratum,
  source.find(x=>x.variant===r.row.variant).total_parameters,nonzero===1?'单类预测':'正常多类预测'];});
table(sheets['种子结果'],['variant','网络','model_seed','测试包数','准确率','Macro-F1','最差RX准确率','最差RX物理ID','总参数量','预测状态'],seedValues,
 {name:'SeedResults',widths:[23,28,18,16,15,15,19,19,17,22],formats:{2:'@',3:'#,##0',4:'0.00%',5:'0.00%',6:'0.00%',7:'@',8:'#,##0'}});

title(sheets['总体汇总'],'纯 CE 神经网络 clean 测试汇总',progressMode?`${raw.snapshot_hkt}；13 种网络；${rows.length} 个完成模型；每个模型 168000 个测试包。`:'历史 E200；14 种网络 × 4 个种子；每个模型 168000 个相同测试包。');
const sums=order.map(v=>{const s=source.find(x=>x.variant===v);return [names[v],s.total_parameters,null,null,null,null,null,null,null,
 progressMode?`${short[v]} (n=${s.actual_seed_count})`:short[v],...(progressMode?[s.original_completed_seeds,s.new_completed_seeds,s.expected_seed_count-s.actual_seed_count,s.completed_model_seeds]:[])];});
table(sheets['总体汇总'],['网络','总参数量','实际seed数','准确率均值','准确率SD（pp）','Macro-F1均值','Macro-F1 SD（pp）','最差RX均值','最差RX SD（pp）','绘图简称',...(progressMode?['原批完成数','新增完成数','尚缺seed数','已完成model_seed']:[])],sums,
 {start:5,name:'NetworkSummary',widths:[29,17,14,19,20,20,22,20,22,27,18,18,18,125],formats:{1:'#,##0',2:'0',3:'0.00%',4:'0.0000',5:'0.00%',6:'0.0000',7:'0.00%',8:'0.0000'}});
let firstSeedRow=2;
for(let i=0;i<order.length;i++){
 const r=6+i,a=firstSeedRow,n=rows.filter(x=>x.row.variant===order[i]).length,b=a+n-1;firstSeedRow=b+1;
 sheets['总体汇总'].getRange(`C${r}:I${r}`).formulas=[[
  `=COUNTA('种子结果'!C${a}:C${b})`,`=AVERAGE('种子结果'!E${a}:E${b})`,`=STDEV('种子结果'!E${a}:E${b})*100`,
  `=AVERAGE('种子结果'!F${a}:F${b})`,`=STDEV('种子结果'!F${a}:F${b})*100`,
  `=AVERAGE('种子结果'!G${a}:G${b})`,`=STDEV('种子结果'!G${a}:G${b})*100`]];
}
const endSummary=5+order.length;
const chart=sheets['总体汇总'].charts.add('bar',[sheets['总体汇总'].getRange(`J5:J${endSummary}`),sheets['总体汇总'].getRange(`D5:D${endSummary}`),sheets['总体汇总'].getRange(`F5:F${endSummary}`)]);
chart.title='Clean accuracy and macro-F1 (actual n in labels)';chart.barOptions.direction='column';chart.barOptions.grouping='clustered';chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=15;
chart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:11}};chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};
chart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};chart.yAxis.title.text='Score (%)';chart.setPosition('A23','K48');
chart.series.items[0].fill='#507398';chart.series.items[1].fill='#BB6C42';

function strata(sheet,filename,kind){
 const input=tables[filename];
 const values=order.flatMap(v=>input.filter(r=>r.variant===v)).map(r=>[r.variant,names[r.variant],r.stratum,r.query_count_per_seed,r.actual_seed_count,r.accuracy_pct/100,r.accuracy_sd_pp,r.macro_f1_pct/100,r.macro_f1_sd_pp]);
 table(sheet,['variant','网络',kind+'物理ID','每seed测试包数','seed数','准确率均值','准确率SD（pp）','Macro-F1均值','Macro-F1 SD（pp）'],values,
  {name:kind==='RX'?'ReceiverResults':'TransmitterResults',widths:[23,28,18,19,12,18,21,19,22],formats:{2:'@',3:'#,##0',4:'0',5:'0.00%',6:'0.0000',7:'0.00%',8:'0.0000'}});
 return input;
}
const receiver=strata(sheets['接收机分层'],'receiver_summary.csv','RX');strata(sheets['发射机分层'],'transmitter_summary.csv','TX');
const rxNames=receiver.filter(r=>r.variant==='residual_fusion').map(r=>r.stratum);
const rxVariants=progressMode?['residual_fusion']:['native','residual_fusion'];
const lastRxCol=progressMode?'M':'N';
const rxHelpers=[['Receiver',...rxVariants.map(v=>progressMode?`CVS (n=${source.find(s=>s.variant===v).actual_seed_count})`:names[v])],...rxNames.map(id=>[id,...rxVariants.map(()=>null)])];
const rxSheet=sheets['接收机分层'];rxSheet.getRange(`L24:${lastRxCol}31`).values=rxHelpers;rxSheet.getRange(`L24:${lastRxCol}31`).format.font={name:'Arial',size:10};rxSheet.getRange(`L24:${lastRxCol}31`).format.columnWidth=20;rxSheet.getRange(`M25:${lastRxCol}31`).setNumberFormat('0.00%');
for(let i=0;i<7;i++){const id=rxNames[i];for(const [j,v] of rxVariants.entries()){
 const targetRow=order.flatMap(x=>receiver.filter(r=>r.variant===x)).findIndex(r=>r.variant===v&&r.stratum===id)+2;
 rxSheet.getRange(`${j===0?'M':'N'}${25+i}`).formulas=[[`=F${targetRow}`]];
}}
const rxChart=rxSheet.charts.add('bar',rxSheet.getRange(`L24:${lastRxCol}31`));rxChart.title='CVS accuracy by held-out receiver';rxChart.barOptions.direction='column';rxChart.barOptions.grouping='clustered';rxChart.titleTextStyle.typeface='Arial';rxChart.titleTextStyle.fontSize=15;
rxChart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:11}};rxChart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:11}};rxChart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};rxChart.setPosition('L2','T21');rxChart.series.items[0].fill=progressMode?'#BB6C42':'#626D78';if(!progressMode)rxChart.series.items[1].fill='#BB6C42';

const paired=tables['paired_differences.csv'].map(r=>[r.control,names[r.control],r.baseline,names[r.baseline],String(r.model_seed),r.accuracy_difference_pp,r.macro_f1_difference_pp]);
table(sheets['配对差'],['对照variant','对照网络','基准variant','基准网络','model_seed','准确率差（pp）','Macro-F1差（pp）'],paired,
 {name:'PairedDifferences',widths:[24,28,24,28,18,22,23],formats:{4:'@',5:'+0.0000;-0.0000;0.0000',6:'+0.0000;-0.0000;0.0000'}});
sheets['配对差'].getRange(`F2:G${paired.length+1}`).conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{font:{color:'#B64342'}}});
const resources=tables['resource_profiles.csv'];
const efficiencyRows=new Map((efficiency?.rows||[]).map(r=>[`${r.variant}:${r.model_seed}`,r]));
if(efficiency&&(efficiencyRows.size!==resources.length||resources.some(r=>!efficiencyRows.has(`${r.variant}:${r.model_seed}`))))throw new Error('Resource rows do not match efficiency input');
const rv=resources.map(r=>[r.variant,names[r.variant],String(r.model_seed),r.hardware,r.total_parameters,r.trainable_parameters,r.gradient_used_parameters,r.resident_state_bytes,r.training_elapsed_seconds,r.training_peak_cuda_allocated_bytes,r.inference_batch1_ms,r.inference_batch128_ms,r.training_batch128_ms,r.full_clean_inference_seconds,r.full_clean_peak_cuda_allocated_bytes,r.conv_linear_macs_per_sample,r.fft_calls_per_sample,
 ...(efficiency?[efficiencyRows.get(`${r.variant}:${r.model_seed}`).matrix_macs,null,null,null,null,null,null,null]:[])]);
table(sheets['资源测量'],['variant','网络','model_seed','GPU','总参数量','可训练参数量','梯度使用参数量','常驻状态（byte）','E200耗时（s）','训练峰值显存（byte）','B1推理（ms）','B128推理（ms）','B128训练（ms）','clean全量推理（s）','clean峰值显存（byte）','原Conv/Linear部分MACs','FFT调用数',
 ...(efficiency?['校正卷积/矩阵MACs','卷积/矩阵FLOPs','B128吞吐（包/s）','B128均摊（µs/包）','常驻状态（MiB）','clean峰值（MiB）','训练峰值（MiB）','FP32权重（MiB）']:[])],rv,
 {name:'ResourceProfiles',widths:[23,28,18,36,19,20,22,24,21,27,21,23,23,24,28,28,20,28,26,25,25,25,25,25,25],formats:{2:'@',4:'#,##0',5:'#,##0',6:'#,##0',7:'#,##0',8:'0.000',9:'#,##0',10:'0.000',11:'0.000',12:'0.000',13:'0.000',14:'#,##0',15:'#,##0',16:'0',...(efficiency?{17:'#,##0',18:'#,##0',19:'0.000',20:'0.000',21:'0.000',22:'0.000',23:'0.000',24:'0.000'}:{})}});
if(efficiency){
 sheets['资源测量'].getRange(`S2:Y${rv.length+1}`).formulas=rv.map((_,i)=>{const r=i+2;return [`=R${r}*2`,`=128000/L${r}`,`=1000*L${r}/128`,`=H${r}/1048576`,`=O${r}/1048576`,`=J${r}/1048576`,`=E${r}*4/1048576`];});
 const es=sheets['效率汇总'];title(es,'13 种网络的性能与资源效率',`同一 RTX 3090 / FP32；每种网络 8 个种子；全部结果保留。CVS 实际为 164225 参数（0.164 M）。`);
 table(es,['网络','实际seed数','参数（M）','常驻状态（MiB）','卷积/矩阵MACs（M）','卷积/矩阵FLOPs（M）','FFT调用/包','B1时延均值（ms）','B1时延SD（ms）','B128时延均值（ms）','B128时延SD（ms）','B128吞吐均值（千包/s）','B128吞吐SD（千包/s）','clean峰值显存（MiB）','训练峰值显存（MiB）','E200耗时（min）','clean全量流程（s）','全量吞吐（千包/s）','准确率','Macro-F1','绘图简称'],
  order.map(v=>[names[v],...Array(19).fill(null),`${short[v]} (n=8)`]),
  {start:5,name:'EfficiencySummary',widths:[29,16,18,25,29,30,19,26,24,29,27,31,30,31,31,24,26,29,19,19,27],formats:Object.fromEntries(Array.from({length:20},(_,i)=>[i,i===0?'@':i===1||i===6?'0':i===18||i===19?'0.00%':'0.000']))});
 for(const [i,v] of order.entries()){
  const r=i+6,ix=resources.flatMap((x,j)=>x.variant===v?[j+2]:[]);
  const refs=c=>ix.map(j=>`'资源测量'!${c}${j}`).join(',');
  const avg=c=>`AVERAGE(${refs(c)})`,std=c=>`STDEV(${refs(c)})`;
  es.getRange(`B${r}:T${r}`).formulas=[[
   `=COUNTA(${refs('C')})`,`=${avg('E')}/1000000`,`=${avg('V')}`,`=${avg('R')}/1000000`,`=${avg('S')}/1000000`,`=${avg('Q')}`,
   `=${avg('K')}`,`=${std('K')}`,`=${avg('L')}`,`=${std('L')}`,`=${avg('T')}/1000`,`=${std('T')}/1000`,
   `=${avg('W')}`,`=${avg('X')}`,`=${avg('I')}/60`,`=${avg('N')}`,`=AVERAGE(${ix.map(j=>`168000/'资源测量'!N${j}`).join(',')})/1000`,
   `='总体汇总'!D${r}`,`='总体汇总'!F${r}`]];
 }
 const efficiencyNotes=[
  '均值与样本 SD 均使用实际 8 个模型种子；吞吐率先逐种子计算再汇总。B128 均摊时延不是单包请求时延。',
  '时延来自既有同硬件资源测量：预热 5 次、20 次前向、CUDA 同步；历史共享 GPU 环境，不能当作独占硬件基准。',
  '卷积/矩阵 FLOPs = 2 × MACs。已补齐 BiLSTM 循环矩阵和 Transformer 融合注意力；原部分计数保留在资源测量 P 列。',
  '该 FLOPs 不含 FFT、归一化、激活、池化、bias 与物理逐元素计算。CVS 每包 2 次 FFT；该数值不是完整网络 FLOPs。',
  '峰值为 CUDA allocated memory，不是进程总显存；MiB=1048576 bytes。clean 全量流程含输入传输、CPU 回读与结果序列化。',
  'CVS 的准确率和参数量优势由实测支持；单包时延及显存并非所有指标最优，保留更快或更省显存的基线。'];
 for(const [i,note] of efficiencyNotes.entries()){es.getRange(`A${22+i}`).values=[[note]];es.getRange(`A${22+i}`).format.font={name:font,size:10,color:'#596B78'};}
 const makeChart=(fields,heading,unit,start,end)=>{
  const ch=es.charts.add('bar',[es.getRange(`U5:U${endSummary}`),...fields.map(c=>es.getRange(`${c}5:${c}${endSummary}`))]);
  ch.title=heading;ch.barOptions.direction='column';ch.barOptions.grouping='clustered';ch.titleTextStyle.typeface='Arial';ch.titleTextStyle.fontSize=15;
  ch.legend={position:'top',textStyle:{typeface:'Arial',fontSize:10}};ch.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};
  ch.yAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};ch.yAxis.title.text=unit;ch.setPosition(start,end);
  ch.series.items[0].fill='#0072B2';if(fields.length>1)ch.series.items[1].fill='#D55E00';
 };
 makeChart(['H','J'],'Inference latency: B1 and B128','ms (lower is better)','A30','K55');
 makeChart(['L'],'Batch-128 throughput','Thousand packets/s (higher is better)','A58','K83');
}
if(deployment){
 if(!efficiency||deployment.rows.length!==208||deployment.trials.length!==41600||deployment.summary.length!==26)throw Error('Incomplete deployment matrix');
 const dr=deployment.rows,dt=deployment.trials;
 table(sheets['请求时延'],['variant','model_seed','批大小','请求序号','端到端时延（ms）'],
  dt.map(r=>[r.variant,String(r.model_seed),r.batch_size,r.trial,r.end_to_end_ms]),
  {name:'DeploymentRequests',widths:[25,18,15,18,26],formats:{1:'@',2:'0',3:'0',4:'0.000000'}});
 table(sheets['部署测量'],['variant','网络','model_seed','批大小','请求数','预热次数','端到端均值（ms）','请求时延SD（ms）','P50（ms）','P95（ms）','P99（ms）','GPU alloc峰值（byte）','GPU reserved峰值（byte）','CPU基线RSS（byte）','CPU推理峰值RSS（byte）','FP32存储（byte）','权重及buffer（byte）','总参数量'],
  dr.map(r=>[r.variant,names[r.variant],String(r.model_seed),r.batch_size,r.trials,r.warmups,...Array(5).fill(null),r.gpu_peak_allocated_bytes,r.gpu_peak_reserved_bytes,r.cpu_baseline_rss_bytes,r.cpu_inference_peak_rss_bytes,r.model_storage_bytes,r.resident_state_bytes,r.total_parameters]),
  {name:'DeploymentMeasurements',widths:[25,29,18,15,16,17,25,27,20,20,20,30,32,28,30,26,30,20],formats:{2:'@',3:'0',4:'0',5:'0',6:'0.000000',7:'0.000000',8:'0.000000',9:'0.000000',10:'0.000000',11:'#,##0',12:'#,##0',13:'#,##0',14:'#,##0',15:'#,##0',16:'#,##0',17:'#,##0'}});
 let first=2;
 for(const [i,r] of dr.entries()){
  const a=first,b=first+r.trials-1;first=b+1;
  const range=`'请求时延'!E${a}:E${b}`;
  sheets['部署测量'].getRange(`G${i+2}:K${i+2}`).formulas=[[`=AVERAGE(${range})`,`=STDEV(${range})`,`=PERCENTILE(${range},0.5)`,`=PERCENTILE(${range},0.95)`,`=PERCENTILE(${range},0.99)`]];
 }
 const es=sheets['效率汇总'];es.getRange('A86').values=[['新增：端到端时延、峰值内存/显存与部署模型存储']];es.getRange('A86').format.font={name:font,size:14,bold:true,color:'#263445'};
 const groups=order.flatMap(v=>[1,128].map(batch=>({v,batch})));
 table(es,['网络','实际seed数','批大小','请求数','端到端均值ms','seed均值SD ms','P50 ms','P95 ms','P99 ms','GPU alloc峰值MiB','GPU reserved峰值MiB','CPU峰值RSS MiB','CPU基线RSS均值MiB','FP32存储MiB','权重及buffer MiB','端到端吞吐 包/s'],
  groups.map(g=>[names[g.v],...Array(15).fill(null)]),
  {start:88,name:'DeploymentSummary',formats:Object.fromEntries(Array.from({length:16},(_,i)=>[i,i===0?'@':i<4?'0':'0.000']))});
 // Preserve existing column widths: the new table uses the established A:P layout.
 const originalWidths=[29,16,18,25,29,30,19,26,24,29,27,31,30,31,31,24];
 originalWidths.forEach((width,i)=>{es.getRange(`${col(i)}88:${col(i)}114`).format.columnWidth=width;});
 es.getRange('A88:P88').format.wrapText=true;es.getRange('A88:P88').format.rowHeight=42;
 for(const [i,g] of groups.entries()){
  const ix=dr.flatMap((r,j)=>r.variant===g.v&&r.batch_size===g.batch?[j+2]:[]);
  const refs=c=>ix.map(j=>`'部署测量'!${c}${j}`).join(',');
  const positions=dt.flatMap((r,j)=>r.variant===g.v&&r.batch_size===g.batch?[j+2]:[]);
  const range=`'请求时延'!E${positions[0]}:E${positions.at(-1)}`,r=i+89;
  es.getRange(`B${r}:P${r}`).formulas=[[
   `=COUNTA(${refs('C')})`,`='部署测量'!D${ix[0]}`,`=COUNT(${range})`,`=AVERAGE(${range})`,`=STDEV(${refs('G')})`,
   `=PERCENTILE(${range},0.5)`,`=PERCENTILE(${range},0.95)`,`=PERCENTILE(${range},0.99)`,
   `=MAX(${refs('L')})/1048576`,`=MAX(${refs('M')})/1048576`,`=MAX(${refs('O')})/1048576`,`=AVERAGE(${refs('N')})/1048576`,
   `=AVERAGE(${refs('P')})/1048576`,`=AVERAGE(${refs('Q')})/1048576`,`=C${r}*1000/E${r}`]];
 }
 const notes=[
  `新实测：RTX3090 / FP32，TF32关闭，GPU0单模型串行；PyTorch ${deployment.notes.torch_version}；每个权重独立进程、CPU 2线程。`,
  ...(deployment.notes.cpu_model?[`CPU：${deployment.notes.cpu_model}；GPU驱动 ${deployment.notes.driver}；Linux ${deployment.notes.kernel}。`]:[]),
  '端到端计时：CPU IQ缓冲构造、传入GPU、完整模型前向（包含FFT等内部操作）、有限值检查、argmax及标签回读CPU。',
  '输入为统一的合成已同步、单位RMS IQ；不包含射频采集/同步前端、磁盘、网络或等待凑批/排队，不读取测试集或truth。',
  '各权重预热30次，B1计时300次、B128计时100次；网络P95/P99合并8个权重的等量请求计算，均值SD为8个权重均值的SD。',
  '显存显示8个权重的最大推理峰值，分别列PyTorch allocated和reserved；不等于驱动/NVIDIA进程总显存。',
  'CPU RSS在单独50次推理窗口每1ms采样，显示8个权重最大值；包括解释器、框架和运行时，可能漏掉更窄瞬时峰值。',
  '模型存储为实际torch.save(FP32 state_dict)序列化字节，包含buffer和格式开销；不含训练状态、优化器或模型代码，未量化。',
  '此前纯前向时延和显存记录保留在上表；新测量范围及资源环境不同，应在各自口径内比较。原12工作表、8张论文图及4张可编辑图表保留。'];
 notes.forEach((note,i)=>{es.getRange(`A${118+i}`).values=[[note]];es.getRange(`A${118+i}`).format.font={name:font,size:10,color:'#596B78'};});
}
const conf=[];
for(const r of rows){const cm=r.scores.results.find(s=>s.dimension==='overall').confusion;for(let i=0;i<6;i++)for(let j=0;j<6;j++)conf.push([r.row.variant,names[r.row.variant],String(r.row.model_seed),raw.classes[i],raw.classes[j],cm[i][j],null,cm[i].reduce((a,b)=>a+b,0)]);}
table(sheets['混淆矩阵'],['variant','网络','model_seed','真实TX','预测TX','样本数','行内比例','真实类总包数'],conf,
 {name:'ConfusionCounts',widths:[23,28,18,16,16,18,18,21],formats:{2:'@',3:'@',4:'@',5:'#,##0',6:'0.00%',7:'#,##0'}});
sheets['混淆矩阵'].getRange(`G2:G${conf.length+1}`).formulas=conf.map((_,i)=>[`=F${i+2}/H${i+2}`]);
const detail=tables['per_seed_metrics.csv'];
table(sheets['分层原始数据'],['row_id','variant','网络','model_seed','view','分层维度','物理ID','测试包数','准确率','Macro-F1'],
 detail.map(r=>[r.row_id,r.variant,names[r.variant],String(r.model_seed),r.view,r.dimension,r.stratum,r.query_count,r.accuracy_pct/100,r.macro_f1_pct/100]),
 {name:'AllScoredMetrics',widths:[40,23,28,18,14,20,17,19,18,18],formats:{3:'@',6:'@',7:'#,##0',8:'0.00%',9:'0.00%'}});

const notes=[
 ['run_id',raw.run_id],['实验范围',progressMode?`固定 E200，13 种网络，${rows.length} 个已完成且独立评分的模型；仅 clean 测试。`:'历史 E200、14 种网络、4 个模型种子；仅 clean 测试。'],
 ['模型种子',progressMode?'计划2026092701至2026092708；各网络实际完成清单见总体汇总和实验进度，split seed 固定392005。':'2026092701至2026092704；split seed 固定392005。'],
 ['数据角色','ManySig；源 RX1/3/4/6/8，day1/2/3；L_s 6300、U_s 56700（不使用）、V 27000（仅验证）。'],
 ['训练','从零训练 200 epoch、10000 次更新；只有身份 CE。固定 E200 checkpoint；无增强或额外损失。'],
 ['测试','同一批 168000 个物理包；7 个未见 RX 各 24000 包；6 个 TX 各 28000 包；逐样本全部 6 类竞争。'],
 ['误差定义','SD 为实际完成模型种子的样本标准差（ddof=1），单位 pp 为百分点；不是标准误或置信区间。'],
 ['最差RX','每个种子先取 7 个 RX 的最低准确率，再计算已完成种子的均值和 SD；并非先平均后取最低。'],
 ['退化结果',`Transformer 的单类预测种子：${seedValues.filter(r=>r[0]==='transformer6'&&r[9]==='单类预测').map(r=>r[2]).join('、')}；准确率 16.6667%、Macro-F1 4.7619%，完整保留。`],
 ['配对差','每条记录为同 model seed 下对照网络减基准网络；只使用双方均已完成评分的种子，单位为百分点。'],
 ['混淆矩阵','每行是真实类与预测类的样本计数；行内比例=样本数/该真实类总包数。TX分层Macro-F1为单真实类子集上的6类宏平均，不是全测试集的该类F1。'],
 ['资源口径',efficiency?'RTX3090/FP32；历史共享GPU测量，预热5次、20次同步前向。卷积/矩阵FLOPs=2×MACs，补齐循环/注意力矩阵；不含FFT等非矩阵操作，CVS有2次FFT/包。完整说明见效率汇总。':'RTX3090/FP32；E200耗时含源V评估。MACs仅含Conv/Linear；显存与状态用byte。'],
 ['比较边界','已暴露代理基准、统一固定训练预算；2D采用16×16 IQ reshape。未提供显著性检验或星地鲁棒性结论。'],
 ['当前种子数',progressMode?`${raw.snapshot_hkt}快照；统计只包含已完成评分种子。CVS 指 residual_fusion。尚缺种子不填零、不剔除负结果。`:'原始4个种子的完成结果；CVS指residual_fusion。'],
 ['数据来源','network_summary.csv、per_seed_metrics.csv、receiver_summary.csv、transmitter_summary.csv、paired_differences.csv、resource_profiles.csv 与冻结评分JSON。'],
 ['使用','总体汇总由种子结果的Excel公式计算，可编辑图表随引用单元格更新。论文图表页为同一快照的高分辨率图片，完整矢量图另行提供。']];
table(sheets['实验说明'],['项目','说明'],notes,{name:'ExperimentContext',widths:[23,116]});sheets['实验说明'].getRange('B2:B17').format.wrapText=true;sheets['实验说明'].getRange('A2:B17').format.rowHeight=44;

if(progressMode){
 const statusText={SCORED:'clean已独立评分',TRAINING:'训练中',PREDICTING_CLEAN:'clean预测中',FROZEN_QUEUED_FOR_CLEAN:'等待clean预测',QUEUED:'等待训练'};
 const progress=raw.progress.map(r=>[r.variant,names[r.variant],String(r.model_seed),r.model_seed<=2026092704?'原批':'新增',statusText[r.status]||r.status,r.epoch,r.steps??(r.status==='SCORED'?10000:null),r.status==='SCORED'?'已纳入':'未纳入']);
 table(sheets['实验进度'],['variant','网络','model_seed','批次','状态','当前epoch','累计更新数','统计范围'],progress,
  {name:'ExperimentProgress',widths:[23,31,18,14,23,17,20,18],formats:{2:'@',5:'0',6:'#,##0'}});
}
let figureCount=0;
if(figureDir){
 const manifest=JSON.parse(await fs.readFile(path.join(figureDir,'manifest.json'),'utf8'));
 const paper=sheets['论文图表'];paper.showGridLines=false;paper.getRange('A1:N450').format.rowHeight=24;
 paper.getRange('A1:N450').format.columnWidth=13;title(paper,'纯 CE clean 测试论文图表',`${raw.snapshot_hkt}；图内 n 为实际完成种子数。`);
 const titles=['总体准确率与Macro-F1','接收机准确率热图','参数量与准确率：单图压缩左轴','架构配对差','CVS接收机与发射机表现','CVS与ResNet-50混淆矩阵','推理时延与吞吐率','参数、卷积/矩阵FLOPs与显存成本'];
 let row=6;
 for(const [i,fig] of manifest.figures.entries()){
  paper.getRange(`A${row}`).values=[[`图 ${i+1}：${titles[i]}`]];paper.getRange(`A${row}`).format.font={name:font,size:13,bold:true,color:'#263445'};
  const bytes=await fs.readFile(path.join(figureDir,'png',fig.stem+'.png'));
  const width=1180,height=width*fig.height_mm/fig.width_mm;
  paper.images.add({dataUrl:'data:image/png;base64,'+bytes.toString('base64'),anchor:{from:{row:row+1,col:0},extent:{widthPx:width,heightPx:height}}});
  row+=Math.ceil(height/24)+6;figureCount++;
 }
}

wb.recalculate();
// Check an actual input edit, then restore the frozen score before export.
const seedCell=sheets['种子结果'].getRange('E2');const originalValue=seedCell.values[0][0];
const before=sheets['总体汇总'].getRange('D6').values[0][0];
seedCell.values=[[originalValue+0.001]];wb.recalculate();
const firstCount=rows.filter(r=>r.row.variant===order[0]).length;
if(Math.abs(sheets['总体汇总'].getRange('D6').values[0][0]-before-0.001/firstCount)>1e-10)throw new Error('Summary did not recalculate');
seedCell.values=[[originalValue]];wb.recalculate();
const summaryCells=sheets['总体汇总'].getRange(`C6:I${endSummary}`).values;
for(let i=0;i<order.length;i++){const expected=source.find(r=>r.variant===order[i]);const actual=summaryCells[i];
 const exp=[expected.actual_seed_count,expected.accuracy_pct/100,expected.accuracy_sd_pp,expected.macro_f1_pct/100,expected.macro_f1_sd_pp,expected.worst_receiver_accuracy_pct/100,expected.worst_receiver_sd_pp];
 for(let j=0;j<7;j++)if(typeof actual[j]!=='number'||Math.abs(actual[j]-exp[j])>1e-9)throw new Error(`Summary mismatch: ${order[i]} column ${j}: ${actual[j]} expected ${exp[j]}`);}
if(efficiency){
 const cells=sheets['效率汇总'].getRange(`B6:T${endSummary}`).values;
 for(const [i,v] of order.entries()){
  const e=efficiency.summary.find(r=>r.variant===v);
  const expected=[8,e.parameters_mean/1e6,e.resident_state_mib_mean,e.matrix_macs_mean/1e6,e.matrix_flops_mean/1e6,e.fft_calls_mean,
   e.inference_b1_ms_mean,e.inference_b1_ms_sd,e.inference_b128_ms_mean,e.inference_b128_ms_sd,e.throughput_b128_samples_s_mean/1000,e.throughput_b128_samples_s_sd/1000,
   e.clean_inference_peak_mib_mean,e.source_training_peak_mib_mean,e.e200_training_minutes_mean,e.clean_pipeline_seconds_mean,e.clean_pipeline_samples_s_mean/1000,
   source.find(r=>r.variant===v).accuracy_pct/100,source.find(r=>r.variant===v).macro_f1_pct/100];
  for(const [j,x] of expected.entries())if(typeof cells[i][j]!=='number'||Math.abs(cells[i][j]-x)>1e-7)throw new Error(`Efficiency mismatch: ${v} column ${j}: ${cells[i][j]} expected ${x}`);
 }
}
if(deployment){
 const actual=sheets['效率汇总'].getRange('B89:P114').values;
 const groups=order.flatMap(v=>[1,128].map(batch=>({v,batch})));
 for(const [i,g] of groups.entries()){
  const e=deployment.summary.find(r=>r.variant===g.v&&r.batch_size===g.batch);
  const expected=[8,g.batch,e.request_count,e.end_to_end_mean_ms,e.seed_mean_sd_ms,e.p50_ms,e.p95_ms,e.p99_ms,e.gpu_peak_allocated_mib,e.gpu_peak_reserved_mib,e.cpu_peak_rss_mib,e.cpu_baseline_rss_mean_mib,e.model_storage_mib,e.resident_state_mib,e.end_to_end_throughput_samples_s];
  for(const [j,x] of expected.entries())if(typeof actual[i][j]!=='number'||Math.abs(actual[i][j]-x)>1e-7)throw Error(`Deployment formula mismatch ${g.v} B${g.batch} col${j}: ${actual[i][j]} vs ${x}`);
 }
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500,summary:'formula error scan'});
await fs.writeFile(path.join(previews,'formula_scan.jsonl'),errors.ndjson);
const inspect=await wb.inspect({kind:'table',range:'总体汇总!A5:J8',include:'values,formulas',tableMaxRows:4,tableMaxCols:10,maxChars:2500});await fs.writeFile(path.join(previews,'summary_inspect.jsonl'),inspect.ndjson);
for(const name of Object.keys(sheets)){
 const range=name==='论文图表'?'A1:N47':name==='效率汇总'?'A2:K20':name==='部署测量'?'A1:K7':name==='请求时延'?'A1:E8':name==='总体汇总'?'A2:M20':name==='实验进度'?'A1:H8':name==='实验说明'?'A1:B9':name==='资源测量'?'A1:K7':'A1:'+col(name==='配对差'?6:name==='混淆矩阵'?7:name==='分层原始数据'?9:name==='种子结果'?9:8)+'7';
 const blob=await wb.render({sheetName:name,range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
for(const [name,range,file] of [['总体汇总','A23:K48','总体图表'],['接收机分层','L2:T21','接收机图表']]){
 const blob=await wb.render({sheetName:name,range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,file+'.png'),new Uint8Array(await blob.arrayBuffer()));}
if(efficiency)for(const [range,file] of [['A30:K55','时延图表'],['A58:K83','吞吐率图表']]){
 const blob=await wb.render({sheetName:'效率汇总',range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,file+'.png'),new Uint8Array(await blob.arrayBuffer()));}
if(deployment)for(const [range,file] of [['A86:I102','端到端汇总'],['J88:P102','部署内存存储']]){
 const blob=await wb.render({sheetName:'效率汇总',range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,file+'.png'),new Uint8Array(await blob.arrayBuffer()));}
await fs.mkdir(path.dirname(outputFile),{recursive:true});const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(outputFile);
console.log(JSON.stringify({status:'VERIFIED',output:outputFile,sheets:sheetNames.length,nativeCharts:efficiency?4:2,paperFigures:figureCount,models:rows.length,metricRecords:detail.length,confusionRows:conf.length,summaryFormulasReconciled:true,efficiencyFormulasReconciled:Boolean(efficiency),deploymentFormulasReconciled:Boolean(deployment),timedRequests:deployment?.trials.length||0}));
