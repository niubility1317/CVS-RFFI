// Build one research workbook from the frozen standard-CE score intermediates.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [inputFile,outputFile]=process.argv.slice(2);
if(!inputFile||!outputFile)throw new Error('Usage: node export_standard_ce_excel.mjs INPUT_JSON NEW_OUTPUT_XLSX');
try {await fs.access(outputFile);throw new Error('Output exists; preserve it and choose a new name.');} catch(e){if(e.code!=='ENOENT')throw e;}
const data=JSON.parse(await fs.readFile(inputFile,'utf8'));
const raw=data.raw,tables=data.tables;
if(raw.run_id!=='20261010-phase1-standard-ce-baselines-manysig-m56-r01'||raw.rows.length!==56)throw new Error('Unexpected benchmark');
const source=tables['network_summary.csv'];
const order=[...source].sort((a,b)=>b.accuracy_pct-a.accuracy_pct).map(r=>r.variant);
const names=Object.fromEntries(source.map(r=>[r.variant,r.variant==='residual_fusion'?'CVS':r.network]));
const short={residual_fusion:'CVS',native:'Original CVS',resnet50:'R50-1D',cvresnet18_1d:'CR18-1D',
 resnet34:'R34-1D',resnet18:'R18-1D',densenet121:'Dense121',vgg16_bn:'VGG16-BN',bilstm3:'BiLSTM',
 resnet18_2d:'R18-2D',convnext_tiny:'ConvNeXt',resnet50_2d:'R50-2D',cvresnet18_2d:'CR18-2D',transformer6:'Transformer'};
const rows=order.flatMap(v=>raw.rows.filter(r=>r.row.variant===v).sort((a,b)=>a.row.model_seed-b.row.model_seed));
const wb=Workbook.create();
const sheets=Object.fromEntries(['总体汇总','种子结果','接收机分层','发射机分层','配对差','资源测量','混淆矩阵','分层原始数据','实验说明'].map(n=>[n,wb.worksheets.add(n)]));
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

title(sheets['总体汇总'],'纯 CE 神经网络 clean 测试汇总','历史 E200；14 种网络 × 4 个种子；每个模型 168000 个相同测试包。');
const sums=order.map(v=>[names[v],source.find(x=>x.variant===v).total_parameters,null,null,null,null,null,null,null,short[v]]);
table(sheets['总体汇总'],['网络','总参数量','seed数','准确率均值','准确率SD（pp）','Macro-F1均值','Macro-F1 SD（pp）','最差RX均值','最差RX SD（pp）','绘图简称'],sums,
 {start:5,name:'NetworkSummary',widths:[29,17,11,19,20,20,22,20,22,23],formats:{1:'#,##0',2:'0',3:'0.00%',4:'0.0000',5:'0.00%',6:'0.0000',7:'0.00%',8:'0.0000'}});
for(let i=0;i<14;i++){
 const r=6+i,a=2+i*4,b=a+3;
 sheets['总体汇总'].getRange(`C${r}:I${r}`).formulas=[[
  `=COUNTA('种子结果'!C${a}:C${b})`,`=AVERAGE('种子结果'!E${a}:E${b})`,`=STDEV('种子结果'!E${a}:E${b})*100`,
  `=AVERAGE('种子结果'!F${a}:F${b})`,`=STDEV('种子结果'!F${a}:F${b})*100`,
  `=AVERAGE('种子结果'!G${a}:G${b})`,`=STDEV('种子结果'!G${a}:G${b})*100`]];
}
const chart=sheets['总体汇总'].charts.add('bar',[sheets['总体汇总'].getRange('J5:J19'),sheets['总体汇总'].getRange('D5:D19'),sheets['总体汇总'].getRange('F5:F19')]);
chart.title='Mean clean accuracy and macro-F1 (four seeds)';chart.barOptions.direction='column';chart.barOptions.grouping='clustered';chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=15;
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
const rxNames=receiver.filter(r=>r.variant==='native').map(r=>r.stratum);
const rxHelpers=[['Receiver','Original CVS','CVS'],...rxNames.map(id=>[id,null,null])];
const rxSheet=sheets['接收机分层'];rxSheet.getRange('L24:N31').values=rxHelpers;rxSheet.getRange('L24:N31').format.font={name:'Arial',size:10};rxSheet.getRange('L24:N31').format.columnWidth=20;rxSheet.getRange('M25:N31').setNumberFormat('0.00%');
for(let i=0;i<7;i++){const id=rxNames[i];for(const [j,v] of ['native','residual_fusion'].entries()){
 const targetRow=order.flatMap(x=>receiver.filter(r=>r.variant===x)).findIndex(r=>r.variant===v&&r.stratum===id)+2;
 rxSheet.getRange(`${j===0?'M':'N'}${25+i}`).formulas=[[`=F${targetRow}`]];
}}
const rxChart=rxSheet.charts.add('bar',rxSheet.getRange('L24:N31'));rxChart.title='CVS accuracy by held-out receiver';rxChart.barOptions.direction='column';rxChart.barOptions.grouping='clustered';rxChart.titleTextStyle.typeface='Arial';rxChart.titleTextStyle.fontSize=15;
rxChart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:11}};rxChart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:11}};rxChart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};rxChart.setPosition('L2','T21');rxChart.series.items[0].fill='#626D78';rxChart.series.items[1].fill='#BB6C42';

const paired=tables['paired_differences.csv'].map(r=>[r.control,names[r.control],r.baseline,names[r.baseline],r.model_seed,r.accuracy_difference_pp,r.macro_f1_difference_pp]);
table(sheets['配对差'],['对照variant','对照网络','基准variant','基准网络','model_seed','准确率差（pp）','Macro-F1差（pp）'],paired,
 {name:'PairedDifferences',widths:[24,28,24,28,18,22,23],formats:{4:'@',5:'+0.0000;-0.0000;0.0000',6:'+0.0000;-0.0000;0.0000'}});
sheets['配对差'].getRange('F2:G105').conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{font:{color:'#B64342'}}});
const resources=tables['resource_profiles.csv'];
const rv=resources.map(r=>[r.variant,names[r.variant],r.model_seed,r.hardware,r.total_parameters,r.trainable_parameters,r.gradient_used_parameters,r.resident_state_bytes,r.training_elapsed_seconds,r.training_peak_cuda_allocated_bytes,r.inference_batch1_ms,r.inference_batch128_ms,r.training_batch128_ms,r.full_clean_inference_seconds,r.full_clean_peak_cuda_allocated_bytes,r.conv_linear_macs_per_sample,r.fft_calls_per_sample]);
table(sheets['资源测量'],['variant','网络','model_seed','GPU','总参数量','可训练参数量','梯度使用参数量','常驻状态（byte）','E200耗时（s）','训练峰值显存（byte）','B1推理（ms）','B128推理（ms）','B128训练（ms）','clean全量推理（s）','clean峰值显存（byte）','Conv/Linear MACs','FFT调用数'],rv,
 {name:'ResourceProfiles',widths:[23,28,18,36,19,20,22,24,21,27,21,23,23,24,28,25,20],formats:{2:'@',4:'#,##0',5:'#,##0',6:'#,##0',7:'#,##0',8:'0.000',9:'#,##0',10:'0.000',11:'0.000',12:'0.000',13:'0.000',14:'#,##0',15:'#,##0',16:'0'}});
const conf=[];
for(const r of rows){const cm=r.scores.results.find(s=>s.dimension==='overall').confusion;for(let i=0;i<6;i++)for(let j=0;j<6;j++)conf.push([r.row.variant,names[r.row.variant],String(r.row.model_seed),raw.classes[i],raw.classes[j],cm[i][j],null,cm[i].reduce((a,b)=>a+b,0)]);}
table(sheets['混淆矩阵'],['variant','网络','model_seed','真实TX','预测TX','样本数','行内比例','真实类总包数'],conf,
 {name:'ConfusionCounts',widths:[23,28,18,16,16,18,18,21],formats:{2:'@',3:'@',4:'@',5:'#,##0',6:'0.00%',7:'#,##0'}});
sheets['混淆矩阵'].getRange(`G2:G${conf.length+1}`).formulas=conf.map((_,i)=>[`=F${i+2}/H${i+2}`]);
const detail=tables['per_seed_metrics.csv'];
table(sheets['分层原始数据'],['row_id','variant','网络','model_seed','view','分层维度','物理ID','测试包数','准确率','Macro-F1'],
 detail.map(r=>[r.row_id,r.variant,names[r.variant],r.model_seed,r.view,r.dimension,r.stratum,r.query_count,r.accuracy_pct/100,r.macro_f1_pct/100]),
 {name:'AllScoredMetrics',widths:[40,23,28,18,14,20,17,19,18,18],formats:{3:'@',6:'@',7:'#,##0',8:'0.00%',9:'0.00%'}});

const notes=[
 ['run_id',raw.run_id],['实验范围','历史 E200、14 种网络、4 个模型种子；仅 clean 测试。与后续 E20 研究区分。'],
 ['模型种子','2026092701、2026092702、2026092703、2026092704；split seed 固定 392005。'],
 ['数据角色','ManySig；源 RX1/3/4/6/8，day1/2/3；L_s 6300、U_s 56700（不使用）、V 27000（仅验证）。'],
 ['训练','从零训练 200 epoch、10000 次更新；只有身份 CE。固定 E200 checkpoint；无增强或额外损失。'],
 ['测试','同一批 168000 个物理包；7 个未见 RX 各 24000 包；6 个 TX 各 28000 包；逐样本全部 6 类竞争。'],
 ['误差定义','SD 为 4 个模型种子的样本标准差（ddof=1），单位 pp 为百分点；不是标准误或置信区间。'],
 ['最差RX','每个种子先取 7 个 RX 的最低准确率，再计算 4 seed 均值和 SD；并非先平均后取最低。'],
 ['退化结果','Transformer 的种子 2026092703 全部预测为 TX 8-20；准确率 16.6667%、Macro-F1 4.7619%，完整保留。'],
 ['配对差','每条记录为同 model seed 下对照网络减基准网络，单位为百分点。'],
 ['混淆矩阵','每行是真实类与预测类的样本计数；行内比例=样本数/该真实类总包数。TX分层Macro-F1为单真实类子集上的6类宏平均，不是全测试集的该类F1。'],
 ['资源口径','RTX3090/FP32；E200耗时含源V评估。MACs仅含Conv/Linear；显存与状态用byte。'],
 ['比较边界','已暴露代理基准、统一固定训练预算；2D采用16×16 IQ reshape。未提供显著性检验或星地鲁棒性结论。'],
 ['当前种子数','本工作簿只有原始 4 个种子的完成结果；新增种子实验尚未并入。CVS 默认指 residual_fusion。'],
 ['数据来源','network_summary.csv、per_seed_metrics.csv、receiver_summary.csv、transmitter_summary.csv、paired_differences.csv、resource_profiles.csv 与冻结评分JSON。'],
 ['使用','总体汇总由种子结果的 Excel 公式计算；原始评分表保持数值。可编辑图表随引用单元格更新。']];
table(sheets['实验说明'],['项目','说明'],notes,{name:'ExperimentContext',widths:[23,116]});sheets['实验说明'].getRange('B2:B17').format.wrapText=true;sheets['实验说明'].getRange('A2:B17').format.rowHeight=44;

wb.recalculate();
// Check an actual input edit, then restore the frozen score before export.
const seedCell=sheets['种子结果'].getRange('E2');const originalValue=seedCell.values[0][0];
const before=sheets['总体汇总'].getRange('D6').values[0][0];
seedCell.values=[[originalValue+0.001]];wb.recalculate();
if(Math.abs(sheets['总体汇总'].getRange('D6').values[0][0]-before-0.00025)>1e-10)throw new Error('Summary did not recalculate');
seedCell.values=[[originalValue]];wb.recalculate();
const summaryCells=sheets['总体汇总'].getRange('C6:I19').values;
for(let i=0;i<14;i++){const expected=source.find(r=>r.variant===order[i]);const actual=summaryCells[i];
 const exp=[4,expected.accuracy_pct/100,expected.accuracy_sd_pp,expected.macro_f1_pct/100,expected.macro_f1_sd_pp,expected.worst_receiver_accuracy_pct/100,expected.worst_receiver_sd_pp];
 for(let j=0;j<7;j++)if(typeof actual[j]!=='number'||Math.abs(actual[j]-exp[j])>1e-9)throw new Error(`Summary mismatch: ${order[i]} column ${j}: ${actual[j]} expected ${exp[j]}`);}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500,summary:'formula error scan'});
await fs.writeFile(path.join(previews,'formula_scan.jsonl'),errors.ndjson);
const inspect=await wb.inspect({kind:'table',range:'总体汇总!A5:J8',include:'values,formulas',tableMaxRows:4,tableMaxCols:10,maxChars:2500});await fs.writeFile(path.join(previews,'summary_inspect.jsonl'),inspect.ndjson);
for(const name of Object.keys(sheets)){
 const range=name==='总体汇总'?'A2:K20':name==='实验说明'?'A1:B9':name==='资源测量'?'A1:K7':'A1:'+col(name==='配对差'?6:name==='混淆矩阵'?7:name==='分层原始数据'?9:name==='种子结果'?9:8)+'7';
 const blob=await wb.render({sheetName:name,range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
for(const [name,range,file] of [['总体汇总','A23:K48','总体图表'],['接收机分层','L2:T21','接收机图表']]){
 const blob=await wb.render({sheetName:name,range,scale:1.25,format:'png'});await fs.writeFile(path.join(previews,file+'.png'),new Uint8Array(await blob.arrayBuffer()));}
await fs.mkdir(path.dirname(outputFile),{recursive:true});const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(outputFile);
console.log(JSON.stringify({status:'VERIFIED',output:outputFile,sheets:9,nativeCharts:2,models:56,metricRecords:784,confusionRows:2016,summaryFormulasReconciled:true}));
