import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const [inputFile,outputFile,previewDir,mode]=process.argv.slice(2);
if(!inputFile||!outputFile||!previewDir)throw Error('INPUT_JSON OUTPUT_XLSX PREVIEW_DIRECTORY [render-only]');
const data=JSON.parse(await fs.readFile(inputFile,'utf8')),figureDir=path.dirname(inputFile),wb=Workbook.create();
await fs.mkdir(previewDir,{recursive:true});
const titles=['双跨汇总','综合汇总','消融对比','种子结果','接收机分层','发射机分层','混淆矩阵','论文图表','实验说明'];
const sheets=Object.fromEntries(titles.map(n=>[n,wb.worksheets.add(n)]));
const font='Microsoft YaHei';
function col(i){let s='';for(i++;i;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s;return s;}
function table(sh,headers,rows,{start=5,widths=[],formats={},name}={}){
 const last=col(headers.length-1),end=start+rows.length;
 sh.getRange(`A${start}:${last}${end}`).values=[headers,...rows];
 sh.getRange(`A${start}:${last}${end}`).format={font:{name:font,size:10,color:'#263445'},rowHeight:23,verticalAlignment:'center'};
 sh.getRange(`A${start}:${last}${start}`).format={fill:'#36556F',font:{name:font,size:10,color:'#FFFFFF',bold:true},rowHeight:31,horizontalAlignment:'center'};
 for(let i=0;i<headers.length;i++)sh.getRange(`${col(i)}${start}:${col(i)}${end}`).format.columnWidth=widths[i]||18;
 for(const [i,format] of Object.entries(formats)){const range=sh.getRange(`${col(Number(i))}${start+1}:${col(Number(i))}${end}`);range.setNumberFormat(format);range.format.horizontalAlignment=format==='@'?'left':'right';}
 sh.tables.add(`A${start}:${last}${end}`,true,name).showBandedRows=true;sh.showGridLines=false;
 if(rows.length>30)sh.freezePanes.freezeRows(start);
}
function title(sh,text,note){sh.getRange('A2').values=[[text]];sh.getRange('A2').format.font={name:font,size:15,bold:true,color:'#263445'};sh.getRange('A2').format.rowHeight=28;if(note){sh.getRange('A3').values=[[note]];sh.getRange('A3').format.font={name:font,size:10,color:'#596B78',italic:true};}}
const order=data.variants,per=[];
for(const scope of ['双跨','综合'])for(const v of order)per.push(...data.per_seed.filter(r=>r.scope===scope&&r.variant===v).sort((a,b)=>a.model_seed-b.model_seed));
const seed=sheets['种子结果'];title(seed,'逐模型、逐种子的固定测试结果','双跨为 D0 × 未见 RX；综合为 D0–D3 × 未见 RX。两个范围分开统计。');
table(seed,['范围','网络','model_seed','测试包数','总参数量','准确率','Macro-F1','最差RX准确率'],per.map(r=>[r.scope,r.network,String(r.model_seed),r.query_count,r.total_parameters,r.accuracy,r.macro_f1,r.worst_rx]),{name:'SeedResults',widths:[12,31,19,17,19,17,17,21],formats:{2:'@',3:'#,##0',4:'#,##0',5:'0.00%',6:'0.00%',7:'0.00%'}});
const expected=[];
const short={sig_time:'PAMF',sig_time_removed:'w/o SIG-TIME',resnet18:'R18-1D',resnet34:'R34-1D',resnet50:'R50-1D',densenet121:'Dense121',vgg16_bn:'VGG16-BN',convnext_tiny:'ConvNeXt',transformer6:'Transformer',bilstm3:'BiLSTM',cvresnet18_1d:'CR18-1D',cvresnet18_2d:'CR18-2D',resnet18_2d:'R18-2D',resnet50_2d:'R50-2D'};
for(const [index,scope] of ['双跨','综合'].entries()){
 const sh=sheets[scope+'汇总'];title(sh,scope==='双跨'?'仅跨日＋跨接收机 clean 测试':'全日期 × 未见接收机 clean 测试',scope==='双跨'?'D0＝2021-03-01；7 个未见 RX；42000 包/模型；纯 CE E200；每网络实际 n＝8。':'D0 至 D3；7 个未见 RX；168000 包/模型；同一批固定权重。此表不包含单独跨日测试。');
 const scopedOrder=[...order].sort((a,b)=>data.summary.find(r=>r.scope===scope&&r.variant===b).accuracy-data.summary.find(r=>r.scope===scope&&r.variant===a).accuracy);
 const rows=scopedOrder.map((v,i)=>[i+1,data.names[v],null,index===0?42000:168000,null,null,null,null,null,null,data.parameters[v]/1e6]);
 table(sh,['名次','网络','实际seed数','每seed包数','准确率均值','准确率SD（pp）','Macro-F1均值','Macro-F1 SD（pp）','最差RX均值','最差RX SD（pp）','参数量（M）'],rows,{name:index===0?'DoubleSummary':'FullSummary',widths:[9,31,15,18,19,22,20,23,21,23,18],formats:{0:'0',2:'0',3:'#,##0',4:'0.00%',5:'0.0000',6:'0.00%',7:'0.0000',8:'0.00%',9:'0.0000',10:'0.000000'}});
 for(const [i,v] of scopedOrder.entries()){
  const row=6+i,start=6+index*order.length*8+order.indexOf(v)*8,end=start+7;
  sh.getRange(`C${row}`).formulas=[[`=COUNTA('种子结果'!C${start}:C${end})`]];
  sh.getRange(`E${row}:J${row}`).formulas=[[
   `=AVERAGE('种子结果'!F${start}:F${end})`,`=STDEV('种子结果'!F${start}:F${end})*100`,
   `=AVERAGE('种子结果'!G${start}:G${end})`,`=STDEV('种子结果'!G${start}:G${end})*100`,
   `=AVERAGE('种子结果'!H${start}:H${end})`,`=STDEV('种子结果'!H${start}:H${end})*100`]];
  const s=data.summary.find(r=>r.scope===scope&&r.variant===v);expected.push({sheet:sh.name,row,values:[s.accuracy,s.accuracy_sd_pp,s.macro_f1,s.macro_f1_sd_pp,s.worst_rx,s.worst_rx_sd_pp]});
 }
 sh.getRange(`M5:M${5+order.length}`).values=[['网络简称'],...scopedOrder.map(v=>[short[v]])];sh.getRange(`M5:M${5+order.length}`).format.columnWidth=19;
 const chart=sh.charts.add('bar',[sh.getRange(`M5:M${5+order.length}`),sh.getRange(`E5:E${5+order.length}`),sh.getRange(`G5:G${5+order.length}`)]);
 chart.title=scope==='双跨'?'Unseen date + unseen receivers':'All target dates + unseen receivers';chart.barOptions.direction='column';chart.barOptions.grouping='clustered';chart.setPosition('A24','K51');
 chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=14;chart.legend={position:'top',textStyle:{typeface:'Arial',fontSize:11}};chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:9}};chart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};
 chart.series.items[0].fill='#365F82';chart.series.items[1].fill='#D55E00';
}
const paired=sheets['消融对比'];title(paired,'新整体网络与仅去除 SIG-TIME 的匹配消融','完整网络减去除版；同模型seed、原划分、纯 CE E200、原默认头。所有正负差值均保留。');
if(data.paired.length){
 table(paired,['范围','model_seed','完整准确率','去除版准确率','准确率差（pp）','完整Macro-F1','去除版Macro-F1','F1差（pp）'],data.paired.map(r=>[r.scope,String(r.model_seed),r.full_accuracy,r.removed_accuracy,null,r.full_macro_f1,r.removed_macro_f1,null]),{name:'PairedAblation',widths:[13,19,21,22,23,22,23,22],formats:{1:'@',2:'0.00%',3:'0.00%',4:'+0.0000;-0.0000;0.0000',5:'0.00%',6:'0.00%',7:'+0.0000;-0.0000;0.0000'}});
 for(let i=0;i<data.paired.length;i++){const r=6+i;paired.getRange(`E${r}`).formulas=[[`=(C${r}-D${r})*100`]];paired.getRange(`H${r}`).formulas=[[`=(F${r}-G${r})*100`]];}
 table(paired,['范围','实际配对数','准确率差均值（pp）','配对差SD（pp）'],['双跨','综合'].map(()=>[null,null,null,null]),{start:27,name:'AblationSummary',widths:[13,20,26,23],formats:{1:'0',2:'+0.0000;-0.0000;0.0000',3:'0.0000'}});
 for(const [i,scope] of ['双跨','综合'].entries()){const r=28+i,start=6+i*8,end=start+7;paired.getRange(`A${r}:D${r}`).values=[[scope,null,null,null]];paired.getRange(`B${r}:D${r}`).formulas=[[`=COUNTA(B${start}:B${end})`,`=AVERAGE(E${start}:E${end})`,`=STDEV(E${start}:E${end})`]];}
}else{paired.getRange('A5').values=[['去除版尚未完成；此处不填零或推测分数。']];}
for(const [dim,label] of [['receiver','接收机分层'],['transmitter','发射机分层']]){
 const sh=sheets[label];title(sh,label+' clean 测试',dim==='transmitter'?'逐发射机准确率即召回率。范围、实际种子数和每seed样本数分别列示。':'逐接收机统计，双跨每 RX 每seed 6000 包，综合为24000包。');
 table(sh,['范围','网络',dim==='receiver'?'RX物理ID':'TX物理ID','实际seed数','每seed包数','准确率均值','准确率SD（pp）',...(dim==='receiver'?['Macro-F1均值','Macro-F1 SD（pp）']:[])],data[dim].map(r=>[r.scope,r.network,r.stratum,r.actual_seed_count,r.query_count_per_seed,r.accuracy,r.accuracy_sd_pp,...(dim==='receiver'?[r.macro_f1,r.macro_f1_sd_pp]:[])]),{name:dim==='receiver'?'ReceiverResults':'TransmitterResults',widths:[12,31,18,17,19,20,23,21,24],formats:{2:'@',3:'0',4:'#,##0',5:'0.00%',6:'0.0000',7:'0.00%',8:'0.0000'}});
}
const cm=sheets['混淆矩阵'];title(cm,'逐种子总体混淆矩阵','原始计数；行是真实发射机、列是预测发射机。综合和双跨分别保存。');
const classes=['14-10','14-7','20-15','20-19','6-15','8-20'];const cmrows=[];
for(const r of data.records.filter(r=>r.dimension==='overall'))for(let i=0;i<6;i++)for(let j=0;j<6;j++)cmrows.push([r.scope,data.names[r.variant],String(r.model_seed),classes[i],classes[j],r.confusion[i][j]]);
table(cm,['范围','网络','model_seed','真实TX','预测TX','计数'],cmrows,{name:'ConfusionMatrices',widths:[13,31,19,18,18,18],formats:{2:'@',3:'@',4:'@',5:'#,##0'}});
const paper=sheets['论文图表'];title(paper,'跨日＋跨接收机论文图表','含全部网络的总体图与每个模型独立的 RX／TX／混淆矩阵图；图内 n＝8。');paper.getRange('A1:N1100').format.rowHeight=24;paper.getRange('A1:N1100').format.columnWidth=13;paper.showGridLines=false;
let y=6;
for(const [i,f] of data.figures.entries()){
 paper.getRange(`A${y}`).values=[[`图 ${i+1}：${f.title}`]];paper.getRange(`A${y}`).format.font={name:font,size:13,bold:true,color:'#263445'};
 const bytes=await fs.readFile(path.join(figureDir,'preview',f.stem+'.png'));const width=1180,height=width*f.height_mm/f.width_mm;
 paper.images.add({dataUrl:'data:image/png;base64,'+bytes.toString('base64'),anchor:{from:{row:y+1,col:0},extent:{widthPx:width,heightPx:height}}});y+=Math.ceil(height/24)+6;
}
const notes=sheets['实验说明'];title(notes,'测试范围与比较说明');
table(notes,['项目','说明'],[
 ['当前PAMF','历史纯CE SIG_TIME完整网络，196001参数；综合8seed准确率81.5965%。本表不混入后续T1/X3/半监督训练。'],
 ['去除SIG-TIME','仅删除sequence_pool的18976参数；其余共有参数初始值一致。参考响应、内部补偿、CFO坐标27置零和原分类头保留。'],
 ['双跨范围','未见日期D0（2021-03-01）＋7个未见接收机：1-1、14-7、2-1、20-1、7-14、7-7、8-8。'],
 ['双跨数量','42000包/模型；每RX6000包，每TX7000包。'],
 ['综合范围','相同7个未见RX，D0–D3全部日期，168000包/模型。原表不含源RX的单独跨日测试。'],
 ['源训练日期','D1/D2/D3＝2021-03-08、2021-03-15、2021-03-23。D0早于训练日期，不称未来日期外推。'],
 ['固定训练','每模型从零E200、10000次更新、纯CE、LS0；只用L_s6300。U_s56700不参与训练，V27000只监测。'],
 ['统计口径','模型种子2026092701至08，split392005固定；均值±样本SD（ddof=1）。单类准确率即召回率。'],
 ['最差RX口径','先在每个seed的7个RX中取最低准确率，再对8个seed计算均值与SD；不同于各RX均值中的最低值。'],
 ['预测复用','已有模型复用原固定预测。新消融逐row训练、冻结、clean预测完整后独立truth-last评分。'],
 ['证据边界','原目标基准此前已暴露。双跨是事后分层复算，不是新独立盲测；结果不反馈训练或选模。'],
 ['原始来源','原综合Excel及其已完成评分；20261014整体SIG_TIME八种子评分；20261026严格去SIG-TIME八种子补跑。'],
 ['历史保护','旧residual_fusion和原Excel保留。本新主表用完整SIG_TIME作为PAMF，并另列仅去SIG-TIME的匹配消融。']
],{name:'ExperimentNotes',widths:[24,145]});
wb.recalculate();
for(const e of expected){const actual=sheets[e.sheet].getRange(`E${e.row}:J${e.row}`).values[0];for(let j=0;j<6;j++)if(Math.abs(actual[j]-e.values[j])>1e-9)throw Error('Summary formula mismatch '+JSON.stringify(e));}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?',options:{useRegex:true,maxResults:12},maxChars:1000})).ndjson);
for(const [name,range] of [['双跨汇总',`A1:K${6+order.length}`],['综合汇总',`A1:K${6+order.length}`],['消融对比','A1:H30'],['种子结果','A1:H11'],['接收机分层','A1:I11'],['发射机分层','A1:G11'],['混淆矩阵','A1:F11'],['实验说明','A1:B18']]){
 const blob=await wb.render({sheetName:name,range,scale:1.2,format:'png'});await fs.writeFile(path.join(previewDir,name+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
if(mode!=='render-only'){
 try{await fs.access(outputFile);throw Error('Output already exists; preserve final workbook');}catch(e){if(e.code!=='ENOENT')throw e;}
 await (await SpreadsheetFile.exportXlsx(wb)).save(outputFile);console.log('EXPORTED '+outputFile);
}
console.log(JSON.stringify({models:order.length,seedRows:per.length,images:data.figures.length,charts:2,sheets:titles.length}));
