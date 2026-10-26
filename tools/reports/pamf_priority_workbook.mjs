import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const base=process.argv[2],input=JSON.parse(await fs.readFile(path.join(base,'weekly_workbook_data.json'),'utf8'));
const wb=Workbook.create(),sheets=new Map(),layouts={},records=new Map(input.records.map(r=>[r.id,r]));
const dir=path.join(base,'workbook_previews');await fs.mkdir(dir,{recursive:true});
const col=n=>{let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const labels=input.labels,colors=input.colors;
const allCharts=sp=>[...(sp.charts||[]),...(sp.jointCharts||[])];
const heads=['数据设置','方法','学习方式','训练星地增强','推理链','测试人群','测试场景','测试集合/配方口径','准确率均值（%）','样本标准差（%）','均值±标准差','模型种子数','种子01准确率（%）','种子02准确率（%）','种子03准确率（%）','说明'];
const protocolName=p=>({'F0':'标准主划分','F1_rx_repartition_r01':'接收机划分一','F2_rx_repartition_r01':'接收机划分二','ManyRx':'十类数据','label01':'1%标注','label05':'5%标注'}[p]||p.replace('m','').replace('-natural','个源接收机'));
for(const sp of input.sheets){
 const sh=wb.worksheets.add(sp.name);sheets.set(sp.name,sh);
 if(sp.charts){
  const charts=allCharts(sp),ids=[...new Set(charts.flatMap(c=>c.series.flatMap(s=>s.ids)).filter(Boolean))],panelRows=Math.ceil(sp.charts.length/2),jr=Math.ceil((sp.jointCharts||[]).length/2),jointTop=18+24*panelRows+5,ph=18+24*panelRows+(jr?5+24*jr:0)+3,pl=ph+sp.partitionData.length;
  let next=pl+5;const blocks=charts.map((c,i)=>{const joint=i>=sp.charts.length,j=joint?i-sp.charts.length:i,top=(joint?jointTop:18)+Math.floor(j/2)*24;const b={head:next,last:next+c.categories.length,from:(j%2?'G':'A')+top,to:(j%2?'L':'F')+(top+19),range:(j%2?'G':'A')+top+':'+(j%2?'L':'F')+(top+19),joint};next=b.last+4;return b;});
  const dh=next+2;layouts[sp.name]={partitionHead:ph,partitionLast:pl,blocks,jointSection:jr?jointTop-2:null,detailHead:dh,dataStart:dh+1,dataLast:dh+ids.length,ids,index:Object.fromEntries(ids.map((id,i)=>[id,dh+1+i])),newLayout:true};
 }else{
  const ph=sp.jointChart?49:22,pl=ph+sp.partitionData.length,ch=pl+5,cl=ch+sp.chart.data.length,dh=cl+5,dl=dh+sp.data.length,jh=dl+5,jl=jh+(sp.jointData||[]).length,jch=sp.jointChart?jl+5:null;
  layouts[sp.name]={partitionHead:ph,partitionLast:pl,chartHead:ch,chartLast:cl,detailHead:dh,dataStart:dh+1,dataLast:dl,jointTableHead:sp.jointChart?jh:null,jointTableLast:sp.jointChart?jl:null,jointChartHead:jch,jointChartLast:jch?jch+sp.jointChart.data.length:null,lastRow:sp.jointChart?jch+sp.jointChart.data.length:dl,newLayout:false};
 }
}
function token(v){const a=v.split(':'),row=layouts[a[1]].dataStart+Number(a[2])-5;return a[0]==='@num'?`='${a[1]}'!${a[3]}${row}`:`=TEXT('${a[1]}'!${a[3]}${row},"0.00")&"±"&TEXT('${a[1]}'!${a[4]}${row},"0.00")`;}
function section(sh,r,text){sh.getRange('A'+r).values=[[text]];sh.getRange('A'+r).format.font={name:'SimSun',size:11,bold:true,color:'#253849'};}
function write(sh,r,head,data){
 const end=r+data.length,last=col(head.length);sh.getRange(`A${r}:${last}${r}`).values=[head];sh.getRange(`A${r+1}:${last}${end}`).values=data.map(a=>a.map(v=>typeof v==='string'&&v.startsWith('@')?null:v));
 data.forEach((a,i)=>a.forEach((v,j)=>{if(typeof v==='string'&&v.startsWith('@'))sh.getRange(col(j+1)+(r+1+i)).formulas=[[token(v)]];}));
 sh.getRange(`A${r}:${last}${r}`).format={font:{name:'SimSun',size:10,bold:true,color:'#253849'},wrapText:true,horizontalAlignment:'center',borders:{top:{style:'medium',color:'#263849'},bottom:{style:'thin',color:'#768A9A'}}};sh.getRange(`A${r}:${last}${r}`).format.rowHeight=34;
 sh.getRange(`A${end}:${last}${end}`).format.borders={bottom:{style:'medium',color:'#263849'}};
 sh.getRange(`A${r+1}:${last}${end}`).format.rowHeight=23;
 head.forEach((h,j)=>{const q=sh.getRange(`${col(j+1)}${r+1}:${col(j+1)}${end}`);if(data.some(a=>typeof a[j]==='number'||typeof a[j]==='string'&&a[j].startsWith('@num:'))){q.setNumberFormat(/种子数|类别|包|池|批量|模型|数量|总数|独立|展示项/.test(h)&&!/准确率|均值|标准差/.test(h)?'0':'0.0000');q.format.horizontalAlignment='right';}else q.format.font.name='SimSun';});
 return `A${r}:${last}${end}`;
}
function styleChart(c,title,type,seriesNames,categories,priority,unit='Accuracy (%)'){
 c.title=title;c.titleTextStyle.typeface='SimSun';c.titleTextStyle.fontSize=15;c.hasLegend=seriesNames.length>1;c.legend={position:'bottom',textStyle:{typeface:'Times New Roman',fontSize:11}};c.xAxis={axisType:'textAxis',textStyle:{typeface:'Times New Roman',fontSize:11}};c.yAxis={numberFormatCode:'0',numberFormatSourceLinked:false,textStyle:{typeface:'Times New Roman',fontSize:11}};c.yAxis.title.text=unit;
 c.yAxis={min:0};if(unit.includes('(%)'))c.yAxis={max:100,majorUnit:20};
 if(type==='bar'){c.barOptions.direction='column';c.barOptions.grouping='clustered';c.barOptions.gapWidth=80;}
 if(type==='line')c.xAxis.title.text='Number of source receivers';
 c.series.items.forEach((s,j)=>{const mi=labels.indexOf(seriesNames[j]),color=mi>=0?colors[mi]:['#86A9C2','#D76B58','#8AA06B','#9B89AE','#C4364E','#E29525'][j%6];s.fill=color;s.valuesFormatCode='0.0';if(type==='line')s.line={fill:color,style:j===5?'dashed':'solid',width:j>=4?2.8:1.6};
  if(priority&&type==='bar'&&seriesNames.length===2)categories.forEach((cat,i)=>{if(cat.startsWith('PAMF')){const ov=s.dataLabelOverrides.add(i);ov.showValue=true;ov.position='outEnd';ov.textStyle.bold=true;ov.textStyle.fontSize=11;}});
 });
}
let chartCount=0;const bindings=[];
for(const [si,sp] of input.sheets.entries()){
 const sh=sheets.get(sp.name),l=layouts[sp.name],last=sp.charts?'P':col(Math.max(12,sp.headers.length)),lastRow=l.lastRow||l.dataLast;sh.showGridLines=false;
 sh.getRange(`A1:${last}${lastRow}`).format={font:{name:'Times New Roman',size:10,color:'#1F2D38'},fill:'#FFFFFF',verticalAlignment:'center'};sh.tabColor=sp.priority?'#B8334B':'#BBC5CF';
 const widths=sp.charts?[26,24,23,22,29,27,22,40,20,20,27,17,22,22,22,48]:Array.from({length:Math.max(12,sp.headers.length)},(_,j)=>sp.widths[j]||22);
 widths.forEach((w,j)=>sh.getRange(`${col(j+1)}1:${col(j+1)}${lastRow}`).format.columnWidth=w);
 sh.getRange('A2').values=[[sp.title||sp.name.replaceAll('_',' ')]];sh.getRange('A2').format.font={name:'SimSun',size:14,bold:true,color:sp.priority?'#A92F45':'#253849'};sh.getRange('A2').format.rowHeight=30;section(sh,4,'实验说明');
 for(let i=0;i<sp.explain.length;i++){const r=i+5;sh.getRange('A'+r).values=[[sp.explain[i][0]]];sh.getRange('A'+r).format.font={name:'SimSun',size:10,bold:true};sh.mergeCells(`B${r}:${sp.charts?'L':'E'}${r}`);sh.getRange('B'+r).values=[[sp.explain[i][1]]];sh.getRange(`B${r}:${sp.charts?'L':'E'}${r}`).format={font:{name:'SimSun',size:10},wrapText:true};sh.getRange('A'+r).format.rowHeight=sp.charts?31:sp.explain[i][1].length>68?44:32;}
 if(sp.charts){
  if(sp.notes){sh.mergeCells('A13:L14');sh.getRange('A13').values=[[sp.notes]];sh.getRange('A13:L14').format={font:{name:'SimSun',size:10,italic:true,color:'#7F4C28'},wrapText:true};}
  if(l.jointSection)section(sh,l.jointSection,'新增独立测试：跨日且跨接收机（目标接收机，2021年3月1日）');
  section(sh,l.partitionHead-2,'对应数据划分：两类测试分别列示，不合并样本或均值');write(sh,l.partitionHead,sp.partitionHeaders,sp.partitionData);sh.getRange(`A${l.partitionHead+1}:I${l.partitionLast}`).format.rowHeight=36;sh.getRange(`C${l.partitionHead+1}:D${l.partitionLast}`).format.wrapText=true;
  const data=l.ids.map(id=>{const r=records.get(id),v=r.seed_values.split(';').map(Number);return [protocolName(r.protocol),r.method,r.learning_regime==='supervised'?'监督':'普通PL半监督',r.channel_augmentation?'开启':'关闭',r.readout==='raw'?'直接识别':'先DURA恢复后识别',r.test==='cross_both'?'目标跨日且跨接收机（3月1日）':'目标跨接收机（3月8/15/23日）',({'clean':'CLEAN','six_practical_mean':'综合星地'}[r.view]||r.view),r.scope==='six_common_target_receivers'?'共同6目标接收机':r.scope==='existing_matched_supervised_no_channel_recipe'?'既有无增强监督配方':'原注册测试人群',r.mean,r.sd,null,3,...v,r.scope==='existing_matched_supervised_no_channel_recipe'?'保留原监督参照配方；当前PL配方缺项':'冻结独立评分统计'];});
  section(sh,l.detailHead-2,'完整结果：独立均值、标准差与可复制列');const addr=write(sh,l.detailHead,heads,data);const t=sh.tables.add(addr,true,`PriorityData${si+1}`);t.style='TableStyleLight1';t.showFilterButton=true;sh.getRange(addr).format.fill='#FFFFFF';sh.getRange(`K${l.dataStart}:K${l.dataLast}`).formulas=data.map((_,i)=>[`=TEXT(I${l.dataStart+i},"0.00")&"±"&TEXT(J${l.dataStart+i},"0.00")`]);
  sh.getRange(`F${l.dataStart}:H${l.dataLast}`).format.wrapText=true;sh.getRange(`A${l.dataStart}:P${l.dataLast}`).format.rowHeight=38;
  allCharts(sp).forEach((ch,i)=>{const b=l.blocks[i];section(sh,b.head-2,(b.joint?'双跨独立表：':'原测试图表数据：')+ch.title);const hdr=['方法/数据设置',...ch.series.map(s=>s.name)],vals=ch.categories.map(cat=>[cat,...ch.series.map(()=>null)]);write(sh,b.head,hdr,vals);ch.series.forEach((ss,j)=>ss.ids.forEach((id,k)=>{if(id)sh.getRange(col(j+2)+(b.head+1+k)).formulas=[[`=I${l.index[id]}`]];}));
   const c=sh.charts.add(ch.type,sh.getRange(`A${b.head}:${col(hdr.length)}${b.last}`));c.setPosition(b.from,b.to);styleChart(c,ch.title+(sp.slug==='main_no_channel'?'（PAMF普通PL缺项）':''),ch.type,ch.series.map(s=>s.name),ch.categories,true,ch.unit);
   // A missing experiment remains an empty source cell, with no fabricated numeric label.
   ch.series.forEach((ss,j)=>ss.ids.forEach((id,k)=>{if(!id){const v=c.series.items[j].dataLabelOverrides.add(k);v.showValue=false;}}));
   chartCount++;bindings.push({sheet:sp.name,panel:i,type:ch.type,test:ch.test,title:ch.title,source:`A${b.head}:${col(hdr.length)}${b.last}`,missing:ch.series.map(s=>s.ids.filter(x=>!x).length)});
  });
 }else{
  sh.mergeCells('A18:E18');sh.getRange('A18').values=[[sp.chart.note]];sh.getRange('A18:E18').format={font:{name:'SimSun',size:9,italic:true},wrapText:true};sh.getRange('A18').format.rowHeight=28;
  section(sh,l.partitionHead-2,'对应数据划分');write(sh,l.partitionHead,sp.partitionHeaders,sp.partitionData);sh.getRange(`A${l.partitionHead+1}:K${l.partitionLast}`).format.rowHeight=34;sh.getRange(`C${l.partitionHead+1}:D${l.partitionLast}`).format.wrapText=true;
  sp.partitionData.forEach((r,i)=>{if(r[0]==='大规模十类数据')sh.getRange(`A${l.partitionHead+1+i}:K${l.partitionHead+1+i}`).format.rowHeight=66;});
  if(sp.partitionNote){sh.mergeCells(`A${l.partitionLast+2}:K${l.partitionLast+2}`);sh.getRange(`A${l.partitionLast+2}`).values=[[sp.partitionNote]];sh.getRange(`A${l.partitionLast+2}:K${l.partitionLast+2}`).format.font={name:'SimSun',size:9,italic:true};}
  section(sh,l.chartHead-2,'图表摘要数据');write(sh,l.chartHead,sp.chart.headers,sp.chart.data);
  if(sp.chart.delta){write(sh,l.chartHead,['方法',sp.chart.headers[1],sp.chart.headers[2],'变化（百分点）'],sp.chart.data.map(r=>[...r,null]));sh.getRange(`D${l.chartHead+1}:D${l.chartLast}`).formulas=sp.chart.data.map((_,i)=>[`=C${l.chartHead+1+i}-B${l.chartHead+1+i}`]);sh.getRange(`D${l.chartHead+1}:D${l.chartLast}`).setNumberFormat('+0.0000;-0.0000;0.0000');}
  section(sh,l.detailHead-2,'完整实验结果');const addr=write(sh,l.detailHead,sp.headers,sp.data);const t=sh.tables.add(addr,true,`SupplementData${si+1}`);t.style='TableStyleLight1';t.showFilterButton=true;sh.getRange(addr).format.fill='#FFFFFF';
  if(sp.copy){const q=sp.copy;sh.getRange(`${col(q.target)}${l.dataStart}:${col(q.target)}${l.dataLast}`).formulas=sp.data.map((_,i)=>[`=TEXT(${col(q.mean)}${l.dataStart+i},"0.00")&"±"&TEXT(${col(q.sd)}${l.dataStart+i},"0.00")`]);}
  if(sp.gain_formula){const q=sp.gain_formula;sh.getRange(`${col(q.target)}${l.dataStart}:${col(q.target)}${l.dataLast}`).formulas=sp.data.map((_,i)=>[`=${col(q.after)}${l.dataStart+i}-${col(q.before)}${l.dataStart+i}`]);sh.getRange(`${col(q.target)}${l.dataStart}:${col(q.target)}${l.dataLast}`).setNumberFormat('+0.0000;-0.0000;0.0000');}
  const c=sh.charts.add(sp.chart.type,sh.getRange(`A${l.chartHead}:${col(sp.chart.headers.length)}${l.chartLast}`));c.setPosition('G4','L18');styleChart(c,sp.chart.title,sp.chart.type,sp.chart.headers.slice(1),sp.chart.data.map(r=>r[0]),false,sp.chart.unit);chartCount++;bindings.push({sheet:sp.name,type:sp.chart.type,title:sp.chart.title,source:`A${l.chartHead}:${col(sp.chart.headers.length)}${l.chartLast}`});
  if(sp.jointChart){
   section(sh,21,'新增独立测试：跨日且跨接收机（目标接收机，2021年3月1日）');
   section(sh,l.jointTableHead-2,'跨日且跨接收机：独立筛选结果表');const a=write(sh,l.jointTableHead,sp.jointHeaders,sp.jointData);const jt=sh.tables.add(a,true,`JointSupplement${si+1}`);jt.style='TableStyleLight1';jt.showFilterButton=true;sh.getRange(a).format.fill='#FFFFFF';
   if(sp.copy){const q=sp.copy;sh.getRange(`${col(q.target)}${l.jointTableHead+1}:${col(q.target)}${l.jointTableLast}`).formulas=sp.jointData.map((_,i)=>[`=TEXT(${col(q.mean)}${l.jointTableHead+1+i},"0.00")&"±"&TEXT(${col(q.sd)}${l.jointTableHead+1+i},"0.00")`]);}
   if(sp.gain_formula){const q=sp.gain_formula;sh.getRange(`${col(q.target)}${l.jointTableHead+1}:${col(q.target)}${l.jointTableLast}`).formulas=sp.jointData.map((_,i)=>[`=${col(q.after)}${l.jointTableHead+1+i}-${col(q.before)}${l.jointTableHead+1+i}`]);sh.getRange(`${col(q.target)}${l.jointTableHead+1}:${col(q.target)}${l.jointTableLast}`).setNumberFormat('+0.0000;-0.0000;0.0000');}
   const jch=sp.jointChart;section(sh,l.jointChartHead-2,'双跨独立图表来源：'+jch.title);write(sh,l.jointChartHead,jch.headers,jch.data);const jc=sh.charts.add(jch.type,sh.getRange(`A${l.jointChartHead}:${col(jch.headers.length)}${l.jointChartLast}`));jc.setPosition('A23','L43');styleChart(jc,jch.title,jch.type,jch.headers.slice(1),jch.data.map(r=>r[0]),false,jch.unit);chartCount++;bindings.push({sheet:sp.name,type:jch.type,test:'cross_both',title:jch.title,source:`A${l.jointChartHead}:${col(jch.headers.length)}${l.jointChartLast}`});
   sh.mergeCells('A45:L46');sh.getRange('A45').values=[[sp.jointExplanation+' 独立筛选表起于第'+l.jointTableHead+'行。']];sh.getRange('A45:L46').format={font:{name:'SimSun',size:10,italic:true,color:'#33566D'},wrapText:true};
  }else if(sp.jointNotApplicable){sh.mergeCells('A19:E19');sh.getRange('A19').values=[[sp.jointNotApplicable]];sh.getRange('A19:E19').format={font:{name:'SimSun',size:9,italic:true},wrapText:true};sh.getRange('A19').format.rowHeight=30;}
 }
 sh.freezePanes.freezeRows(2);sh.freezePanes.freezeColumns(1);
}
// Test one primary helper dependency and restore its exact frozen value.
const sp=input.sheets.find(s=>s.slug==='main_channel_on'),l=layouts[sp.name],sh=sheets.get(sp.name),src=sh.getRange('I'+l.dataStart),old=src.values[0][0],dest=sh.getRange('B'+(l.blocks[0].head+1)),prior=dest.values[0][0];src.values=[[old+1]];wb.recalculate();if(Math.abs(dest.values[0][0]-prior-1)>1e-9)throw Error('Primary chart helper failed recalculation');src.values=[[old]];wb.recalculate();
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:2000});await fs.writeFile(path.join(base,'formula_scan.json'),scan.ndjson);
await fs.writeFile(path.join(base,'workbook_layouts.json'),JSON.stringify(layouts,null,2));await fs.writeFile(path.join(base,'chart_bindings.json'),JSON.stringify(bindings,null,2));
for(const sp of input.sheets){
 if(process.argv[3]==='lines-only'&&!(sp.charts&&sp.charts[0].type==='line'))continue;
 if(process.argv[3]==='missing-only'&&sp.slug!=='main_no_channel')continue;
 if(process.argv[3]==='units-only'&&!(sp.chart&&sp.chart.unit!=='Accuracy (%)'))continue;
 const l=layouts[sp.name];let ranges=sp.charts?[['explanation','A1:L14'],...l.blocks.map((b,i)=>['panel'+(i+1),b.range]),['partition',`A${l.partitionHead-2}:I${l.partitionLast}`],['results',`A${l.detailHead-2}:P${Math.min(l.dataLast,l.dataStart+5)}`]]:[['overview','A1:L18']];
 if(process.argv[3]==='joint-only'){const jr=sp.charts?Math.min(...l.ids.filter(id=>records.get(id).test==='cross_both').map(id=>l.index[id])):0;ranges=sp.charts?[...l.blocks.flatMap((b,i)=>b.joint?[['panel'+(i+1),b.range]]:[]),['partition',`A${l.partitionHead-2}:I${l.partitionLast}`],['joint-results',`A${jr}:P${Math.min(l.dataLast,jr+5)}`]]:sp.jointChart?[['joint-chart','A21:L46'],['joint-table',`A${l.jointTableHead-2}:L${Math.min(l.jointTableLast,l.jointTableHead+6)}`]]:[];}
 for(const [suffix,range] of ranges){const b=await wb.render({sheetName:sp.name,range,scale:1,format:'png'});await fs.writeFile(path.join(dir,sp.name+'-'+suffix+'.png'),new Uint8Array(await b.arrayBuffer()));}
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(input.output);console.log(JSON.stringify({status:'EXPORTED',sheets:sheets.size,native_charts:chartCount,primary_recalculation_verified:true,output:input.output}));
