import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const base=process.argv[2] || path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/,'$1');
const input=JSON.parse(await fs.readFile(path.join(base,'weekly_workbook_data.json'),'utf8'));
const wb=Workbook.create(),sheets=new Map(),layouts={};
const previewDir=path.join(base,'workbook_previews');await fs.mkdir(previewDir,{recursive:true});
const col=n=>{let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
for(const sp of input.sheets){
 sheets.set(sp.name,wb.worksheets.add(sp.name));
 const ph=22,pl=ph+sp.partitionData.length,ch=pl+5,cl=ch+sp.chart.data.length,dh=cl+5;
 layouts[sp.name]={partitionHead:ph,partitionLast:pl,chartHead:ch,chartLast:cl,detailHead:dh,dataStart:dh+1,dataLast:dh+sp.data.length};
}
function resolve(token){
 const bits=token.split(':'),kind=bits[0],name=bits[1],row=layouts[name].dataStart+Number(bits[2])-5;
 if(kind==='@copy')return `=TEXT('${name}'!${bits[3]||'I'}${row},"0.00")&"±"&TEXT('${name}'!${bits[4]||'J'}${row},"0.00")`;
 if(kind==='@num')return `='${name}'!${bits[3]}${row}`;
 throw new Error('Unknown linked value: '+token);
}
function write(sh,startRow,headers,data){
 const last=col(headers.length),end=startRow+data.length;
 sh.getRange(`A${startRow}:${last}${startRow}`).values=[headers];
 sh.getRange(`A${startRow+1}:${last}${end}`).values=data.map(r=>r.map(v=>typeof v==='string'&&v.startsWith('@')?null:v));
 for(let j=0;j<headers.length;j++){
   if(data.some(r=>typeof r[j]==='string'&&r[j].startsWith('@'))){
     data.forEach((r,i)=>{if(typeof r[j]==='string'&&r[j].startsWith('@'))sh.getRange(`${col(j+1)}${startRow+1+i}`).formulas=[[resolve(r[j])]];});
   }
 }
 sh.getRange(`A${startRow}:${last}${end}`).format.fill='#FFFFFF';
 sh.getRange(`A${startRow}:${last}${startRow}`).format={font:{name:'SimSun',size:10,bold:true,color:'#000000'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,borders:{top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
 sh.getRange(`A${startRow}:${last}${startRow}`).format.rowHeight=34;
 sh.getRange(`A${end}:${last}${end}`).format.borders={bottom:{style:'medium',color:'#000000'}};
 sh.getRange(`A${startRow+1}:${last}${end}`).format.rowHeight=22;
 for(let j=0;j<headers.length;j++){
   const rs=sh.getRange(`${col(j+1)}${startRow+1}:${col(j+1)}${end}`);
   const numeric=data.some(r=>typeof r[j]==='number'||(typeof r[j]==='string'&&r[j].startsWith('@num:')));
   if(numeric){rs.setNumberFormat(/种子|类别|包|池|批量|模型|数量|总数|独立|展示项/.test(headers[j])&&!/均值|标准差/.test(headers[j])?'0':'0.0000');rs.format.horizontalAlignment='right';}
   else if(data.some(r=>typeof r[j]==='string'&&/[\u3400-\u9fff]/.test(r[j])))rs.format.font.name='SimSun';
   if(/均值±|可复制/.test(headers[j]))rs.format.horizontalAlignment='center';
 }
 return `A${startRow}:${last}${end}`;
}
function section(sh,row,title){
 sh.getRange(`A${row}`).values=[[title]];sh.getRange(`A${row}`).format.font={name:'SimSun',size:11,bold:true,color:'#000000'};
}
let chartCount=0;
for(const [index,sp] of input.sheets.entries()){
 const sh=sheets.get(sp.name),l=layouts[sp.name],endCol=col(Math.max(11,sp.headers.length));
 sh.showGridLines=false;sh.getRange(`A1:${endCol}${l.dataLast}`).format={font:{name:'Times New Roman',size:10,color:'#000000'},fill:'#FFFFFF',verticalAlignment:'center'};
 const widths=Array.from({length:Math.max(11,sp.headers.length)},(_,i)=>sp.widths[i]||21);
 widths.forEach((v,j)=>sh.getRange(`${col(j+1)}1:${col(j+1)}${l.dataLast}`).format.columnWidth=v);
 sh.getRange('A2').values=[[sp.name.replaceAll('_',' ')]];sh.getRange('A2').format.font={name:'SimSun',size:14,bold:true};sh.getRange('A2').format.rowHeight=27;
 section(sh,4,'实验说明');
 for(let i=0;i<sp.explain.length;i++){
   const r=i+5;sh.getRange(`A${r}`).values=[[sp.explain[i][0]]];sh.getRange(`A${r}`).format.font={name:'SimSun',size:10,bold:true};
   sh.mergeCells(`B${r}:E${r}`);sh.getRange(`B${r}`).values=[[sp.explain[i][1]]];sh.getRange(`B${r}:E${r}`).format={font:{name:'SimSun',size:10},wrapText:true,verticalAlignment:'center'};
   sh.getRange(`A${r}:E${r}`).format.rowHeight=sp.explain[i][1].length>68?44:32;
 }
 sh.mergeCells('A18:E18');sh.getRange('A18').values=[[sp.chart.note]];sh.getRange('A18:E18').format={font:{name:'SimSun',size:9,italic:true},wrapText:true};sh.getRange('A18').format.rowHeight=28;
 section(sh,20,'数据划分：计数为每个物理记录集合，测试为每场景包数');
 write(sh,l.partitionHead,sp.partitionHeaders,sp.partitionData);
 sh.getRange(`A${l.partitionHead+1}:K${l.partitionLast}`).format.rowHeight=34;
 sh.getRange(`C${l.partitionHead+1}:D${l.partitionLast}`).format={font:{name:'Times New Roman',size:10},wrapText:true};
 sh.getRange(`B${l.partitionHead+1}:D${l.partitionLast}`).format.horizontalAlignment='center';
 sh.getRange(`H${l.partitionHead+1}:H${l.partitionLast}`).format.horizontalAlignment='center';
 sp.partitionData.forEach((r,i)=>{if(r[0]==='大规模十类数据')sh.getRange(`A${l.partitionHead+1+i}:K${l.partitionHead+1+i}`).format.rowHeight=66;});
 if(sp.partitionNote){sh.mergeCells(`A${l.partitionLast+2}:K${l.partitionLast+2}`);sh.getRange(`A${l.partitionLast+2}`).values=[[sp.partitionNote]];sh.getRange(`A${l.partitionLast+2}:K${l.partitionLast+2}`).format.font={name:'SimSun',size:9,italic:true};}
 section(sh,l.chartHead-2,'结果摘要与图表数据');
 write(sh,l.chartHead,sp.chart.headers,sp.chart.data);
 if(sp.chart.delta){sh.getRange(`D${l.chartHead}`).values=[['变化（百分点）']];sh.getRange(`D${l.chartHead}`).format={font:{name:'SimSun',size:10,bold:true},horizontalAlignment:'center',borders:{top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};sh.getRange(`D${l.chartLast}`).format.borders={bottom:{style:'medium',color:'#000000'}};sh.getRange(`D${l.chartHead+1}:D${l.chartLast}`).formulas=sp.chart.data.map((_,i)=>[`=C${l.chartHead+1+i}-B${l.chartHead+1+i}`]);sh.getRange(`D${l.chartHead+1}:D${l.chartLast}`).setNumberFormat('+0.0000;-0.0000;0.0000');}
 section(sh,l.detailHead-2,'完整实验结果：可筛选，保留均值、标准差与复制列');
 const address=write(sh,l.detailHead,sp.headers,sp.data);const t=sh.tables.add(address,true,`ExperimentData${index+1}`);t.style='TableStyleLight1';t.showFilterButton=true;
 sh.getRange(address).format.fill='#FFFFFF';
 if(sp.copy){const {target,mean,sd}=sp.copy;sh.getRange(`${col(target)}${l.dataStart}:${col(target)}${l.dataLast}`).formulas=sp.data.map((_,i)=>[`=TEXT(${col(mean)}${l.dataStart+i},"0.00")&"±"&TEXT(${col(sd)}${l.dataStart+i},"0.00")`]);}
 if(sp.gain_formula){const g=sp.gain_formula;sh.getRange(`${col(g.target)}${l.dataStart}:${col(g.target)}${l.dataLast}`).formulas=sp.data.map((_,i)=>[`=${col(g.after)}${l.dataStart+i}-${col(g.before)}${l.dataStart+i}`]);sh.getRange(`${col(g.target)}${l.dataStart}:${col(g.target)}${l.dataLast}`).setNumberFormat('+0.0000;-0.0000;0.0000');}
 const chart=sh.charts.add(sp.chart.type,sh.getRange(`A${l.chartHead}:${col(sp.chart.headers.length)}${l.chartLast}`));
 chart.setPosition('G4','L18');chart.title=sp.chart.title;chart.titleTextStyle.typeface='SimSun';chart.titleTextStyle.fontSize=15;
 chart.hasLegend=sp.chart.headers.length>2;chart.legend={position:'bottom',textStyle:{typeface:'Times New Roman',fontSize:11}};
 chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Times New Roman',fontSize:11}};
 chart.yAxis={numberFormatCode:'0.0',numberFormatSourceLinked:false,textStyle:{typeface:'Times New Roman',fontSize:11}};
 chart.yAxis.title.text=sp.chart.unit;
 if(sp.chart.type==='line')chart.xAxis.title.text=sp.chart.headers[0];
 if(sp.chart.type==='bar'){chart.barOptions.direction=sp.chart.orientation;chart.barOptions.grouping='clustered';chart.barOptions.gapWidth=85;}
 const colors=['#333333','#999999','#666666','#BBBBBB','#111111','#777777'];
 chart.series.items.forEach((series,j)=>{series.fill=colors[j%colors.length];if(sp.chart.type==='line')series.line={fill:colors[j%colors.length],style:j%2?'dashed':'solid',width:2};});
 chartCount++;sh.freezePanes.freezeRows(2);sh.freezePanes.freezeColumns(1);
}
// Test the summary and chart-data chain, then restore the frozen source.
const testSpec=input.sheets.find(s=>s.name==='半监督_标准对比'),token=testSpec.chart.data[0][1].split(':'),sourceName=token[1],row=layouts[sourceName].dataStart+Number(token[2])-5;
const src=sheets.get(sourceName),cell=src.getRange(`${token[3]}${row}`),old=cell.values[0][0],summary=sheets.get(testSpec.name),dest=summary.getRange(`B${layouts[testSpec.name].chartHead+1}`),before=dest.values[0][0];
cell.values=[[old+1]];wb.recalculate();if(Math.abs(dest.values[0][0]-before-1)>1e-9)throw new Error('Linked summary did not update');cell.values=[[old]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});
await fs.writeFile(path.join(base,'formula_scan.json'),errors.ndjson);
await fs.writeFile(path.join(base,'workbook_layouts.json'),JSON.stringify(layouts,null,2));
const inspections=[];
for(const sp of input.sheets){
 const sh=sheets.get(sp.name),l=layouts[sp.name];inspections.push({sheet:sp.name,charts:sh.charts.items.map(c=>({title:String(c.title.text),series:c.series.items.map(s=>({values:s.formula,categories:s.categoryFormula}))}))});
 for(const [suffix,range] of [['overview','A1:L18'],['partition',`A20:K${l.partitionLast}`],['results',`A${l.chartHead-2}:${col(Math.max(sp.chart.headers.length,sp.headers.length))}${Math.min(l.dataLast,l.dataStart+5)}`]]){
  const view=await wb.render({sheetName:sp.name,range,scale:1,format:'png'});await fs.writeFile(path.join(previewDir,sp.name+'-'+suffix+'.png'),new Uint8Array(await view.arrayBuffer()));
 }
}
await fs.writeFile(path.join(base,'chart_bindings.json'),JSON.stringify(inspections,null,2));
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(input.output);
console.log(JSON.stringify({status:'EXPORTED',output:input.output,sheets:sheets.size,charts:chartCount,summary_recalculation_verified:true}));
