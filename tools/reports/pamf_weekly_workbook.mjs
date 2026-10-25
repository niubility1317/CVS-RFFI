import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const base=process.argv[2] || path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/,'$1');
const input=JSON.parse(await fs.readFile(path.join(base,'weekly_workbook_data.json'),'utf8'));
const wb=Workbook.create();
const previewDir=path.join(base,'workbook_previews');await fs.mkdir(previewDir,{recursive:true});
const col=n=>{let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const sheets=new Map();
const summary=wb.worksheets.add('周报摘要');sheets.set('周报摘要',summary);
for(const sp of input.sheets)sheets.set(sp.name,wb.worksheets.add(sp.name));
function baseStyle(sh,end){
  sh.showGridLines=false;
  sh.getRange(`A1:${end}`).format.font={name:'Times New Roman',size:10,color:'#000000'};
  sh.getRange(`A1:${end}`).format.fill='#FFFFFF';
  sh.getRange(`A1:${end}`).format.verticalAlignment='center';
  sh.getRange('A2').format.font={name:'SimSun',size:14,bold:true,color:'#000000'};
}
for(const [index,sp] of input.sheets.entries()){
  const sh=sheets.get(sp.name),last=col(sp.headers.length),end=sp.data.length+4;
  baseStyle(sh,`${last}${end}`);
  sh.getRange('A2').values=[[sp.name.replaceAll('_',' ')]];
  sh.getRange('A3').values=[[sp.note]];
  sh.getRange('A3').format.font={name:'SimSun',size:9,italic:true,color:'#333333'};
  sh.getRange(`A4:${last}4`).values=[sp.headers];
  sh.getRange(`A5:${last}${end}`).values=sp.data.map(r=>r.map(v=>typeof v==='string'&&v.startsWith('@copy:')?null:v));
  const table=sh.tables.add(`A4:${last}${end}`,true,`ResearchTable${index+1}`);table.style='TableStyleLight1';table.showFilterButton=true;
  sh.getRange(`A4:${last}${end}`).format.fill='#FFFFFF';
  sh.getRange(`A4:${last}4`).format={fill:'#FFFFFF',font:{name:'SimSun',size:10,bold:true,color:'#000000'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,borders:{top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
  sh.getRange(`A${end}:${last}${end}`).format.borders={bottom:{style:'medium',color:'#000000'}};
  sh.getRange(`A4:${last}4`).format.rowHeight=34;
  sh.getRange(`A5:${last}${end}`).format.rowHeight=21;
  sp.widths.forEach((width,j)=>sh.getRange(`${col(j+1)}1:${col(j+1)}${end}`).format.columnWidth=width);
  for(let j=0;j<sp.headers.length;j++){
    const sample=sp.data.find(r=>typeof r[j]==='number');
    if(sample){const range=sh.getRange(`${col(j+1)}5:${col(j+1)}${end}`);range.setNumberFormat(/种子|数量|批量|种子数/.test(sp.headers[j])?'0':'0.0000');range.format.horizontalAlignment='right';}
    else if(sp.data.some(r=>typeof r[j]==='string'&&/[\u3400-\u9fff]/.test(r[j])))sh.getRange(`${col(j+1)}5:${col(j+1)}${end}`).format.font.name='SimSun';
  }
  if(sp.copy){const {target,mean,sd}=sp.copy;sh.getRange(`${col(target)}5:${col(target)}${end}`).formulas=sp.data.map((_,i)=>[`=TEXT(${col(mean)}${i+5},"0.00")&"±"&TEXT(${col(sd)}${i+5},"0.00")`]);sh.getRange(`${col(target)}5:${col(target)}${end}`).format.horizontalAlignment='center';}
  if(sp.name==='主比较_11设置'){
    sh.getRange(`C5:H${end}`).formulas=sp.data.map(r=>r.slice(2).map(x=>{const [_,name,row]=x.split(':');return `=TEXT('${name}'!F${row},"0.00")&"±"&TEXT('${name}'!G${row},"0.00")`;}));
  }
  if(sp.name==='DURA泛用性'){
    sh.getRange(`H5:H${end}`).formulas=sp.data.map((_,i)=>[`=G${i+5}-F${i+5}`]);
    sh.getRange(`H5:H${end}`).setNumberFormat('+0.0000;-0.0000;0.0000');
  }
  sh.freezePanes.freezeRows(4);sh.freezePanes.freezeColumns(2);
}
baseStyle(summary,'H27');
summary.getRange('A2').values=[['PAMF 与对比方法周报摘要']];
summary.getRange('A3').values=[['截至2026-10-25；标准接收机划分、跨接收机测试；单位为准确率%。']];
summary.getRange('A4:G4').values=[['方法','CLEAN均值','CLEAN标准差','六景均值','六景标准差','CLEAN可复制','六景可复制']];
summary.getRange('A4:G4').format={fill:'#FFFFFF',font:{name:'SimSun',size:10,bold:true,color:'#000000'},horizontalAlignment:'center',wrapText:true,borders:{top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
summary.getRange('A4:G4').format.rowHeight=30;
for(let i=0;i<input.summary.length;i++){
  const r=input.summary[i],n=i+5;summary.getRange(`A${n}`).values=[[r.method]];
  summary.getRange(`B${n}:G${n}`).formulas=[[`='主比较_数值'!F${r.clean}`,`='主比较_数值'!G${r.clean}`,`='主比较_数值'!F${r.practical}`,`='主比较_数值'!G${r.practical}`,`=TEXT(B${n},"0.00")&"±"&TEXT(C${n},"0.00")`,`=TEXT(D${n},"0.00")&"±"&TEXT(E${n},"0.00")`]];
}
summary.getRange('B5:E10').setNumberFormat('0.0000');summary.getRange('A5:G10').format.rowHeight=25;
summary.getRange('F5:G10').format.horizontalAlignment='center';
summary.getRange('A10:G10').format.borders={bottom:{style:'medium',color:'#000000'}};
summary.getRange('A12').values=[['接入DURA提升']];summary.getRange('B12').formulas=[['=D10-D9']];summary.getRange('B12').setNumberFormat('+0.0000;-0.0000;0.0000');summary.getRange('C12').values=[['百分点']];
summary.getRange('A14:D18').values=[['实验分组','逻辑模型数','当前报告范围','章节位置'],['外部无信道增强',264,'目标接收机CLEAN','无增强CLEAN控制'],['外部有信道增强',264,'CLEAN与六种practical','有增强控制与主比较'],['PAMF监督控制',33,'CLEAN与六种practical','PAMF自身对照'],['PAMF普通伪标签',33,'CLEAN与六种practical','PAMF、DURA+PAMF']];
summary.getRange('A14:D14').format={fill:'#FFFFFF',font:{name:'SimSun',size:10,bold:true,color:'#000000'},horizontalAlignment:'center',borders:{top:{style:'medium',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
summary.getRange('A18:D18').format.borders={bottom:{style:'medium',color:'#000000'}};
summary.getRange('A14:D18').format.rowHeight=26;
summary.getRange('A20:B25').values=[['主表推理链','前五种方法直接识别；仅DURA+PAMF先恢复。'],['独立模型计数','主比较165个识别器、198个模型与推理链展示项。'],['DURA泛用性','其他方法接DURA单独列出；保留CLEAN损伤和负收益。'],['本次报告新增','训练0、预测0、真值评分0、DURA训练0。'],['统计定义','三模型种子样本标准差；先种子内配对再汇总。'],['结果边界','WiSig地面代理与信道仿真，不能视作真实在轨验证。']];
summary.getRange('A20:B25').format.rowHeight=24;
summary.getRange('A12').format.font.name='SimSun';summary.getRange('C12').format.font.name='SimSun';
summary.getRange('A14:A25').format.font.name='SimSun';summary.getRange('B20:B25').format.font.name='SimSun';summary.getRange('C15:D18').format.font.name='SimSun';
const sw=[28,19,23,27,19,24,24];sw.forEach((v,i)=>summary.getRange(`${col(i+1)}1:${col(i+1)}27`).format.columnWidth=v);
// Verify dependent summary updates, then restore the frozen numerical source.
const sh=sheets.get('主比较_数值'),sourceRow=input.summary[0].practical,cell=sh.getRange(`F${sourceRow}`),before=cell.values[0][0];
const expected=summary.getRange('D5').values[0][0];cell.values=[[before+1]];wb.recalculate();
if(Math.abs(summary.getRange('D5').values[0][0]-(expected+1))>1e-9)throw new Error('Linked summary did not recalculate');
cell.values=[[before]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:2000});
await fs.writeFile(path.join(base,'workbook_formula_scan.json'),errors.ndjson);
const inspection=await wb.inspect({kind:'table',range:'周报摘要!A4:G12',include:'values,formulas',tableMaxRows:9,tableMaxCols:7,maxChars:5000});
await fs.writeFile(path.join(base,'workbook_summary_inspection.json'),inspection.ndjson);
for(const name of sheets.keys()){
 const range=name==='周报摘要'?'A1:G25':name==='主比较_11设置'?'A1:H14':'A1:J11';
 const view=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(previewDir,name+'.png'),new Uint8Array(await view.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(input.output);
console.log(JSON.stringify({status:'EXPORTED',output:input.output,sheets:sheets.size,summary_recalculation_verified:true}));
