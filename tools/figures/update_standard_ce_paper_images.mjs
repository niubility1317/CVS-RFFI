// Refresh only the paper-image layout in an existing research workbook.
import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';

const [input,output,figureDir,previewDir]=process.argv.slice(2);
if(!input||!output||!figureDir||!previewDir)throw new Error('Expected INPUT OUTPUT FIGURES PREVIEWS');
try{await fs.access(output);throw new Error('Preserve existing output before editing');}catch(e){if(e.code!=='ENOENT')throw e;}
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const paper=wb.worksheets.getItem('论文图表');
await fs.mkdir(previewDir,{recursive:true});
const before=await wb.render({sheetName:'论文图表',range:'A90:N134',scale:1,format:'png'});
await fs.writeFile(path.join(previewDir,'before.png'),new Uint8Array(await before.arrayBuffer()));
const manifest=JSON.parse(await fs.readFile(path.join(figureDir,'manifest.json'),'utf8'));
if(manifest.figures.length!==6||Object.keys(manifest.seeds_by_network).length!==13||Object.values(manifest.seeds_by_network).some(s=>s.length!==8))throw new Error('Unexpected result snapshot');
paper.deleteAllDrawings();
paper.getRange('A6:A350').values=Array.from({length:345},()=>[null]);
const titles=['总体准确率与Macro-F1','接收机准确率热图','参数量与准确率：全模型及局部放大','架构配对差','CVS接收机与发射机表现','CVS与ResNet-50混淆矩阵'];
let row=6,parameterRow;
for(const [i,figure] of manifest.figures.entries()){
    paper.getRange(`A${row}`).values=[[`图 ${i+1}：${titles[i]}`]];
    paper.getRange(`A${row}`).format.font={name:'Microsoft YaHei',size:13,bold:true,color:'#263445'};
    const bytes=await fs.readFile(path.join(figureDir,'png',figure.stem+'.png'));
    const width=1180,height=width*figure.height_mm/figure.width_mm;
    paper.images.add({dataUrl:'data:image/png;base64,'+bytes.toString('base64'),
        anchor:{from:{row:row+1,col:0},extent:{widthPx:width,heightPx:height}}});
    if(i===2)parameterRow=row;
    row+=Math.ceil(height/24)+6;
}
const after=await wb.render({sheetName:'论文图表',range:`A${parameterRow}:N${parameterRow+42}`,scale:1,format:'png'});
await fs.writeFile(path.join(previewDir,'after.png'),new Uint8Array(await after.arrayBuffer()));
const exported=await SpreadsheetFile.exportXlsx(wb);await exported.save(output);
console.log(JSON.stringify({status:'EXPORTED',paperFigures:6,parameterRow,output}));
