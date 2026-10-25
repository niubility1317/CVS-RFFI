'use strict';
const katex=require('./vendor/katex/katex.min.js');
function checkMath(chapters){
 const issues=[];let formulas=0;
 for(const c of chapters){
  const body=[c.lead||'',c.html].join('\n');
  if(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\t]/.test(body))issues.push(c.id+': unexpected control character');
  const visible=body.replace(/<(?:code|pre)\b[^>]*>[\s\S]*?<\/(?:code|pre)>/g,'');
  const text=visible.replace(/\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)/g,(_,display,inline)=>{
   const tex=display||inline;
   try{katex.renderToString(tex,{throwOnError:true,strict:'ignore'});formulas++;}catch(e){issues.push(c.id+': '+e.message);}
   return ' FORMATTEDMATH ';
  }).replace(/<[^>]*>/g,' ');
  const leftover=text.match(/\\[A-Za-z()[\]]|\b(?:operatorname|mathrm|ldots|partial)\b|\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9{]/g);
  if(leftover)issues.push(c.id+': unformatted math or identifier: '+[...new Set(leftover)].join(', '));
 }
 return{formulas,issues};
}
module.exports={checkMath};
