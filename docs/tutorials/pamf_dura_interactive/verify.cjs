/* Focused checks for the tutorial's mathematical examples and offline bundle. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=__dirname;
global.window={COURSE_CHAPTERS:[]};
for(const name of ['foundation-content','pamf-content','dura-content','reference-content','labs']){
 vm.runInThisContext(fs.readFileSync(path.join(root,name+'.js'),'utf8'),{filename:name+'.js'});
}
const math=window.CourseMath,near=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
assert.equal(window.COURSE_CHAPTERS.length,20);
assert.equal(new Set(window.COURSE_CHAPTERS.map(c=>c.id)).size,20);
const slots=window.COURSE_CHAPTERS.flatMap(c=>[...c.html.matchAll(/data-lab="([^"]+)"/g)].map(m=>m[1]));
assert.equal(slots.length,20);assert.equal(new Set(slots).size,20);
slots.forEach(name=>assert.ok(window.CourseLabs.names.includes(name),name));
const {formulas,issues}=require('./math-source-check.cjs').checkMath(window.COURSE_CHAPTERS);
assert.deepEqual(issues,[],'Raw mathematical notation must not bypass the renderer');
assert.ok(formulas>690,'Regression: all inline mathematics must remain formatted');
const sincChapter=window.COURSE_CHAPTERS.find(c=>c.id==='p-time');
assert.ok(sincChapter.html.includes('\\int_{-f_c}^{f_c}'));
assert.ok(sincChapter.html.includes('\\sin(2\\pi f_ct)'));
assert.ok(sincChapter.html.includes('\\operatorname{sinc}'));
assert.ok(sincChapter.html.includes('t_n='));
assert.ok(sincChapter.html.includes('w[n]='));
assert.deepEqual(math.conv([1,2,0,1],[1,.5]),[1,2.5,1,1]);
assert.deepEqual(math.conv([1,2,0,1],[1,.5],true),[2,2.5,1,1]);
const forward=[[0,0],[1,0],[1,1],[0,1],[0,0]],reverse=forward.slice().reverse();
assert.deepEqual(math.signature(forward).first,[0,0]);near(math.signature(forward).area[0],1);near(math.signature(reverse).area[0],-1);
near(math.signature(forward.slice(0,1)).area[0],0);
const p=math.softmax([1002,1001,1000]),q=math.softmax([2,1,0]);p.forEach((v,i)=>near(v,q[i]));near(p.reduce((a,b)=>a+b,0),1);near(math.kl(p,p),0);
const target=[1,0,0];near(math.ce(q,target),-Math.log(q[0]));
const x=math.map(.8,1.2,.4,.01),derivative=2*.8*(.8*x-1.2)/.01+2*(x-.4)/(.1169459182403555**2);near(derivative,0);
assert.ok(Math.abs(math.map(.8,1.2,.4,1)-.4)<Math.abs(math.map(.8,1.2,.4,.001)-.4));
for(const u of [[-3,2],[3,2],[-1,-8],[0,0]]){const g=math.guard([1,0],u);assert.ok(math.dot(g,[1,0])>=-1e-12);assert.ok(math.norm(g)<=1+1e-12);}
assert.deepEqual(math.guard([0,0],[1,2]),[0,0]);
for(const c of [-1.5,0,1,1.5])for(const start of [[0,0],[-.7,-.3],[1,-1]]){
 const trace=math.cg([[4,c],[c,3]],[1,2],start,2),last=trace[2];assert.ok(last.res<1e-10);near(last.x[0],(3-2*c)/(12-c*c));near(last.x[1],(8-c)/(12-c*c));
}
const N=32,re=Array.from({length:N},(_,n)=>Math.cos(2*Math.PI*3*n/N)),im=Array.from({length:N},(_,n)=>Math.sin(2*Math.PI*3*n/N));
const spectrum=math.dft(re,im);near(spectrum.r[3],N);near(spectrum.i[3],0);near(spectrum.r[29],0);
assert.equal(window.COURSE_QUIZ.length,10);window.COURSE_QUIZ.forEach(q=>assert.ok(q.correct>=0&&q.correct<q.options.length));
const output=fs.readFileSync(path.join(root,'dist/index.html'),'utf8');
assert.ok(output.startsWith('<!doctype html>'));assert.ok(!/url\(fonts\//.test(output));assert.ok(!/<(?:script|link|iframe)[^>]+(?:src|href)=["']https?:\/\//.test(output));assert.ok(!/\/\* (?:APP_JS|PAMF_CONTENT|DURA_CONTENT) \*\//.test(output));
console.log(JSON.stringify({status:'PASS',chapters:20,interactive_examples:slots.length,validated_formulas:formulas,questions:10,math:'FIR, DFT, signed area, stable Softmax, CE/KL, scalar MAP, gradient projection, warm-start CG',offline_bytes:Buffer.byteLength(output)},null,2));
