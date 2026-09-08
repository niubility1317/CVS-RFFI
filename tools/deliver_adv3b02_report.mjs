// Use the standard portable reader/probes with real elapsed browser time on
// Windows, where dump-DOM virtual time outruns the reader's load/rAF bootstrap.
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import assert from 'node:assert/strict';
const plugin='E:/codex/home/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599/skills/build-report/scripts/';
const out='E:/type10-7/automation_reports/CV-SincNet/adv3b02_all_batches_full_report_20260908/';
const diagnostic='E:/type10-7/local_artifacts/adv3b02_all_batches_full_report_20260908/';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/lh594/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {deliverPortableArtifact}=await import(pathToFileURL(plugin+'deliver_portable_artifact.mjs'));
const {extractPortableChartSvgs}=await import(pathToFileURL(plugin+'extract_portable_chart_svgs.mjs'));
const {verifyPortableArtifactStructure}=await import(pathToFileURL(plugin+'verify_portable_artifact.mjs'));
const {resolveChromiumExecutable}=await import(pathToFileURL(plugin+'portable_browser_helpers.mjs'));
const {injectPortableVerifierProbe,buildPortableVerifierHarness,parsePortableVerifierDump}=await import(pathToFileURL(plugin+'portable_browser_cli.mjs'));
const browser=await chromium.launch({executablePath:resolveChromiumExecutable(),headless:true});
async function readBrowser(url,selector,viewport,colorScheme='light'){
  const page=await browser.newPage({viewport,colorScheme,reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',async route=>{
    const protocol=new URL(route.request().url()).protocol;
    return ['file:','data:','blob:','about:'].includes(protocol)?route.continue():route.abort();
  });
  try{
    await page.goto(url,{waitUntil:'load',timeout:15000});
    await page.locator(selector).waitFor({state:'attached',timeout:20000});
    if(selector==='#data-analytics-portable-verifier-result'){
      const widths=[];
      for(const frame of page.frames().slice(1))widths.push(await frame.evaluate(()=>({width:innerWidth,wide:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+2).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,160),right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).slice(0,80)})));
      writeFileSync(diagnostic+'reader_layout_diagnostics.json',JSON.stringify(widths,null,2)+'\n','utf8');
    }
    assert.equal(errors.length,0,JSON.stringify(errors));
    return await page.content();
  }catch(e){
    await page.screenshot({path:diagnostic+'reader_failure.png',fullPage:false});
    e.message+='; errors='+JSON.stringify(errors)+'; state='+await page.evaluate(()=>document.documentElement.dataset.dataAnalyticsPortableReader);
    throw e;
  }finally{await page.close();}
}
async function realTimeDump(options){
  const args=options.arguments,url=args.find(a=>a.startsWith('file:'));
  const size=args.find(a=>a.startsWith('--window-size=')).split('=')[1].split(',').map(Number);
  const dark=args.includes('--blink-settings=preferredColorScheme=0');
  return {stdout:await readBrowser(url,'meta[data-portable-chart-extraction]',{width:size[0],height:size[1]},dark?'dark':'light'),stderr:''};
}
async function verifyWithStandardProbes(options){
  const structure=verifyPortableArtifactStructure(options),html=readFileSync(options.htmlPath,'utf8');
  const directory=mkdtempSync(join(tmpdir(),'adv3b02-reader-qa-'));
  const channel=randomUUID();
  const viewports=[{name:'desktop',width:1440,height:1000},{name:'mobile',width:390,height:844}];
  const frames=viewports.map(viewport=>({viewport,html:injectPortableVerifierProbe(html,{actionTimeoutMs:5000,channel,checkSource:viewport.name==='desktop',expectedCounts:structure.counts,readerRoot:'#data-analytics-portable-reader-root',readyTimeoutMs:15000,title:structure.title,viewport})}));
  try{
    const path=join(directory,'harness.html');
    writeFileSync(path,buildPortableVerifierHarness({channel,frames,timeoutMs:21000}),'utf8');
    const dump=await readBrowser(pathToFileURL(path).href,'#data-analytics-portable-verifier-result',{width:1846,height:1000});
    const result=parsePortableVerifierDump(dump);
    writeFileSync(out+'reader_probe_results.json',JSON.stringify(result,null,2)+'\n','utf8');
    if(result.results?.some(r=>!r.ok)){
      const p=await browser.newPage({viewport:{width:390,height:844}});
      await p.goto(pathToFileURL(options.htmlPath).href,{waitUntil:'load'});
      await p.waitForFunction(()=>document.documentElement.dataset.dataAnalyticsPortableReader==='ready',null,{timeout:15000});
      const widths=await p.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+2).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent.slice(0,200),right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})).slice(0,120)}));
      writeFileSync(diagnostic+'reader_mobile_overflow.json',JSON.stringify(widths,null,2)+'\n','utf8');
      await p.screenshot({path:diagnostic+'reader_mobile_failure.png',fullPage:false});await p.close();
    }
    assert.equal(result.ok,true,JSON.stringify(result));
    assert.equal(result.results.length,2);
    for(const r of result.results)assert.equal(r.ok,true,JSON.stringify(r));
    return {...structure,sourceDialog:'verified',sourceInteraction:'verified',viewports:result.results,timings:{},runner:'Playwright real-time, unchanged standard portable probes'};
  }finally{assert.ok(directory.startsWith(join(tmpdir(),'adv3b02-reader-qa-')));rmSync(directory,{recursive:true,force:true});}
}
try{
  const result=await deliverPortableArtifact({inputPath:out+'artifact.json',outputPath:out+'report.html',readyTimeoutMs:15000,actionTimeoutMs:5000,timeoutMs:45000},{extract:options=>extractPortableChartSvgs({...options,runDump:realTimeDump}),verify:verifyWithStandardProbes});
  writeFileSync(out+'html_delivery_validation.json',JSON.stringify(result,null,2)+'\n','utf8');
  console.log(JSON.stringify(result));
}catch(e){console.error(JSON.stringify({error:e.message,details:e.details,result:e.deliveryResult}));process.exitCode=1;}
finally{await browser.close();}
