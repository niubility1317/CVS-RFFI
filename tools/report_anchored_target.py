"""Render verified all-candidate target replay with complete group tables."""
import json,csv,shutil
from pathlib import Path
B=Path('analysis/core90_anchored_target_all_20260912')
d=json.loads((B/'scores.json').read_text());a=json.loads((B/'independent_recount.json').read_text());rows=d['records']
names=['A0','A1','A2','A3','A4','C_angle','C_angle_keep','P1','A5-0','A5-25','A5-50','A5-75','A5-100','A6'];scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
assert len(rows)==3360 and len(d['confusions'])==56 and a['recount_verified']
errors=[];loglines=0
for p in (B/'logs').glob('*.log'):
    lines=p.read_text(encoding='utf-8').splitlines();loglines+=len(lines)
    errors += [dict(file=p.name,line=x) for x in lines if any(s in x for s in ['Traceback','Error:','CUDA out of memory'])]
assert not errors
for scene in scenes:
    s=json.loads((B/scene/'seal.json').read_text());assert s['status']=='SEALED' and s['rows']==168000 and s['candidates']==names
def get(n,s,g='all',v='all'):return next(x for x in rows if (x['candidate'],x['scene'],x['group'],x['value'])==(n,s,g,v))
for n in names:
    for s in scenes:
        allrow=get(n,s)
        for g in ['RX','TX','day','RX_TX']:
            subset=[x for x in rows if x['candidate']==n and x['scene']==s and x['group']==g]
            assert sum(x['rows'] for x in subset)==allrow['rows']
            assert sum(x['correct'] for x in subset)==allrow['correct']
def pc(x):return 'N/A' if x is None else f'{100*x:.4f}'
def tab(h,r):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,x))+'|' for x in r])
main=[]
for n in names:
    vals=[get(n,s)['closed_accuracy'] for s in scenes];base=[get('A0',s)['closed_accuracy'] for s in scenes]
    main.append([n]+[pc(v) for v in vals]+[pc(sum(vals[1:])/3),f'{100*sum(v-b for v,b in zip(vals[1:],base[1:]))/3:+.4f}'])
out=['# 全14候选目标域详细测试结果','',
'状态：VERIFIED / SCORED。所有候选均已实际完成；56份预测封存后独立连接真值，随后再次从保存的预测独立复核overall正确数、接受数、接受正确数，全部一致。','',
'Run：core90_anchored_target_all_s392005_20260912_r1。实际代码18befc4b2c1d0d2cfcc1936af664678d8c29dc26。冻结source run：core90_anchored_s392005_20260911_r1。seed392005；不重新训练、校准、适应或选模。A0是冻结H0对照；P1使用full输入。','',
'## 测试口径','',
'ManySig equalized=1，256点中心裁剪、RMS归一化、center=false。目标RX=0/2/5/7/9/10/11，day=0/1/2/3，六TX，每个场景168000个相同物理ID；14候选×4场景共9408000次候选决策。每RX每场景24000条，每TX每场景28000条，每RX×TX每场景4000条。LEO随机数沿用sha256(seed,scene,opaque ID)逐记录CPU生成。','',
'这是用户明确授权的全部冻结候选历史目标集复测，各场景为同一集合的不同压力测试。结果是描述性比较，不能当成独立确认、真实在轨验证或target驱动选模依据；不据本表调参、晋级、重训或选择性重跑。','',
'## 全样本闭集准确率','',
'以下单位为%；包含被拒识样本的原始top-class判断。LEO均值为三个等样本量弱信道场景的算术平均，不混入clean。Δ单位为百分点。','',
tab(['候选','clean','clear','低仰角','雨衰','LEO均值','ΔLEO vs H0'],main),'',
'## 拒识覆盖率与接受准确率','',
'每格为coverage / accepted accuracy（%）。接受准确率仅统计被接受样本，不等同于全样本准确率；拒识阈值沿用V冻结值，没有按target重新设为90%覆盖率。','',
tab(['候选']+scenes,[[n]+[pc(get(n,s)['coverage'])+' / '+pc(get(n,s)['accepted_accuracy']) for s in scenes] for n in names]),'',
'## 相对H0救回与损伤','',
'每格为救回/损伤/净正确数；分别指H0错→候选对、H0对→候选错以及二者之差。同一物理样本跨场景重复计入时，不作独立样本显著性声明。','',
tab(['候选']+scenes,[[n]+[('/'.join(str(x[k]) for k in ['rescue','harm','net'])) for s in scenes for x in [next(z for z in a['paired'] if z['candidate']==n and z['scene']==s)]] for n in names]),'',
'## 按接收机的全部候选准确率','']
for s in scenes:
    out += [f'### {s}','',tab(['候选']+[str(rx) for rx in [0,2,5,7,9,10,11]],[[n]+[pc(get(n,s,'RX',str(rx))['closed_accuracy']) for rx in [0,2,5,7,9,10,11]] for n in names]),'']
out += ['## 按TX的全部候选准确率','']
for s in scenes:
    out += [f'### {s}','',tab(['候选']+a['tx_mapping'],[[n]+[pc(get(n,s,'TX',str(tx))['closed_accuracy']) for tx in range(6)] for n in names]),'']
out += ['## 最差接收机与最差RX×TX','',tab(['候选','场景','最差RX准确率','最差RX×TX准确率'],[[n,s,pc(min(x['closed_accuracy'] for x in rows if x['candidate']==n and x['scene']==s and x['group']=='RX')),pc(min(x['closed_accuracy'] for x in rows if x['candidate']==n and x['scene']==s and x['group']=='RX_TX'))] for n in names for s in scenes]),'',
'## 执行与完整性','',f'完整读取{len(list((B/"logs").glob("*.log")))}份日志，共{loglines}行，无Traceback/OOM/运行错误。四个worker退出码均为0；所有56份场景预测均SEALED；scorer打印ALL_SCORED，协调器写入SCORED。3360条分组记录按RX/TX/day/RX×TX重新求和均与overall一致；保存56个6×6混淆矩阵。','',
tab(['场景','共享预测阶段秒数'],[[s,f"{json.loads((B/s/'seal.json').read_text())['seconds']:.3f}"] for s in scenes]),'',
'计时包含共享IQ读取、信道生成、骨干前向、14候选推理及保存，不是单候选或单kernel延迟。各scene GPU0/1/2/3单worker，原GPU4任务未变。','',
'## 文件','',
'scores.csv为全部3360条overall/RX/TX/day/RX×TX统计；scores.json另含56个混淆矩阵；independent_recount.json保存56组相对H0的配对救回/损伤及独立复核。原始预测.pt保留在N607本run的各scene目录，日志已完整收集。','']
text='\n'.join(out);(B/'detailed_target_results.md').write_text(text,encoding='utf-8')
verification=dict(status='VERIFIED',rows=3360,confusions=56,candidate_decisions=9408000,full_log_lines=loglines,errors=errors,independent_recount=True,group_sums=True)
(B/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
r=Path('automation_reports/CV-SincNet/core90_anchored_target_all_s392005_20260912_r1')
shutil.copyfile(B/'detailed_target_results.md',r/'detailed_target_results.md')
with (r/'report.md').open('a',encoding='utf-8') as f:f.write('\n## 完成与交付\n\n状态VERIFIED/SCORED，全部14候选×4场景完成。56份预测封存后独立评分，独立复核与分组求和检查通过，无运行错误。详见[detailed_target_results.md](detailed_target_results.md)。\n')
root=Path('E:/type10-7');shutil.copytree(B,root/B,dirs_exist_ok=True);shutil.copytree(r,root/r,dirs_exist_ok=True)
print(tab(['候选','clean','clear','低仰角','雨衰','LEO均值','ΔLEO'],main));print(json.dumps(verification))
