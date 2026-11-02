"""Independent post-prediction Phase1 accuracy, F1, group and DURA-pair audit."""
import json,sys,pathlib,numpy as np
r=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');name=sys.argv[1];p=r/'runs'/name/'flow/test_prediction';s=json.loads((p.parent/'test_summary.json').read_text());m=json.loads((p/'complete.json').read_text());views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
assert m['status']=='PREDICTIONS_COMPLETE' and m['count']==168000 and m['views']==views and m['TX_truth_reads']==0 and m['model_updates']==0 and s['status']=='ANALYZED' and s['all_predictions_before_truth'] and not s['target_feedback']
basep=r/'runs/20261101-phase1-dura-trisigtime-baseline-s2026092701-r01/flow/test_prediction';ids=np.load(p/'physical_ids.npy');assert ids.shape==(168000,) and len(set(ids.tolist()))==168000 and np.array_equal(ids,np.load(basep/'physical_ids.npy'));pred={};dura={}
for v in views:
 raw=np.load(p/(v+'.raw.npy'));rest=np.load(p/(v+'.restored.npy'));comp=np.load(basep/(v+'.restored.npy'));assert raw.shape==rest.shape==comp.shape==(168000,) and raw.dtype==rest.dtype==comp.dtype==np.int64 and np.array_equal(raw,np.load(basep/(v+'.raw.npy'))) and all(np.all((z>=0)&(z<6)) for z in (raw,rest,comp));pred[v]=(raw,rest);dura[v]=comp
# No truth is opened until every prediction array and common baseline ID is checked.
truth=json.loads((r/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json').read_text());labels=np.asarray([truth[str(i)]['label'] for i in ids]);rx=np.asarray([truth[str(i)]['receiver'] for i in ids]);day=np.asarray([truth[str(i)]['day'] for i in ids]);clean=float(np.mean(pred['clean'][0]==labels)*100);assert abs(clean-s['raw_clean_accuracy'])<1e-9
with (r/'Dataset_WigSig/ManySig.pkl').open('rb') as f:header=f.read(400)
dates=['2021_03_01','2021_03_08','2021_03_15','2021_03_23'];pos=[header.index(x.encode()) for x in dates];assert pos==sorted(pos);source=json.loads((r/'runs/20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0/source_contract.json').read_text());assert source['source_days']==[1,2,3] and source['source_rxs']==[1,3,4,6,8];names=['1-1','1-19','14-7','18-2','19-2','2-1','2-19','20-1','3-19','7-14','7-7','8-8'];buf=header[header.index(b'rx_list'):header.index(b'capture_date_list')];loc=[buf.index(x.encode()+b'q') for x in names];assert loc==sorted(loc) and set(np.unique(rx))==set(names)-{names[i] for i in source['source_rxs']}
masks={'cross_rx':np.isin(day,dates[1:]),'cross_both':day==dates[0]};assert masks['cross_rx'].sum()==126000 and masks['cross_both'].sum()==42000 and np.all(masks['cross_rx']^masks['cross_both']);groups={(x['view'],x['dimension'],x['value']):x for x in s['groups']};out={'status':'PASS','run':name,'count':len(ids),'raw_clean_accuracy':clean,'views':{},'subgroups':{},'group_checks':0,'common_raw_and_ids_match_DURA':True}
def near(a,b):assert abs(float(a)-float(b))<1e-9,(a,b)
for v in views:
 a,b=pred[v];accuracy=float(np.mean(b==labels)*100);f1=float(np.mean([2*np.sum((b==i)&(labels==i))/max(1,np.sum(b==i)+np.sum(labels==i)) for i in range(6)])*100);item=s['views'][v]
 for key,value in [('raw_accuracy',np.mean(a==labels)*100),('restored_accuracy',accuracy),('raw_clean_gap_pp',clean-accuracy),('macro_f1',f1)]:near(item[key],value)
 for typ,vec in [('TX',labels),('RX',rx),('day',day)]:
  for value in np.unique(vec):
   sel=vec==value;g=groups[(v,typ,str(value))];assert g['count']==int(sel.sum());near(g['raw'],np.mean(a[sel]==labels[sel])*100);near(g['restored'],np.mean(b[sel]==labels[sel])*100);out['group_checks']+=1
 q=dura[v];out['views'][v]={'raw_accuracy':float(np.mean(a==labels)*100),'restored_accuracy':accuracy,'gap_pp':clean-accuracy,'relative_gap_percent':(clean-accuracy)/clean*100,'macro_f1':f1,'dura_accuracy':float(np.mean(q==labels)*100),'repairs_vs_dura':int(np.sum((q!=labels)&(b==labels))),'damages_vs_dura':int(np.sum((q==labels)&(b!=labels)))}
assert out['group_checks']==len(groups)==119
for name,sel in masks.items():
 b=float(np.mean(pred['clean'][0][sel]==labels[sel])*100);item={'count':int(sel.sum()),'raw_clean_accuracy':b,'views':{}}
 for v in views:
  a,z=pred[v];acc=float(np.mean(z[sel]==labels[sel])*100);item['views'][v]={'raw_accuracy':float(np.mean(a[sel]==labels[sel])*100),'restored_accuracy':acc,'gap_pp':b-acc,'relative_gap_percent':(b-acc)/b*100}
 out['subgroups'][name]=item
out['all_six_within_3pp']=all(out['views'][v]['gap_pp']<=3 for v in views[1:]);out['all_six_within_relative_3percent']=all(out['views'][v]['relative_gap_percent']<=3 for v in views[1:]);print(json.dumps(out,allow_nan=False))
