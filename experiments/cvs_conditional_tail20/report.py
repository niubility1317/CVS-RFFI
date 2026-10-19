import time
from . import design as d
def source_decision():
 values={r['arm']:d.read(d.BASE/r['row_id']/'source_qualification/complete.json') for r in d.rows()};clean={r['arm']:d.read(d.BASE/r['row_id']/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for r in d.rows()}
 for arm in ('B0','B1'):
  values[arm]=d.read(d.reference(arm)/'source_qualification/complete.json');clean[arm]=d.read(d.reference(arm)/'source/completion.json')['final_source_metrics']['source_val_accuracy']
 midlow=lambda arm:sum(values[arm]['source_accuracies'][i][j] for i in range(2) for j in (1,2))/4
 inc={a:100*(midlow('TAIL2')-midlow(a)) for a in ('CE2','B1')};dc=100*(clean['TAIL2']-clean['B0']);passed=min(inc.values())>=.5 and dc>=-.2
 result=dict(status='SOURCE_DECISION_FIXED',timestamp=time.time(),source_qualified=passed,midlow_increment_pp=inc,fullVclean_net_pp=dc,fullVclean=clean,readouts=values,formal_scores_read=False,decision='E50 source eligible; matched continuation rules required first' if passed else 'CLOSE_FIXED_TWO_DRAW_CONDITIONAL_TAIL_RECIPE; no K/mixture/seed rescue',goal_complete=False)
 d.write(d.BASE/'source_matrix_decision.json',result);return result
def collect():
 source=d.read(d.BASE/'source_matrix_decision.json');rows={}
 for arm in ('B0','B1','CE2','TAIL2'):
  path=d.reference(arm) if arm in ('B0','B1') else d.BASE/(arm+'-s'+str(d.SEED));data=d.read(path/'scoring/scores.json')['results'];overall={a['view']:a for a in data if a['dimension']=='overall'}
  rows[arm]=dict(overall=overall,worst_RX={v:min(a['accuracy'] for a in data if a['view']==v and a['dimension']=='receiver') for v in d.VIEWS},reused=arm in ('B0','B1'))
 result=dict(status='ANALYZED',run_id=d.RUN,source_decision=source,rows=rows,differences={v:{'TAIL2_minus_'+b+'_pp':100*(rows['TAIL2']['overall'][v]['accuracy']-rows[b]['overall'][v]['accuracy']) for b in ('B0','B1','CE2')} for v in d.VIEWS},suburban_gap_pp={v:100*(rows['TAIL2']['overall']['clean']['accuracy']-rows['TAIL2']['overall'][v]['accuracy']) for v in d.SCENES[:3]},model_seed_count=1,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'analysis.json',result);return result
