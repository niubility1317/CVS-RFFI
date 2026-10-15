"""Post-training analysis only; immutable source, predictions and scores are reused."""
import argparse,os
from . import design as d
from .predict import source_proof
from .score import source_evidence,finish
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('source','summary'),required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
 out=d.BASE/'analysis_recovery_r01'
 if a.mode=='source':
  for row in d.training_rows():source_proof(d.config(row))
  out.mkdir(exist_ok=False);q=source_evidence(out)
  d.write(out/'recovery.json',dict(status='SOURCE_ANALYSIS_COMPLETE',source_run=d.RUN,source_execution_commit=d.read(d.BASE/'experiment.json')['execution_commit'],analysis_code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),reason='GPU repeated backward cannot require bitwise equality; unchanged allclose tolerance retained',training_or_optimizer_updates=0,head_fit_updates=0,query_read=False,target_metrics_read=False))
  print(dict(status=q['status'],source_only=True,output=str(out)))
 else:
  q=d.read(out/'source_qualification.json')
  if (out/'analysis.json').exists():raise FileExistsError('Analysis already exists; reconcile instead of repeating')
  result=finish(out,qualification=q)
  d.write(out/'completion.json',dict(status='ANALYZED',source_run=d.RUN,models=result['models'],prediction_count=result['prediction_count'],metric_records=result['metric_records'],goal_84=False,training_or_optimizer_updates=0,head_fit_updates=0,truth_last=True))
  print(dict(status=result['status'],models=result['models'],goal_84=False))
if __name__=='__main__':main()
