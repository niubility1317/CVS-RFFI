"""Full source-log parser, including live complete-prefix records; no target IO."""
import argparse,csv,gzip,json,math,re
from pathlib import Path

def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

def finite(value):
 if isinstance(value,float):return math.isfinite(value)
 if isinstance(value,dict):return all(finite(v) for v in value.values())
 if isinstance(value,list):return all(finite(v) for v in value)
 return True

def summarize(base,rows):
 result=[]
 for row in rows:
  folder=Path(base)/row['row_id']/'source';path=folder/'epoch_metrics.jsonl'
  epochs=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.is_file() else []
  if epochs and [v['epoch'] for v in epochs]!=list(range(1,len(epochs)+1)):raise ValueError('Missing/duplicate source epoch')
  if any(not finite(v) for v in epochs):raise ValueError('Nonfinite source epoch')
  step_count=0;last_step=0;first_U=None;component_audits=[];ranges={};live_gzip_tail=False
  step_file=folder/'step_metrics.jsonl.gz'
  if step_file.is_file():
   try:
    with gzip.open(step_file,'rt',encoding='utf-8') as stream:
     for line in stream:
      if not line.endswith('\n'):
       live_gzip_tail=True;break
      value=json.loads(line)
      if not finite(value):raise ValueError('Nonfinite source step')
      if value['step']!=last_step+1:raise ValueError('Missing/duplicate source step')
      step_count+=1;last_step=value['step']
      detail=value['details']
      if detail.get('U_confidence') is not None and first_U is None:first_U=value['step']
      if value.get('gradient_audit'):component_audits.append(dict(step=value['step'],**value['gradient_audit']))
      for key,metric in value['metrics'].items():
       if isinstance(metric,(int,float)):
        r=ranges.setdefault(key,[metric,metric]);r[0]=min(r[0],metric);r[1]=max(r[1],metric)
   except EOFError:live_gzip_tail=True
  done=read(folder/'completion.json');text_path=Path(base)/'logs'/('train-'+row['row_id']+'.log')
  text=text_path.read_text(encoding='utf-8',errors='replace') if text_path.is_file() else ''
  markers={key:len(re.findall(pattern,text,re.I)) for key,pattern in dict(traceback=r'Traceback \(most recent call last\)',OOM=r'out of memory|CUDA error',killed=r'\bKilled\b',warnings=r'Warning|warn\(').items()}
  if done and (len(epochs)!=40 or last_step!=8880 or live_gzip_tail):raise ValueError('Completed source log incomplete')
  csv_path=folder/'epoch_metrics.csv';csv_rows=list(csv.DictReader(csv_path.open(encoding='utf-8'))) if csv_path.is_file() else []
  if len(csv_rows)!=len(epochs):raise ValueError('Source epoch CSV/JSON record count differs')
  for csv_row,epoch in zip(csv_rows,epochs):
   if int(csv_row['epoch'])!=epoch['epoch'] or abs(float(csv_row['V_accuracy'])-epoch['V_accuracy'])>1e-12:raise ValueError('Source CSV/JSON values differ')
  final=epochs[-1] if epochs else None
  result.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],source_complete=done is not None,parsed_epoch_count=len(epochs),parsed_step_count=step_count,last_parsed_step=last_step,live_gzip_tail=live_gzip_tail,first_U_student_step=first_U,full_stdout_bytes=len(text.encode('utf-8')),stdout_markers=markers,step_metric_ranges=ranges,gradient_audits=component_audits,epochs=epochs,final_epoch=final,physical_holdout=read(folder/'physical_holdout.json'),teacher_diagnostics=read(folder/'source_teacher_diagnostics.json'),post_readapt_U_scan=read(folder/'post_readapt_U_scan.json')))
 return dict(status='SOURCE_LOGS_PARSED',scope='all complete source records through snapshot; live gzip EOF explicitly recorded',target_files_read=False,rows=result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--rows',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 value=summarize(a.base,json.loads(Path(a.rows).read_text()));Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
