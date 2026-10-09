"""CPU-only independent scoring of all existing immutable predictions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cvs_receiver_residual_test_recovery import design as d,evaluate as ev


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    root=Path.cwd();project=Path(d.PROJECT)
    write(root/'owner.json',dict(pid=os.getpid(),argv=sys.argv,cwd=str(root),
        launch_owner=d.OWNER,commit=(root/'release_commit.txt').read_text().strip(),started=time.time()))
    try:
        marker=project/'releases'/d.RELEASE/'all_predictions_complete.json'
        old=json.loads(marker.read_text())
        if old['status']!='ALL_PREDICTIONS_COMPLETE' or old['rows']!=32 or old['views']!=7:
            raise ValueError('Original all32 prediction marker missing')
        # Validate both complete prediction matrices before either independent scorer.
        for run in d.SCORE_RUNS:
            ev.configure(run['run_id']);ev.validate_predictions()
        write(root/'all_predictions_verified.json',dict(status='ALL_PREDICTIONS_COMPLETE',rows=32,views=7,
            original_marker=str(marker),original_marker_mtime=marker.stat().st_mtime,at=time.time(),
            predictions_reused=True,training=False,query_fit=False))
        for run in d.SCORE_RUNS:
            base=project/'runs'/run['run_id'];base.mkdir(exist_ok=False)
            write(base/'lineage.json',dict(**run,scorer_commit=(root/'release_commit.txt').read_text().strip(),
                prediction_release=d.RELEASE,training=False,query_fit=False,predictions_reused=True))
            command=[sys.executable,'-u','-m','experiments.cvs_receiver_residual_test_recovery.evaluate',
                '--mode','score','--run',run['run_id']]
            with (base/'scorer.stdout.log').open('x') as log:
                subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
            done=json.loads((base/'scoring_complete.json').read_text())
            if done['result_rows']!=1568 or done['day_result_rows']!=448 or done['independent_recount']!='VERIFIED':
                raise ValueError('Complete score coverage differs')
            write(base/'completion.json',dict(status='ANALYZED',rows=16,views=7,
                parent_run_id=run['parent_run_id'],prediction_run_id=run['prediction_run_id'],
                disposition=run['disposition'],independent_recount='VERIFIED',target_feedback_forbidden=True))
            print(json.dumps(dict(run_id=run['run_id'],status='ANALYZED',scored_rows=1568,day_rows=448)),flush=True)
        write(root/'completion.json',dict(status='ANALYZED',rows=32,views=7,predictions_reused=True,
            all_predictions_before_truth=True,training=False))
    except Exception as error:
        write(root/'failure.json',dict(status='FAILED',error=repr(error),no_retry=True));raise


if __name__=='__main__':main()
