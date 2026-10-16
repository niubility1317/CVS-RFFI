"""Author the existing minimal registration schema; no experiment side effects."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN='20261016-phase1-F-timing-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_timing_sourceRX1_e20_20261016_r01'


def spec():
    old=ROOT/'automation_reports/CV-SincNet/20261016-phase1-sinc-bandgain-rdrop-sourcefold-s2026092701-m3-r01/experiment.json'
    c=json.loads(old.read_text(encoding='utf-8'))
    c.update(run_id=RUN,group_id='phase1-F-IQ-timing-Ag-As',display_name='F 基座轻度分数延时增强：E20 源 RX1 留出',description='A distinct IQ timing hypothesis, compared with matched six-forward raw slots. No ordinary-augmentation reorganization, spectral-response bank, feature alignment, target or new U mechanism.',aliases=['F fractional timing Ag'],tags=['F','IQ_timing','Ag','RDrop','source_only','E20','default_head','held_RX1'],authorization='User20261016 authorizes joint Ag/As research on F/E20 and asks for a direction distinct from the referenced urban-training chat. First bounded source falsification before any new target result.')
    c['code']=dict(commit='Actual Git HEAD in release_commit.txt and resolved_config',checkout=str(ROOT),environment='ordinary N607 CVS-RFFI; CPU2/loader0',cwd='PROJECT/releases/'+RELEASE)
    c['data'].update(representation='Existing native RMS IQ256; new bounded fractional timing in L only; no satellite at this source stage',roles=dict(L_s='5040 fit;1260 held RX1 original L for fixed-final development only;144 fit anchor reads unused',U_s='45360 hidden labels; original weak/strong/oldgate/ownEMA rule',V='21600 fitRX health only'),leo_config_ref=None)
    c['checkpoint'].update(contract_check_ref='No inherited checkpoint. Own scratch initial CP no-query smoke first; original four-RX source exclusion before student/ownEMA creation. Code-only dependency AF09554c3a.',provenance_verdict='Scratch F; no parent/teacher/EMA/checkpoint inheritance',selection_rule='Fixed final20 student default head only')
    c['execution'].update(launch_owner='codex/root/F-timing-Ag-20261016',remote_run_root='PROJECT/runs/'+RUN,remote_log_root='PROJECT/runs/'+RUN+'/logs',local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='CVS-RFFI/bin/python -u experiments/cvs_iq_timing20/runner.py')
    c['expected_artifacts']=['initial_smoke.pt/final_ssdg.pth private','source_contract/initialization/resolved config/actual native args/module paths','complete native epoch and step logs private','training_sequence.jsonl.gz and timing_steps.jsonl.gz private','timing_progress/timing_result/source_readouts/FINAL','queue_state/launch/result/completion','independent complete source confusion/paired recount and cost analysis']
    c['metrics_plan']=dict(metric_names=['held-source accuracy','MacroF1','CE','perTX/day','seen V health','full E20 curves','actual forwards/updates/walltime'],dimensions=['B00/B10','seed2026092701','heldRX1 day1/2/3 6TX'],prediction_ref='Each row source/source_readouts/FINAL.json; source labels only',scorer_ref='Independent saved source-decision recount; no target truth')
    template=c['rows'][0];c['rows']=[]
    for arm in ('B00','B10'):
        row=copy.deepcopy(template);rowid=arm+'-RX1-s2026092701'
        row.update(row_id=rowid,method='F/X3/unifiedSinc raw slots'+(' + bounded fractional timing' if arm=='B10' else ''),purpose='primary' if arm=='B10' else 'matched_control',config_ref='experiments/cvs_iq_timing20/design.py:config('+arm+')',resolved_config_ref='PROJECT/runs/'+RUN+'/'+rowid+'/source/resolved_config.json',scenario='source_RX1_true_holdout',optimizer='Original F CENTER_UNITS AdamW and physicalHz globalclip5, unchanged origin WD',budget_ref='Scratch20=13L+7U,4440updates,3RDrop pairs/physical L,weights.5/.25/.25,3409920 L forwards;original U/EMA additional;no target',output_root='PROJECT/runs/'+RUN+'/'+rowid+'/source',log_path='PROJECT/runs/'+RUN+'/logs/'+rowid+'.stdout.log',command='CVS-RFFI/bin/python -u experiments/cvs_iq_timing20/runner.py --worker '+rowid,expected_artifacts=c['expected_artifacts'])
        row['seeds']['augmentation']=20261016071;row['seed_notes']='Model and old native U/sampler streams retain2026092701;new timing uses private generator seed20261016071+step;split392005;no support or target eval.'
        c['rows'].append(row)
    c['notes']=[
        'Fixed Ag: eachL uniformdelay[-.125,.125] samples,33tap Hann-sinc,reflect,no circular roll;restore input RMS. Fixed engineering hypothesis, not measured receiver replacement or guaranteed identity invariance. No PA/DAC/CFO/noise/multipath/clipping.',
        'Firstrawslot.5,Agslot.25,As-inactive rawslot.25; eachslot has2 stochastic passes of same generated IQ;CEmean+symKL1,LS.01;total weighted scale1. Bothrows6Lforwards, no crossview KL or new U terms.',
        'Native use_aug=false disables ordinary L augmentation generation only. Native U weak/strong/ownEMA/oldgate remains unchanged. This is a newly matched F raw-slot recipe, not the historical ordinary-aug F performance control.',
        'Primary B10 sourcequalification:heldaccuracy gain>=1pp,macroF1>=0pp,seenVmean loss<=.5pp,completehealthy20/finalLCE<=.5. FAIL closesfixedcandidate:no strength/seed/RX/window rescue. PASS permits separately registered confirmation and fullsource2x2;no target in currentrun.',
        'Future fullsource factorial B00/B10/B01/B11 uses identical three slots/6Lforward normalization,raw.5/g.25/s.25;As drawn directlyfromraw,never repeated fullchannel;AgAs composition deferred. Suburban2pp is an exploration objective,not an asserted hard userthreshold. No new fullsource or target work yet.',
        'REJECTED_EXTRA_GATE: generic designapproval,blankettests and extra review layers are nonblocking under the user exclusive minimal workflow. One focused P0/P1 before release remains.',
        'Local signal and realF synthetic checks only; PyTorch2.1 CPU emits existing NumPy2 bridge warning. Tests do not use NumPy tensor bridging. Remote run uses existing verified CVS-RFFI environment without installs.'
    ]
    c.pop('registered_at',None)
    return c


if __name__=='__main__':
    output=ROOT/'local_artifacts/F_timing_preparation_r01/spec.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(spec(),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(output)
