"""Source developer evidence only; never reads target scores or registry summaries."""
from . import design as d


def pilot(records):
    if set(records)!=set(d.ARMS):raise ValueError('Keep all eight pilot configurations, including negative controls')
    baseline=records['A0'];eligible=[];differences=[]
    for arm in d.ARMS[1:]:
        value=records[arm]
        delta=dict(arm=arm,clean=value['clean_accuracy']-baseline['clean_accuracy'],
            low=value['scene_accuracy']['practical_low_urban']-baseline['scene_accuracy']['practical_low_urban'],
            urban=value['urban_mean']-baseline['urban_mean'],worst=value['worst_urban_RX']-baseline['worst_urban_RX'])
        delta['eligible']=(delta['clean']>=-.005 and delta['low']>=.01 and delta['urban']>=0. and delta['worst']>=-.01)
        differences.append(delta)
        if delta['eligible']:eligible.append(arm)
    return dict(source_only=True,query_read=False,source_role='L_dev',criterion_frozen_before_target=True,
        family_expandable=bool(eligible),eligible_signal_arms=eligible,differences=differences,
        expansion_matrix='all eight configs x both remaining predefined seeds, without arm filtering',
        single_seed_best_claim=False,
        criterion='At least one nonzero auxiliary: final50 low +1pp, clean >= -0.5pp, urban >=0pp, worst urban RX >=-1pp vs matched A0')


def from_sources():
    records={}
    for row in d.rows():
        source=d.BASE/row['row_id']/'source'
        frozen=d.read(source/'source_frozen.json');done=d.read(source/'training_mechanism_complete.json')
        history=d.read(source/'source_learning_progress.json')
        if frozen['epoch']!=50 or frozen['selection']!='final_only' or not history['source_only'] or history['target_feedback']:
            raise ValueError('Only fixed final50 source developer evidence may advance family')
        if done['updates']!=11100 or done['head_fit_updates'] or done['U_truth_read']:
            raise ValueError('Incomplete or unauthorized source training')
        value=history['records'][-1]
        if value['epoch']!=50 or value['source_role']!='L_dev' or value['query_read']:
            raise ValueError('Need actual held-out L developer final metrics')
        records[row['arm']]=value
    return pilot(records)
