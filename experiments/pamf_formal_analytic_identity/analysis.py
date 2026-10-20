"""Reuse the unchanged source probe maths; rename only the three readouts."""
from . import design as d
from experiments.pamf_compact_identity_probe.analysis import analyze as legacy_analyze


def analyze(features,ids,labels,rx,day):
    if set(features)!={(v,r,g) for v in d.VIEWS for r in d.READOUTS for g in d.GROUPS}:
        raise ValueError('All fixed formal analytic cells required')
    old={(view,d.ALIASES[row],group):value for (view,row,group),value in features.items()}
    result=legacy_analyze(old,ids,labels,rx,day)
    reverse={value:key for key,value in d.ALIASES.items()}
    for collection in ('records','query_label_negative_controls','comparisons'):
        for record in result[collection]:
            for field in ('candidate','control'):
                if field in record:record[field]=reverse[record[field]]
    result.update(target_decisions_changed=False,selected_candidate=None)
    return result
