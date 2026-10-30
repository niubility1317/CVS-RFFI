from copy import deepcopy
from experiments.pamf_dura_efficiency.stage_compare import prior_decision


def case(correct,error):return dict(teacher_correct=correct,metrics={'feature_cosine_error':{'mean':error}})


def test_prior_runtime_requires_both_views_and_scene_protection():
    cells={epoch:{'all':{'cases':{'native':case(480,.1),'prior':case(486,.09)}},
        'scene:clean':{'cases':{'native':case(70,.01),'prior':case(70,.01)}}} for epoch in ('1','20')}
    # The matched control cannot substitute for a failing head-guided candidate.
    summary={'headFeature':cells,'noVarAux':deepcopy(cells)}
    assert prior_decision(summary)['head_prior_runtime_eligible']
    cells['20']['scene:clean']['cases']['prior']['teacher_correct']=69
    assert not prior_decision(summary)['head_prior_runtime_eligible']
    cells['20']['scene:clean']['cases']['prior']['teacher_correct']=70
    cells['1']['all']['cases']['prior']['teacher_correct']=485
    assert prior_decision(summary)['cached_view_passed']==[False,True]
