from experiments.pamf_dura_efficiency.stage_fidelity import solver_decision


def case(correct,error):return dict(teacher_correct=correct,metrics={'feature_cosine_error':{'mean':error}})


def test_solver_rule_requires_both_views_and_no_scene_damage():
    summary={}
    for epoch in ('1','20'):
        summary[epoch]={
            'all':{'cases':{'native':case(450,.1),'CG4':case(456,.09),'CG8':case(458,.08),'CG16':case(460,.07)}},
            'scene:clean':{'cases':{'native':case(70,.01),'CG4':case(70,.01),'CG8':case(70,.01),'CG16':case(70,.01)}},
            'scene:practical_low_urban':{'cases':{'native':case(50,.2),'CG4':case(51,.18),'CG8':case(52,.17),'CG16':case(53,.16)}}}
    assert solver_decision(summary)['smallest_qualified_CG']==4
    summary['20']['scene:clean']['cases']['CG4']['teacher_correct']=69
    assert solver_decision(summary)['smallest_qualified_CG']==8
    summary['1']['all']['cases']['CG8']['teacher_correct']=455
    assert solver_decision(summary)['smallest_qualified_CG']==16
    summary['20']['all']['cases']['CG16']['metrics']['feature_cosine_error']['mean']=.091
    assert solver_decision(summary)['smallest_qualified_CG'] is None
