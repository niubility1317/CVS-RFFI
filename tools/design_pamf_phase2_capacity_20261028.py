"""Metadata-only Phase2 capacity arithmetic. Never reads IQ, models or truth."""
from collections import Counter
import argparse
import json
from pathlib import Path

def rotate(values,shift):
    return [values[(i-shift)%len(values)] for i in range(len(values))]

def design():
    rx,classes,models,draws,paths=7,26,4,5,2
    blocks=[dict(id=f'MIX{i+1:02d}',support_pool=6,query=7,
                 support_by_date=rotate([2,2,1,1],i%4),query_by_date=rotate([2,1,2,2],i%4)) for i in range(8)]
    single=[dict(id=i,scenario=s,support_pool=24,query=8,support_by_date=[6]*4,query_by_date=[2]*4)
            for i,s in [('D-LS','practical_low_suburban'),('D-MU','practical_mid_urban'),('D-LU','practical_low_urban')]]
    grouping={1:[[i] for i in range(8)],5:[[i] for i in range(8)],10:[[0,2],[1,3],[4,6],[5,7]],20:[[0,1,2,3],[4,5,6,7]]}
    per_K=[]
    for k,groups in grouping.items():
        q=7*len(groups[0]);pool=6*len(groups[0])
        per_K.append(dict(K=k,mixed_tasks=len(groups),mixed_groups=[[blocks[i]['id'] for i in g] for g in groups],
                          candidate_support_per_class=pool,used_support_per_class=k,query_per_class=q,
                          all_scene_tasks=len(groups)+3,B_old_support=6*k,C20_total_support=26*k,
                          mixed_A_B_old_query=6*q,mixed_C20_old_query=6*q,mixed_C20_new_query=20*q,mixed_C20_total_query=26*q,
                          single_A_B_old_query=48,single_C20_total_query=208))
    total_per_combo=sum(b['support_pool']+b['query'] for b in blocks+single)
    day_support=[sum(b['support_by_date'][d] for b in blocks+single) for d in range(4)]
    day_query=[sum(b['query_by_date'][d] for b in blocks+single) for d in range(4)]
    core_task_per_seed_path=sum(x['all_scene_tasks'] for x in per_K)*rx*draws
    core_B=core_task_per_seed_path*models*paths*15
    core_C=core_task_per_seed_path*models*paths*4*29
    core_A=11*rx*models*paths*15
    all_scene_slice_groups=4+4+1+3*3
    slice_full_draw_B_per_method=all_scene_slice_groups*models*rx*draws*paths
    mixed_slice_full_draw_B_per_method=(4+4+1)*models*rx*draws*paths
    mixed_slice_one_draw_B_per_method=(4+4+1)*models*rx*paths
    direct_contexts=(4+1)*models*rx*paths
    seq_contexts=direct_contexts*15
    full_all_method_C=core_task_per_seed_path*models*paths*4
    rows=[]
    def row(i,name,A,B,C,physical_scope,note):
        rows.append(dict(id=i,name=name,A=A,B=B,C=C,fit_slots=B+C,stage_prediction_slots=A+B+C,
                         additional_physical_records=0,physical_scope=physical_scope,notes=note))
    row('E01','29条完整链路',core_A,core_B,core_C,'全部200条/RX/TX；K1/5/10/20；N0/2/5/10/20；4F×7RX×5draw×2paths',
        '15个B前缀、29个C后缀；A逐基础池复用。完整方法/预算不缩短。')
    B=8*slice_full_draw_B_per_method
    row('E02','8个共享PAMF控制',0,B,2*B,'第一四块综合＋三个单景；K1/5/20；N0/5/20；4F×7RX×5draw×2paths',
        'S1/2/3/4/8/9/10/11；同片MILKR从E01复用，S5/6在E03。')
    full_two_B=2*core_task_per_seed_path*models*paths
    delete_nine_B=9*mixed_slice_full_draw_B_per_method
    row('E03','MILKR机制消融',0,full_two_B+delete_nine_B,4*full_two_B+2*delete_nine_B,
        'M2/M3全矩阵；九个机制变体第一四块综合、K1/5/20、N0/5/20、4F×7RX×5draw×2paths',
        'M0复用主行；M1历史整体方法仅条件参考。')
    B=12*mixed_slice_one_draw_B_per_method//paths
    row('E04','12个非默认参数点',0,B,B,'第一四块综合；K1/5/20；N0/20；4F×7RX×1draw；P0',
        '三个单因素轴各四个非默认点；默认点复用。全部报告，不按query择优。')
    row('E05','22条等数据直接C',0,0,22*direct_contexts,'第一四块综合；K5/20；N20；4F×7RX×1draw×2paths',
        '标签与B+C相同；无中间B更新。')
    row('E06','4条原生无B的A→C',0,0,4*direct_contexts,'同E05','原生新类输入权限不同，不解释为等标签B轨迹消融。')
    row('E07','15系统分次注册',0,0,seq_contexts*17,'第一四块综合；K5/20；4F×7RX×1draw×2paths',
        '1×20/2×10/4×5/10×2共17个C阶段；相同终态与中间状态按方法精确复用。')
    row('E08','15系统三个额外类顺序',0,0,seq_contexts*12,'同E07；仅4×5日程；三个额外类顺序',
        '默认顺序在E07；顺序不改变物理数据。')
    B=4*slice_full_draw_B_per_method//draws
    C=2*B
    A=7*rx*models*paths*4
    row('E09','8个原代码辅助版本',A,B,C,'第一四块综合＋三个单景；K1/5/20；N0/5/20；4F×7RX×1draw×2paths',
        '沿r05最多四个不同source/A、四条B、四条C版本的复用前上界；不合法/未接通者N/A且0活跃槽位。')
    B=2*mixed_slice_one_draw_B_per_method
    row('E10','DANN/MCD通用控制',4*rx*models*paths*2,B,4*B,
        '第一四块综合；K1/5/20；N0/5/20；4F×7RX×1draw×2paths',
        '两B前缀各接NCM/JointCE；通用RFFI绑定扩展，非新增RFFI论文。')
    main_K5_context=11*rx*models*draws*paths
    main_K20_context=5*rx*models*draws*paths
    totals=dict(fit_slots=sum(r['fit_slots'] for r in rows),stage_prediction_slots=sum(r['stage_prediction_slots'] for r in rows),
                core_fit_slots=core_B+core_C,core_stage_slots=core_A+core_B+core_C,
                core_B_tasks_per_model_path=core_task_per_seed_path,
                first_main_K5_N20=dict(A=core_A,B=15*main_K5_context,C=29*main_K5_context,fit_slots=44*main_K5_context),
                second_main_K20_N20=dict(A=0,B=15*main_K20_context,C=29*main_K20_context,fit_slots=44*main_K20_context))
    decisions=dict(A=80*6*rx*models*paths*15,
                   B=80*6*rx*models*draws*paths*4*15,
                   C=80*rx*models*draws*paths*4*(8+11+16+26)*29)
    decisions['total']=sum(decisions.values());totals['core_individual_query_decisions']=decisions
    return dict(schema='pamf_phase2_capacity_r06',date='2026-10-28',status='CONTINGENCY_ONLY_NOT_PRIMARY',launch_ready=False,
                superseded_primary_planning_by='configs/pamf_phase2_full_eq_planning_r07_20261028.json',
                supersedes='r05 data sizes, K-dependent group counts and E01-E10 budgets; method S/B/C identities unchanged',
                scope='Existing labeled MILKR; practical residual; six fixed mixture probabilities; three pure scenarios; P0/P1',
                per_RX_TX_available=200,dates=4,per_date_available=50,source_roles=dict(L_s=6300,U_s_hidden_TX=56700,V=27000),
                targets=dict(RX=7,old_classes=6,new_classes=20,total_classes=26),atomic_mixed_blocks=blocks,single_scene_pools=single,
                per_K=per_K,physical_allocation_seed=2026102800,support_order_seed=2026102840,
                support_repetitions=dict(count=5,labels=[2026092711,2026092712,2026092713,2026092714,2026092715],
                    base_pool_rule='One seed-frozen 6-record order per RX/TX/base block, rotated by offsets0/1/2/3/4; K1 uses first1, K>=5 uses first5 per member block',
                    single_rule='One seed-frozen 24-record order per RX/TX/single scene, rotated by offsets0/5/10/15/20; nested firstK',
                    independent_data_replicates=False),
                allocation=dict(per_RX_TX=total_per_combo,support_candidates_per_RX_TX=sum(b['support_pool'] for b in blocks+single),
                    query_per_RX_TX=sum(b['query'] for b in blocks+single),support_by_date=day_support,query_by_date=day_query,
                    mixed_records=104*rx*classes,mixed_support_candidates=48*rx*classes,mixed_queries=56*rx*classes,
                    single_scene_each_records=32*rx*classes,single_scene_each_support_candidates=24*rx*classes,single_scene_each_queries=8*rx*classes,
                    total_records=total_per_combo*rx*classes,total_support_candidates=120*rx*classes,total_queries=80*rx*classes,
                    raw_float32_IQ_bytes=total_per_combo*rx*classes*2*256*4,both_P0_P1_IQ_bytes=total_per_combo*rx*classes*2*256*4*2),
                supplementary_slices=dict(first_quartet=['MIX01','MIX02','MIX03','MIX04'],
                    K1_K5_groups=4,K10_groups=2,K20_groups=1,with_three_singles_K1_K5_groups=7,with_three_singles_K20_groups=4,
                    primary_draw_count=5,thin_supplement_draw_count=1),
                reporting=dict(primary_cell='model/RX/atomic_base_pool/draw/K/N/path, even when the fitting task merges multiple base pools',
                    H='compute old/new accuracy and harmonic mean within each fixed atomic report cell, then aggregate',
                    task_H='optional secondary, explicitly task-pooled; do not mix with atomic-cell H for K curves',
                    independence='Eight report slices at every K do not imply eight independent fits; K20 has two fitting groups per RX/draw'),
                budget_ledger=rows,budget_summary=totals,
                rules=['Every physical record receives one channel realization only; pure and atomic mixed pools are disjoint',
                    'Within a fixed K the primary mixed tasks are disjoint; across K they reuse the same fixed base observations',
                    'All methods, model seeds, support repetitions, C suffixes, ablations and P0/P1 share the same legal physical pools',
                    'No same-record second overlay to produce new G20/G25 or extra mixture replicas',
                    'No query becomes support, and only current registered classes/support are visible at each A/B/C stage',
                    'Logical stage slots are not optimizer updates, independent observations or measured GPU hours',
                    'Missing interfaces are N/A, never zero accuracy; preserve native budgets and source permissions'])

def verify(x):
    assert x['allocation']['per_RX_TX']==200
    assert x['allocation']['support_by_date']==[30]*4 and x['allocation']['query_by_date']==[20]*4
    # Simulated record identities establish that every available record is used once.
    free=[list(range(50)) for _ in range(4)];used=[];roles={}
    for block in x['atomic_mixed_blocks']+x['single_scene_pools']:
        values={}
        for role,key in [('support','support_by_date'),('query','query_by_date')]:
            take=[]
            for day,n in enumerate(block[key]):
                take.extend((day,v) for v in free[day][:n]);free[day]=free[day][n:]
            values[role]=take;used.extend(take)
        assert not set(values['support'])&set(values['query']);roles[block['id']]=values
    assert len(set(used))==len(used)==200 and not any(free)
    all_mix_query={v for b in x['atomic_mixed_blocks'] for v in roles[b['id']]['query']}
    for r in x['per_K']:
        assert len(r['mixed_groups'])=={1:8,5:8,10:4,20:2}[r['K']]
        assert sorted(b for g in r['mixed_groups'] for b in g)==[b['id'] for b in x['atomic_mixed_blocks']]
        q=[v for g in r['mixed_groups'] for b in g for v in roles[b]['query']]
        assert len(q)==len(set(q))==56 and set(q)==all_mix_query
        assert r['used_support_per_class']<=r['candidate_support_per_class']
        if r['K']>=5:assert 5*len(r['mixed_groups'][0])==r['K']
    assert x['allocation']['total_records']==36400
    assert x['allocation']['total_support_candidates']+x['allocation']['total_queries']==36400
    assert x['budget_summary']['core_fit_slots']==1247120
    assert x['budget_summary']['fit_slots']==1683584
    assert x['budget_summary']['stage_prediction_slots']==1694840
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--check',type=Path);a=p.parse_args()
    x=design();verify(x)
    if a.check:
        expected=json.loads(a.check.read_text(encoding='utf-8'));assert expected==x,'Saved capacity design differs from arithmetic'
    if a.output:a.output.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',physical_records=x['allocation']['total_records'],budgets=x['budget_summary']),ensure_ascii=False))

if __name__=='__main__':main()
