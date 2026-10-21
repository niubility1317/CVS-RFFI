"""Bind reused training/evaluation to this immutable matrix in each process."""
def bind():
    from .plan import make_plan
    from experiments.phase1_external_dg_aug import source, evaluation
    plan=make_plan();rows={r['row_id']:r for r in plan['rows']}
    source._PLAN=plan;source._ROWS=rows
    evaluation._PLAN=plan;evaluation._ROWS=rows
    return source,evaluation
