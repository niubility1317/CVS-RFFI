"""Exact parent inheritance and own final default-head identity checks."""

def final_metadata(c, done, frozen, initial, contract, role_ids):
    if (done.get('config') != c or done.get('epoch') != 10
            or done.get('optimizer_steps') != 2220 or done.get('target_access')
            or done.get('target_evaluated', False) or done.get('head_fit_updates', 0) != 0
            or frozen.get('config') != c or frozen.get('selection') != 'final_only'
            or frozen.get('target_access')):
        raise ValueError('Own fixed10 final/default-head source freeze differs')
    if (initial.get('scratch_only') is not False or initial.get('target_contact')
            or initial.get('checkpoint_sources') != c['checkpoint_sources']
            or initial.get('source_roles') != 'EXACT_MATCH' or contract.get('role_ids') != role_ids):
        raise ValueError('Own exact-source corresponding parent provenance differs')

def checkpoint(c, cp):
    if (cp.get('epoch') != 10 or cp.get('a1_ema_successful_updates') != 2220
            or cp.get('run_id') != c['run_id'] or cp.get('candidate_id') != c['row_id']
            or cp.get('checkpoint_selection') != 'final_only' or cp.get('ema_model') is None):
        raise ValueError('Own final checkpoint identity/default EMA differs')
    args = cp.get('args', {})
    if (args.get('baseline_ckpt') != c['parent_source'] + '/final_ssdg.pth'
            or args.get('teacher_ckpt') or args.get('pseudo_temporal_mode')!='batch_neighbor'
            or not args.get('pseudo_temporal_gate')):
        raise ValueError('Unexpected corresponding parent or external teacher')
