from experiments.cvs_oracle80.data import oracle_native,supervised_split,check_roles
from . import design as d

def source_contract():
    from pathlib import Path
    value=d.read(d.SOURCE);check_roles(value['role_ids'])
    if value['status']!='VALIDATED_ONCE' or value['oracle_data']!=d.config(d.rows()[0])['oracle_data'] or not value['source_target_disjoint']:raise ValueError('Existing source contract differs')
    if not Path(d.SOURCE).is_file():raise FileNotFoundError(d.SOURCE)
    return dict(status='REUSED_VALIDATED_ONCE',role_counts={k:len(v) for k,v in value['role_ids'].items()},steps_per_epoch=563,query_read=False,data_revalidated=False)
