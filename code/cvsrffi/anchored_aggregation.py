"""Aggregate completed A6 source runs without loading a model or fitting anything."""
import csv,json,hashlib
from pathlib import Path
from .anchored_pipeline import validate_config
from .anchored_reporting import assess_source_promotion,save_json
from .anchored_fit import write_csv


def aggregate_source_runs(inputs,output,config,contract_path):
    config=validate_config(config);output=Path(output)
    if output.exists():raise FileExistsError('preserve existing aggregate; use a new output directory')
    contract_bytes=Path(contract_path).read_bytes();contract=json.loads(contract_bytes.decode('utf-8'))
    contract_digest=hashlib.sha256(contract_bytes).hexdigest()
    if any(contract.get(k)!=config[k] for k in ('source_receivers','source_days','split_seed')):
        raise ValueError('aggregate source contract mismatch')
    classes=contract.get('tx_mapping')
    if not isinstance(classes,list) or len(classes)!=6 or len(set(classes))!=6:
        raise ValueError('CORE90 frozen six-class tx_mapping required')
    if len(inputs)!=len(config['head_seeds']):raise ValueError('three completed source runs required')
    reports={};sources=[];reference=None;combined=[]
    def behavior(c):return {k:v for k,v in c.items() if k not in {'profile','cache'}}
    for value in inputs:
        path=Path(value);manifest=json.loads((path/'protocol_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('stage')!='fuse' or manifest.get('candidate')!='A6' or manifest.get('status')!='SOURCE_STAGE_COMPLETE':
            raise ValueError('completed nested A6 fuse stage required')
        seed=manifest.get('head_seed')
        if seed not in config['head_seeds'] or seed in reports:raise ValueError('unexpected or duplicate head seed')
        incoming=validate_config(manifest['config'])
        if behavior(incoming)!=behavior(config):raise ValueError('aggregate method configuration mismatch')
        identity=manifest.get('cache_identity',{})
        expected=dict(data_contract_sha256=contract_digest,preprocessing_version=config['cache']['preprocessing_version'],
                      view_recipe={'version':'paired_leo_v1','seed':config['view_seed']},split_seed=config['split_seed'],role='L_s')
        if any(identity.get(k)!=v for k,v in expected.items()) or not identity.get('checkpoint_sha256'):
            raise ValueError('aggregate cache/contract identity mismatch')
        if reference is not None and identity!=reference:raise ValueError('aggregate checkpoint/data identity mismatch')
        reference=identity
        if manifest.get('source_only') is not True or manifest.get('backbone_updated') is not False or manifest.get('support_used') is not False:
            raise ValueError('source-only frozen-backbone evidence required')
        activation=json.loads((path/'activation.json').read_text(encoding='utf-8'))
        if activation.get('status')!='SOURCE_NESTED_EVALUATED':raise ValueError('nested evaluation not completed')
        with (path/'nested_source_metrics.csv').open(encoding='utf-8',newline='') as f:
            records=list(csv.DictReader(f))
        for row in records:
            for key,value in list(row.items()):
                if key not in {'candidate','group','value'}:
                    try:row[key]=float(value) if value else None
                    except (ValueError,TypeError):pass
        reports[seed]=records;combined.extend(dict(head_seed=seed,**row) for row in records)
        sources.append(dict(head_seed=seed,directory=str(path.resolve())))
    verdict=assess_source_promotion(reports,config['head_seeds'],required_rx=config['source_receivers'],
                                    required_tx=range(len(classes)),required_days=config['source_days'])
    output.mkdir(parents=True)
    save_json(output/'source_promotion.json',verdict)
    write_csv(output/'all_seed_source_metrics.csv',combined)
    if 'comparisons' in verdict:write_csv(output/'a6_vs_inner_a5.csv',verdict['comparisons'])
    save_json(output/'protocol_manifest.json',dict(stage='aggregate',status=verdict['status'],candidate='A6',
        config=config,cache_identity=reference,tx_mapping=classes,inputs=sources,source_only=True,
        additional_fits=0,confirmation=False,backbone_seed_replications=1))
    (output/'report.md').write_text('# CORE90三seed source汇总\n\n状态：`'+verdict['status']+'`。\n\n'
        '仅汇总已有嵌套source预测；新增拟合次数为0。三个head seed共享一个H0，不能替代多个骨干seed或独立确认。\n\n'
        '- [完整指标](all_seed_source_metrics.csv)\n- [判定与完整性检查](source_promotion.json)\n'+
        ('- [A6相对inner选定A5的逐组差值](a6_vs_inner_a5.csv)\n' if 'comparisons' in verdict else ''),encoding='utf-8')
    return verdict
