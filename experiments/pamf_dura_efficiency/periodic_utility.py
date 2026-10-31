"""Fixed source-IQ component counterfactuals; not a deployable restoration."""
import torch

START,STOP,PERIOD=80,160,20
CASES=('native','periodic','residual','matched_global')


def periodic_projection(value):
    if value.ndim!=3 or value.shape[1:]!=(2,256):raise ValueError('Source Bx2x256 IQ required')
    out=torch.zeros_like(value)
    center=value[:,:,START:STOP].reshape(-1,2,4,PERIOD)
    out[:,:,START:STOP]=center.mean(2,keepdim=True).expand(-1,-1,4,-1).reshape(-1,2,STOP-START)
    return out


@torch.no_grad()
def counterfactuals(native,clean):
    if clean.shape!=native.shape:raise ValueError('Paired source IQ required')
    periodic=periodic_projection(clean-native)
    window=torch.zeros_like(native);window[:,:,START:STOP]=(clean-native)[:,:,START:STOP]
    residual=window-periodic
    error=clean-native
    radius=residual.flatten(1).norm(dim=1)
    scale=radius/error.flatten(1).norm(dim=1).clamp_min(1e-20)
    matched=error*scale[:,None,None]
    cases=dict(native=native,periodic=native+periodic,residual=native+residual,matched_global=native+matched)
    details=dict(periodic_delta=periodic,residual_delta=residual,matched_delta=matched,window_error=window)
    return cases,details


def decision(views):
    if set(views)!={'1','20'}:raise ValueError('Both fixed source views required')
    records=[]
    for view,groups in views.items():
        if set(k for k in groups if k.startswith('rx:'))!={'rx:'+str(r) for r in (1,3,4,6,8)}:
            raise ValueError('All five source RX required')
        scenes={k:v for k,v in groups.items() if k.startswith('scene:')}
        if len(scenes)!=7 or 'scene:clean' not in scenes or groups['all']['count']!=512:raise ValueError('All seven fixed source scenes/512 required')
        row=groups['all']['cases'];native=row['native'];residual=row['residual'];matched=row['matched_global']
        gain=residual['teacher_correct']-native['teacher_correct'];extra=residual['teacher_correct']-matched['teacher_correct']
        rx_ok=all(groups[k]['cases']['residual']['teacher_correct']>=groups[k]['cases']['native']['teacher_correct'] for k in groups if k.startswith('rx:'))
        clean_ok=scenes['scene:clean']['cases']['residual']['teacher_correct']>=scenes['scene:clean']['cases']['native']['teacher_correct']
        scene_ok=all(v['cases']['residual']['teacher_correct']>=v['cases']['native']['teacher_correct']-(0 if k=='scene:clean' else 1) for k,v in scenes.items())
        feature_ok=residual['feature_cosine_error']<=native['feature_cosine_error']
        passed=gain>=4 and extra>=2 and rx_ok and clean_ok and scene_ok and feature_ok
        records.append(dict(view=view,gain_vs_native=gain,gain_vs_equal_energy_global=extra,RX_preserved=bool(rx_ok),clean_scene_preserved=bool(clean_ok),scene_protection=bool(scene_ok),feature_error_not_higher=bool(feature_ok),passed=bool(passed)))
    return dict(eligible=all(r['passed'] for r in records),views=records,automatic_G_training=False,primary='residual only;periodic descriptive;no alternate window/period/threshold')
