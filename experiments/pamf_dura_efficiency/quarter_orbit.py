"""Four exact IQ quarter-turns with output phase restored before averaging."""
import torch


def rotate_iq(value,turns):
    if value.ndim!=3 or value.shape[1:]!=(2,256):raise ValueError('IQ[B,2,256] required')
    k=turns%4
    if k==0:return value
    if k==1:return torch.stack((-value[:,1],value[:,0]),1)
    if k==2:return -value
    return torch.stack((value[:,1],-value[:,0]),1)


def quarter_orbit(model,received):
    inputs=torch.cat([rotate_iq(received,k) for k in range(4)],0)
    output=model(inputs)
    if output.shape!=inputs.shape:raise ValueError('Restorer must preserve the IQ batch')
    output=output.reshape(4,len(received),2,256)
    routes=torch.stack([rotate_iq(output[k],-k) for k in range(4)])
    return routes.mean(0),routes


def decide(summary):
    if set(summary)!={'clean','1','20'}:raise ValueError('Full source clean and both cached views required')
    clean=summary['clean']['all'];clean_ok=clean['cases']['orbit']['teacher_correct']>=clean['cases']['native']['teacher_correct']
    rows=[]
    for view in ['1','20']:
        groups=summary[view];allcase=groups['all'];scenes={k:v for k,v in groups.items() if k.startswith('scene:')}
        if clean['count']!=512 or allcase['count']!=512 or len(scenes)!=7 or 'scene:clean' not in scenes:
            raise ValueError('Complete source512/seven scene cohort required')
        native=allcase['cases']['native'];orbit=allcase['cases']['orbit']
        ratio=orbit['iq_nmse']/native['iq_nmse'];gain=orbit['teacher_correct']-native['teacher_correct']
        no_damage=all(v['cases']['orbit']['teacher_correct']-v['cases']['native']['teacher_correct']>=
            (0 if k=='scene:clean' else -1) for k,v in scenes.items())
        rows.append(dict(view=view,IQ_ratio=ratio,choice_gain=gain,scene_preserved=no_damage,passed=bool(ratio<=.9 and gain>=6 and no_damage)))
    return dict(clean_preserved=bool(clean_ok),views=rows,eligible=bool(clean_ok and all(r['passed'] for r in rows)))
