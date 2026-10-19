import numpy as np
def score(logits,y,rx,day,scene=None):
 out=[]
 dimensions=[('overall','ALL',np.ones(len(y),bool))]+[(kind,str(v),values==v) for kind,values in [('receiver',rx),('day',day),('transmitter',y)] for v in sorted(set(values))]
 if scene is not None:dimensions += [('training_scene',str(v),scene==v) for v in sorted(set(scene))]
 p=np.asarray(logits).argmax(1)
 for kind,v,take in dimensions:
  t=y[take];q=p[take];cm=np.bincount(t*6+q,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);out.append(dict(dimension=kind,stratum=v,count=int(take.sum()),accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()),confusion=cm.tolist()))
 return out
def decision(rows):
 r=rows['FULL_As_JOINT_FINAL30']['views'];acc=lambda v,m:r[v]['conditions'][m][0]['accuracy'];fresh=lambda m:sum(acc(v,m) for v in ('mid_draw0','low_draw0','mid_draw1','low_draw1'))/4;a=100*(fresh('E')-fresh('NATIVE'));b=100*(fresh('E')-fresh('I'));c=100*(acc('raw','E')-acc('raw','NATIVE'))
 return dict(E_minus_NATIVE_fresh_midlow_pp=a,E_minus_I_fresh_midlow_pp=b,E_minus_NATIVE_raw_L_pp=c,supports_boundary_hypothesis=a>=.5 and b>=.5 and c>=-.2,automatic_training=False,automatic_extension=False,formal_scores_read=False)
