"""Run the single immutable local G0; no NN/checkpoint/radio-data machinery."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .iq import apply_allpass,response
from .fixtures import records,dense_reference,reflected_reference
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf8")
def energy(x):return float(np.sum(np.asarray(x,dtype=np.float64)**2))
def run(out,commit):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    write(out/"resolved_config.json",dict(config=d.config(),code_commit=commit,pid=os.getpid(),
        cwd=os.getcwd(),argv=sys.argv,python=sys.executable,numpy=np.__version__,GPU=None,
        real_IQ=0,checkpoint_reads=0,NN=0,updates=0))
    try:
        result=evaluate(out)
        result.update(wall_seconds=time.perf_counter()-started)
        write(out/"summary.json",result)
        write(out/"completion.json",dict(status="G0_COMPLETE",qualified=result["qualified"],
            source_only=False,synthetic_only=True,real_IQ=0,CP=0,NN=0,updates=0,
            code_commit=commit,wall_seconds=result["wall_seconds"]))
        print(json.dumps(result),flush=True)
        return result
    except Exception as e:
        write(out/"failure.json",dict(status="TECHNICAL_FAILED",error=repr(e),no_automatic_retry=True,
            NN=0,IQ=0,CP=0,updates=0))
        raise
def evaluate(out):
    rows=[];frequency={};w=np.linspace(-np.pi,np.pi,4097)
    for a in d.ALPHAS:
        h=response(a,w);phase=np.unwrap(np.angle(h))
        slope=float(np.dot(w,phase)/np.dot(w,w));intercept=float(np.mean(phase))
        residual=phase-intercept-slope*w
        fraction=float(np.dot(residual,residual)/np.dot(phase,phase)) if a else 0.
        frequency[str(a)]=dict(alpha=a,grid_count=len(w),magnitude_error=float(np.max(np.abs(np.abs(h)-1))),
            max_abs_phase=float(np.max(np.abs(phase))),best_linear_phase_slope=slope,
            nonlinear_phase_fraction=fraction,phase_bound=2*np.arcsin(abs(a)))
    for r in records():
        raw=r["iq"];original=energy(raw)
        for a in d.ALPHAS:
            view=apply_allpass(raw,a)
            ideal=dense_reference(r["full"],a)
            same_padding=reflected_reference(raw,a)
            delta=energy(view.astype(np.float64)-raw)
            error=energy(view.astype(np.float64)-ideal)
            inner_error=energy((view.astype(np.float64)-ideal)[:,:,d.CONTEXT:-d.CONTEXT])
            row=dict(case_id=r["case_id"],family=r["family"],alpha=a,
                raw_energy=original,change_energy=delta,artificial_context_error_energy=error,
                relative_change_RMS=float(np.sqrt(delta/original)),
                power_error=abs(energy(view)/original-1),
                boundary_RMS=float(np.sqrt(error/original)),
                boundary_fraction_of_change=error/delta if delta else 0.,
                interior_error_RMS=float(np.sqrt(inner_error/original)),
                dense_error_RMS=float(np.sqrt(energy(view.astype(np.float64)-same_padding)/original)),
                identity_exact=bool(np.array_equal(view,raw)),all_finite=bool(np.isfinite(view).all()))
            rows.append(row)
    if len(rows)!=96:raise ValueError("All registered observations required")
    active=[r for r in rows if r["alpha"]]
    pooled=sum(r["artificial_context_error_energy"] for r in active)/sum(r["change_energy"] for r in active)
    limits=d.LIMITS
    checks=dict(all_96_finite=all(r["all_finite"] for r in rows),
        alpha_zero_identity=all(r["identity_exact"] for r in rows if not r["alpha"]),
        unit_magnitude=all(v["magnitude_error"]<=limits["magnitude_error"] for v in frequency.values()),
        phase_bound=all(v["max_abs_phase"]<=v["phase_bound"]+1e-12 for v in frequency.values()),
        not_pure_linear_phase=all(v["nonlinear_phase_fraction"]>=limits["nonlinear_phase_fraction"] for v in frequency.values() if v["alpha"]),
        bounded_waveform=all(r["relative_change_RMS"]<=limits["relative_change"] for r in rows),
        bounded_power=all(r["power_error"]<=limits["relative_power_error"] for r in rows),
        independent_dense_match=all(r["dense_error_RMS"]<=limits["dense_error"] for r in rows),
        bounded_artificial_context=all(r["boundary_RMS"]<=limits["boundary_RMS"] for r in rows),
        pooled_change_not_boundary=pooled<=limits["pooled_boundary_fraction"],
        interior_consistency=all(r["interior_error_RMS"]<=limits["interior_error"] for r in rows))
    write(out/"case_metrics.json",rows);write(out/"frequency_summary.json",frequency)
    return dict(status="G0_ANALYZED",run_id=d.RUN,qualified=all(checks.values()),checks=checks,
        cases=32,observations=96,active_observations=64,alphas=list(d.ALPHAS),
        maxima={k:max(r[k] for r in rows) for k in ("relative_change_RMS","power_error","boundary_RMS","dense_error_RMS","interior_error_RMS")},
        pooled_boundary_fraction=pooled,
        boundary_dominated_counterexamples=[dict(case_id=r["case_id"],alpha=r["alpha"],fraction=r["boundary_fraction_of_change"]) for r in active if r["boundary_fraction_of_change"]>.5],
        NN=0,IQ=0,checkpoint_reads=0,updates=0,target_access=False,identity_efficacy_proved=False,
        goal_complete=False,on_result=d.config()["on_pass" if all(checks.values()) else "on_fail"])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--commit",required=True)
    a=p.parse_args();run(a.output,a.commit)
