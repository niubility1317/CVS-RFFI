import copy
import hashlib
import numpy as np
import pytest
import cvsrffi.cid_bpp_phase1_nested_lodo as nested
import cvsrffi.cid_bpp_fixed_held_spike as spike
from cvsrffi.cid_bpp_fixed_held_spike import ARMS, CIDBPPFixedHeldError, build_packet, predict_packet, score_packet
from cvsrffi.r2a_fixed_held_four_arm import DUAL_ARCHIVE_SCHEMA, COVERAGE_SCHEMA

SHA=hashlib.sha256(b"cid-bpp-coverage").hexdigest()
BINDING={"archive_schema":DUAL_ARCHIVE_SCHEMA,"coverage_schema":COVERAGE_SCHEMA,"archive_sha256":"a"*64,"manifest_sha256":"b"*64,"coverage_sha256":SHA}

def archive():
 r={k:[] for k in("z_id","z_dom","labels","receiver_ids","day_ids","physical_ids","scenario_names")}; classes=[f"c{i}" for i in range(6)]
 for ri in range(7):
  for ci,c in enumerate(classes):
   for si,s in enumerate(("leo_clear_weak","leo_low_elev_weak","leo_rain_weak")):
    for n in range(8):
     x=np.zeros(160,np.float32);x[ci]=1;x[20+si]=.02*(n+1);x[40+ri]=.01*(ci+1);x[90+n]=.003*(ri+1)
     d=np.zeros(160,np.float32);d[ri]=1;d[70+si]=.2
     for k,v in (("z_id",x),("z_dom",d),("labels",c),("receiver_ids",f"r{ri}"),("day_ids",f"d{n%4}"),("physical_ids",f"r{ri}-{c}-{s}-{n}"),("scenario_names",s)):r[k].append(v)
 return {**{k:np.asarray(v) for k,v in r.items()},"class_ids":np.asarray(classes)}

def query(packet,a):
 ids=sorted({x for r in packet["rows"] for x in r["query_ids"]}); idx={x:i for i,x in enumerate(a["physical_ids"].tolist())};return ids,np.asarray([a["z_id"][idx[x]] for x in ids],np.float32)
def resign_packet(p):p["packet_sha256"]=spike._digest({k:v for k,v in p.items() if k!="packet_sha256"})
def resign_prediction(p):p["COMMIT"]=spike._digest({k:v for k,v in p.items() if k!="COMMIT"})
def resign_truth(t):t["truth_sha256"]=spike._digest({k:v for k,v in t.items() if k!="truth_sha256"})

@pytest.fixture(scope="module")
def built():
 original=nested.FAMILIES
 try:
  nested.FAMILIES=(original[0],) # permitted test-only acceleration; production remains frozen at eight
  a=archive(); p,t=build_packet(a,coverage_sha256=SHA,artifact_binding=BINDING); ids,z=query(p,a); pred=predict_packet(p,ids,z)
  return a,p,t,ids,z,pred
 finally:nested.FAMILIES=original

def test_frozen_family_constants_and_primary_sorting_difference():
 assert len(nested.FAMILIES)==8 and nested.FROZEN_FAMILY_COUNT==8
 assert tuple(x[0] for x in nested.FAMILIES)==("F00","F01","F02","F03","F04","F05","F06","F07")
 # A wins under min(mean delta)=.15, while the rejected mean(per-episode min)
 # would incorrectly prefer B (.10 over 0).
 a={"M0":.0,"M_DA":.15,"M_HEAD":.15,"M_JOINT":.30}; b={"M0":.0,"M_DA":.10,"M_HEAD":.10,"M_JOINT":.20}
 assert nested.primary_family_score(a)>.10 and nested.primary_family_score(a)>nested.primary_family_score(b)
 assert np.mean((0.,0.))<np.mean((.10,.10))

def test_build_predict_score_receipt_and_inner_authority(built):
 a,p,t,ids,z,pred=built
 assert len(p["rows"])==18 and len(p["lock_groups"])==6 and p["contract"]["selection_family_count"]==8
 for held,g in p["lock_groups"].items():
  receipt=g["selection_receipt"]
  assert receipt["production_family_count"]==8 and receipt["evaluated_family_count"]==1
  assert len(receipt["family_scores"])==1 and receipt["exclusion"]["outer_held_receiver"]==p["held_receiver"]
  assert receipt["selected_family_inner_audit_and_jackknife"]
  assert all(e["fp32_teacher_audit"]["C5"]["aggregate_top1_agreement"]>=.995 for e in receipt["selected_family_inner_audit_and_jackknife"])
 for r in p["rows"]:
  assert r["c5"]["resource"]["jackknife_count"] in (0,5)
  assert r["c6"]["resource"]["quantization_gate"]["authority"]=="selected_inner_outer_excluded_held_day_query"
 metrics=score_packet(p,pred,t,commit=pred["COMMIT"],truth_sha256=t["truth_sha256"])
 assert len(metrics)==72 and {x["arm"] for x in metrics}==set(ARMS)
 for i in range(0,72,4):
  q={x["arm"]:x for x in metrics[i:i+4]};e=q["M_JOINT"]["H_old_new"]-q["M_DA"]["H_old_new"]-q["M_HEAD"]["H_old_new"]+q["M0"]["H_old_new"]
  assert all(x["I_syn"]==e for x in q.values())

def test_outer_exclusion_isolation(built):
 a,p,_,_,_,_=built; held=p["held_receiver"]; pseudo=p["classes"][0]
 outer_rows=[r for r in p["rows"] if r["pseudo_new"]==pseudo]
 b={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in a.items()}
 mask=(b["receiver_ids"].astype(str)==held)|(b["labels"].astype(str)==pseudo)
 b["z_id"][mask]*=-17.; b["labels"][b["receiver_ids"].astype(str)==held]="outer-mutated"
 original=nested.FAMILIES
 try:
  nested.FAMILIES=(original[0],)
  s=nested.select_nested_lodo(b,held_receiver=held,outer_pseudo_new=pseudo,outer_row_ids=[r["row_id"] for r in outer_rows],outer_support_ids=[x for r in outer_rows for x in r["support_physical_ids"]["C4"]+r["support_physical_ids"]["C5"]],outer_query_ids=[x for r in outer_rows for x in r["query_ids"]])
 finally:nested.FAMILIES=original
 assert s.selection_receipt_sha256==p["lock_groups"][pseudo]["selection_receipt_sha256"]

def test_packet_negative_bpp_quant_resource_and_jackknife_fallback(built,monkeypatch):
 a,p,_,_,_,_=built
 bad=copy.deepcopy(p); bad["rows"][0]["c6"]["bpp_cid_wire_sha256"]="0"*64; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 bad=copy.deepcopy(p); bad["rows"][0]["c6"]["resource"]["mac_ledger"]["build_total_mac"]=spike.MAX_BUILD_MAC+1; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 bad=copy.deepcopy(p); bad["rows"][0]["c6"]["resource"]["quantization_gate"]["authority"]="outer_target_support"; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 bad=copy.deepcopy(p); g=bad["lock_groups"][bad["classes"][0]]; g["selection_receipt"]["exclusion"]["outer_support_ids"][0]="forged-support"; g["selection_receipt_sha256"]=spike._digest(g["selection_receipt"]); resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 bad=copy.deepcopy(p); bad["lock_groups"][bad["classes"][0]]["bpp_c6"]["inverse_gamma_a0"]=3.; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 bad=copy.deepcopy(p); bad["rows"][0]["c6"]["resource"]["build_elapsed_ns"]=spike.MAX_BUILD_NS+1; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(bad,*query(p,a))
 monkeypatch.setattr(spike,"_projectors",lambda *_:[])
 original=nested.FAMILIES
 try:
  nested.FAMILIES=(original[0],); q,_=build_packet(a,coverage_sha256=SHA,artifact_binding=BINDING)
 finally:nested.FAMILIES=original
 assert all(r["c6"]["resource"]["jackknife_fallback"]=="jackknife_no_direction" for r in q["rows"])
 spike._verify_packet(q)
 low_overlap=[np.outer(v,v) for v in np.eye(spike.K,spike.Z_DIM)]
 monkeypatch.setattr(spike,"_projectors",lambda *_:low_overlap)
 try:
  nested.FAMILIES=(original[0],); low,_=build_packet(a,coverage_sha256=SHA,artifact_binding=BINDING)
 finally:nested.FAMILIES=original
 assert all(r[s]["resource"]["jackknife_fallback"]=="jackknife_overlap" for r in low["rows"] for s in ("c5","c6"))
 spike._verify_packet(low)
 bad=copy.deepcopy(low); bad["rows"][0]["c6"]["resource"]["jackknife_fallback"]="none"; resign_packet(bad)
 with pytest.raises(CIDBPPFixedHeldError):spike._verify_packet(bad)

def test_predict_truth_and_row_reorder_negatives(built):
 a,p,t,ids,z,pred=built
 with pytest.raises(CIDBPPFixedHeldError):predict_packet(p,ids,z.astype(np.float64))
 bad=copy.deepcopy(pred); bad["rows"][1],bad["rows"][2]=bad["rows"][2],bad["rows"][1]; resign_prediction(bad)
 with pytest.raises(CIDBPPFixedHeldError):score_packet(p,bad,t,commit=bad["COMMIT"],truth_sha256=t["truth_sha256"])
 bad=copy.deepcopy(pred); bad["rows"][1]["row_id"]=bad["rows"][0]["row_id"]; resign_prediction(bad)
 with pytest.raises(CIDBPPFixedHeldError):score_packet(p,bad,t,commit=bad["COMMIT"],truth_sha256=t["truth_sha256"])
 badt=copy.deepcopy(t); row=badt["rows"][0]; key=next(iter(row["query_labels"])); value=row["query_labels"].pop(key); row["query_labels"]["unknown-id"]=value; resign_truth(badt)
 with pytest.raises(CIDBPPFixedHeldError):score_packet(p,pred,badt,commit=pred["COMMIT"],truth_sha256=badt["truth_sha256"])
 bad=copy.deepcopy(pred); bad["performance"]["aggregate_four_arm_mac"]+=1; resign_prediction(bad)
 with pytest.raises(CIDBPPFixedHeldError):score_packet(p,bad,t,commit=bad["COMMIT"],truth_sha256=t["truth_sha256"])
