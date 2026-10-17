"""Private label-independent receiver-processing draw;no global RNG."""
import hashlib,json
import numpy as np
SEED=20261017941
def choice(pid,epoch,step):
 if len(pid)!=5 or tuple(map(int,pid))!=tuple(pid) or pid[0]!=1 or pid[1] not in (1,3,4,6,8) or pid[2] not in (1,2,3) or epoch not in range(21,31) or step not in range(4441,6661) or epoch!=(step-1)//222+1:raise ValueError('Original fullsource physical metadata and cumulativeE21..30 required')
 namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S';raw=json.dumps([SEED,namespace,'receiver_MMSE',tuple(map(int,pid))],ensure_ascii=False,separators=(',',':'));seed=int.from_bytes(hashlib.sha256(raw.encode('utf8')).digest()[:8],'little')
 return int(np.random.default_rng(seed).integers(0,2))
