import numpy as np
from phase1_source_multicandidate import metrics

def main():
    truth=np.array([0,1,1,0]);mask=np.array([True,True,True,False]);pred={'raw':np.array([0,0,0,0]),'dura':np.array([0,0,1,0]),'semantic':np.array([0,1,0,0]),'curriculum':np.array([0,1,1,1])};r=metrics(pred,truth,mask)
    assert r['count']==3 and r['dura']['correct']==2 and r['semantic']['repairs_vs_dura']==1 and r['semantic']['damages_vs_dura']==1 and r['oracle_any_correct']==3
    print('VERIFIED paired source candidates and nondeployable oracle count')
if __name__=='__main__':main()
