import numpy as np
from phase1_direct_source_paired import paired_counts

def main():
    base=np.array([0,0,1,0]);candidate=np.array([0,1,0,1]);truth=np.array([0,1,1,0]);valid=np.array([True,True,True,False])
    assert paired_counts(base,candidate,truth,valid)==dict(count=3,baseline=2,candidate=2,repairs=1,damages=1)
    print('VERIFIED paired source repair and damage counts')
if __name__=='__main__':main()
