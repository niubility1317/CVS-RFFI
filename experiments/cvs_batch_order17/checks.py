"""Focused L-only and unchanged-observation counterexamples; zero model/data reads."""
from collections import Counter
from .reorder import BLOCK,BATCH,reorder,metrics,labeled_records

def fixture():
 cells=[(t,1,r,1,0,t*4+i,t) for t in range(6) for i,r in enumerate((3,4,6,8))]
 flat=[cells[i%24] for i in range(768)]
 return [flat[i:i+128] for i in range(0,768,128)]
def main():
 b=fixture();a=reorder(b,0);assert a==reorder(b,0) and a!=reorder(b,1);assert Counter(x for row in a for x in row)==Counter(x for row in b for x in row)
 for t in range(6):
  for r in (3,4,6,8):
   n=[sum((x[0],x[2])==(t,r) for x in row) for row in a];assert max(n)-min(n)<=1
 for transform in (lambda x:x[:-1],lambda x:[x[0][:-1],*x[1:]],lambda x:[[tuple([6,*x[0][0][1:]]),*x[0][1:]],*x[1:]],lambda x:[[tuple([x[0][0][0],1,1,*x[0][0][3:]]),*x[0][1:]],*x[1:]]):
  try:reorder(transform(b),0)
  except ValueError:pass
  else:raise AssertionError('Illegal batch/label/heldRX accepted')
 class Hidden(dict):
  def __getitem__(self,k):
   if k=='ids':raise AssertionError('U metadata accessed')
   return super().__getitem__(k)
 assert list(labeled_records([Hidden(role='U'),dict(role='L',ids=[1])]))==[[1]]
 one=[(0,1,3,1,i,i,0) for i in range(128)];z=metrics(one);assert z['valid_anchors']==0 and z['positive_min']==0
 assert metrics(a[0])['valid_anchors']==128
 print('PASS: exact physical multiplicities; deterministic private seed; per-cell quotas; malformed/held/label rejection; U IDs untouched; no manufactured valid anchors. NN=0/realIQ=0/updates=0')
if __name__=='__main__':main()
