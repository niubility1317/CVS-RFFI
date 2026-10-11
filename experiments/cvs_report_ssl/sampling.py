"""RX/day round-robin U coverage, without inspecting TX or opaque UID content."""
import torch


class DomainStream:
    def __init__(self,rx,day,seed):
        if rx.shape!=day.shape or rx.ndim!=1 or len(rx)==0:
            raise ValueError('Expected source RX/day arrays')
        pairs=list(zip(rx.cpu().tolist(),day.cpu().tolist()))
        self.keys=sorted(set(pairs));self.members=[torch.tensor([i for i,p in enumerate(pairs) if p==k],dtype=torch.long) for k in self.keys]
        if len({len(m) for m in self.members})!=1:
            raise ValueError('This registered equal-domain stream requires equal observed cell sizes')
        self.count=len(pairs);self.generator=torch.Generator().manual_seed(seed)
        self.pass_index=1;self.domain_cursor=0;self._shuffle()

    def _shuffle(self):
        self.pools=[members[torch.randperm(len(members),generator=self.generator)] for members in self.members]
        self.positions=[0]*len(self.keys)

    def batch(self,size=128):
        if size<=0:raise ValueError('Positive U batch required')
        indices,passes=[],[]
        for _ in range(size):
            if sum(self.positions)==self.count:
                self.pass_index+=1;self._shuffle()
            d=self.domain_cursor
            if self.positions[d]>=len(self.pools[d]):raise ValueError('Unequal traversal reached before global pass complete')
            indices.append(int(self.pools[d][self.positions[d]]));passes.append(self.pass_index)
            self.positions[d]+=1;self.domain_cursor=(d+1)%len(self.keys)
        return torch.tensor(indices,dtype=torch.long),torch.tensor(passes,dtype=torch.long)

    def state_dict(self):
        return dict(keys=self.keys,count=self.count,pools=[p.clone() for p in self.pools],positions=list(self.positions),pass_index=self.pass_index,domain_cursor=self.domain_cursor,generator=self.generator.get_state())

    def load_state_dict(self,state):
        if state['keys']!=self.keys or state['count']!=self.count or len(state['pools'])!=len(self.keys):raise ValueError('Source domain ownership differs')
        for m,p,pos in zip(self.members,state['pools'],state['positions']):
            if not torch.equal(m.sort().values,p.sort().values) or not 0<=pos<=len(p):raise ValueError('Invalid source permutation/cursor')
        if len(state['positions'])!=len(self.keys) or not 0<=state['domain_cursor']<len(self.keys) or state['pass_index']<1:raise ValueError('Invalid global traversal state')
        self.pools=[p.clone() for p in state['pools']];self.positions=list(state['positions']);self.domain_cursor=state['domain_cursor'];self.pass_index=state['pass_index'];self.generator.set_state(state['generator'])
