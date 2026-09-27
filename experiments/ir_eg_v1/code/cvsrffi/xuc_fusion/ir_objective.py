"""Pure native loss assembly with explicitly registered adversarial leaves."""
from dataclasses import replace
import torch
import torch.nn.functional as F
from .ir_head import head_logits


def register_domain_call(ctx, logits, key, ids, domain, coefficient, grl):
    tape = getattr(ctx, 'ir_tape', None)
    if tape is None: return
    matches=[i for i,r in enumerate(tape.records) if r['logits'] is logits]
    if len(matches)!=1: raise ValueError('registered CE must identify exactly one native head call')
    if len(domain)==0: raise ValueError('empty registered domain call')
    tape.register(key, tuple(str(x) for x in ids), domain,
        torch.full((len(domain),), float(coefficient)/len(domain),device=domain.device),
        float(grl),call_index=matches[0])
    ctx.ir_call_indices[key]=matches[0]


class ObjectivePacket:
    def __init__(self, tape, assembly, phi_predictor, call_indices):
        self.head_batch=tape.batch()
        self.phi_predictor=tuple(p.detach().clone() for p in phi_predictor)
        self._assembly=assembly
        self._inputs={key:tape.records[i]['post_grl_input'] for key,i in call_indices.items()}
        self._released=False

    def assemble(self, phi_leaf):
        if self._released: raise RuntimeError('packet already released')
        if any(not p.is_leaf or not p.requires_grad for p in phi_leaf):
            raise ValueError('response head must consist of independent differentiable leaves')
        leaves={}
        for call in self.head_batch.calls:
            # This intentionally retains the native post-GRL graph, including
            # the zero-gradient U edge. The detached batch is for head solving.
            live=replace(call,z_detached=self._inputs[call.key])
            logits=head_logits(phi_leaf,live)
            # Native CE uses mean before applying the scalar coefficient.
            # Only uniform within-call native reductions are supported here.
            weight=call.sample_weight
            if not torch.equal(weight,weight[0].expand_as(weight)):
                raise ValueError('unsupported nonuniform native CE reduction')
            leaves[call.key]=F.cross_entropy(logits.float(),call.domain)
        return self._assembly(leaves)

    def release(self):
        self._assembly=None
        self._inputs.clear()
        self._released=True


def build_objective_packet(ctx):
    if not callable(getattr(ctx,'ir_assembly',None)): raise ValueError('objective has no pure native assembly')
    return ObjectivePacket(ctx.ir_tape,ctx.ir_assembly,ctx.ir_phi_predictor,ctx.ir_call_indices)


def graph_activity(loss, parameters):
    """Read autograd connectivity without executing a backbone backward."""
    wanted={id(p):i for i,p in enumerate(parameters)}
    active=[False]*len(parameters)
    pending=[loss.grad_fn];seen=set()
    while pending:
        node=pending.pop()
        if node is None or node in seen: continue
        seen.add(node)
        variable=getattr(node,'variable',None)
        if variable is not None and id(variable) in wanted: active[wanted[id(variable)]]=True
        pending.extend(edge[0] for edge in node.next_functions)
    return tuple(active)
