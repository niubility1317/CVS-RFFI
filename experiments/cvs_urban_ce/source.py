import argparse
import torch
from experiments.cvs_urban_ce import design
from experiments.cvs_urban_ce.runtime import installed
from experiments.cvs_phase1_stack import source

def train(c):
    source.validate=design.validate;source.make_args=design.make_args;source.installed=installed
    torch.cuda.reset_peak_memory_stats()
    source.train(c)
    path=design.Path(c['output_root'])/'completion.json';done=design.read(path)
    done['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    done['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    done['cost_scope']='full source.train including own smoke, validation and final freeze'
    design.write(path,done)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    train(design.read(p.parse_args().config))
