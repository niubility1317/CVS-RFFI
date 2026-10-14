import json
from . import design as d
from .qualification import check
from experiments.cvs_signature_order50.preflight import main as smoke
def main():
 qualification=check();smoke();v=d.read(d.ROOT/'preflight.json');v.update(source_qualification=qualification);d.write(d.ROOT/'preflight.json',v);print(json.dumps(v))
if __name__=='__main__':main()
