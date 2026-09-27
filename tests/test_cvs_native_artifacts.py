import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cvs_native_artifacts',ROOT/'tools/cvs_native_artifacts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SourceContractTests(unittest.TestCase):
    def test_rejects_inheritance_target_access_and_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)
            contract=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1],source_days=[1],ratios=[.07,.63,.3],split_seed=392005,num_classes=6)
            docs={'source_contract.json':dict(contract,native_role_comparison='EXACT_MATCH'),
                  'initialization.json':dict(scratch_only=True,checkpoint_sources=[],target_contact=False),
                  'completion.json':dict(target_evaluated=False,epochs=200,status='TRAINING_COMPLETE'),
                  'resolved_config.json':dict(seed=392005,checkpoint_selection='final_only',a1_periodic_target_start=0,a1_final_weak_reference=False)}
            def save():
                for name,value in docs.items():(source/name).write_text(json.dumps(value),encoding='utf-8')
            save();module.verify_source(source,contract,392005)
            for file,key,value in [('initialization.json','checkpoint_sources',['old']),('initialization.json','target_contact',True),
                                   ('completion.json','epochs',100),('resolved_config.json','seed',1),
                                   ('source_contract.json','role_ids',{'L_s':['wrong']})]:
                old=docs[file][key];docs[file][key]=value;save()
                with self.assertRaises(ValueError):module.verify_source(source,contract,392005)
                docs[file][key]=old


if __name__=='__main__':unittest.main()
