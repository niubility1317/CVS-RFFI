import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('full_early',Path(__file__).with_name('test_full_completed_rows.py'))
early=importlib.util.module_from_spec(spec);spec.loader.exec_module(early)

class ContractTest(unittest.TestCase):
    def test_only_six_frozen_same_contract_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def put(path,value):path.write_text(json.dumps(value),encoding='utf-8')
            put(root/'pipeline_state.json',{'rows':{r:{'status':'TRAINING_COMPLETE'} for r in early.ROWS}})
            put(root/'source_contract.json',{'role_ids':{'L':[1],'U':[2],'V':[3]}})
            for rid in early.ROWS:
                row=root/rid;row.mkdir()
                put(row/'completion.json',{'epochs':200,'steps':44400})
                put(row/'initialization.json',{'scratch_only':True,'target_contact':False,'checkpoint_sources':[]})
                (row/'source_contract.json').write_bytes((root/'source_contract.json').read_bytes())
                (row/'final_ssdg.pth').touch()
            early.validate_source(root)
            self.assertEqual(set(early.ROWS),{'F-A1','F-M14','F-M11','F-M05','F-M08','F-M12'})
            for path,value in [
                (root/'F-A1/completion.json',{'epochs':199,'steps':44400}),
                (root/'F-A1/completion.json',{'epochs':200,'steps':9800}),
                (root/'F-A1/initialization.json',{'scratch_only':True,'target_contact':True,'checkpoint_sources':[]}),
                (root/'F-A1/initialization.json',{'scratch_only':True,'target_contact':False,'checkpoint_sources':['other']}),
                (root/'F-A1/source_contract.json',{'role_ids':{'L':[9]}}),
            ]:
                old=path.read_bytes();put(path,value)
                with self.assertRaises(ValueError):early.validate_source(root)
                path.write_bytes(old)

if __name__=='__main__':unittest.main()
