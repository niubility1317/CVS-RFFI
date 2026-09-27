import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('early', Path(__file__).with_name('test_completed_rows.py'))
early = importlib.util.module_from_spec(spec)
spec.loader.exec_module(early)

class ContractTest(unittest.TestCase):
    def test_frozen_source_only_and_reject_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def put(path, data):
                path.write_text(json.dumps(data), encoding='utf-8')
            put(root/'pipeline_state.json', {'rows': {r: {'status':'TRAINING_COMPLETE'} for r in early.ROWS}})
            put(root/'source_contract.json', {'role_ids': {'L':[1], 'U':[2], 'V':[3]}})
            for rid in early.ROWS:
                row = root/rid; row.mkdir()
                put(row/'completion.json', {'epochs':200})
                put(row/'initialization.json', {'scratch_only':True, 'target_contact':False, 'checkpoint_sources':[]})
                (row/'source_contract.json').write_bytes((root/'source_contract.json').read_bytes())
                (row/'final_ssdg.pth').touch()
            early.validate_source(root)
            self.assertNotIn('M09', early.ROWS)
            self.assertNotIn('M10', early.ROWS)
            for path, value in [
                (root/'M00/completion.json', {'epochs':199}),
                (root/'M00/initialization.json', {'scratch_only':True,'target_contact':True,'checkpoint_sources':[]}),
                (root/'M00/source_contract.json', {'role_ids':{'L':[4]}}),
            ]:
                original = path.read_bytes()
                put(path, value)
                with self.assertRaises(ValueError):
                    early.validate_source(root)
                path.write_bytes(original)

if __name__ == '__main__':
    unittest.main()
