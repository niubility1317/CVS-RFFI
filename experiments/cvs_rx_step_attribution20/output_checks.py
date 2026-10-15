"""Focused regression for preserving an existing run's output evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import runner, source_probe


class OutputOwnership(unittest.TestCase):
    def invoke(self, root, failure):
        request=root.parent/'request.json'
        request.write_text(json.dumps(dict(output_root=str(root),source_release=str(root.parent/'previous'))),encoding='utf-8')
        with patch.object(sys,'argv',['runner',str(request)]),patch.object(source_probe,'run',side_effect=failure):
            with self.assertRaises(FileExistsError):runner.main()

    def test_existing_partial_output_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output';root.mkdir()
            original=b'{"status":"FAILED","reason":"original scientific failure"}\n'
            (root/'failure.json').write_bytes(original)
            (root/'partial.json').write_bytes(b'original partial evidence\n')
            before={p.name:p.read_bytes() for p in root.iterdir()}
            self.invoke(root,FileExistsError('Preserve the existing attribution output'))
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_existing_completed_output_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output';root.mkdir()
            (root/'summary.json').write_bytes(b'original summary\n')
            before={p.name:p.read_bytes() for p in root.iterdir()}
            self.invoke(root,FileExistsError('Preserve the existing attribution output'))
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_new_owned_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output'
            def fail(request,out):
                (out/'partial.json').write_bytes(b'new partial evidence\n')
                raise RuntimeError('owned failure')
            with patch.object(source_probe.d,'validate'),patch.object(source_probe.d,'OUTPUT',str(root)),patch.object(source_probe,'_run_owned',side_effect=fail,create=True):
                with self.assertRaisesRegex(RuntimeError,'owned failure'):source_probe.run({})
            failure=json.loads((root/'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['reason'],'owned failure')
            self.assertEqual((root/'partial.json').read_bytes(),b'new partial evidence\n')


if __name__=='__main__':
    if '--existing-only' in sys.argv:
        suite=unittest.TestSuite(OutputOwnership(name) for name in ('test_existing_partial_output_unchanged','test_existing_completed_output_unchanged'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main(verbosity=2)
