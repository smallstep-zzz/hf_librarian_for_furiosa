import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from hf_librarian.continuous import row_families,aggregate
class ContinuousTests(unittest.TestCase):
 def test_rows_are_bounded_distinct_real_batch_prefill_gaps(self):
  groups=row_families();self.assertEqual(len(groups),8)
  self.assertTrue(all(1<=v<=4096 for rows in groups for v in rows))
  self.assertEqual(groups[0],[3,6,12,24,48,96,192,384,768])
  self.assertEqual(len({tuple(r) for r in groups}),8)
 def test_aggregate_does_not_refit_unchanged_teacher_keys(self):
  from hf_librarian.campaign import fingerprint
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);p=root/'waves/1/models/x/cpu/collection/sample_index.json';p.parent.mkdir(parents=True);p.write_text(json.dumps([{'key':'a','contexts':[]}]))
   state={'trained_keys':fingerprint(['a'])}
   with patch('hf_librarian.continuous.command') as command:
    aggregate({'history_roots':[]},root,state)
   command.assert_not_called()

class StopTests(unittest.TestCase):
 def test_preexisting_stop_does_not_admit_a_campaign(self):
  from hf_librarian.continuous import run
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'STOP.json').write_text('{"stop":true}')
   plan=root/'plan.json';plan.write_text(json.dumps({'service_root':temp,'catalog':str(root/'catalog.json')}))
   with patch('hf_librarian.continuous.environment',return_value={}),patch('hf_librarian.continuous.start') as start:
    run(plan)
   start.assert_not_called();self.assertTrue((root/'finished.json').is_file())

if __name__=='__main__':unittest.main()
