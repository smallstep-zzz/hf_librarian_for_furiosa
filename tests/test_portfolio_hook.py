import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from hf_librarian.portfolio_hook import after_wave
class Hook(unittest.TestCase):
 def test_discovery_failure_does_not_run_npu_or_stop_corpus(self):
  with tempfile.TemporaryDirectory() as d:
   with patch('hf_librarian.portfolio_hook.environment',return_value={}),patch('hf_librarian.portfolio_hook.subprocess.Popen'),patch('tplmachine.research.jobs.wait_process',return_value=2):
    r=after_wave({'portfolio_root':d,'sdk_python':'/sdk/python'},Path(d))
   self.assertEqual(len(r['steps']),1);self.assertFalse(r['runtime_promotion'])
 def test_tcl_change_does_not_replay_corpus_but_generator_change_does(self):
  from hf_librarian.continuous import corpus_rules
  with tempfile.TemporaryDirectory() as d:
   b=Path(d)/'tplmachine';(b/'research').mkdir(parents=True)
   for name,text in [('research/improvement_model.py','generator'),('aten_kernels.py','mm'),('research/campaign_worker.py','def collect_phase():\n pass\n'),('mlp_regions.py','tcl-v1')]:
    (b/name).write_text(text)
   state={};original=corpus_rules(d,state);(b/'mlp_regions.py').write_text('tcl-v2')
   self.assertEqual(corpus_rules(d,state),original)
   (b/'research/improvement_model.py').write_text('generator-v2');self.assertNotEqual(corpus_rules(d,state),original)
 def test_pause_does_not_admit_new_comparison(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'PAUSE_AFTER_WAVE.json').touch()
   r=after_wave({'service_root':d,'portfolio_root':d},Path(d))
   self.assertEqual(r['status'],'skipped_for_pause')
 def test_adding_other_aten_kernels_keeps_mm_corpus_signature(self):
  from hf_librarian.continuous import corpus_rules
  with tempfile.TemporaryDirectory() as d:
   b=Path(d)/'tplmachine';(b/'research').mkdir(parents=True)
   for name,text in [('research/improvement_model.py','gen'),('aten_kernels.py','def contraction():\n pass\ndef mm():\n pass\n'),('research/campaign_worker.py','def collect_phase():\n pass\n')]:
    (b/name).write_text(text)
   state={};old=corpus_rules(d,state)
   with (b/'aten_kernels.py').open('a') as f:f.write('\ndef bmm():\n pass\n')
   self.assertEqual(corpus_rules(d,state),old)
