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
