import json,tempfile,unittest,sys
from unittest.mock import patch
from pathlib import Path
from hf_librarian.continuous_control import control,checkpoint
from hf_librarian.campaign import save

class ContinuousControl(unittest.TestCase):
 def test_pause_does_not_stop_active_wave_and_receipt_preserves_progress(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);plan=root/'plan.json';save(plan,{'service_root':str(root)})
   wave=root/'waves/4';save(root/'state.json',{'wave':4,'active':str(wave),'attempted':{'a':{'status':'submitted'}}})
   result=control('pause',plan)
   self.assertEqual(result['status'],'pause_requested');self.assertFalse((root/'STOP.json').exists())
   self.assertFalse((wave/'STOP.json').exists());self.assertFalse((root/'pause-receipt.json').exists())
   save(root/'state.json',{'wave':4,'attempted':{'a':{'status':'submitted'}}})
   result=checkpoint(root,plan)
   self.assertEqual(result['completed_wave'],4);self.assertIsNone(result['active'])
   self.assertEqual(result['attempted_contracts'],1)
   self.assertEqual(control('status',plan)['pause-receipt.json']['status'],'paused')

 def test_native_manager_drains_active_wave_before_pause_receipt(self):
  from hf_librarian.continuous import run
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);wave=root/'waves/4';plan=root/'plan.json'
   save(plan,{'service_root':d,'catalog':str(root/'catalog.json')})
   save(root/'state.json',{'wave':4,'active':str(wave),'attempted':{'a':{}}})
   control('pause',plan);events=[]
   def wait(job):
    self.assertFalse((root/'pause-receipt.json').exists());events.append('drained')
   with patch('hf_librarian.continuous.threading.Thread'),patch('hf_librarian.continuous.environment',return_value={}),patch('hf_librarian.continuous.aggregate'),patch('hf_librarian.continuous.start') as start,patch('tplmachine.research.jobs.wait',side_effect=wait):
    run(plan)
   self.assertEqual(events,['drained']);start.assert_not_called()
   r=json.loads((root/'pause-receipt.json').read_text());self.assertEqual(r['completed_wave'],4);self.assertIsNone(r['active'])

 def test_failed_training_does_not_claim_trained_keys(self):
  from hf_librarian.continuous import aggregate
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);save(root/'waves/1/cpu/collection/sample_index.json',[{'key':'new','contexts':[{'operator':'aten.mm.default'}]}])
   state={'trained_keys':'previous'}
   plan={'history_roots':[],'feedback_lock':str(root/'feedback.lock'),'data_root':str(root/'data')}
   failure={'receipt':{'returncode':1},'result':{'status':'failed'}}
   with patch('hf_librarian.continuous.command',return_value=failure):aggregate(plan,root,state)
   self.assertEqual(state['trained_keys'],'previous');self.assertTrue((root/'review-training.json').exists())

 def test_resume_uses_librarian_interpreter_not_sdk_without_librarian_dependencies(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);plan=root/'plan.json';save(plan,{'service_root':d,'sdk_python':'/sdk/no-json5'})
   save(root/'state.json',{'wave':3,'attempted':{'saved':{}}});save(root/'PAUSE_AFTER_WAVE.json',{})
   with patch('hf_librarian.continuous_control.environment',return_value={}),patch('tplmachine.research.jobs.launch',return_value={'started':True}) as launch:
    self.assertTrue(control('resume',plan)['started'])
   self.assertEqual(launch.call_args.args[1][0],sys.executable)
   self.assertEqual(json.loads((root/'state.json').read_text())['wave'],3)
   self.assertFalse((root/'PAUSE_AFTER_WAVE.json').exists())

 def test_new_region_labels_do_not_refit_matrix_prior(self):
  from hf_librarian.continuous import aggregate
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);save(root/'waves/1/cpu/collection/sample_index.json',[{'key':'pair','contexts':[{'operator':'region.gated_mlp_pair'}]}])
   state={'trained_keys':'previous'}
   with patch('hf_librarian.continuous.command') as command:aggregate({'history_roots':[]},root,state)
   command.assert_not_called();self.assertEqual(state['trained_keys'],'previous')

if __name__=='__main__':unittest.main()
