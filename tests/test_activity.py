import json,tempfile,unittest
from pathlib import Path
from hf_librarian.activity import snapshot
from hf_librarian.campaign import save
class ActivityTests(unittest.TestCase):
    def test_failed_supervisor_reports_bounded_error_without_starting_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save(root/'request.json',{'command':['python','something']})
            save(root/'result.json',{'status':'failed'})
            (root/'worker.log').write_text('x'*10000+'TypeError: direction')
            result=snapshot(root)
            self.assertEqual(result['phase'],'process_failed')
            self.assertFalse(result['process_alive'])
            self.assertLessEqual(len(result['error_excerpt']),1800)
            self.assertIn('direction',result['error_excerpt'])
    def test_overnight_discovery_uses_supported_hub_call(self):
        import inspect
        from huggingface_hub import HfApi
        inspect.signature(HfApi.list_models).bind(HfApi(),pipeline_tag='text-generation',sort='downloads',limit=300)
    def test_absent_job_is_not_reported_as_running(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):snapshot(temp)
if __name__=='__main__':unittest.main()
