import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('static_failures',Path(__file__).parents[1]/'hf_librarian/static_failures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FailureTests(unittest.TestCase):
    def test_rule_gap_does_not_claim_sdk_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.fxb.plan').mkdir()
            m.write(root/'model.fxb.plan/preflight.json',{'code':'EXECUTION_RULE_GAP','failure_groups':[{'op':'aten.conv2d.default','reason':'no public EDF region rule'}]})
            r={'stage':'export_static','error_type':'EDFExportError','error':'changing node number'}
            m.failure_details(r,root)
            self.assertIn('SDK support unknown',r['diagnosis']['root_cause_status'])
            first=r['signature'];r['error']='different path and node';m.failure_details(r,root)
            self.assertEqual(first,r['signature'])
    def test_unconfirmed_and_unknown_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'control').mkdir()
            r={'case':'control','stage':'execute','status':'failed','error_type':'WorkerExit','error':'signal','npu_execution':'unknown','cpu_fallback':'unknown'}
            m.failure_details(r,root/'control');m.write(root/'control/result.json',r)
            rows=m.summarize(root)
            self.assertEqual(rows[0]['diagnosis']['root_cause_status'],'unconfirmed')
            self.assertIn('CPU fallback=unknown',(root/'REPORT.md').read_text())
    def test_publication_reuses_marker_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker='<!-- hf-librarian-static:resnet -->'
            m.write(root/'issues.json',[{'marker':marker,'title':'test','body':marker+' evidence'}])
            replies=[json.dumps({'items':[{'number':3,'body':marker}]}),json.dumps({'html_url':'https://example.test/3#comment'})]
            with patch.object(m.subprocess,'check_output',side_effect=replies) as call:
                m.publish(root,'owner/repo')
                self.assertIn('repos/owner/repo/issues/3/comments',call.call_args.args[0])
            with patch.object(m.subprocess,'check_output') as call:
                m.publish(root,'owner/repo');call.assert_not_called()
    def test_publication_search_error_does_not_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m.write(root/'issues.json',[{'marker':'x','title':'t','body':'b'}])
            with patch.object(m.subprocess,'check_output',side_effect=m.subprocess.CalledProcessError(1,'gh')) as call:
                with self.assertRaises(m.subprocess.CalledProcessError): m.publish(root,'owner/repo')
                self.assertEqual(call.call_count,1)
    def test_atomic_unicode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';m.write(p,{'오류':'원본'});m.write(p,{'오류':'재현'})
            self.assertEqual(json.loads(p.read_text()),{'오류':'재현'})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

if __name__=='__main__':unittest.main()
