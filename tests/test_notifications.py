import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from hf_librarian.notifications import event,mac_notify
class NotificationsTests(unittest.TestCase):
    def test_terminal_event_is_durable_and_deduplicated(self):
        for state,phase,expected in [('completed','finished','completed'),('failed','process_failed','failed'),('completed','review_required','review_required')]:
            with self.subTest(state=state,phase=phase),tempfile.TemporaryDirectory() as temp:
                with patch('hf_librarian.activity.snapshot',return_value={'status':state,'phase':phase,'reason':'test'}):
                    a=event(temp);b=event(temp)
                self.assertEqual(a,b);self.assertEqual(a['kind'],expected)
                self.assertEqual(len((Path(temp)/'notifications/events.jsonl').read_text().splitlines()),1)
    def test_running_job_does_not_emit_completion(self):
        with tempfile.TemporaryDirectory() as temp,patch('hf_librarian.activity.snapshot',return_value={'status':'running'}):
            with self.assertRaises(ValueError):event(temp)
            self.assertFalse((Path(temp)/'notifications/event.json').exists())
    def test_model_text_is_passed_as_argument_not_applescript(self):
        with patch('hf_librarian.notifications.subprocess.run') as run:
            mac_notify({'kind':'failed','phase':'bad " quoted','job':'/job'})
        argv=run.call_args.args[0]
        self.assertNotIn('bad',argv[2]);self.assertIn('bad',argv[-1])
if __name__=='__main__':unittest.main()
