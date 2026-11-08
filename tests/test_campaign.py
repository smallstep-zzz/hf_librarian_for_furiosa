import concurrent.futures
import json
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from hf_librarian.campaign import Ledger, estimate, lease, repair, run, save, npu_task


class CampaignTests(unittest.TestCase):
    def test_config_intake_downloads_no_weights_and_reuses_pinned_hash(self):
        from hf_librarian import Catalog,Librarian
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalog=root/'catalog.json'
            save(catalog,{'schema_version':1,'settings':{'max_concurrent_downloads':2,'min_free_gib':.001},
                'categories':{'test':'test'},'blacklist':{},'models':[{'id':'org/model','categories':['test']}]})
            path=root/'cache/snapshot/config.json';path.parent.mkdir(parents=True);path.write_text('{}')
            info=SimpleNamespace(sha='a'*40,safetensors=SimpleNamespace(total=100),siblings=[SimpleNamespace(rfilename='config.json',size=2)],card_data={'license':'apache-2.0'})
            with patch('hf_librarian.library.HfApi') as api,patch('huggingface_hub.hf_hub_download',return_value=str(path)) as download,patch('hf_librarian.library.snapshot_download',side_effect=AssertionError('weights requested')):
                api.return_value.model_info.return_value=info
                with Librarian(Catalog(catalog),root/'cache') as library:
                    self.assertEqual(library.request(model_id='org/model',profile='config'),str(path.parent))
                    self.assertEqual(library.request(model_id='org/model',profile='config'),str(path.parent))
                self.assertEqual(download.call_count,1)

    def test_eta_excludes_lock_wait(self):
        rows=[{'state':'completed','cpu':{'cpu_s':10},'npu':{'npu_active_s':2,'npu_wait_s':9999}},
              {'state':'ready'},{'state':'ready'}]
        eta=estimate(rows,2,100)
        self.assertEqual(eta['eta_s'],13)
        self.assertNotIn('9999',json.dumps(eta))

    def test_three_attempt_limit_and_previous_reason_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            for i in range(3):
                repair(temp,'group','begin',{'hypothesis':'reason'+str(i),'action':'change'+str(i)})
                result=repair(temp,'group','record',{'hypothesis':'reason'+str(i),'action':'change'+str(i),
                     'validation_status':'failed','evidence_path':'evidence.json'})
            self.assertTrue(result['requires_user_decision'])
            self.assertEqual(len(repair(temp,'group','history')['attempts']),3)
            with self.assertRaises(ValueError):repair(temp,'group','begin',{'hypothesis':'more','action':'fourth'})

    def test_npu_lease_serializes_workers_but_records_wait_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);active=0;peak=0;guard=threading.Lock()
            def command(plan,argv,output,fd=None):
                nonlocal active,peak
                if argv[0]=='execute':
                    with guard:active+=1;peak=max(peak,active)
                    time.sleep(.05)
                    with guard:active-=1
                return {'receipt':{'returncode':0},'result':{'status':'passed'},'result_path':str(root/'audit.json')}
            plan={'root':temp,'npu_lock':str(root/'npu.lock'),'feedback_lock':str(root/'feedback.lock'),'data_root':str(root/'data')}
            with patch('hf_librarian.campaign.command',side_effect=command),patch('hf_librarian.campaign.npu_owners',return_value=[]),concurrent.futures.ThreadPoolExecutor(2) as pool:
                rows=list(pool.map(lambda name:npu_task(plan,name,'prepared.json'),['one','two']))
            self.assertEqual(peak,1)
            self.assertGreater(max(r['npu_wait_s'] for r in rows),.02)
            self.assertTrue(all(r['npu_active_s']<.1 for r in rows))

    def test_controller_adapts_without_an_agent_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);models=['model'+str(i) for i in range(10)]
            plan={'root':temp,'models':models,'catalog':str(root/'catalog.json'),'tclmachine_source':temp,
                  'data_root':str(root/'data'),'feedback_lock':str(root/'feedback.lock'),'max_parallel':5,'npu':True}
            save(root/'plan.json',plan)
            save(root/'catalog.json',{'schema_version':1,'categories':{},'models':[],'blacklist':{},'settings':{}})
            ledger=Ledger(root)
            for m in models:ledger.put(m,'pending')
            ledger.close()
            class Library:
                def __init__(self,*a,**kw):pass
                def __enter__(self):return self
                def __exit__(self,*a):pass
                def request_async(self,model_id,profile):
                    path=root/'snapshots'/model_id;path.mkdir(parents=True);save(path/'config.json',{'model':model_id})
                    f=concurrent.futures.Future();f.set_result(str(path));return f
            def cpu(plan,model,snapshot):
                time.sleep(.04)
                return {'receipt':{'returncode':0},'result':{'status':'passed','prepared_probe':'prepared'},'cpu_s':.04}
            def npu(*args):return {'receipt':{'returncode':0},'result':{'status':'passed'},'npu_active_s':.001,'npu_wait_s':0}
            with patch('hf_librarian.library.Librarian',Library),patch('hf_librarian.campaign.cpu_task',side_effect=cpu),\
                 patch('hf_librarian.campaign.npu_task',side_effect=npu),patch('hf_librarian.campaign.memory_slots',return_value=5):
                self.assertEqual(run(root),0)
            result=json.loads((root/'summary.json').read_text())
            self.assertEqual(result['counts'],{'completed':10})
            self.assertEqual(result['peak_parallel'],5)
            self.assertEqual([h['limit'] for h in result['autoscale']],[3,5])


if __name__=='__main__':unittest.main()
