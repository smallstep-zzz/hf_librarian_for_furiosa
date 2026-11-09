import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hf_librarian import Catalog, Librarian

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'catalog.json'
        self.path.write_text(json.dumps({'schema_version':1,'settings':{'cache_limit_gib':1,'min_free_gib':1,'max_model_gib':1,'max_concurrent_downloads':2},'categories':{'test':'Test'},'blacklist':{},'models':[{'id':'org/model','categories':['test'],'priority':1}]}))
        self.catalog = Catalog(self.path)
    def test_catalog_and_blacklist(self):
        self.catalog.add_category('new','New')
        self.catalog.add_model({'id':'org/second','categories':['new']})
        self.assertEqual(len(Catalog(self.path).models('new')), 1)
        self.catalog.set_blacklist('org/model', reason='excluded')
        with Librarian(self.catalog, self.temp.name) as lib:
            with self.assertRaises(PermissionError): lib.request(model_id='org/model')
            with self.assertRaises(LookupError): lib.select('test')
        self.catalog.remove_model('org/second')
        self.assertIsNone(Catalog(self.path).get('org/second'))
    def test_offline_reuse_without_hub(self):
        import hashlib
        key = hashlib.sha256(json.dumps(['org/model','main',None,[]],sort_keys=True).encode()).hexdigest()
        cache = Path(self.temp.name)/'cache'; cache.mkdir()
        snapshot = cache/'snapshot'; snapshot.mkdir(); (snapshot/'model.safetensors').write_text('weights')
        markers = cache/'.hf-librarian'; markers.mkdir()
        (markers/(key+'.json')).write_text(json.dumps({'path':str(snapshot),'files':['model.safetensors']}))
        events = []
        with patch('hf_librarian.library.HfApi', side_effect=AssertionError('network accessed')):
            with Librarian(self.catalog,cache,on_event=events.append) as lib:
                self.assertEqual(lib.request(category='test'),str(snapshot))
        self.assertEqual(events[0]['event'],'available')

    def test_zero_download_reuses_shared_cache_but_new_bytes_obey_budget(self):
        from types import SimpleNamespace as S
        snapshot=Path(self.temp.name)/'snapshot';snapshot.mkdir();(snapshot/'model.safetensors').write_text('weights')
        info=S(sha='commit',siblings=[S(rfilename='model.safetensors',size=7)])
        for needed in (False,True):
            with self.subTest(needed=needed),patch('hf_librarian.library.HfApi') as api,patch('hf_librarian.library.snapshot_download',return_value=[S(file_size=7,will_download=needed)]),patch('hf_librarian.library.subprocess.run',return_value=S(returncode=0,stdout=str(snapshot)+'\n',stderr='')) as run:
                api.return_value.model_info.return_value=info
                with Librarian(self.catalog,Path(self.temp.name)/str(needed)) as lib:
                    if needed:
                        with patch.object(lib,'usage',return_value={'cache_bytes':2*1024**3,'limit_bytes':1024**3,'free_bytes':4*1024**3}),self.assertRaises(RuntimeError):lib.request(model_id='org/model')
                        run.assert_not_called()
                    else:
                        with patch.object(lib,'usage',side_effect=AssertionError('no growth: do not scan/reject shared cache')):self.assertEqual(lib.request(model_id='org/model'),str(snapshot))

if __name__ == '__main__': unittest.main()

class VerificationTests(unittest.TestCase):
    def test_version_mismatch_is_recorded_without_build(self):
        from hf_librarian import Verifier
        with tempfile.TemporaryDirectory() as temp:
            binary=Path(temp)/'fxb'
            binary.write_text('#!/bin/sh\necho "fxb v1.2.3"\n')
            binary.chmod(0o755)
            class FakeCatalog:
                def get(self, model): return {'id':model}
                def blacklist(self): return {}
            class FakeLibrarian:
                catalog=FakeCatalog()
                def request(self, **kwargs): raise AssertionError('must not download')
            v=Verifier(FakeLibrarian(),temp)
            result=v.verify('org/model',executable=str(binary),version='9.9.9')
            self.assertEqual(result['status'],'blocked')
            self.assertEqual(result['actual_version'],'1.2.3')
            self.assertEqual(v.history('org/model')[0]['id'],result['id'])
