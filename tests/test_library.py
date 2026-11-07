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

if __name__ == '__main__': unittest.main()
