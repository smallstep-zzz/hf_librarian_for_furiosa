import io,json,tempfile,unittest,urllib.error
from hf_librarian.hub_research import HubResearch,ResearchLimit
class Tests(unittest.TestCase):
 def test_cache_and_cooldown(self):
  with tempfile.TemporaryDirectory() as root:
   calls=[]
   def transport(request,timeout):
    calls.append(request.full_url)
    if 'good' in request.full_url:return io.BytesIO(json.dumps({'id':'org/good','sha':'abc','siblings':[]}).encode())
    raise urllib.error.HTTPError(request.full_url,429,'limited',{'Retry-After':'30'},None)
   t=[1000]; api=HubResearch(root,lambda:t[0],transport)
   self.assertFalse(api.model('org/good')['cache_hit'])
   self.assertTrue(api.model('org/good')['cache_hit'])
   self.assertEqual(len(calls),1)
   t[0]+=6
   with self.assertRaises(ResearchLimit):api.model('org/limited')
   t[0]+=6
   with self.assertRaises(ResearchLimit):api.model('org/another')
   self.assertEqual(len(calls),2)
