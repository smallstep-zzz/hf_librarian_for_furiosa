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
 def test_sdk_429_stops_later_calls(self):
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as root:
   count=[]
   class HubFailure(Exception):
    response=SimpleNamespace(status_code=429,headers={'Retry-After':'900'})
   def fail():
    count.append(1)
    raise HubFailure()
   api=HubResearch(root,lambda:1000)
   with self.assertRaises(ResearchLimit): api.call(fail)
   with self.assertRaises(ResearchLimit): api.call(fail)
   self.assertEqual(len(count),1)
   state=json.loads((__import__('pathlib').Path(root)/'state.json').read_text())
   self.assertEqual(state['cooldown_until'],1900)
