"""Conservative, cross-process Hub metadata lookup; never downloads model files."""
import fcntl
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

class ResearchLimit(RuntimeError):
    pass

class HubResearch:
    def __init__(self, root=None, clock=time.time, transport=None):
        self.root = Path(root or os.environ.get('HF_LIBRARIAN_RESEARCH_CACHE', Path.home()/'.cache/hf-librarian/research'))
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.transport = transport or urllib.request.urlopen
    def model(self, model_id, max_age=86400):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model_id):
            raise ValueError('Expected namespace/model')
        with (self.root/'lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now=self.clock(); state_path=self.root/'state.json'
            try: state=json.loads(state_path.read_text())
            except FileNotFoundError: state={}
            cached=state.get('models',{}).get(model_id)
            if cached and now-cached['at']<max_age: return {**cached['data'],'cache_hit':True}
            if now < state.get('cooldown_until',0): raise ResearchLimit('Hub cooldown active; try after '+str(int(state['cooldown_until'])))
            recent=[x for x in state.get('requests',[]) if now-x<300]
            if len(recent)>=20: raise ResearchLimit('Local Hub research budget exhausted; try after '+str(int(recent[0]+300)))
            if recent and now-recent[-1]<5: raise ResearchLimit('Minimum request spacing is 5 seconds; retry later')
            recent.append(now); state['requests']=recent; state_path.write_text(json.dumps(state))
            url='https://huggingface.co/api/models/'+urllib.parse.quote(model_id,safe='/')+'?blobs=true'
            try:
                with self.transport(urllib.request.Request(url,headers={'User-Agent':'hf-librarian-research/1.0'}),timeout=20) as response:
                    info=json.load(response)
            except urllib.error.HTTPError as e:
                if e.code==429:
                    try: delay=max(300,int(e.headers.get('Retry-After','300')))
                    except ValueError: delay=300
                    state['cooldown_until']=now+delay;state_path.write_text(json.dumps(state))
                    raise ResearchLimit('Hub returned HTTP 429; cooldown recorded') from e
                raise
            result={'id':info.get('id',model_id),'revision':info.get('sha'),'gated':info.get('gated'),
                    'private':info.get('private'),'license':(info.get('cardData') or {}).get('license'),
                    'files':[{'name':x.get('rfilename'),'size':x.get('size')} for x in info.get('siblings',[])]}
            state.setdefault('models',{})[model_id]={'at':now,'data':result}
            state_path.write_text(json.dumps(state));return {**result,'cache_hit':False}
