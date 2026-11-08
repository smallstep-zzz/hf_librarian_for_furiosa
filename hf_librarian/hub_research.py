"""Conservative shared Hub request guard and cached metadata lookup."""
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
    def _save(self, state):
        temporary=self.root/f'.state-{os.getpid()}'
        temporary.write_text(json.dumps(state));os.replace(temporary,self.root/'state.json')
    def _locked(self, operation, cache_key=None, max_age=86400):
        with (self.root/'lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            now=self.clock(); state_path=self.root/'state.json'
            try: state=json.loads(state_path.read_text())
            except FileNotFoundError: state={}
            cached=state.get('models',{}).get(cache_key) if cache_key else None
            if cached and now-cached['at']<max_age: return {**cached['data'],'cache_hit':True}
            if now < state.get('cooldown_until',0): raise ResearchLimit('Hub cooldown active until '+str(int(state['cooldown_until'])))
            recent=[x for x in state.get('requests',[]) if now-x<300]
            if len(recent)>=20: raise ResearchLimit('Local Hub research budget exhausted until '+str(int(min(recent)+300)))
            if recent and now-max(recent)<5: raise ResearchLimit('Minimum request spacing is 5 seconds')
            recent.append(now);state['requests']=recent;self._save(state)
            try: result=operation()
            except Exception as e:
                response=None if isinstance(e,urllib.error.HTTPError) else getattr(e,'response',None)
                status=getattr(e,'code',None) or getattr(response,'status_code',None)
                if status==429:
                    headers=getattr(e,'headers',None) or getattr(response,'headers',{}) or {}
                    try: delay=max(300,int(headers.get('Retry-After','300')))
                    except (ValueError,TypeError): delay=300
                    state['cooldown_until']=now+delay;self._save(state)
                    raise ResearchLimit('Hub HTTP 429; cooldown recorded') from e
                raise
            if cache_key:
                state.setdefault('models',{})[cache_key]={'at':now,'data':result};self._save(state)
                return {**result,'cache_hit':False}
            return result
    def call(self, operation):
        """Guard one logical SDK call; SDK-internal requests remain outside this count."""
        try:
            return self._locked(operation)
        except ResearchLimit as exc:
            if str(exc) != 'Minimum request spacing is 5 seconds': raise
            time.sleep(5)
            return self._locked(operation)
    def model(self, model_id, max_age=86400):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model_id): raise ValueError('Expected namespace/model')
        def fetch():
            url='https://huggingface.co/api/models/'+urllib.parse.quote(model_id,safe='/')+'?blobs=true'
            with self.transport(urllib.request.Request(url,headers={'User-Agent':'hf-librarian-research/1.0'}),timeout=20) as response:
                info=json.load(response)
            return {'id':info.get('id',model_id),'revision':info.get('sha'),'gated':info.get('gated'),
                    'private':info.get('private'),'license':(info.get('cardData') or {}).get('license'),
                    'files':[{'name':x.get('rfilename'),'size':x.get('size')} for x in info.get('siblings',[])]}
        return self._locked(fetch,model_id,max_age)
