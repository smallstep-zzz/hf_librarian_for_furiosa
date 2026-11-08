"""Popularity-first bounded discovery; detached campaigns, event completion waits."""
import argparse,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from .catalog import Catalog
from .campaign import start,save
from .hub_research import HubResearch,ResearchLimit

def run(plan_path,deadline):
    from tplmachine.research.jobs import wait
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['root']+'-overnight');root.mkdir(exist_ok=True)
    catalog=Catalog(plan['catalog']);seen=set(plan['models']);api=HfApi();gate=HubResearch();round_id=0
    previous=plan.get('previous_root')
    if previous:
        import sqlite3
        for db_path in Path(previous).glob('round-*/campaign.sqlite'):
            with sqlite3.connect(db_path) as db:
                seen.update(row[0] for row in db.execute("SELECT model FROM items WHERE state='completed'"))
    end=datetime.fromisoformat(deadline).replace(tzinfo=ZoneInfo('Asia/Seoul')).timestamp()
    save(root/'activity.json',{'phase':'starting','deadline':deadline,'updated':time.time()})
    for difficulty,ceiling in [('basic',2_000_000_000),('intermediate',8_000_000_000),('advanced',20_000_000_000)]:
        if time.time()>=end:break
        candidates=[]; rate_limited=False
        save(root/'activity.json',{'phase':'discovery','difficulty':difficulty,'deadline':deadline,'updated':time.time()})
        # Hub supplies ranking, not a coordinator-maintained model list.
        try:
            discovered=gate.call(lambda: list(api.list_models(pipeline_tag='text-generation',sort='downloads',limit=30)))
        except ResearchLimit as exc:
            save(root/'review-needed.json',{'reason':'Hub research rate limit','error':str(exc)});return
        for model in discovered:
            if time.time()>=end:break
            if model.id in seen or getattr(model,'gated',False):continue
            try:
                save(root/'activity.json',{'phase':'model_metadata','model':model.id,'difficulty':difficulty,'selected':len(candidates),'deadline':deadline,'updated':time.time()})
                existing=catalog.get(model.id)
                info=gate.call(lambda: api.model_info(model.id,revision=existing.get('revision','main') if existing else 'main',files_metadata=True))
                if getattr(info,'gated',False) or getattr(info,'private',False):continue
                count=getattr(getattr(info,'safetensors',None),'total',None)
                if not count or count>ceiling or (difficulty!='basic' and count<=ceiling/4):continue
                if any(x in model.id.lower() for x in ('gguf','gptq','awq')):continue
                # Pure configuration preflight; no weights, no NPU or LLM.
                from tplmachine.research.improvement_model import contracts
                config=Path(gate.call(lambda: hf_hub_download(model.id,'config.json',revision=info.sha)))
                if config.stat().st_size>1024**2:continue
                tasks=contracts(json.loads(config.read_text()),model.id,[1])
                if tasks[0]['parameter_upper_estimate']>ceiling:continue
                if not existing:catalog.add_model({'id':model.id,'categories':['language-generation'],'revision':info.sha,
                    'priority':len(candidates),'difficulty':difficulty,'npu_status':'untested',
                    'source':'https://huggingface.co/'+model.id,'notes':'Hub downloads ranking; config projection research only'})
                candidates.append(model.id);seen.add(model.id)
                if len(candidates)>=30:break
            except ResearchLimit as exc:
                save(root/'review-needed.json',{'reason':'Hub research rate limit','error':str(exc)});rate_limited=True;break
            except Exception as exc:save(root/'discovery-error.json',{'model':model.id,'error':str(exc)[:500]})
        for offset in range(0,len(candidates),5):
            if time.time()>=end:break
            round_id+=1;batch=dict(plan,root=str(root/f'round-{round_id}'),models=candidates[offset:offset+5],task_timeout_s=min(1800,max(1,int(end-time.time()))))
            save(root/'activity.json',{'phase':'campaign','campaign_root':batch['root'],'models':batch['models'],'difficulty':difficulty,'round':round_id,'deadline':deadline,'updated':time.time()})
            start(batch);wait(Path(batch['root'])/'job')
            summary=json.loads((Path(batch['root'])/'summary.json').read_text())
            save(root/'progress.json',{'difficulty':difficulty,'round':round_id,'summary':summary,'deadline':deadline})
            # Do not pretend unknown kernel defects are automatically repairable.
            if summary['counts'].get('failed',0)>len(batch['models'])/2:
                save(root/'review-needed.json',{'round':round_id,'reason':'majority failure; harder tier withheld','root':batch['root']})
                return
        if rate_limited:return
    save(root/'finished.json',{'rounds':round_id,'finished':time.time(),'deadline':deadline})

def main():
    p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--deadline',required=True);a=p.parse_args();run(a.plan,a.deadline)
if __name__=='__main__':main()
