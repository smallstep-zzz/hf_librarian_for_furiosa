"""Useful queued improvement work; existing catalog only, no autonomous model research.

Bounded CPU campaigns, a single shared NPU lease, cumulative teacher-only fitting.
No LLM calls, no rerun of an unchanged attempted contract, no hidden kernel repair.
"""
import argparse,ctypes,json,os,time,threading,shutil
from pathlib import Path
from .campaign import start,save,lease,command,environment,fingerprint
from .catalog import Catalog
from .library import Librarian


def row_families():
    # Fill realistic non-power-of-two batch/prefill gaps, then wait for new intake.
    return [[min(4096,base*2**i) for i in range(9)] for base in (3,5,7,9,11,13,15,17)]


def wait_for_input(root,catalog,timeout=300):
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0:raise OSError('inotify unavailable')
    try:
        for directory in (Path(root),Path(catalog).resolve().parent):
            if libc.inotify_add_watch(fd,os.fsencode(directory),0x100|0x80|0x8)<0:raise OSError('watch failed')
        import select
        if select.select([fd],[],[],timeout)[0]:os.read(fd,65536)
    finally:os.close(fd)


def aggregate(plan,root,state):
    from tplmachine.improvement import register_compiler_prior
    roots=[Path(p) for p in plan.get('history_roots',[])]+[root/'waves']
    paths=sorted({str(p) for base in roots for p in base.rglob('sample_index.json')
                  if p.parent.name=='collection'})
    keys=set()
    for path in paths:
        keys.update(x['key'] for x in json.loads(Path(path).read_text()))
    identity=fingerprint(sorted(keys))
    if not keys or identity==state.get('trained_keys'):return
    output=root/'training'/identity
    save(root/'training-indexes.json',paths)
    with lease(plan['feedback_lock']):
        if (output/'result.json').is_file():
            prior=json.loads((output/'result.json').read_text());result={'receipt':{'returncode':0 if prior['status']=='passed' else 1},'result':prior,'result_path':str(output/'result.json')}
        else:result=command(dict(plan,root=str(root)),['train','--indexes',str(root/'training-indexes.json'),
                    '--data-root',plan['data_root'],'--output',str(output)],output)
        if result['receipt']['returncode']==0:
            result['registry']=register_compiler_prior(output/'models/compiler-estimate',plan['model_registry'],
                                                       result['result'].get('models',[]))
        state['trained_keys']=identity;state['last_training']=result
        save(root/'state.json',state)


def run(plan_path):
    from tplmachine.research.jobs import wait
    from tplmachine.research.improvement_model import contracts
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['service_root']);root.mkdir(parents=True,exist_ok=True)
    os.environ.update(environment(plan))
    state_path=root/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'attempted':{},'wave':0}
    def forward_stop():
        while not (root/'finished.json').exists():
            if (root/'STOP.json').exists():
                current=json.loads(state_path.read_text()) if state_path.exists() else {}
                if current.get('active'):save(Path(current['active'])/'STOP.json',{'stop':True})
                return
            wait_for_input(root,plan['catalog'])
    with lease(root/'manager.lock'):
        threading.Thread(target=forward_stop,daemon=True).start()
        if state.get('active'):
            wait(Path(state['active'])/'job');state.pop('active');save(state_path,state)
        aggregate(plan,root,state)
        while not (root/'STOP.json').exists():
            disk=shutil.disk_usage(plan['data_root'])
            if disk.used/disk.total>=.90 or disk.free<2*1024**3:
                save(root/'review-needed.json',{'reason':'storage reserve reached; no cache deletion'});break
            catalog=Catalog(plan['catalog']);rules=fingerprint({str(p):p.read_text() for p in
                (Path(plan['tclmachine_source'])/'tplmachine/research/improvement_model.py',
                 Path(plan['tclmachine_source'])/'tplmachine/aten_kernels.py',
                 Path(plan['tclmachine_source'])/'tplmachine/research/campaign_worker.py')})
            chosen=[];keys=[];chosen_rows=None
            save(root/'activity.json',{'phase':'queue_planning','updated':time.time(),'wave':state['wave']})
            with Librarian(catalog,max_concurrent=2) as lib:
                # Catalog upkeep is delegated to its owner. No model IDs hardcoded here.
                models=sorted(catalog.models(),key=lambda m:(m.get('difficulty','medium') not in ('easy','basic'),m.get('priority',100),m['id']))
                for rows in row_families():
                    for model in models:
                        if (root/'STOP.json').exists():break
                        if model['id'] in catalog.blacklist() or model.get('quantization')=='gguf':continue
                        key=None
                        try:
                            snapshot=lib.request(model_id=model['id'],profile='config')
                            cfg=json.loads((Path(snapshot)/'config.json').read_text())
                            tasks=contracts(cfg,model['id'],rows)
                            signature=sorted({tuple(t['shape']) for t in tasks})
                            key=fingerprint({'shapes':signature,'rules':rules})
                            if key in state['attempted'] or key in keys:continue
                            chosen.append(model['id']);keys.append(key);chosen_rows=rows
                            if len(chosen)>=5:break
                        except Exception as exc:
                            save(root/'last-intake-error.json',{'model':model['id'],'error':str(exc)[:1000],'time':time.time()})
                    if chosen:break
            if not chosen:
                save(root/'activity.json',{'phase':'waiting_for_new_catalog_or_rules','reason':'all admitted unique row contracts attempted; no redundant NPU loops','updated':time.time(),'wave':state['wave']})
                wait_for_input(root,plan['catalog']);continue
            state['wave']+=1;wave=root/'waves'/str(state['wave'])
            batch=dict(plan,root=str(wave),models=chosen,row_buckets=chosen_rows)
            save(root/'activity.json',{'phase':'campaign','campaign_root':str(wave),'models':chosen,'rows':chosen_rows,'wave':state['wave'],'updated':time.time()})
            start(batch);state['active']=str(wave)
            for key in keys:state['attempted'][key]={'models':chosen,'wave':state['wave'],'status':'submitted'}
            save(state_path,state)
            wait(wave/'job');state.pop('active');save(state_path,state)
            # Failed cases remain receipts for FHA, not an implicit LLM/repair loop.
            summary=json.loads((wave/'summary.json').read_text())
            if summary['counts'].get('failed') or summary['counts'].get('blocked'):
                save(root/'review-latest.json',{'wave':state['wave'],'counts':summary['counts'],'root':str(wave)})
            aggregate(plan,root,state)
        save(root/'finished.json',{'status':'stopped','time':time.time(),'waves':state['wave']})

def main():
    p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();run(a.plan)
if __name__=='__main__':main()
