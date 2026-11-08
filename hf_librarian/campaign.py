"""Detached, bounded TCLMachine campaigns. FHA only starts/queries/reviews on demand."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import re
import subprocess
import sys
import time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(temporary,path)


@contextmanager
def lease(path,stop_file=None):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    before=time.monotonic()
    with path.open('a') as handle:
        if stop_file is None:fcntl.flock(handle,fcntl.LOCK_EX)
        else:
            while True:
                if Path(stop_file).exists():raise InterruptedError('stop requested while awaiting resource')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.1) # server resource wait, never an LLM polling loop
        yield handle,time.monotonic()-before


class Ledger:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        with lease(self.root/'ledger-init.lock'):
            self.db=sqlite3.connect(self.root/'campaign.sqlite',timeout=30)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('CREATE TABLE IF NOT EXISTS items (model TEXT PRIMARY KEY, state TEXT, record TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS repairs (group_id TEXT, attempt INTEGER, record TEXT, PRIMARY KEY(group_id,attempt))')
            self.db.commit()

    def put(self,model,state,**record):
        record=dict(record,model=model,state=state,updated=time.time())
        with self.db:self.db.execute('INSERT OR REPLACE INTO items VALUES(?,?,?)',(model,state,json.dumps(record)))
        return record

    def rows(self):return [json.loads(row[0]) for row in self.db.execute('SELECT record FROM items ORDER BY model')]
    def close(self):self.db.close()


def environment(plan):
    return dict(os.environ,PYTHONPATH=os.pathsep.join([plan['tclmachine_source'],*plan.get('pythonpath',[]),
        str(Path(__file__).resolve().parents[1])]),RAYON_NUM_THREADS='5',OMP_NUM_THREADS='1',PYTHONFAULTHANDLER='1',
        TCLMACHINE_PRIOR_REGISTRY=str(plan.get('model_registry',Path.home()/'.local/share/tclmachine/compiler-priors')))


def start(plan):
    root=Path(plan['root']).resolve()
    if root.exists():raise ValueError('Use a new campaign root; prior results remain immutable')
    for key in ('sdk_python','tclmachine_source','catalog'):
        if not Path(plan[key]).is_absolute() or not Path(plan[key]).exists():raise ValueError('Missing absolute '+key)
    from .catalog import Catalog
    catalog=Catalog(plan['catalog'])
    registry=Path(plan.get('registry',Path.home()/'.local/share/hf-librarian/campaigns'))
    models=plan.get('models',[])
    if not models:
        categories=plan.get('categories',[plan['category']] if plan.get('category') else [])
        if not categories:raise ValueError('Supply exact models or categories')
        seen=Counter()
        if registry.exists():
            for marker in registry.glob('*.json'):
                previous=Path(json.loads(marker.read_text())['root'])/'plan.json'
                if previous.exists():seen.update(json.loads(previous.read_text()).get('models',[]))
        candidates=[m for m in catalog.models() if set(m['categories'])&set(categories)
            and m['id'] not in catalog.blacklist() and m.get('quantization')!='gguf'
            and not all('.gguf' in pattern for pattern in m.get('include') or ['config.json'])]
        models=[m['id'] for m in sorted(candidates,key=lambda m:(seen[m['id']],m.get('priority',100),m['id']))[:plan.get('model_limit',10)]]
    if not 1<=len(models)<=100 or len(set(models))!=len(models):raise ValueError('Use1..100 distinct models')
    if not 2<=plan.get('max_parallel',5)<=5:raise ValueError('Parallel ceiling must be2..5')
    for key in ('data_root','npu_lock','feedback_lock'):
        if not Path(plan[key]).is_absolute():raise ValueError('Absolute canonical '+key+' required')
    version=fingerprint({name:hashlib.sha256((Path(plan['tclmachine_source'])/name).read_bytes()).hexdigest()
        for name in ('tplmachine/research/improvement_model.py','tplmachine/research/campaign_worker.py',
                     'tplmachine/improvement.py','tplmachine/cost_provider.py','tplmachine/official_tools.py')})
    plan=dict(plan,root=str(root),models=models,source_version=version,started=time.time(),max_parallel=plan.get('max_parallel',5))
    if plan.get('npu',True):
        check=subprocess.run([plan['sdk_python'],'-c','import json; from fxbex.settings import lock_path; print(json.dumps(str(lock_path())))'],
            env=environment(plan),capture_output=True,text=True,timeout=30,check=True)
        canonical=Path(json.loads(check.stdout.strip().splitlines()[-1])).resolve()
        if Path(plan['npu_lock']).resolve()!=canonical:
            raise ValueError('NPU lock must match the existing canonical SDK/FXBEX lease: '+str(canonical))
    root.mkdir(parents=True);save(root/'plan.json',plan)
    ledger=Ledger(root)
    for model in models:ledger.put(model,'pending')
    ledger.close()
    for model in models:
        if not catalog.get(model) or model in catalog.blacklist():raise ValueError('Missing/blocked model: '+model)
    os.environ.update(environment(plan))
    sys.path[:0]=environment(plan)['PYTHONPATH'].split(os.pathsep)
    from tplmachine.research.jobs import launch
    result=launch(root/'job',[sys.executable,'-m','hf_librarian.campaign','run',str(root)])
    save(root/'launch.json',result)
    registry.mkdir(parents=True,exist_ok=True);save(registry/(fingerprint(str(root))+'.json'),{'root':str(root),'started':plan['started']})
    return dict(result,status='started',monitoring='none; inspect on demand',root=str(root))


def command(plan,argv,output,*,fd=None):
    from tplmachine.research.watchdog import observe
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    env=environment(plan)
    env['TPL_COLLECTION_STOP_FILE']=str(Path(plan['root'])/'STOP.json')
    if fd is not None:
        env.update(FXBEX_NPU_LOCK=str(plan['npu_lock']),FXBEX_NPU_LOCK_FD=str(fd.fileno()))
    receipt=observe([plan['sdk_python'],'-m','tplmachine.research.campaign_worker',*argv],
        output.with_name(output.name+'.worker.log'),timeout=plan.get('task_timeout_s',1800),env=env,
        pass_fds=() if fd is None else (fd.fileno(),))
    result_path=output/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'failed','stage':'process',
        'first_error':receipt.get('log_tail','worker exited without a result')[-1500:]}
    return {'receipt':receipt,'result':result,'result_path':str(result_path)}


def cpu_task(plan,model,snapshot):
    output=Path(plan['root'])/'models'/fingerprint(model)[:16]/'cpu'
    before=time.monotonic()
    argv=['collect','--config',str(Path(snapshot)/'config.json'),'--model',model,
        '--data-root',plan['data_root'],'--output',str(output),'--workers',str(plan.get('compile_workers',4))]
    if plan.get('row_buckets'):argv+=['--rows',*[str(v) for v in plan['row_buckets']]]
    result=command(plan,argv,output)
    return dict(result,cpu_s=time.monotonic()-before,snapshot=snapshot,
                revision=Path(snapshot).name,config_sha256=hashlib.sha256((Path(snapshot)/'config.json').read_bytes()).hexdigest())


def npu_task(plan,model,prepared,queued_at=None):
    queue_wait=max(0,time.monotonic()-queued_at) if queued_at else 0
    output=Path(plan['root'])/'models'/fingerprint(model)[:16]/'npu'
    with lease(plan['npu_lock'],Path(plan['root'])/'STOP.json') as (fd,wait_s):
        owners=npu_owners()
        if owners:
            return {'receipt':{'returncode':75},'result':{'status':'blocked','stage':'resource_busy','owners':owners},
                    'npu_wait_s':wait_s,'npu_queue_wait_s':queue_wait,'npu_active_s':0,'performance_label':False}
        before=time.monotonic()
        save(Path(plan['root'])/'npu-active.json',{'state':'active','model':model,'started':time.time()})
        try:
            result=command(plan,['execute','--prepared',str(prepared),'--output',str(output)],output,fd=fd)
            active_s=time.monotonic()-before
        finally:save(Path(plan['root'])/'npu-active.json',{'state':'idle','finished':time.time()})
    with lease(plan['feedback_lock']) as (_,feedback_wait):
        record=command(plan,['record','--audit',result['result_path'],'--data-root',plan['data_root'],
                             '--output',str(output/'feedback')],output/'feedback')
    return dict(result,npu_wait_s=wait_s,npu_queue_wait_s=queue_wait,npu_active_s=active_s,feedback_wait_s=feedback_wait,
                feedback=record,performance_label=False)


def npu_owners():
    text=subprocess.check_output(['furiosa-smi','ps'],text=True,timeout=10)
    return [int(pid) for pid in re.findall(r'^\|\s*(\d+)\s*\|',text,re.MULTILINE)]


def memory_slots(plan):
    from tplmachine.research.watchdog import memory_budget
    budget=memory_budget()
    available=budget['available_bytes']
    if available is None:return 1
    return max(1,min(plan['max_parallel'],int((available-budget['reserve_bytes'])//(plan.get('worker_gib',6)*1024**3))))


def run(root):
    from .library import Librarian
    from .catalog import Catalog
    root=Path(root);plan=json.loads((root/'plan.json').read_text());ledger=Ledger(root)
    env=environment(plan);sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
    os.environ.update(env)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    rows={r['model']:r for r in ledger.rows()};pending=deque(plan['models']);success=0;limit=2;peak=0
    futures={};ready=deque();history=[]
    try:
        with Librarian(Catalog(plan['catalog']),max_concurrent=2) as library,ThreadPoolExecutor(max_workers=5) as cpu,ThreadPoolExecutor(max_workers=1) as npu:
            downloads={library.request_async(model_id=m,profile='config'):m for m in pending};pending.clear()
            futures.update({f:('prepare',m) for f,m in downloads.items()})
            while futures or ready:
                if (root/'STOP.json').exists():
                    ready.clear()
                    # Already submitted children complete and retain evidence; no new admission.
                    for model,row in list(rows.items()):
                        if row['state'] in ('pending','ready'):rows[model]=ledger.put(model,'cancelled')
                allowed=min(limit,memory_slots(plan));running=sum(kind=='cpu' for kind,_ in futures.values())
                while ready and running<allowed and not (root/'STOP.json').exists():
                    model,snapshot=ready.popleft();rows[model]=ledger.put(model,'cpu',started=time.time())
                    futures[cpu.submit(cpu_task,plan,model,snapshot)]=('cpu',model);running+=1;peak=max(peak,running)
                save(root/'progress.json',{'phase':'working','parallel_limit':allowed,'peak_parallel':peak,
                    'completed_cpu':success,'last_update':time.time(),'npu_tasks_pending_or_running':sum(k=='npu' for k,_ in futures.values()),
                    'npu_parallel_limit':1,
                    'ready':len(ready),'autoscale':history})
                if not futures:break
                done,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    kind,model=futures.pop(future)
                    try:
                        value=future.result()
                        if kind=='prepare':
                            config=Path(value)/'config.json';intake=root/'intake'/fingerprint(model)[:16]
                            intake.mkdir(parents=True,exist_ok=True)
                            (intake/'config.json').write_bytes(config.read_bytes())
                            save(intake/'source.json',{'model':model,'revision':Path(value).name,
                                'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'profile':'config'})
                            rows[model]=ledger.put(model,'ready',snapshot=value,revision=Path(value).name)
                            ready.append((model,value))
                        elif kind=='cpu':
                            if value['receipt']['returncode']!=0 or value['result'].get('status')=='failed':
                                rows[model]=ledger.put(model,'failed',stage='cpu',cpu=value);continue
                            success+=1
                            proposed=3 if success>=2 else 2
                            if success>=5:proposed=5
                            if proposed>limit:
                                limit=min(proposed,plan['max_parallel']);history.append({'successful_cpu':success,'limit':limit,'time':time.time()})
                            prepared=value['result'].get('prepared_probe')
                            if (root/'STOP.json').exists():rows[model]=ledger.put(model,'cancelled',cpu=value)
                            elif plan.get('npu',True) and prepared:
                                rows[model]=ledger.put(model,'npu_queued',cpu=value)
                                futures[npu.submit(npu_task,plan,model,prepared,time.monotonic())]=('npu',model)
                            else:rows[model]=ledger.put(model,'completed',cpu=value,npu='not_requested')
                        else:
                            previous=rows[model].get('cpu')
                            state=('blocked' if value['result'].get('status')=='blocked' else
                                ('completed' if value['receipt']['returncode']==0 and value['result'].get('status')=='passed' else 'failed'))
                            rows[model]=ledger.put(model,state,cpu=previous,npu=value)
                    except Exception as exc:
                        rows[model]=ledger.put(model,'cancelled' if isinstance(exc,InterruptedError) else 'failed',stage=kind,error=type(exc).__name__+': '+str(exc)[:1200])
        # A single serialized commit/fit stage; estimates never replace a runtime decision.
        indexes=[]
        for row in rows.values():
            path=root/'models'/fingerprint(row['model'])[:16]/'cpu/collection/sample_index.json'
            if path.exists():indexes.append(str(path))
        save(root/'training-indexes.json',indexes)
        training=None
        if indexes and not (root/'STOP.json').exists():
            with lease(plan['feedback_lock']) as (_,wait_s):
                output=root/'training'
                training=command(plan,['train','--indexes',str(root/'training-indexes.json'),
                    '--data-root',plan['data_root'],'--output',str(output)],output)
                training['feedback_wait_s']=wait_s
                if training['receipt']['returncode']==0 and training['result'].get('proposal_status')=='proposal_better_than_constant':
                    from tplmachine.improvement import register_compiler_prior
                    training['registry']=register_compiler_prior(output/'models/compiler-estimate',
                        plan.get('model_registry',Path.home()/'.local/share/tclmachine/compiler-priors'),
                        training['result'].get('models',[]))
        counts=dict(Counter(row['state'] for row in rows.values()))
        train_failed=bool(training and training['receipt']['returncode']!=0)
        save(root/'summary.json',{'status':'stopped' if (root/'STOP.json').exists() else ('completed_with_issues' if counts.get('failed') or counts.get('blocked') or train_failed else 'completed'),
             'counts':counts,'models':len(rows),'peak_parallel':peak,'autoscale':history,'training':training,
             'llm_calls':0,'model_weights_updated':False,'scope':'configuration projection cost corpus and optional representative mm NPU probes',
             'finished':time.time()})
        return 0
    except BaseException as exc:
        save(root/'process_failure.json',{'status':'failed','error':type(exc).__name__,'message':str(exc)[:1500],'time':time.time()})
        raise
    finally:ledger.close()


def estimate(rows,parallel,now=None,npu_enabled=True):
    now=now or time.time()
    cpu=[r['cpu']['cpu_s'] for r in rows if isinstance(r.get('cpu'),dict)]
    npu=[r['npu']['npu_active_s'] for r in rows if isinstance(r.get('npu'),dict)]
    cpu_mean=sum(cpu)/len(cpu) if cpu else 30.
    npu_mean=sum(npu)/len(npu) if npu else 10.
    unfinished=[r for r in rows if r['state'] not in ('completed','failed','cancelled')]
    cpu_left=sum(max(0,cpu_mean-(now-r.get('started',now))) if r['state']=='cpu' else cpu_mean
                 for r in unfinished if r['state']!='npu_queued')/max(1,parallel)
    npu_left=len(unfinished)*npu_mean if npu_enabled else 0
    eta=max(cpu_left,npu_left)+3 if unfinished else 0
    return {'eta_s':round(eta),'range_s':[round(eta*.5),round(eta*2)],'cpu_mean_s':cpu_mean,
        'npu_active_mean_s':npu_mean,'basis':'remaining count × observed phase mean / admitted CPU slots; serialized NPU; waits excluded',
        'seed_estimate':not cpu or (npu_enabled and not npu)}


def status(root):
    root=Path(root)
    if not (root/'campaign.sqlite').is_file():raise FileNotFoundError('No campaign ledger at '+str(root))
    ledger=Ledger(root)
    try:rows=ledger.rows()
    finally:ledger.close()
    progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
    summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else None
    plan=json.loads((root/'plan.json').read_text())
    alive=False;reason=None
    try:
        meta=json.loads((root/'job/running.json').read_text());pid=meta['supervisor_pid']
        alive=os.uname().nodename==meta['host'] and Path('/proc/'+str(pid)+'/stat').read_text().split()[21]==meta['proc_start']
    except (OSError,KeyError,ValueError):reason='supervisor not observed; check process_failure.json and job receipt'
    state=summary['status'] if summary else ('running' if alive else 'process_missing')
    if summary:reason=None
    return {'status':state,'process_alive':alive,'process_note':reason,'counts':dict(Counter(r['state'] for r in rows)),
        'parallel':progress.get('parallel_limit',2),'eta':estimate(rows,progress.get('parallel_limit',2),npu_enabled=plan.get('npu',True)),
        'summary':summary,'root':str(root),'checked_once':True}


def history(root,model=None):
    root=Path(root)
    if not (root/'campaign.sqlite').exists():
        entries=sorted(root.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:100]
        return {'campaigns':[history(json.loads(p.read_text())['root'],model) for p in entries],
                'limit':100,'note':'per-model configuration coverage, not fine-tuning of HF model weights'}
    ledger=Ledger(root)
    try:rows=ledger.rows()
    finally:ledger.close()
    out=[]
    for row in rows:
        if model and row['model']!=model:continue
        cpu=row.get('cpu',{});audit=row.get('npu',{})
        out.append({'model':row['model'],'state':row['state'],'revision':cpu.get('revision',row.get('revision')),
            'config_sha256':cpu.get('config_sha256'),'cpu_s':cpu.get('cpu_s'),
            'npu_status':audit.get('result',{}).get('status') if isinstance(audit,dict) else audit,
            'npu_wait_s':audit.get('npu_wait_s') if isinstance(audit,dict) else None,
            'cpu_result':cpu.get('result_path'),'npu_result':audit.get('result_path') if isinstance(audit,dict) else None,
            'coverage':'configuration-derived projection corpus; representative initialized mm only'})
    summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
    training=summary.get('training') or {}
    plan=json.loads((root/'plan.json').read_text())
    return {'root':str(root),'models':out,'objective_score':training.get('result'),'source_version':plan.get('source_version'),
            'model_weights_updated':False,'runtime_performance_improvement':'not established by compiler surrogate'}


def issues(root,group=None,detail='summary'):
    root=Path(root)
    # Prefer canonical TCLMachine workspace report, then individual failed workers.
    plan=json.loads((root/'plan.json').read_text())
    source=Path(plan['data_root'])/'improvement.sqlite'
    sys.path[:0]=environment(plan)['PYTHONPATH'].split(os.pathsep)
    from tplmachine.improvement import Workspace
    workspace=Workspace(plan['data_root'])
    try:report=workspace.report(root/'review')
    finally:workspace.close()
    if group is None:
        for item in report['groups']:
            item['automatic_handling']='retain baseline; collect diverse data' if item['cause']=='surrogate_shape_generalization_gap' else None
            item['review_owner']=None if item['automatic_handling'] else 'FHA'
        return {'unique_issue_groups':report['unique_issue_groups'],'issue_occurrences':report['issue_occurrences'],
                'groups':report['groups'][:5],'scope':'shared workspace issue history',
                'all_groups':str(root/'review/issue_index.json')}
    entry=next((g for g in json.loads((root/'review/issue_index.json').read_text()) if g['group_id']==group),None)
    if entry is None:raise KeyError('Unknown issue group')
    packet=json.loads(Path(entry['packet']).read_text())
    topic={'workload_generation':'projection_rules','compile':'tcl_compiler','numerical':'precision',
           'npu_execution':'runtime_binding','trace_evidence':'execution_evidence',
           'surrogate_validation':'prediction_training','resource_admission':'memory_admission'}.get(packet['stage'],'orchestration')
    entry['knowledge_topic']=topic
    packet['knowledge_topic']=topic
    if detail=='summary':return entry
    if detail=='repro':return {k:packet[k] for k in ('group_id','occurrences','facts','representatives','hypothesis')}
    return packet


def repair(root,group,action,record=None):
    ledger=Ledger(root)
    try:
        with lease(Path(root)/'repair.lock'):
            old=[json.loads(r[0]) for r in ledger.db.execute('SELECT record FROM repairs WHERE group_id=? ORDER BY attempt',(group,))]
            if action=='history':return {'group_id':group,'attempts':old,'remaining':max(0,3-len(old))}
            if not record or not all(record.get(k) for k in ('hypothesis','action')):raise ValueError('hypothesis and action are required')
            if action=='begin':
                if len(old)>=3:raise ValueError('Three attempts exhausted; aggregate issues and ask user for a new plan')
                if old and old[-1]['status']=='in_progress':raise ValueError('Record validation before another repair')
                if old and old[-1]['status']=='verified':raise ValueError('Already resolved; use the new evidence group for a new problem')
                value=dict(record,attempt=len(old)+1,status='in_progress',time=time.time())
                with ledger.db:ledger.db.execute('INSERT INTO repairs VALUES(?,?,?)',(group,len(old)+1,json.dumps(value)))
            else:
                if not old or old[-1]['status']!='in_progress':raise ValueError('No active repair')
                if record.get('validation_status') not in ('passed','failed') or not record.get('evidence_path'):
                    raise ValueError('Actual validation status and evidence path required')
                if record['validation_status']=='passed' and not Path(record['evidence_path']).is_file():
                    raise ValueError('Successful validation needs an existing evidence file')
                value=dict(old[-1],validation_record=record,status='verified' if record['validation_status']=='passed' else 'failed')
                with ledger.db:ledger.db.execute('UPDATE repairs SET record=? WHERE group_id=? AND attempt=?',(json.dumps(value),group,len(old)))
            return {'group_id':group,'latest':value,'requires_user_decision':value['status']=='failed' and len(old)>=3}
    finally:ledger.close()


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','status','issues','history','repair','stop'])
    p.add_argument('root');p.add_argument('--group');p.add_argument('--detail',choices=['summary','repro','evidence'],default='summary')
    p.add_argument('--repair-action',choices=['begin','record','history'],default='history');p.add_argument('--record')
    a=p.parse_args()
    if a.action=='start':result=start(json.loads(Path(a.root).read_text()))
    elif a.action=='run':return run(a.root)
    elif a.action=='status':result=status(a.root)
    elif a.action=='issues':result=issues(a.root,a.group,a.detail)
    elif a.action=='history':result=history(a.root)
    elif a.action=='stop':save(Path(a.root)/'STOP.json',{'stop':True});result={'stop_requested':True}
    else:result=repair(a.root,a.group,a.repair_action,json.loads(Path(a.record).read_text()) if a.record else None)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
