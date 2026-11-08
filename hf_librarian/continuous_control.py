"""Wave-boundary pause, checkpoint receipt, and detached resume."""
import json,os,time,sys
from pathlib import Path
from .campaign import save,environment


def root_for(plan_path):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    return plan_path,plan,Path(plan['service_root'])


def checkpoint(root,plan_path,status='paused'):
    root=Path(root);state=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {}
    active=state.get('active')
    if active and (Path(active)/'job/result.json').exists():active=None
    receipt={'status':status,'time':time.time(),'plan':str(Path(plan_path).resolve()),
             'completed_wave':state.get('wave',0),'active':active,'next_wave':state.get('wave',0)+1,
             'attempted_contracts':len(state.get('attempted',{})),
             'state':str(root/'state.json'),'resume_command':'hf-librarian continuous resume '+str(Path(plan_path).resolve()),
             'resume_policy':'reuse attempted-contract fingerprints and existing receipts; no data deletion'}
    wave=root/'waves'/str(state.get('wave',0))
    for filename,key in (('summary.json','last_wave_counts'),('job/result.json','last_wave_job')):
        p=wave/filename
        if p.exists():
            value=json.loads(p.read_text());receipt[key]=value.get('counts',{}) if key=='last_wave_counts' else {k:value.get(k) for k in ('status','returncode')}
    receipt['completed_wave_means']='finished processing; failed/blocked cases are not successful models'
    save(root/'pause-receipt.json',receipt);return receipt


def control(action,plan_path):
    path,plan,root=root_for(plan_path);root.mkdir(parents=True,exist_ok=True)
    if action=='pause':
        save(root/'PAUSE_AFTER_WAVE.json',{'requested':time.time(),'plan':str(path)})
        return {'status':'pause_requested','boundary':'after active wave and feedback; not immediate stop',
                'receipt':str(root/'pause-receipt.json'),'resume_command':'hf-librarian continuous resume '+str(path)}
    if action=='status':
        result={}
        for name in ('activity.json','pause-receipt.json','finished.json'):
            p=root/name
            if p.exists():result[name]=json.loads(p.read_text())
        result['pause_requested']=(root/'PAUSE_AFTER_WAVE.json').exists()
        return result
    if action!='resume':raise ValueError('unknown continuous action')
    import fcntl
    with (root/'manager.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('manager is still active/draining; wait for paused receipt')
        state=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {}
        if state.get('active'):
            job=Path(state['active'])/'job'
            if not (job/'result.json').exists():raise ValueError('active wave has not completed')
        for name in ('PAUSE_AFTER_WAVE.json','STOP.json','finished.json','pause-receipt.json'):
            p=root/name
            if p.exists():p.rename(root/(name+'.resumed-'+str(time.time_ns())))
        os.environ.update(environment(plan))
        # Existing completion supervisor survives SSH disconnects.
        from tplmachine.research.jobs import launch
        return launch(root/'resume-jobs'/str(time.time_ns()),
                      [sys.executable,'-m','hf_librarian.continuous',str(path)])
