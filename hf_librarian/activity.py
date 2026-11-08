"""One-shot job/phase diagnostics; no monitor, no model execution, bounded output."""
import json,os,sqlite3,time
from pathlib import Path

def read(path):
    return json.loads(path.read_text()) if path.is_file() else None

def snapshot(job):
    job=Path(job).resolve();request=read(job/'request.json')
    if not request:raise ValueError('Not a supervisor job directory: '+str(job))
    done=read(job/'result.json');running=read(job/'running.json');alive=False
    if running and running.get('host')==os.uname().nodename:
        try:alive=Path('/proc/'+str(running['supervisor_pid'])+'/stat').read_text().split()[21]==running['proc_start']
        except OSError:pass
    result={'job':str(job),'process_alive':alive,'status':done['status'] if done else ('running' if alive else 'process_missing'),
            'checked_once':True,'phase':None,'active':[]}
    argv=request['command']
    try:
        plan=read(Path(argv[argv.index('hf_librarian.overnight')+1]));root=Path(plan['root']+'-overnight')
    except (ValueError,IndexError,TypeError):root=job.parent
    activity=read(root/'activity.json') or {};result.update(activity)
    if activity.get('updated'):result['last_activity_age_s']=round(time.time()-activity['updated'])
    review=read(root/'review-needed.json');finished=read(root/'finished.json')
    if review:result.update(phase='review_required',reason=review['reason'],review=review)
    elif finished:result.update(phase='finished',completion=finished)
    campaign=activity.get('campaign_root')
    if campaign and (Path(campaign)/'campaign.sqlite').is_file():
        from .campaign import status
        state=status(campaign)
        result['campaign']={k:state[k] for k in ('status','counts','parallel','eta')}
        with sqlite3.connect('file:'+str(Path(campaign)/'campaign.sqlite')+'?mode=ro',uri=True) as db:
            rows=[json.loads(r[0]) for r in db.execute('SELECT record FROM items')]
        result['active']=[{'model':r['model'],'stage':r['state']} for r in rows if r['state'] in ('cpu','npu_queued','ready','pending')][:20]
        result['npu']=read(Path(campaign)/'npu-active.json') or {'state':'no_execution_recorded'}
        result['stage_legend']={'cpu':'TCL generation / official estimation / representative EDF compilation',
                                'npu_queued':'queued or executing; consult npu field','ready':'waiting for CPU/RAM admission','pending':'config preparation'}
        result['failures']=[{'model':r['model'],'stage':r.get('stage'),
            'error':r.get('error') or r.get('npu',{}).get('result',{}).get('first_error') or r.get('cpu',{}).get('result',{}).get('first_error')}
            for r in rows if r['state'] in ('failed','blocked')][:5]
    if done and done['status'] not in ('completed',):
        path=job/'worker.log'
        if path.exists():
            with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-1800));result['error_excerpt']=stream.read().decode(errors='replace')
        result['phase']='process_failed'
    result['next_action']='inspect error/review before restart' if result['status'] in ('failed','process_missing','crashed') or review else 'none; query again only when needed'
    return result
