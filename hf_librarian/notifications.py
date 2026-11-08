"""Durable terminal-job events and optional desktop delivery, without job polling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def event(job):
    from .activity import snapshot
    from .campaign import save,lease
    job=Path(job).resolve();folder=job/'notifications';folder.mkdir(exist_ok=True)
    with lease(folder/'event.lock'):
        path=folder/'event.json'
        if path.exists():return json.loads(path.read_text())
        state=snapshot(job)
        if state['status']=='running':raise ValueError('Job is still running; no terminal event')
        kind='failed' if state['status']!='completed' else ('review_required' if state['phase']=='review_required' else 'completed')
        value={'schema':'hf-librarian.event.v1','id':hashlib.sha256(str(job).encode()).hexdigest(),
               'kind':kind,'job':str(job),'time':time.time(),'phase':state['phase'] or 'unknown',
               'counts':state.get('campaign',{}).get('counts',{}),
               'reason':state.get('reason') or state.get('error_excerpt','')[-1000:]}
        save(path,value)
        with (folder/'events.jsonl').open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        return value


def wait_event(job):
    # Existing Linux pidfd/inotify completion wait; no periodic model/status query.
    from tplmachine.research.jobs import wait
    try:wait(job)
    except Exception:
        # A supervisor crash without a receipt is still an incident worth recording.
        from .activity import snapshot
        if snapshot(job)['status']=='running':raise
    return event(job)


def mac_notify(value):
    title='TCLMachine: '+{'completed':'작업 완료','failed':'작업 오류','review_required':'검토 필요'}[value['kind']]
    body='상태: '+value['phase']+' · '+str(value.get('counts',{}))+'\n'+value['job']
    # AppleScript receives arguments as data; no model text is interpolated into code.
    script='on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv) sound name "Glass"\nend run'
    subprocess.run(['osascript','-e',script,title,body],check=True,timeout=15,capture_output=True,text=True)


def desktop(host,job,remote_python,state_root,remote_pythonpath=""):
    root=Path(state_root).expanduser();root.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256((host+'\0'+job).encode()).hexdigest();receipt=root/(key+'.json')
    if receipt.exists():return
    import shlex
    argv=[remote_python,'-m','hf_librarian.notifications','wait',job]
    if remote_pythonpath:argv=['env','PYTHONPATH='+remote_pythonpath,*argv]
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',host,shlex.join(argv)]
    while True:
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode==0:
            try:
                value=json.loads(result.stdout.strip().splitlines()[-1])
                if value.get('schema')!='hf-librarian.event.v1':raise ValueError('unexpected event schema')
                # Persist inbox first: notification permission errors must not lose the event.
                inbox=root/(key+'.event.json');inbox.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
                try:
                    mac_notify(value)
                    receipt.write_text(json.dumps({'event':value['id'],'delivery':'macOS request accepted','time':time.time()})+'\n')
                except Exception as exc:
                    (root/(key+'.delivery-error.json')).write_text(json.dumps({'error':str(exc)[:500],'inbox':str(inbox)})+'\n')
                return
            except (ValueError,IndexError) as exc:
                (root/(key+'.connection-error.json')).write_text(json.dumps({'error':str(exc)[:500]})+'\n')
        else:
            (root/(key+'.connection-error.json')).write_text(json.dumps({'returncode':result.returncode,'stderr':result.stderr[-1000:]})+'\n')
        # Connection recovery only; never poll server job status.
        time.sleep(60)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for name in ('wait','record'):
        q=s.add_parser(name);q.add_argument('job')
    q=s.add_parser('desktop');q.add_argument('host');q.add_argument('job');q.add_argument('--remote-python',default='/root/miniconda3/bin/python3');q.add_argument('--state-root',default='~/.local/state/hf-librarian/notifications');q.add_argument('--remote-pythonpath',default='')
    a=p.parse_args()
    if a.action=='desktop':desktop(a.host,a.job,a.remote_python,a.state_root,a.remote_pythonpath)
    else:print(json.dumps(wait_event(a.job) if a.action=='wait' else event(a.job),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
