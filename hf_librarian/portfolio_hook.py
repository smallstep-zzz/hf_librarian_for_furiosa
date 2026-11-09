"""Opt-in after-wave competition; all CPU/NPU corpus children already drained."""
import os,subprocess,sys,time
from pathlib import Path
from .campaign import environment,save


def after_wave(plan,wave,discover=True):
    if plan.get('service_root') and (Path(plan['service_root'])/'PAUSE_AFTER_WAVE.json').exists():
        return {'status':'skipped_for_pause','runtime_promotion':False}
    root=Path(plan['portfolio_root']);root.mkdir(parents=True,exist_ok=True);results=[];started=time.time()
    actions=([('discover',['--wave',str(wave)])] if discover else [])+[('drain',['--limit',str(plan.get('portfolio_jobs_per_wave',1))])]
    for action,extra in actions:
        if plan.get('service_root'):
            save(Path(plan['service_root'])/'activity.json',{'phase':'portfolio_'+action,'queue_root':str(root),'wave_root':str(wave),'updated':time.time()})
        command=[plan['sdk_python'],'-m','tplmachine.research.portfolio',action,'--root',str(root),*extra]
        with (Path(wave)/('portfolio-'+action+'.log')).open('a') as log:
            child=subprocess.Popen(command,env=dict(environment(plan),TCL_PORTFOLIO_ISOLATED='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            from tplmachine.research.jobs import wait_process
            try:code=wait_process(child,3600)
            except TimeoutError:
                import signal
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
        results.append({'action':action,'returncode':code})
        if code:break
    summary=None
    if results and results[-1]['action']=='drain' and results[-1]['returncode']==0 and (root/'last-drain.json').exists() and (root/'last-drain.json').stat().st_mtime>=started:
        import json
        last=json.loads((root/'last-drain.json').read_text());summary={'statuses':[x['status'] for x in last['results']],'receipt':str(root/'last-drain.json')}
    pending=0
    if (root/'queue.sqlite').exists():
        import sqlite3
        db=sqlite3.connect((root/'queue.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:pending=db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0]
        finally:db.close()
    receipt={'remaining_pending':pending,'competition':summary,'wave':str(wave),'steps':results,'runtime_promotion':False,'time':time.time()}
    save(Path(wave)/'portfolio-receipt.json',receipt);return receipt
