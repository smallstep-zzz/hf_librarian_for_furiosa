"""Opt-in after-wave competition; all CPU/NPU corpus children already drained."""
import os,subprocess,sys,time
from pathlib import Path
from .campaign import environment,save


def after_wave(plan,wave):
    root=Path(plan['portfolio_root']);root.mkdir(parents=True,exist_ok=True);results=[]
    for action,extra in [('discover',['--wave',str(wave)]),('drain',['--limit',str(plan.get('portfolio_jobs_per_wave',1))])]:
        command=[plan['sdk_python'],'-m','tplmachine.research.portfolio',action,'--root',str(root),*extra]
        with (Path(wave)/('portfolio-'+action+'.log')).open('a') as log:
            child=subprocess.Popen(command,env=environment(plan),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            from tplmachine.research.jobs import wait_process
            try:code=wait_process(child,3600)
            except TimeoutError:
                import signal
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
        results.append({'action':action,'returncode':code})
        if code:break
    summary=None
    if (root/'last-drain.json').exists():
        import json
        last=json.loads((root/'last-drain.json').read_text());summary={'statuses':[x['status'] for x in last['results']],'receipt':str(root/'last-drain.json')}
    receipt={'competition':summary,'wave':str(wave),'steps':results,'runtime_promotion':False,'time':time.time()}
    save(Path(wave)/'portfolio-receipt.json',receipt);return receipt
