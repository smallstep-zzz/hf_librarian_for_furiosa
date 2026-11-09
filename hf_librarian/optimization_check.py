"""Librarian-owned bounded optimization qualification; kernel waits, no LLM polling."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .campaign import save,environment


def start(plan_path,source,output,limit=6):
    import sys
    if not 1<=limit<=12:raise ValueError('limit must be 1..12')
    if Path(output).exists():raise FileExistsError(output)
    from .continuous_control import control
    from tplmachine.research.jobs import launch
    plan=json.loads(Path(plan_path).read_text());os.environ.update(environment(dict(plan,tclmachine_source=source)))
    control('pause',plan_path)
    return launch(str(output)+'-job',[sys.executable,'-m','hf_librarian.optimization_check','--plan',str(Path(plan_path).resolve()),'--source',str(Path(source).resolve()),'--output',str(Path(output).resolve()),'--limit',str(limit)])


def run(plan_path,source,output,limit=6):
    plan=json.loads(Path(plan_path).read_text());root=Path(output);root.mkdir(parents=True,exist_ok=False)
    service=Path(plan['service_root']);save(root/'activity.json',{'phase':'waiting_for_wave_boundary','time':time.time()})
    # A paused manager releases this lock after all active wave work and feedback.
    with (service/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        finished=json.loads((service/'finished.json').read_text());state=json.loads((service/'state.json').read_text())
        if finished.get('status')!='paused' or state.get('active'):raise RuntimeError('service must reach a drained pause')
        wave=service/'waves'/str(state['wave'])
    env=environment(dict(plan,tclmachine_source=source))
    if plan.get('optimizer_safety_evidence'):env['TCL_OPTIMIZER_SAFETY_EVIDENCE']=str(plan['optimizer_safety_evidence'])
    save(root/'activity.json',{'phase':'hardware_comparison','wave':str(wave),'time':time.time()})
    from tplmachine.research.jobs import wait_process
    with (root/'comparison.log').open('w') as log:
        child=subprocess.Popen([plan['sdk_python'],'-m','tplmachine.research.adaptive_validation','--wave',str(wave),'--service',str(service),'--output',str(root/'comparison'),'--limit',str(limit)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=wait_process(child,3600)
        except TimeoutError:
            import signal
            os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
    comparison=root/'comparison/result.json'
    result={'status':'completed' if code==0 else 'failed','returncode':code,'comparison':str(comparison),'service_remains_paused':True}
    if code==0 and comparison.is_file():
        report=json.loads(comparison.read_text())
        if report.get('cases'):
            save(Path(plan['portfolio_root'])/'adaptive-qualification.json',report)
            result['adopted']=sum(c.get('status')=='adopt' for c in report['cases'])
    save(root/'result.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=6);a=p.parse_args();print(json.dumps(run(a.plan,a.source,a.output,a.limit)))
if __name__=='__main__':main()
