"""Detached native test supervisor with Linux pidfd compatibility and crash receipt."""
import subprocess,json,os,select,signal,time,ctypes
from pathlib import Path
signal.signal(signal.SIGHUP,signal.SIG_IGN)
r=Path('/root/fury-validation/reports');started=time.time();p=None;result={'status':'failed'}

def pidfd(pid):
 if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
 libc=ctypes.CDLL(None,use_errno=True)
 if hasattr(libc,'pidfd_open'):fd=libc.pidfd_open(pid,0)
 elif os.uname().machine in ('x86_64','aarch64'):fd=libc.syscall(434,pid,0)
 else:raise OSError('pidfd unsupported on this host')
 if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
 return fd

try:
 with (r/'native.log').open('w') as log:
  p=subprocess.Popen(['/root/.venvs/fury-tools/bin/python','/root/fury-validation/native_test.py'],stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',RAYON_NUM_THREADS='8'))
  (r/'running.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':p.pid}))
  fd=pidfd(p.pid)
  try:
   if not select.select([fd],[],[],1800)[0]:os.killpg(p.pid,signal.SIGKILL);result['timeout']=True
  finally:os.close(fd)
  code=p.wait();result.update(returncode=code,status='completed' if code==0 else 'failed')
except BaseException as exc:
 result['error']=type(exc).__name__+': '+str(exc)[:500]
 if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
finally:
 result['wall_s']=time.time()-started;(r/'supervisor-result.json').write_text(json.dumps(result))
