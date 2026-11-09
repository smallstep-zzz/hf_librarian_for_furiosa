"""Server25: existing FXB, one model load, repeated real native generations."""
import json,os,signal,time,traceback,resource
from pathlib import Path
signal.signal(signal.SIGHUP,signal.SIG_IGN);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/root/fury-validation/reports');root.mkdir(parents=True,exist_ok=True)
r={'model':'Qwen/Qwen3-8B-FP8','status':'failed','generations':[],'evidence_scope':'native FXB backend and process/device mapping; no per-operator trace API exposed'}
llm=None;start=time.monotonic()
try:
 from furiosa_llm import LLM,SamplingParams
 from fxbx import hug
 from furiosa_llm.metadata.config_types import LoggerConfig
 snapshot=json.load(open(root/'librarian.json'))['snapshot']
 Wrapped=hug(LLM);r['wrapped_class']=Wrapped is not LLM
 llm=Wrapped(snapshot,fxb='/root/qwen3-8b-fp8.fxb',max_model_len=512,logger_config=LoggerConfig(log_stats=True))
 r['load_s']=time.monotonic()-start
 import subprocess
 r['device_processes']=subprocess.check_output(['furiosa-smi','ps'],text=True,timeout=10)
 for text,expected in [('Reply with only the number: what is 2+2?','4'),('Reply with exactly this word: READY','READY')]:
  begin=time.monotonic();values=llm.chat([{'role':'user','content':text}],sampling_params=SamplingParams(temperature=0,max_tokens=32),chat_template_kwargs={'enable_thinking':False})
  output=values[0].outputs[0].text;r['generations'].append({'prompt':text,'text':output,'expected_present':expected in output,'request_s':time.monotonic()-begin,'tokens':len(values[0].outputs[0].token_ids)})
 history=[{'role':'user','content':'My name is Minsoo. Remember it and reply OK.'}]
 for text in (None,'What is my name? Reply only my name.'):
  if text:history.append({'role':'user','content':text})
  begin=time.monotonic();values=llm.chat(history,sampling_params=SamplingParams(temperature=0,max_tokens=32),chat_template_kwargs={'enable_thinking':False})
  output=values[0].outputs[0].text;history.append({'role':'assistant','content':output});r['generations'].append({'history_turn':len(history)//2,'text':output,'request_s':time.monotonic()-begin,'tokens':len(values[0].outputs[0].token_ids)})
 r['history_passed']='minsoo' in output.lower();r['status']='passed' if all(x.get('expected_present',True) for x in r['generations']) and r['history_passed'] else 'quality_failed'
except BaseException as exc:r['error']=type(exc).__name__+': '+str(exc)[:1600];traceback.print_exc()
finally:
 if llm is not None:
  try:llm.shutdown()
  except Exception as exc:r['shutdown_error']=str(exc)[:300]
 r['total_s']=time.monotonic()-start;(root/'native-result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
