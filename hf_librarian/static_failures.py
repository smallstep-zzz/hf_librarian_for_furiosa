"""Offline-cache static FXB campaign. Run with python -m hf_librarian.static_failures."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import datetime
import socket
from pathlib import Path
import subprocess
import sys
import time
import traceback

# One small checkpoint per distinct structure; IDs are fixture provenance, not compiler rules.
CASES = {
    'control': ('', 'linear/relu; harness control'),
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', 'encoder attention, layer norm, embedding'),
    'deit': ('facebook/deit-tiny-patch16-224', 'patch projection, vision attention, position embedding'),
    'resnet': ('microsoft/resnet-18', 'convolution, batch norm, residual, pooling'),
    'segformer': ('nvidia/segformer-b0-finetuned-ade-512-512', 'multiscale segmentation, resize, depthwise convolution'),
    'whisper': ('openai/whisper-tiny', 'audio Conv1d, encoder-decoder cross attention'),
    'lfm2': ('LiquidAI/LFM2-350M', 'short convolution and attention hybrid'),
    'qwen35': ('Qwen/Qwen3.5-0.8B', 'gated delta/recurrent and attention hybrid'),
}

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(model_id):
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    # Pinned librarian downloads can intentionally have no refs/main. Never guess among revisions.
    snapshots = Path(HF_HUB_CACHE)/('models--'+model_id.replace('/', '--'))/'snapshots'
    candidates = sorted(p for p in snapshots.glob('*') if p.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    return Path(snapshot_download(model_id, local_files_only=True))


def fixture(case):
    import torch
    from transformers import (AutoModel, AutoModelForImageClassification,
                              AutoModelForSemanticSegmentation, AutoModelForCausalLM,
                              WhisperForConditionalGeneration)
    torch.manual_seed(2718)
    model_id, features = CASES[case]
    source = snapshot(model_id) if model_id else None
    identity = {'model_id': model_id, 'features': features, 'snapshot': str(source),
                'revision': source.name if source else None, 'input_scope': 'synthetic deterministic smoke input; not task accuracy'}
    if source and (source/'config.json').exists():
        identity['config_sha256'] = digest(source/'config.json')
    opts = {'local_files_only': True, 'trust_remote_code': False}
    class Text(torch.nn.Module):
        def __init__(self, model, causal=False):
            super().__init__(); self.model=model; self.causal=causal
        def forward(self, ids, mask):
            if self.causal:
                return self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
            return self.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
    class Vision(torch.nn.Module):
        def __init__(self, model): super().__init__(); self.model=model
        def forward(self, pixels): return self.model(pixel_values=pixels, return_dict=False)[0]
    class Audio(torch.nn.Module):
        def __init__(self, model): super().__init__(); self.model=model
        def forward(self, features, decoder_ids):
            return self.model(input_features=features, decoder_input_ids=decoder_ids, use_cache=False, return_dict=False)[0]
    if case == 'control':
        model=torch.nn.Sequential(torch.nn.Linear(8,12),torch.nn.ReLU(),torch.nn.Linear(12,8))
        inputs=(torch.randn(4,8),)
    elif case in ('minilm','lfm2','qwen35'):
        cls=AutoModel if case=='minilm' else AutoModelForCausalLM
        loaded=cls.from_pretrained(source,attn_implementation='eager',**opts)
        # int32 IDs exercise the qualified static lookup boundary; preserve actual checkpoint config.
        model=Text(loaded,case!='minilm'); inputs=(torch.arange(16,dtype=torch.int32).reshape(1,16),torch.ones(1,16,dtype=torch.int32))
        identity['scope']='text-only fixed-length prefill; no cache/decode or full-context claim'
    elif case in ('deit','resnet','segformer'):
        cls=AutoModelForSemanticSegmentation if case=='segformer' else AutoModelForImageClassification
        loaded=cls.from_pretrained(source,**opts)
        size=64 if case=='segformer' else 224
        model=Vision(loaded); inputs=(torch.randn(1,3,size,size),)
        identity['scope']='single fixed resolution logits; segmentation deliberately 64x64 smoke geometry' if case=='segformer' else 'single 224x224 image logits'
    else:
        loaded=WhisperForConditionalGeneration.from_pretrained(source,attn_implementation='eager',**opts)
        model=Audio(loaded); inputs=(torch.zeros(1,loaded.config.num_mel_bins,loaded.config.max_source_positions*2),torch.tensor([[loaded.config.decoder_start_token_id]],dtype=torch.int32))
        identity['scope']='full encoder + one decoder token; zero mel fixture, no transcription quality claim'
    model=model.bfloat16().eval().requires_grad_(False)
    inputs=tuple(x.bfloat16().contiguous() if x.is_floating_point() else x.contiguous() for x in inputs)
    return model,inputs,identity


def failure_details(record, root):
    preflight=root/'model.fxb.plan/preflight.json'
    if preflight.exists():
        record['preflight']=json.loads(preflight.read_text())
    code=record.get('error_code') or record.get('preflight',{}).get('code')
    if code=='EXECUTION_RULE_GAP':
        record['diagnosis']={'observed_reason':'정적 실행 계획의 공유 EDF 규칙이 해당 연산/shape/dtype/layout을 수용하지 못함',
            'root_cause_status':'preflight guard confirmed; underlying SDK support unknown',
            'proposed_fix':'failure_groups별 최소 그래프를 분리하고 동일 shape/dtype의 공유 규칙을 추가·수치/trace 검증한다. 모델명 예외는 추가하지 않는다.',
            'next_evidence':['graph.json의 해당 node와 preflight source를 대조','최초 gap을 고친 뒤 재실행하여 뒤에 가려진 실패 수집','CPU 기준값, 각 step NPU trace, 새 프로세스 재적재 검증']}
    elif code=='LAYOUT_UNSUPPORTED':
        record['diagnosis']={'observed_reason':record.get('error'),
            'root_cause_status':'tensor binding layout guard confirmed; required packing implementation unconfirmed',
            'proposed_fix':'graph.json에서 해당 노드의 shape/stride/consumer를 확인하고 명시적인 NPU packing 또는 검증된 layout 전달 규칙을 추가한다. 호스트 contiguous로 모델 연산을 숨기지 않는다.',
            'next_evidence':['실패 텐서 shape/dtype/stride와 producer/consumer 연결','최소 비연속 텐서 재현','수정된 packing의 CPU 비교와 NPU trace']}
    else:
        record['diagnosis']={'observed_reason':record.get('error','증거 부족'), 'root_cause_status':'unconfirmed',
            'proposed_fix':'traceback의 최초 실패 경계를 최소 재현하고 입력/환경/계약을 확인한 후 수정한다.',
            'next_evidence':['traceback.txt와 stage events 확인','수정 전후 같은 revision/입력/SDK로 재현','compile success와 NPU execution을 별도로 확인']}
    key={'stage':record['stage'],'code':code,'error_type':record.get('error_type'),
         'groups':sorted((x['op'],x['reason']) for x in record.get('preflight',{}).get('failure_groups',[]))}
    record['signature']=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]


def worker(case, root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists(): raise FileExistsError('attempt already exists; use a new output directory')
    record={'schema':'hf-librarian.static-failure.v1','case':case,'status':'running','stage':'environment',
            'npu_execution':'not_run','cpu_fallback':'unknown','events':[], 'source_hashes':{},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'host':socket.gethostname(),'collector_sha256':digest(__file__)}
    started=time.monotonic()
    def stage(name):
        record['stage']=name
        record['events'].append({'stage':name,'elapsed_s':time.monotonic()-started})
        write(root/'result.json',record)
    try:
        import torch, tclmachine, fxbx
        from tplmachine.graph_capture import capture_for_export
        record['versions']={p:importlib.metadata.version(p) for p in ('torch','transformers','furiosa-torch','fxbx')}
        record['versions']['tclmachine']=tclmachine.__version__
        import tplmachine.execution_plan as owner
        for mod in (tclmachine,owner,fxbx):
            path=Path(mod.__file__);record['source_hashes'][str(path)]=digest(path)
        try:
            record['compiler_version']=subprocess.check_output(['furiosa-tcc','--version'],text=True,stderr=subprocess.STDOUT,timeout=10).strip()
        except (OSError,subprocess.SubprocessError) as exc:
            record['compiler_version']='unknown: '+type(exc).__name__
        tree=Path(owner.__file__).parent
        record['producer_tree_sha256']=hashlib.sha256(''.join(str(p.relative_to(tree))+digest(p) for p in sorted(tree.rglob('*.py'))).encode()).hexdigest()
        import tarfile
        with tarfile.open(root/'sources.tar.gz','w:gz') as archive:
            for folder in (tree,Path(tclmachine.__file__).parent,Path(fxbx.__file__).parent,Path(fxbx.__file__).parent.parent/'fxbex'):
                for file in sorted(folder.rglob('*.py')):
                    archive.add(file,arcname=str(file.relative_to(folder.parent)))
            archive.add(__file__,arcname='executed_collector.py')
        record['source_archive_sha256']=digest(root/'sources.tar.gz')
        torch.set_num_threads(2)
        stage('fixture')
        model,inputs,identity=fixture(case);record['identity']=identity
        record['inputs']=[{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()} for x in inputs]
        stage('capture')
        ep=capture_for_export(model,inputs)
        nodes=[{'node':n.name,'op':n.op,'target':str(n.target),'source':n.meta.get('stack_trace',''),
                'module_stack':str(n.meta.get('nn_module_stack',{})), 'tensor':str(n.meta.get('tensor_meta','')),
                'shape':[str(v) for v in n.meta['val'].shape] if isinstance(n.meta.get('val'),torch.Tensor) else None,
                'dtype':str(n.meta['val'].dtype) if isinstance(n.meta.get('val'),torch.Tensor) else None,
                'stride':[str(v) for v in n.meta['val'].stride()] if isinstance(n.meta.get('val'),torch.Tensor) else None} for n in ep.graph.nodes]
        write(root/'graph.json',nodes)
        record['operator_counts']=dict(collections.Counter(n['target'] for n in nodes if n['op']=='call_function'))
        stage('export_static')
        report=tclmachine.export_model(ep,inputs,root/'model.fxb',compile_workers=2,
            artifact_root=root/'edfs',source_metadata={**identity,'architecture':type(model).__qualname__})
        write(root/'export.json',report)
        stage('cpu_reference')
        with torch.no_grad(): expected=model(*inputs)
        stage('reload')
        from unittest.mock import patch
        from furiosa.torch.profiler import Profiler
        # A new instance validates archive loading; fresh process is a separate, still-required acceptance gate.
        with patch('furiosa.torch.compiler.compile',side_effect=AssertionError('runtime compilation forbidden')):
            with fxbx.GraphFXB(root/'model.fxb',weights=model.state_dict()) as runtime:
                stage('execute')
                with Profiler(str(root/'trace.json')): actual=runtime.predict(inputs)
                stage('numerical')
                expected=(expected,) if isinstance(expected,torch.Tensor) else tuple(expected)
                assert len(actual)==len(expected),'output arity differs'
                for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=.02,rtol=.02)
        stage('trace_gate')
        events=json.loads((root/'trace.json').read_text())['traceEvents']
        steps={e['name'] for e in events if e.get('name','').startswith('execution_plan::')}
        npu={e['name'].split('::Task::',1)[1] for e in events if '::Task::execution_plan::' in e.get('name','')}
        cpu=sum(e.get('name')=='run_by_cpu' for e in events)
        record['trace']={'steps':sorted(steps),'npu_steps':sorted(npu),'cpu_routes':cpu}
        record['cpu_fallback']='detected' if cpu else 'no_trace_routes'
        if cpu: raise RuntimeError('CPU_FALLBACK_DETECTED')
        if not steps or steps!=npu: raise RuntimeError('NPU_STEP_EVIDENCE_MISSING')
        record.update(status='smoke_passed',npu_execution='confirmed_per_step',acceptance='limited smoke; changed inputs/repeats/state/fresh-process still required')
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc),error_code=getattr(exc,'code',None))
        (root/'traceback.txt').write_text(traceback.format_exc())
        record['traceback_frames']=[{'file':f.filename,'line':f.lineno,'function':f.name,'source':f.line} for f in traceback.extract_tb(exc.__traceback__)]
        failure_details(record,root)
    finally:
        record['elapsed_s']=time.monotonic()-started
        record['evidence_hashes']={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.json','.txt') and p.name!='result.json'}
        write(root/'result.json',record)
    return record


def summarize(root):
    root=Path(root)
    rows=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
    write(root/'summary.json',{'schema':'hf-librarian.static-campaign.v1','attempts':rows})
    lines=['# 정적 FXB 실패 수집 결과','', '관측된 guard/예외와 근본 원인 추정을 구분합니다. 규칙 부재는 SDK/NPU 미지원 판정이 아닙니다.', '', '|모델/구조|결과|실패 단계|코드|', '|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['case']} ({CASES[r['case']][1]})|{r['status']}|{r['stage']}|{r.get('error_code') or r.get('error_type','')}|")
    for r in rows:
        lines += ['',f"## {r['case']}", '', f"- 증거: `{root/r['case']}`",f"- 입력/모델: `{json.dumps(r.get('identity',{}),ensure_ascii=False)}`",f"- 판정: NPU={r.get('npu_execution')}; CPU fallback={r.get('cpu_fallback')}",f"- 재현: `python -m hf_librarian.static_failures run --out /root/.cache/hf-librarian/static-rerun-새이름 --cases {r['case']}`"]
        if r['status']=='failed':
            lines += [f"- 오류: `{r.get('error','')[:1500]}`", f"- 진단: {json.dumps(r.get('diagnosis',{}),ensure_ascii=False)}", f"- 지문: `{r.get('signature')}`"]
            for g in r.get('preflight',{}).get('failure_groups',[]):
                lines += ['',f"### `{g['op']}` × {g['occurrences']}",f"조건: {g['reason']}",f"노드: `{g['node']}`",'```text',g.get('source',''),g.get('module_stack',''),'```']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    drafts=[]
    for r in rows:
        if r['status']!='failed': continue
        marker='<!-- hf-librarian-static:'+r['case']+' -->'
        section='## '+r['case']+'\n'
        body=marker+'\n'+('\n'.join(lines).split(section,1)[1].split('\n## ',1)[0])
        body+='\n\n버전: `'+json.dumps(r.get('versions',{}))+'`\n'
        body+='\n컴파일러: `'+r.get('compiler_version','unknown')+'`\n'
        body+='\n실패 프레임: `'+json.dumps(r.get('traceback_frames',[])[-12:],ensure_ascii=False)+'`\n'
        body+='\n입력: `'+json.dumps(r.get('inputs',[]))+'`\n'
        body+='\n소스 지문: `'+json.dumps(r.get('source_hashes',{}))+'`\n'
        body+='\n전체 graph/preflight/traceback/원본 로그와 SHA256은 위 증거 경로에 보존됩니다. SDK 업그레이드, 모델별 우회, 자동 수정은 수행하지 않았습니다.\n'
        drafts.append({'marker':marker,'title':'[정적 FXB] '+r['case']+': '+str(r.get('error_code') or r.get('error_type')),'body':body})
    write(root/'issues.json',drafts)
    return rows


def publish(root, repository):
    """Explicit gh-authenticated publication; stable per-case marker, new evidence as comments."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):
        raise ValueError('repository must be owner/name')
    root=Path(root)
    # Fail closed: an authentication/search error never becomes permission to create duplicates.
    def gh(*args):
        return json.loads(subprocess.check_output(['gh','api',*args],text=True))
    receipts=json.loads((root/'github-receipts.json').read_text()) if (root/'github-receipts.json').exists() else []
    for issue in json.loads((root/'issues.json').read_text()):
        body_hash=hashlib.sha256(issue['body'].encode()).hexdigest()
        if any(r.get('marker')==issue['marker'] and r.get('body_sha256')==body_hash for r in receipts): continue
        q=f"repo:{repository} in:body \"{issue['marker']}\""
        matches=[r for r in gh('-X','GET','search/issues','-f','q='+q)['items'] if issue['marker'] in (r.get('body') or '')]
        if len(matches)>1: raise ValueError('multiple matching issues; resolve duplicate markers first')
        payload=root/'issue-payload.json'
        write(payload,{'body':issue['body']} if matches else {'title':issue['title'],'body':issue['body']})
        endpoint=f"repos/{repository}/issues"+(f"/{matches[0]['number']}/comments" if matches else '')
        result=gh('-X','POST',endpoint,'--input',str(payload))
        receipts.append({'marker':issue['marker'],'url':result['html_url'],'body_sha256':body_hash})
        write(root/'github-receipts.json',receipts)
    return receipts


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--out',required=True);run.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES));run.add_argument('--timeout',type=int,default=900)
    w=sub.add_parser('worker');w.add_argument('--case',choices=CASES,required=True);w.add_argument('--out',required=True)
    s=sub.add_parser('report');s.add_argument('--out',required=True)
    pub=sub.add_parser('publish');pub.add_argument('--out',required=True);pub.add_argument('--repo',required=True)
    args=p.parse_args()
    if args.action=='publish':print(json.dumps(publish(args.out,args.repo)));return
    if args.action=='worker':worker(args.case,args.out);return
    if args.action=='report':summarize(args.out);return
    if args.timeout<=0:p.error('timeout must be positive')
    root=Path(args.out).absolute();root.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','FURIOSA_REQUIRE_NPU':'1',
         'FXBX_TRACE_STEPS':'1','RAYON_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
    for case in dict.fromkeys(args.cases):
        target=root/case;target.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--case',case,'--out',str(target)]
        write(target/'command.json',{'argv':command,'cwd':os.getcwd(),'environment':{k:env[k] for k in ('PYTHONPATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','FXBX_TRACE_STEPS') if k in env}})
        with (target/'process.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
        result=target/'result.json'
        if not result.exists() or json.loads(result.read_text()).get('status')=='running':
            r=json.loads(result.read_text()) if result.exists() else {'case':case,'stage':'worker_start'}
            r.update(status='failed',error_type='Timeout' if code==124 else 'WorkerExit',error=f'worker exit {code}',npu_execution='unknown',cpu_fallback='unknown')
            failure_details(r,target);write(result,r)
        write(target/'process.json',{'exit_code':code})
        rows=summarize(root)
        print(json.dumps({'case':case,'status':next(r['status'] for r in rows if r['case']==case)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
