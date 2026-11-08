"""Run real FXB/FXBEX commands and retain evidence, never infer NPU execution."""
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

class Verifier:
    def __init__(self, librarian, results_dir=None):
        self.librarian = librarian
        self.root = Path(results_dir or os.environ.get('HF_LIBRARIAN_RESULTS', Path.home()/'.cache/hf-librarian/verification')).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'history.sqlite3'
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE IF NOT EXISTS results (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, backend TEXT NOT NULL, started_at TEXT NOT NULL, payload TEXT NOT NULL)')
    def history(self, model_id=None, backend=None):
        query='SELECT payload FROM results WHERE 1=1'; args=[]
        if model_id: query+=' AND model_id=?'; args.append(model_id)
        if backend: query+=' AND backend=?'; args.append(backend)
        with sqlite3.connect(self.db) as db:
            return [json.loads(r[0]) for r in db.execute(query+' ORDER BY started_at DESC',args)]
    def verify(self, model_id, backend='fxb', executable=None, version=None, source=None, timeout=1800, build_args=None):
        if backend not in ('fxb','fxbex'): raise ValueError('backend must be fxb or fxbex')
        if timeout <= 0: raise ValueError('timeout must be positive')
        if not self.librarian.catalog.get(model_id): raise LookupError('Model is not in catalog')
        if model_id in self.librarian.catalog.blacklist(): raise PermissionError('Model is blacklisted')
        run_id=uuid.uuid4().hex
        run_dir=self.root/run_id; run_dir.mkdir()
        record={'id':run_id,'model_id':model_id,'backend':backend,'started_at':datetime.now(timezone.utc).isoformat(),
                'requested_version':version or 'latest', 'actual_version':None,'status':'blocked',
                'artifact':None,'model_revision':None,'source':str(source) if source else None,
                'build_args':list(build_args or []),'npu_execution':'untested','log':str(run_dir/'build.log')}
        output=run_dir/('model.fxb' if backend=='fxb' else 'model.fxbex')
        chunks=[]
        def run(argv, limit):
            chunks.append('$ '+json.dumps(argv)+'\n')
            started=time.monotonic()
            try:
                result=subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=limit)
                chunks.append(result.stdout or '')
                chunks.append(f'\nexit={result.returncode}, seconds={time.monotonic()-started:.2f}\n')
                return result.returncode
            except subprocess.TimeoutExpired as e:
                chunks.append((e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or ''))
                chunks.append('\nTIMEOUT\n')
                raise
        try:
            binary=shutil.which(executable or backend)
            if not binary: raise RuntimeError(f'{backend} executable not found')
            record['executable']=str(Path(binary).resolve())
            v=subprocess.run([binary,'version' if backend=='fxb' else '--version'],capture_output=True,text=True,timeout=30,check=True)
            raw=(v.stdout+' '+v.stderr).strip(); record['version_output']=raw
            match=re.search(r'\b(?:v)?(\d+(?:\.\d+)+(?:[-+][\w.]+)?)',raw)
            if not match: raise RuntimeError(f'Cannot parse version: {raw}')
            record['actual_version']=match.group(1)
            if version and version!=record['actual_version']: raise RuntimeError(f'Requested {version}, executable reports {record["actual_version"]}')
            if not version:
                package = 'furiosa-llm' if backend == 'fxb' else 'fxbex'
                try:
                    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/json', timeout=15) as response:
                        latest=json.load(response)['info']['version']
                except Exception as e:
                    raise RuntimeError(f'Cannot confirm latest {package} version: {e}; specify an installed version explicitly')
                record['latest_version']=latest
                if latest != record['actual_version']:
                    raise RuntimeError(f'Latest {package} is {latest}; selected executable is {record["actual_version"]}. Select an executable with the latest version or explicitly request the installed version.')
            if backend == 'fxb':
                try:
                    first=Path(binary).open('rb').readline().decode().strip()
                    interpreter=first[2:].split()[0] if first.startswith('#!') else None
                    if interpreter and Path(interpreter).is_file():
                        probe=subprocess.run([interpreter,'-c',"import importlib.metadata as m; print(m.version('furiosa-tcc'))"],capture_output=True,text=True,timeout=30,check=True)
                        record['compiler_version']=probe.stdout.strip()
                    else: record['compiler_version']=None
                except Exception: record['compiler_version']=None
            if backend=='fxbex' and not source: raise RuntimeError('FXBEX requires a prepared compiled directory; it cannot compile a Hugging Face model')
            snapshot=Path(self.librarian.request(model_id=model_id)).resolve()
            record['snapshot']=str(snapshot); record['model_revision']=snapshot.name
            if backend=='fxb':
                argv=[binary,'build',str(snapshot),str(output)]+list(build_args or [])
            else:
                source=Path(source).expanduser().resolve()
                if not source.is_dir(): raise RuntimeError('FXBEX source must be a prepared compiled directory')
                # Caller supplies the compiled directory; its provenance must identify this model/revision.
                manifest=source/'manifest.json'
                if not manifest.is_file(): raise RuntimeError('FXBEX source lacks manifest.json')
                manifest_data=json.loads(manifest.read_text())
                provenance=manifest_data.get('provenance') or {}
                if provenance.get('source_url', '').rstrip('/') != 'https://huggingface.co/' + model_id or provenance.get('revision') != snapshot.name:
                    raise RuntimeError('FXBEX manifest provenance does not match model and cached revision')
                argv=[binary,'build',str(source),'-o',str(output),'--source-model',model_id,'--source-revision',snapshot.name]+list(build_args or [])
            record['status']='failed'; record['command']=argv
            if run(argv,timeout)!=0: raise RuntimeError('Build command failed; see log')
            if not output.is_file() or output.stat().st_size==0: raise RuntimeError('Build did not produce a nonempty artifact')
            inspect=[binary,'show' if backend=='fxb' else 'inspect',str(output)]
            if run(inspect,120)!=0: raise RuntimeError('Artifact inspection failed; see log')
            record['artifact']=str(output);record['artifact_bytes']=output.stat().st_size
            h=hashlib.sha256()
            with output.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
            record['sha256']=h.hexdigest();record['status']='passed'
        except subprocess.TimeoutExpired: record['status']='timeout';record['error']='Command timed out'
        except Exception as e: record['error']=str(e)
        finally:
            (run_dir/'build.log').write_text(''.join(chunks),encoding='utf-8')
            record['finished_at']=datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db) as db:
                db.execute('INSERT INTO results VALUES (?,?,?,?,?)',(run_id,model_id,backend,record['started_at'],json.dumps(record,ensure_ascii=False)))
        return record
