import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('static_failures',Path(__file__).parents[1]/'hf_librarian/static_failures.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FailureTests(unittest.TestCase):
    def test_rule_gap_does_not_claim_sdk_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.fxb.plan').mkdir()
            m.write(root/'model.fxb.plan/preflight.json',{'code':'EXECUTION_RULE_GAP','failure_groups':[{'op':'aten.conv2d.default','reason':'no public EDF region rule'}]})
            r={'stage':'export_static','error_type':'EDFExportError','error':'changing node number'}
            m.failure_details(r,root)
            self.assertIn('SDK support unknown',r['diagnosis']['root_cause_status'])
            first=r['signature'];r['error']='different path and node';m.failure_details(r,root)
            self.assertEqual(first,r['signature'])
    def test_capability_deferral_names_the_consumer_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.fxb.plan').mkdir()
            deferrals={'supports_frozen_buffer_guards':2}
            m.write(root/'model.fxb.plan/preflight.json',{
                'code':'EXECUTION_RULE_GAP','static_preparation':{'consumer_capability_deferrals':deferrals},
                'failure_groups':[{'op':'aten.cos.default','reason':'no public EDF region rule'}]})
            record={'stage':'export_static','error_type':'EDFExportError'}
            m.failure_details(record,root)
            self.assertEqual(record['diagnosis']['consumer_capability_deferrals'],deferrals)
            self.assertIn('--consumer-source',record['diagnosis']['proposed_fix'])

    def test_explicit_consumer_source_wins_and_rejects_a_preloaded_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();legacy=root/'installed';source=root/'selected source'
            for directory,capability in ((legacy,False),(source,True)):
                (directory/'fxbx').mkdir(parents=True);(directory/'fxbex').mkdir()
                (directory/'fxbx/__init__.py').write_text('from fxbex.execution_fxb import ExecutionFXB\n')
                (directory/'fxbex/__init__.py').write_text('')
                (directory/'fxbex/execution_fxb.py').write_text(
                    f'class ExecutionFXB:\n    supports_frozen_buffer_guards={capability!r}\n')
            script="""
import importlib.util,json,os,sys
spec=importlib.util.spec_from_file_location('collector',sys.argv[1]);collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
if sys.argv[3]=='preloaded':import fxbx
api,engine,root=collector.load_consumer(sys.argv[2] or None)
print(json.dumps({'capability':api.ExecutionFXB.supports_frozen_buffer_guards,'engine':engine.__file__,'path':os.environ['PYTHONPATH']}))
"""
            command=[sys.executable,'-c',script,m.__file__]
            env={**os.environ,'PYTHONPATH':str(legacy)}
            baseline=json.loads(subprocess.check_output(command+['','fresh'],env=env,text=True))
            self.assertFalse(baseline['capability'])
            self.assertEqual(baseline['path'],str(legacy))
            selected=json.loads(subprocess.check_output(command+[str(source),'fresh'],env=env,text=True))
            self.assertTrue(selected['capability'])
            self.assertEqual(Path(selected['engine']),source/'fxbex/execution_fxb.py')
            self.assertEqual(selected['path'].split(os.pathsep),[str(source),str(legacy)])
            shadowed=subprocess.run(command+[str(source),'preloaded'],env=env,text=True,capture_output=True)
            self.assertNotEqual(shadowed.returncode,0)
            self.assertIn('but imports resolved to',shadowed.stderr)

    def test_run_records_and_forwards_source_and_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'consumer';output=root/'attempt';artifacts=root/'shared edfs'
            (source/'fxbx').mkdir(parents=True);(source/'fxbex').mkdir()
            for name in ('fxbx/__init__.py','fxbex/__init__.py','fxbex/execution_fxb.py'):
                (source/name).touch()
            def run_worker(command, **options):
                self.assertEqual(options['env']['PYTHONPATH'].split(os.pathsep),[str(source),'producer'])
                m.write(output/'control/result.json',{'case':'control','status':'smoke_passed','stage':'trace_gate'})
                return SimpleNamespace(wait=lambda timeout:0)
            argv=[m.__file__,'run','--out',str(output),'--cases','control',
                  '--consumer-source',str(source),'--artifact-root',str(artifacts)]
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'PYTHONPATH':'producer'}), \
                    patch.object(m.subprocess,'Popen',side_effect=run_worker) as launch,patch('builtins.print'):
                m.main()
            command=launch.call_args.args[0]
            receipt=json.loads((output/'control/command.json').read_text())
            self.assertEqual(receipt['artifact_root'],str(artifacts))
            self.assertEqual(receipt['argv'],command)
            with patch.object(sys,'argv',command[1:]),patch.object(m,'worker') as worker:
                m.main()
            worker.assert_called_once_with('control',str(output/'control'),source,artifacts)

    def test_runtime_weights_include_nonpersistent_buffer_aliases_and_preserve_layout(self):
        try:import torch
        except ImportError:self.skipTest('optional PyTorch')
        model=torch.nn.Linear(2,2).eval().requires_grad_(False)
        phase=torch.arange(6,dtype=torch.float32).reshape(2,3).t()
        model.register_buffer('phase',phase,persistent=False)
        model.register_buffer('phase_alias',phase,persistent=False)
        self.assertNotIn('phase',model.state_dict())
        bound=m.runtime_weights(model)
        self.assertEqual(set(bound),{'weight','bias','phase','phase_alias'})
        self.assertEqual(bound['phase'].stride(),phase.stride())
        self.assertEqual(bound['phase_alias'].data_ptr(),phase.data_ptr())
        self.assertTrue(torch.equal(bound['phase'],phase))

    def test_unconfirmed_and_unknown_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'control').mkdir()
            r={'case':'control','stage':'execute','status':'failed','error_type':'WorkerExit','error':'signal','npu_execution':'unknown','cpu_fallback':'unknown'}
            m.failure_details(r,root/'control');m.write(root/'control/result.json',r)
            rows=m.summarize(root)
            self.assertEqual(rows[0]['diagnosis']['root_cause_status'],'unconfirmed')
            self.assertIn('CPU fallback=unknown',(root/'REPORT.md').read_text())
    def test_publication_reuses_marker_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);marker='<!-- hf-librarian-static:resnet -->'
            m.write(root/'issues.json',[{'marker':marker,'title':'test','body':marker+' evidence'}])
            replies=[json.dumps({'items':[{'number':3,'body':marker}]}),json.dumps({'html_url':'https://example.test/3#comment'})]
            with patch.object(m.subprocess,'check_output',side_effect=replies) as call:
                m.publish(root,'owner/repo')
                self.assertIn('repos/owner/repo/issues/3/comments',call.call_args.args[0])
            with patch.object(m.subprocess,'check_output') as call:
                m.publish(root,'owner/repo');call.assert_not_called()
    def test_publication_search_error_does_not_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m.write(root/'issues.json',[{'marker':'x','title':'t','body':'b'}])
            with patch.object(m.subprocess,'check_output',side_effect=m.subprocess.CalledProcessError(1,'gh')) as call:
                with self.assertRaises(m.subprocess.CalledProcessError): m.publish(root,'owner/repo')
                self.assertEqual(call.call_count,1)
    def test_trace_requires_every_step(self):
        events=[{'name':'execution_plan::a'},{'name':'execution_plan::b'},{'name':'x::Task::execution_plan::a'}]
        self.assertEqual(m.trace_evidence(events)['npu_execution'],'partial_or_unmatched')
        self.assertEqual(m.trace_evidence([])['npu_execution'],'not_observed')
    def test_cpu_fallback_is_visible_even_with_npu_steps(self):
        events=[{'name':'execution_plan::a'},{'name':'x::Task::execution_plan::a'},{'name':'run_by_cpu'}]
        r=m.trace_evidence(events)
        self.assertEqual(r['npu_execution'],'confirmed_per_step')
        self.assertEqual(r['cpu_routes'],1)
    def test_atomic_unicode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';m.write(p,{'오류':'원본'});m.write(p,{'오류':'재현'})
            self.assertEqual(json.loads(p.read_text()),{'오류':'재현'})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

if __name__=='__main__':unittest.main()
