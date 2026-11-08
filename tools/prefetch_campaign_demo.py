"""Prepare only configurations for a bounded small-model campaign."""
import concurrent.futures
import json
from pathlib import Path
import shutil

from hf_librarian import Catalog, Librarian

MODELS = ['Qwen/Qwen3-0.6B','HuggingFaceTB/SmolLM2-135M-Instruct',
          'TinyLlama/TinyLlama-1.1B-Chat-v1.0','microsoft/phi-2','state-spaces/mamba-130m-hf',
          'HuggingFaceTB/SmolLM2-360M-Instruct','HuggingFaceTB/SmolLM2-1.7B-Instruct',
          'Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-1.5B','Qwen/Qwen3-1.7B']


def main():
    root=Path('/root/TPLMachine/reports/librarian-campaign-20261108')
    path=root/'catalog.json'
    if not path.exists():shutil.copyfile(Catalog().path,path)
    catalog=Catalog(path)
    for model in MODELS:
        if not catalog.get(model):catalog.add_model({'id':model,'categories':['language-generation'],
            'revision':'main','include':['config.json','*.safetensors'],'exclude':[],
            'priority':60,'npu_status':'untested','source':'https://huggingface.co/'+model,
            'notes':'configuration-only estimator diversity; no checkpoint download'})
    results=[]
    with Librarian(catalog,max_concurrent=2) as library:
        futures={library.request_async(model_id=model,profile='config'):model for model in MODELS}
        for future in concurrent.futures.as_completed(futures):
            try:results.append({'model':futures[future],'path':future.result(),'status':'prepared'})
            except Exception as exc:results.append({'model':futures[future],'status':'failed','error':str(exc)[:500]})
    (root/'prefetch-all.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({'models':len(results),'prepared':sum(r['status']=='prepared' for r in results)}))


if __name__=='__main__':main()
