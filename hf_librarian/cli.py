import argparse
import json
import sys
from . import Catalog, Librarian, Verifier, __version__

def main():
    p = argparse.ArgumentParser(prog="hf-librarian")
    p.add_argument("--catalog")
    sub = p.add_subparsers(dest="command", required=True)
    activity=sub.add_parser("activity",help="One-shot supervisor and active model stages");activity.add_argument("job")
    sub.add_parser("categories")
    ls = sub.add_parser("list"); ls.add_argument("--category")
    sub.add_parser("settings"); sub.add_parser("status"); sub.add_parser("version"); sub.add_parser("blacklist")
    get = sub.add_parser("get"); group = get.add_mutually_exclusive_group(required=True); group.add_argument("--id"); group.add_argument("--category")
    pre = sub.add_parser("prefetch"); pre.add_argument("--category", action="append", required=True); pre.add_argument("--workers", type=int)
    cat = sub.add_parser("add-category"); cat.add_argument("name"); cat.add_argument("description")
    add = sub.add_parser("add-model"); add.add_argument("json", help="JSON object or @file")
    rm = sub.add_parser("remove-model"); rm.add_argument("id")
    block = sub.add_parser("block"); block.add_argument("id"); block.add_argument("--reason", default="")
    unblock = sub.add_parser("unblock"); unblock.add_argument("id")
    setting = sub.add_parser("set"); setting.add_argument("key"); setting.add_argument("value", help="JSON value")
    verify = sub.add_parser("verify"); verify.add_argument("--id", required=True); verify.add_argument("--backend", choices=["fxb", "fxbex"], default="fxb"); verify.add_argument("--executable"); verify.add_argument("--version"); verify.add_argument("--source"); verify.add_argument("--timeout", type=int, default=1800); verify.add_argument("--build-arg", action="append", default=[]); verify.add_argument("--results-dir")
    hist = sub.add_parser("history"); hist.add_argument("--id"); hist.add_argument("--backend", choices=["fxb", "fxbex"]); hist.add_argument("--results-dir")
    get.add_argument('--profile',choices=['weights','config'],default='weights')
    pre.add_argument('--profile',choices=['weights','config'],default='weights')
    campaign=sub.add_parser('campaign');campaign.add_argument('action',choices=['start','status','issues','history','repair','stop'])
    campaign.add_argument('root');campaign.add_argument('--group');campaign.add_argument('--detail',choices=['summary','repro','evidence'],default='summary')
    campaign.add_argument('--model');campaign.add_argument('--repair-action',choices=['begin','record','history'],default='history');campaign.add_argument('--record')
    args = p.parse_args(); c = Catalog(args.catalog)
    try:
        if args.command=='activity':
            from .activity import snapshot
            result=snapshot(args.job)
        elif args.command=='campaign':
            from . import campaign as controller
            from pathlib import Path
            if args.action=='start':result=controller.start(json.loads(Path(args.root).read_text()))
            elif args.action=='status':result=controller.status(args.root)
            elif args.action=='issues':result=controller.issues(args.root,args.group,args.detail)
            elif args.action=='history':result=controller.history(args.root,args.model)
            elif args.action=='repair':result=controller.repair(args.root,args.group,args.repair_action,json.loads(Path(args.record).read_text()) if args.record else None)
            else:controller.save(Path(args.root)/'STOP.json',{'stop':True});result={'stop_requested':True}
        elif args.command == "categories": result = c.categories()
        elif args.command == "list": result = c.models(args.category)
        elif args.command == "settings": result = c.settings()
        elif args.command == "version": result = {"package": __version__, "schema": c.version()}
        elif args.command == "blacklist": result = c.blacklist()
        elif args.command == "add-category": result = c.add_category(args.name, args.description)
        elif args.command == "add-model":
            from pathlib import Path
            import json5
            value = Path(args.json[1:]).read_text() if args.json.startswith("@") else args.json
            result = c.add_model(json5.loads(value))
        elif args.command == "remove-model": result = c.remove_model(args.id)
        elif args.command == "block": result = c.set_blacklist(args.id, reason=args.reason)
        elif args.command == "unblock": result = c.set_blacklist(args.id, False)
        elif args.command == "set": result = c.update_settings(**{args.key: json.loads(args.value)})
        else:
            def event(e): print(json.dumps(e, ensure_ascii=False), file=sys.stderr, flush=True)
            with Librarian(c, max_concurrent=getattr(args, "workers", None), on_event=event) as lib:
                if args.command == "verify":
                    result = Verifier(lib,args.results_dir).verify(args.id,args.backend,args.executable,args.version,args.source,args.timeout,args.build_arg)
                    print(json.dumps(result,ensure_ascii=False,indent=2))
                    if result['status'] != 'passed': sys.exit(1)
                    return
                elif args.command == "history": result = Verifier(lib,args.results_dir).history(args.id,args.backend)
                elif args.command == "status": result = lib.usage()
                elif args.command == "get": result = lib.request(model_id=args.id, category=args.category,profile=args.profile)
                else:
                    futures = [lib.request_async(category=x,profile=args.profile) for x in args.category]
                    result = [f.result() for f in futures]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr); sys.exit(1)


if __name__=='__main__':main()
