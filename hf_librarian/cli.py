import argparse
import json
import sys
from . import Catalog, Librarian, __version__

def main():
    p = argparse.ArgumentParser(prog="hf-librarian")
    p.add_argument("--catalog")
    sub = p.add_subparsers(dest="command", required=True)
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
    args = p.parse_args(); c = Catalog(args.catalog)
    try:
        if args.command == "categories": result = c.categories()
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
                if args.command == "status": result = lib.usage()
                elif args.command == "get": result = lib.request(model_id=args.id, category=args.category)
                else:
                    futures = [lib.request_async(category=x) for x in args.category]
                    result = [f.result() for f in futures]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr); sys.exit(1)
