import concurrent.futures
import hashlib
import json
import fnmatch
import os
import shutil
import subprocess
import threading
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from .catalog import Catalog

GIB = 1024 ** 3

class Librarian:
    def __init__(self, catalog=None, cache_dir=None, max_concurrent=None, on_event=None):
        self.catalog = catalog or Catalog()
        self.cache = Path(cache_dir or os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE)).expanduser().resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.settings = self.catalog.settings()
        workers = max_concurrent if max_concurrent is not None else self.settings["max_concurrent_downloads"]
        if workers < 1: raise ValueError("max_concurrent must be positive")
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.on_event = on_event or (lambda event: None)
        self._guard = threading.Lock()
        self._reserved = 0
        self._inflight = {}

    def close(self): self.pool.shutdown(wait=True)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def emit(self, event, model_id, **kwargs):
        try: self.on_event({"event": event, "model_id": model_id, **kwargs})
        except Exception: pass
    def usage(self):
        size = 0
        for root, dirs, files in os.walk(self.cache):
            for name in files:
                p = Path(root) / name
                try:
                    if not p.is_symlink(): size += p.stat().st_size
                except FileNotFoundError: pass
        return {"cache_bytes": size, "free_bytes": shutil.disk_usage(self.cache).free,
                "limit_bytes": int(self.settings["cache_limit_gib"] * GIB), "cache_dir": str(self.cache)}
    def select(self, category):
        models = [m for m in self.catalog.models(category) if m["id"] not in self.catalog.blacklist()]
        if not models: raise LookupError(f"No eligible model in category {category}")
        return sorted(models, key=lambda m: (m.get("priority", 100), m["id"]))[0]["id"]
    def request(self, model_id=None, category=None):
        return self.request_async(model_id, category).result()
    def request_async(self, model_id=None, category=None):
        if (model_id is None) == (category is None): raise ValueError("Specify exactly one of model_id or category")
        model_id = model_id or self.select(category)
        if model_id in self.catalog.blacklist(): raise PermissionError(f"Blacklisted: {model_id}")
        model = self.catalog.get(model_id)
        if not model: raise LookupError(f"Model not in catalog: {model_id}")
        with self._guard:
            future = self._inflight.get(model_id)
            if future is None or future.done():
                future = self.pool.submit(self._download, model)
                self._inflight[model_id] = future
            return future
    def _download(self, model):
        model_id = model["id"]
        try:
            revision = model.get("revision", "main")
            patterns = model.get("include")
            excludes = model.get("exclude", [])
            key = hashlib.sha256(json.dumps([model_id, revision, patterns, excludes], sort_keys=True).encode()).hexdigest()
            marker = self.cache / ".hf-librarian" / (key + ".json")
            if marker.exists():
                try:
                    state = json.loads(marker.read_text())
                    local = Path(state["path"])
                    if local.is_dir() and all((local / name).is_file() for name in state["files"]):
                        self.emit("available", model_id, path=str(local))
                        return str(local)
                except (OSError, ValueError, KeyError, TypeError): pass
            self.emit("missing", model_id)
            info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
            commit = info.sha
            files = [s for s in info.siblings if (not patterns or any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)) and not any(fnmatch.fnmatch(s.rfilename, p) for p in excludes)]
            if not files or any(s.size is None for s in files): raise RuntimeError("Cannot determine selected file sizes")
            if not any(s.rfilename.endswith((".safetensors", ".bin", ".pt", ".gguf", ".onnx")) for s in files):
                raise RuntimeError("Selection contains no recognized weight file")
            total = sum(s.size for s in files)
            if total > self.settings["max_model_gib"] * GIB: raise RuntimeError("Model exceeds max_model_gib")
            plan = snapshot_download(model_id, revision=commit, cache_dir=str(self.cache), allow_patterns=patterns, ignore_patterns=excludes, dry_run=True)
            needed = sum(item.file_size for item in plan if item.will_download)
            with self._guard:
                stats = self.usage()
                if stats["cache_bytes"] + self._reserved + needed > stats["limit_bytes"] or stats["free_bytes"] - self._reserved - needed < self.settings["min_free_gib"] * GIB:
                    raise RuntimeError("Insufficient cache budget or disk space")
                self._reserved += needed
            try:
                self.emit("downloading", model_id, bytes=needed)
                args = ["hf", "download", model_id, "--revision", commit, "--cache-dir", str(self.cache), "--quiet"]
                for pattern in patterns or []: args.extend(["--include", pattern])
                for pattern in excludes: args.extend(["--exclude", pattern])
                proc = subprocess.run(args, text=True, capture_output=True)
                if proc.returncode:
                    raise RuntimeError(f"hf download failed: {proc.stderr[-2000:]}")
                path = proc.stdout.strip().splitlines()[-1]
                if not Path(path).is_dir() or any(not (Path(path) / s.rfilename).is_file() for s in files):
                    raise RuntimeError("Download is incomplete")
                marker.parent.mkdir(exist_ok=True)
                temp = marker.with_suffix(".tmp")
                temp.write_text(json.dumps({"model_id": model_id, "commit": commit, "path": path, "files": [s.rfilename for s in files]}))
                os.replace(temp, marker)
                self.emit("completed", model_id, path=path)
                return path
            finally:
                with self._guard: self._reserved -= needed
        except Exception as exc:
            self.emit("failed", model_id, error=str(exc))
            raise
