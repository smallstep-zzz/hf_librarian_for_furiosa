import copy
import json
import os
import tempfile
from pathlib import Path

import json5

DEFAULT = Path(__file__).with_name("catalog.json")

class Catalog:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("HF_LIBRARIAN_CATALOG", DEFAULT))
        self.data = json5.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("schema_version") != 1:
            raise ValueError("Unsupported catalog schema")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=".catalog-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name): os.unlink(name)

    def categories(self): return copy.deepcopy(self.data["categories"])
    def settings(self): return copy.deepcopy(self.data["settings"])
    def version(self): return self.data["schema_version"]
    def blacklist(self): return copy.deepcopy(self.data["blacklist"])
    def models(self, category=None, **filters):
        return [copy.deepcopy(m) for m in self.data["models"]
                if (category is None or category in m["categories"])
                and all(m.get(k) == v for k, v in filters.items())]
    def get(self, model_id):
        return next((m for m in self.models() if m["id"] == model_id), None)
    def add_category(self, name, description):
        if not name or name in self.data["categories"]: raise ValueError("Category already exists or is empty")
        self.data["categories"][name] = description
        self.save()
    def add_model(self, model):
        model = copy.deepcopy(model)
        if not model.get("id") or self.get(model["id"]): raise ValueError("Model ID missing or already exists")
        if not model.get("categories") or any(c not in self.data["categories"] for c in model["categories"]):
            raise ValueError("Unknown or missing category")
        self.data["models"].append(model)
        self.save()
    def remove_model(self, model_id):
        if not self.get(model_id): raise KeyError(model_id)
        self.data["models"] = [m for m in self.data["models"] if m["id"] != model_id]
        self.save()
    def set_blacklist(self, model_id, blocked=True, reason=""):
        if blocked: self.data["blacklist"][model_id] = reason
        else: self.data["blacklist"].pop(model_id, None)
        self.save()
    def update_settings(self, **values):
        unknown = set(values) - set(self.data["settings"])
        if unknown: raise ValueError(f"Unknown settings: {unknown}")
        settings = {**self.data["settings"], **values}
        if settings["max_concurrent_downloads"] < 1 or min(settings["cache_limit_gib"], settings["min_free_gib"], settings["max_model_gib"]) <= 0:
            raise ValueError("Settings must be positive")
        self.data["settings"] = settings
        self.save()
