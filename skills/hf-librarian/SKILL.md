---
name: hf-librarian
description: Research, classify, and maintain Hugging Face test models for Furiosa compiler and NPU coverage using hf-librarian. Use when asked to add model coverage, prefetch an announced model, or find models by task and test properties.
---

# Hugging Face model librarian

For one known error or a small rule edit, FHA should call FuryNo5 directly.
Use this librarian for model/category preparation and unattended multi-model
improvement campaigns. It is not a mandatory hop for every test.

`get --profile config` pins and downloads only config.json for typed projection
collection; it does not load model weights or execute remote model code. The
default `weights` profile preserves existing download behaviour.

Start `hf-librarian campaign start PLAN.json` using the plan format in README.
Return after supervisor readiness. Do not keep FHA/LLM watching or polling it.
The host continues after SSH disconnect. Only when asked about progress/completion,
query `campaign status ROOT` once on that host: it checks actual process identity
and reports arithmetic ETA with a range. Network failure means connectivity
unknown; missing process/receipt is a process incident, not a model failure.
`campaign stop ROOT` writes a server stop signal.

CPU collections run in separate SDK processes, starting at2, then3 after2
successes and5 after5, subject to effective-memory admission. NPU execution uses
one shared canonical FXBEX lease across campaigns; feedback/training commits use
a different shared lease. Queue/lease waits never enter performance labels.
These initialized projection audits do not prove whole-model support or speed.

Review `campaign history ROOT` for model/revision/coverage/objective scores, and
`campaign issues ROOT` for counts/top groups. Request one `--group ID --detail repro`
or `evidence` only when needed. Reusable generation/routing/runtime defects belong
to FHA's shared-component maintainer, not repeated per-model TCL generation.
Automatic baseline retention is not a manual repair case. Repair history retains
why/change/actual validation and permits at most3 attempts; aggregate unresolved
groups into one user decision. No automatic paid LLM call is made by the manager.

Cost policy: official Furiosa estimates first; on failure, a compatible validated
FuryNo5 predictor may return a tagged fallback. Predictions are not real NPU
latency or training truth. Without a compatible prior return unknown.

Use the installed `hf-librarian` command. Do not edit catalog JSON directly. `hf-librarian categories`, `list --category CATEGORY`, `settings`, `status`, and `blacklist` show current state. Model IDs are exact Hugging Face IDs.

When asked to research models, check the current Hugging Face model card and file list. Choose varied architectures and a representative range of size, quantization, novelty, rare operations, and paper implementations. Record the source URL and reason for selection. Verify the license and gated status before recommending use. The catalog records candidates, not proven NPU support; set `npu_status` to `untested` until a real compiler/runtime test proves otherwise. Estimated file size is not runtime memory. Check selected files, loader requirements, and Hub sizes; do not select multiple redundant weight formats without reason.

Add a category with `hf-librarian add-category NAME DESCRIPTION`. Add a model with `hf-librarian add-model @model.json`, where the input is a small object such as:

```json
{"id":"org/model", "categories":["language-generation"], "revision":"main", "include":["*.json","*.safetensors"], "exclude":[], "estimated_gib":1.0, "quantization":"bf16", "difficulty":"medium", "rarity":"common", "paper_implementation":false, "npu_status":"untested", "priority":50, "notes":"why this model", "source":"https://huggingface.co/org/model"}
```

`hf-librarian remove-model ID`, `block ID --reason REASON`, and `unblock ID` manage membership and exclusions. Removing or blocking does not delete cached files. `hf-librarian get --id ID` downloads a known model and prints its snapshot path; `get --category NAME` selects the highest-priority eligible model. `prefetch --category NAME --category NAME` downloads in parallel. Follow an explicit request to prefetch an announced future test model after adding it. Do not fill the cache merely to reach its limit. Respect the configured limit, free-space reserve, gated model access, and blacklist; report failures accurately. Use Python API examples in the project README for integration.

For build coverage use `hf-librarian verify --id ORG/MODEL --backend fxb` and `history --id ORG/MODEL`. Default verification requires the latest PyPI version to match the installed executable; for a specific installed release provide both `--version` and, if needed, `--executable`. FXBEX is a separate packaging step, not a compiler: `--backend fxbex --source COMPILED_DIR` requires a producer-prepared manifest whose provenance matches the model and revision. Report blocked, failed, timeout, and passed separately, with exact versions, inputs, and logs. An artifact alone does not prove model execution on the NPU.

For “server looks quiet / what is running”, use `hf-librarian activity JOB_ROOT`
once. It distinguishes actual supervisor death from model metadata preparation,
CPU preparation, NPU queue/execution, completion and review-required stop. A launch
readiness receipt alone is not proof the controller stayed alive. Use its bounded
error excerpt before restarting. Do not claim full-night progress from startup.

When the user authorizes completion/error notifications, attach the reusable
`hf_librarian.notifications wait JOB_ROOT` as a detached server event recorder.
It uses existing kernel completion events, not status polling. A separately
started macOS `notifications.py desktop HOST JOB_ROOT` receiver delivers one
native notification, retaining inbox/delivery receipts and retrying SSH only on
connection failure. Keep notifications separate from the workload; loss of Mac
connectivity must not cancel it. Native notification acceptance is not proof the
user saw the banner. Do not promise a closed Codex chat wakes automatically.

For continued useful work after a model acceptance, use the continuous manager
with the existing catalog, not another hand-maintained model list. It queues new
row contracts, serializes NPU/feedback access, deduplicates attempts by shape/rule
version, and refits on combined official teacher rows only when keys change.
Leave prepared catalog discovery to its owner. Keep the detached manager running
and return after one readiness/activity check. `activity JOB_ROOT` reports its
current wave; STOP.json propagates to active workers. Watch for disk/memory gates.
No auto-repair/LLM is hidden in this service. An exhausted queue waits for new
catalog/rule input; it does not churn old tests to keep utilization high.


## Hub research rate limits

Prefer existing catalog entries and cached public model-card evidence before requesting Hub metadata. For a specific model, call `from hf_librarian.hub_research import HubResearch; HubResearch().model('org/model')`. This caches results for 24 hours, coordinates its own local processes, limits starts to 20 per 5 minutes and at least 5 seconds apart, and records a cooldown on HTTP 429. A `ResearchLimit` is a stop signal: do not immediately retry, fan out across models, or switch credentials/IP to evade the limit. The `overnight` discovery path uses this shared local guard and stops on a limit. The helper does not control other applications sharing the IP or SDK-internal requests; coordinate those separately, and do not assume the guard can guarantee the absence of HTTP 429. Metadata and a public file listing do not prove that every weight is downloadable or that the model runs on NPU. Never download weights solely to populate the catalog unless asked.

For optimizer changes use `optimization-check PLAN --source TCL_SOURCE --output NEW_ROOT`.
It requests a drained-wave pause, runs as a detached event-driven job, and compares
fixed/adaptive/cached closed regions from HF configuration snapshots. It validates
fresh numerical/NPU gates and the selected artifact's runtime binding. This is not
whole-model support. Exact qualified shape/source/environment cases are automatically
used by the normal portfolio hook; hold cases retain the old path. Review results
and resume the corpus once; do not run a polling watcher or invoke an LLM per case.

For historical failures, group by canonical cause before repair. Keep original
observations; attach scoped repair evidence with `campaign repair begin/record`.
Issue summaries include latest repair receipts without erasing occurrence counts.
Use shard-aware sample-index readers in aggregation/training; never raise general
JSON limits to load a cumulative index. Pass plan.npu_lock to SDK children as
FXBEX_NPU_LOCK so canonical inode checks agree with inherited scheduler leases.
Compile deadline failures get one quiet retry after CPU drain before classifying
support. Finite tiny-VE corpus routing is not a universal ATen runtime replacement.
