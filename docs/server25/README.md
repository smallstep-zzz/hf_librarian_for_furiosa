# Server25 installation and native model smoke — 2026-11-09

Server5 is `ssh furiosa-npu`; it continues its existing librarian improvement work.
Server25 is `ssh -F /Users/dong-jin/Documents/GitHub/furiosa_25/.ssh/config furiosa-npu-25`.
No Server5 NPU resources are used by this smoke.

Installed in /root/.venvs/fury-tools, reusing SDK system packages:
fxbx0.7.0, hf-librarian0.2.0 from this repository, huggingface-hub1.33.0,
Transformers5.22.1, torch2.10.0, furiosa-llm2026.4.0. SDK/driver unchanged.
`pip check`: no broken requirements. Global fxbx/hf-librarian command symlinks
were added only where absent; /root/fury-validation/env.sh selects the environment.

The librarian selected the existing Qwen/Qwen3-8B-FP8 snapshot at immutable revision
220b46e3b2180893580a4454f21f22d3ebb187d3. No new weights were downloaded or deleted.
A real shared-cache issue was fixed: an over-budget preexisting cache cannot block
zero-byte reuse. New downloads still enforce both cache budget and free space.

Using `from fxbx import hug; LLM=hug(furiosa_llm.LLM)` and existing
/root/qwen3-8b-fp8.fxb, two independent model loads passed four generations each.
Latest load7.823s; request times0.050/0.028/0.104/0.047s for2/2/10/4 output tokens.
Arithmetic returned4, exact-word test returnedREADY, multi-turn history retained
Minsoo. These are short functional smokes, not a throughput or long-context benchmark.
Context512 tested. Native process was mapped to npu3 cores0–7; shutdown released
all device ownership. No per-operator native trace API was exposed by the installed
Python LLM interface, so full CPU/NPU operator coverage remains unverified.
This tests the official native-FXB route through fxbX, not custom TCL packages;
furiosa.torch is not installed and was not added/upgraded.

The first supervisor assumed os.pidfd_open existed in the Conda build. The model
still passed and released the device, but no supervisor receipt was produced.
The supervisor now uses Linux libc/syscall compatibility, always records errors,
and kills its child group on monitoring failure. The second real run produced a
completed receipt (9.06s) and zero outstanding NPU processes. Raw logs remain on
Server25 under /root/fury-validation/reports. Small JSON summaries are stored here.

Reproduce on Server25:

```bash
source /root/fury-validation/env.sh
python /root/fury-validation/native_test.py
# Detached and event-driven, with persistent completion/error receipt:
nohup python /root/fury-validation/supervise.py > /root/fury-validation/reports/supervisor.log 2>&1 < /dev/null &
```

Source copies: tools/server25/. Model weights and FXB are external, not in Git.
