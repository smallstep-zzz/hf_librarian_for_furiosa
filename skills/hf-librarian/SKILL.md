---
name: hf-librarian
description: Research, classify, and maintain Hugging Face test models for Furiosa compiler and NPU coverage using hf-librarian. Use when asked to add model coverage, prefetch an announced model, or find models by task and test properties.
---

# Hugging Face model librarian

Use the installed `hf-librarian` command. Do not edit catalog JSON directly. `hf-librarian categories`, `list --category CATEGORY`, `settings`, `status`, and `blacklist` show current state. Model IDs are exact Hugging Face IDs.

When asked to research models, check the current Hugging Face model card and file list. Choose varied architectures and a representative range of size, quantization, novelty, rare operations, and paper implementations. Record the source URL and reason for selection. Verify the license and gated status before recommending use. The catalog records candidates, not proven NPU support; set `npu_status` to `untested` until a real compiler/runtime test proves otherwise. Estimated file size is not runtime memory. Check selected files, loader requirements, and Hub sizes; do not select multiple redundant weight formats without reason.

Add a category with `hf-librarian add-category NAME DESCRIPTION`. Add a model with `hf-librarian add-model @model.json`, where the input is a small object such as:

```json
{"id":"org/model", "categories":["language-generation"], "revision":"main", "include":["*.json","*.safetensors"], "exclude":[], "estimated_gib":1.0, "quantization":"bf16", "difficulty":"medium", "rarity":"common", "paper_implementation":false, "npu_status":"untested", "priority":50, "notes":"why this model", "source":"https://huggingface.co/org/model"}
```

`hf-librarian remove-model ID`, `block ID --reason REASON`, and `unblock ID` manage membership and exclusions. Removing or blocking does not delete cached files. `hf-librarian get --id ID` downloads a known model and prints its snapshot path; `get --category NAME` selects the highest-priority eligible model. `prefetch --category NAME --category NAME` downloads in parallel. Follow an explicit request to prefetch an announced future test model after adding it. Do not fill the cache merely to reach its limit. Respect the configured limit, free-space reserve, gated model access, and blacklist; report failures accurately. Use Python API examples in the project README for integration.
