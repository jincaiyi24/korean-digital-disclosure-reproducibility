"""Private paired BERT/FinBERT candidate. No training on import or by default.

Requires a separately approved, sealed current-teacher dataset and verified local
pretrained files. Test performance is first computed after dev selection is sealed.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time

LABELS = ("BMODD", "NonBMODD", "Exclude")
LABEL_ID = {name: index for index, name in enumerate(LABELS)}
REQUIRED_VERSIONS = {"torch": "2.7.1+cu126", "transformers": "4.57.1", "tokenizers": "0.22.1",
    "safetensors": "0.6.2", "numpy": "2.2.6", "huggingface-hub": "0.36.2"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty records are not a completed experiment")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def disable_network():
    def denied(*a, **k):
        raise RuntimeError("NETWORK_DISABLED_PRIVATE_PAIRED_BENCHMARK")
    socket.socket.connect = denied; socket.create_connection = denied
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")


def metrics(targets, predicted):
    if len(targets) != len(predicted) or not targets:
        raise ValueError("Aligned nonempty predictions required")
    matrix = [[0] * 3 for _ in LABELS]
    for actual, guess in zip(targets, predicted, strict=True):
        matrix[LABEL_ID[actual]][LABEL_ID[guess]] += 1
    rows = []
    for i, name in enumerate(LABELS):
        support = sum(matrix[i]); n_pred = sum(matrix[j][i] for j in range(3)); tp = matrix[i][i]
        p = tp / n_pred if n_pred else 0.0; r = tp / support if support else 0.0
        f1 = 2 * tp / (support + n_pred) if support + n_pred else 0.0
        rows.append({"class": name, "precision": p, "recall": r, "f1": f1,
            "support": support, "predicted_support": n_pred})
    return {"macro_f1": sum(r["f1"] for r in rows) / 3,
        "agreement_with_current_rules": sum(matrix[i][i] for i in range(3)) / len(targets),
        "class_metrics": rows, "confusion_rows_teacher_columns_student": matrix,
        "target_type": "current_rule_pseudo_label_NOT_human_gold"}


def accumulation_group_samples(train_n, batch_size, accumulation, step):
    start = (step // accumulation) * accumulation
    size = min(train_n - start * batch_size, accumulation * batch_size)
    if size <= 0:
        raise ValueError("Invalid accumulation step")
    return size


def dev_improves(score, best_score):
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Invalid dev Macro-F1")
    return score > best_score


def protected_unchanged(protected):
    try:
        return all(Path(path).is_file() and sha(path) == h for path, h in protected.items())
    except (OSError, ValueError):
        return False


def valid_probabilities(logits, expected_rows, torch):
    if logits.ndim != 2 or tuple(logits.shape) != (expected_rows, 3) or not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite or misaligned model logits")
    p = torch.softmax(logits, dim=1)
    if (not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not torch.all(torch.abs(p.sum(dim=1) - 1) <= 1e-5)):
        raise FloatingPointError("Invalid model probability matrix")
    return p.cpu().tolist()


def begin_fixed_test(output, best_dir):
    seal_path = output / "BEST_MODEL_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (seal["status"] != "DEV_SELECTED_BEST_CHECKPOINT_SEALED_BEFORE_TEST"
            or seal["epochs_completed"] != 3 or seal["test_evaluations"] != 0
            or seal["best_epoch"] not in (1, 2, 3)
            or Path(seal["checkpoint_directory"]).resolve() != best_dir.resolve()):
        raise ValueError("Test access requires completed, sealed dev selection")
    if not best_dir.resolve().is_relative_to(output.resolve()):
        raise ValueError("Best checkpoint escaped its own model run")
    files = seal["checkpoint_files"]
    if "model.safetensors" not in files:
        raise ValueError("Saved real selected model required")
    for name, expected in files.items():
        p = (best_dir / name).resolve()
        if not p.is_relative_to(best_dir.resolve()) or sha(p) != expected:
            raise ValueError("Best checkpoint bytes changed before test")
    dump(output / "TEST_EVALUATION_STARTED.json", {"status": "ONE_FIXED_TEST_PASS_STARTED",
        "best_model_seal_sha256": sha(seal_path), "utc": timestamp(),
        "test_based_model_selection": False})


def validate(args):
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    contract = json.loads(args.training_contract.read_text(encoding="utf-8"))
    policy = json.loads(args.head_policy.read_text(encoding="utf-8"))
    seal_path = args.dataset / "DATASET_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["status"] != "CURRENT_RULE_2700_DATASET_SEALED" or seal["rows"] != 2700 or seal["classifier_calls"] != 2700:
        raise ValueError("Completed real current-teacher dataset required")
    if seal["protocol_sha256"] != sha(args.protocol) or not seal["protected_unchanged"]:
        raise ValueError("Dataset protocol/closure differs")
    if (contract["teacher_protocol_sha256"] != sha(args.protocol)
            or contract["dataset_seal_sha256"] != sha(seal_path)
            or contract["teacher_replay_receipt_sha256"] != sha(args.dataset / "CURRENT_TEACHER_REPLAY_RECEIPT.json")):
        raise ValueError("Actual teacher replay is not bound to the training contract")
    if any(policy[k] != contract[k] for k in ("teacher_protocol_sha256", "dataset_seal_sha256", "teacher_replay_receipt_sha256")):
        raise ValueError("Common-head policy is not bound to the actual teacher dataset")
    budget = contract["training_budget"]
    if budget["seeds"] != protocol["training"]["training_seeds"] or any(
            budget[k] != protocol["training"][k] for k in budget if k != "seeds"):
        raise ValueError("Paired training budget changed")
    protected = {str(args.protocol.resolve()): sha(args.protocol), str(seal_path.resolve()): sha(seal_path),
        str(args.models.resolve()): sha(args.models), str(Path(__file__).resolve()): sha(__file__),
        str(args.training_contract.resolve()): sha(args.training_contract),
        str(args.head_policy.resolve()): sha(args.head_policy),
        str((args.dataset / "CURRENT_TEACHER_REPLAY_RECEIPT.json").resolve()): contract["teacher_replay_receipt_sha256"]}
    for name, h in seal["files"].items():
        p = (args.dataset / name).resolve()
        if not p.is_relative_to(args.dataset.resolve()) or sha(p) != h:
            raise ValueError("Sealed teacher data changed")
        protected[str(p)] = h
    spec = json.loads(args.models.read_text(encoding="utf-8"))
    if set(spec["models"]) != {"bert", "finbert"}:
        raise ValueError("Exactly the paired two models required")
    for role, model in spec["models"].items():
        if any(model[k] != protocol["models"][role][k] for k in ("model_id", "revision")):
            raise ValueError("Pinned model identity differs")
        root = Path(model["model_dir"]).resolve()
        provenance = Path(model["provenance_receipt"]).resolve()
        if sha(provenance) != model["provenance_receipt_sha256"]:
            raise ValueError("Verified model acquisition receipt changed")
        protected[str(provenance)] = sha(provenance)
        files = model["verified_files"]
        if len({r["file"] for r in files}) != len(files) or "model.safetensors" not in {r["file"] for r in files}:
            raise ValueError("Unique verified safetensors model artifacts required")
        for row in files:
            relative = Path(row["file"])
            p = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not p.is_relative_to(root):
                raise ValueError("Model artifact path escaped")
            if p.stat().st_size != row["size"] or sha(p) != row["sha256"]:
                raise ValueError("Verified model bytes changed")
            protected[str(p)] = row["sha256"]
    versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
    if versions != REQUIRED_VERSIONS or sys.version_info[:3] != (3, 12, 14):
        raise ValueError("Use the separately verified private benchmark Python environment")
    return protocol, spec, protected


def load_split(dataset, split, required_n):
    rows = read_csv(dataset / (split + "_LOCAL_ONLY.csv"))
    if len(rows) != required_n or len({r["sentence_id"] for r in rows}) != required_n or any(r["split"] != split for r in rows):
        raise ValueError("Fixed split identity/size changed")
    if {r["label"] for r in rows} != set(LABELS):
        raise ValueError("Missing current class; do not rebalance")
    for row in rows:
        if hashlib.sha256(row["input_text"].encode("utf-8")).hexdigest() != row["input_text_sha256"]:
            raise ValueError("Metadata-plus-original-sentence input changed")
    return rows


def state_hash(module, torch, excluded_prefixes=()):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_source_encoder(model, model_dir, loading, torch):
    """Verify actual loaded tensor values, not only a before/after reset hash."""
    from safetensors import safe_open
    state = model.base_model.state_dict()
    buffers = dict(model.base_model.named_buffers())
    checks = []; expected_missing = {"classifier.weight", "classifier.bias"}
    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name, value in state.items():
            key = "bert." + name
            if name.startswith("pooler.") and key not in keys:
                expected_missing.add(key)
                continue
            if key not in keys:
                raise ValueError("Pinned source is missing a real loaded encoder tensor")
            expected = source.get_tensor(key)
            actual = value.detach().cpu()
            if (expected.shape != actual.shape or expected.dtype != actual.dtype
                    or not torch.equal(expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8))):
                raise ValueError("Loaded embedding/encoder/present-pooler differs from pinned safetensors")
            checks.append(key)
        for key in keys:
            if not key.startswith("bert.") or key in checks:
                continue
            name = key[len("bert."):]
            if name != "embeddings.position_ids" or name not in buffers:
                raise ValueError("Unexplained pretrained encoder or buffer key")
            expected = source.get_tensor(key); actual = buffers[name].detach().cpu()
            if (expected.shape != actual.shape or expected.dtype != actual.dtype
                    or not torch.equal(expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8))):
                raise ValueError("Nonpersistent position buffer differs from pinned source")
            checks.append(key)
        allowed_unused = {"cls.predictions.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
            "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias",
            "cls.seq_relationship.weight", "cls.seq_relationship.bias", "bert.embeddings.position_ids"}
        if (set(loading["missing_keys"]) != expected_missing
                or set(loading["unexpected_keys"]) - allowed_unused
                or loading.get("mismatched_keys") or loading.get("error_msgs")):
            raise ValueError("Unexpected pretrained missing/unused/mismatched keys")
        return {"all_loaded_embedding_encoder_and_source_present_pooler_tensors_exact": True,
            "source_bert_tensor_keys_verified": checks, "source_bert_tensor_count": len(checks),
            "source_has_pooler": "bert.pooler.dense.weight" in keys,
            "nonpersistent_position_buffer_verified": "bert.embeddings.position_ids" in checks,
            "loading_info": loading}


def initialize_common_heads(model, seed, torch, expected_initial_heads=None):
    if not isinstance(model.classifier, torch.nn.Linear) or model.classifier.out_features != 3:
        raise ValueError("Explicit new three-class linear head required")
    before = state_hash(model.base_model, torch, excluded_prefixes=("pooler.",))
    torch.manual_seed(seed)
    model._init_weights(model.bert.pooler.dense)
    model._init_weights(model.classifier)
    after = state_hash(model.base_model, torch, excluded_prefixes=("pooler.",))
    if before != after:
        raise AssertionError("Common head initialization changed pretrained backbone")
    heads = {"pooler": state_hash(model.bert.pooler, torch), "classifier": state_hash(model.classifier, torch)}
    if expected_initial_heads is not None and expected_initial_heads != heads:
        raise ValueError("Same-seed pair has different new pooler/classifier initialization")
    return before, heads


def token_audit(rows, tokenizer, max_length):
    if not tokenizer.is_fast:
        raise ValueError("Fast native tokenizer offsets required for real retained-body audit")
    counts = []; prefix_counts = []; body_counts = []; unknown = 0; tokens = 0
    retained_body = []; truncated_body = []; records = []
    for row in rows:
        prefix = f"기업명: {row['corp_name']}\n보고연도: {row['year']}\n문장: "
        if row["input_text"] != prefix + row["sentence"]:
            raise ValueError("Exact agreed metadata input format required")
        full = tokenizer(row["input_text"], truncation=False, return_offsets_mapping=True, return_special_tokens_mask=True)
        clipped = tokenizer(row["input_text"], truncation=True, max_length=max_length,
            return_offsets_mapping=True, return_special_tokens_mask=True)
        ids = full["input_ids"]
        body_start = len(prefix)
        full_body = sum(not special and end > body_start and end > start
            for (start, end), special in zip(full["offset_mapping"], full["special_tokens_mask"], strict=True))
        kept_body = sum(not special and end > body_start and end > start
            for (start, end), special in zip(clipped["offset_mapping"], clipped["special_tokens_mask"], strict=True))
        full_prefix = sum(not special and 0 < end <= body_start
            for (start, end), special in zip(full["offset_mapping"], full["special_tokens_mask"], strict=True))
        kept_prefix = sum(not special and 0 < end <= body_start
            for (start, end), special in zip(clipped["offset_mapping"], clipped["special_tokens_mask"], strict=True))
        if not full_body or not kept_body:
            raise ValueError("Prefix/truncation leaves no sentence-body token; do not train")
        retained_body.append(kept_body); truncated_body.append(full_body - kept_body)
        records.append({"sentence_id": row["sentence_id"], "split": row["split"],
            "input_text_sha256": row["input_text_sha256"], "full_token_count": len(ids),
            "retained_token_count": len(clipped["input_ids"]), "full_prefix_tokens": full_prefix,
            "retained_prefix_tokens": kept_prefix, "full_body_tokens": full_body,
            "retained_body_tokens": kept_body, "removed_body_tokens": full_body - kept_body,
            "full_special_tokens": sum(full["special_tokens_mask"]),
            "retained_special_tokens": sum(clipped["special_tokens_mask"])})
        counts.append(len(ids)); prefix_counts.append(len(tokenizer(prefix, truncation=False)["input_ids"]))
        body_counts.append(len(tokenizer(row["sentence"], truncation=False)["input_ids"]))
        unknown += sum(i == tokenizer.unk_token_id for i in ids); tokens += len(ids)
    return {"rows": len(rows), "max_length": max_length, "over_max_length": sum(x > max_length for x in counts),
        "full_tokens_min_max": [min(counts), max(counts)], "prefix_tokens_min_max": [min(prefix_counts), max(prefix_counts)],
        "body_tokens_min_max": [min(body_counts), max(body_counts)], "unknown_tokens": unknown,
        "all_untruncated_tokens": tokens, "unknown_token_fraction": unknown / tokens,
        "retained_body_tokens_min_max": [min(retained_body), max(retained_body)],
        "body_tokens_removed_by_truncation": sum(truncated_body),
        "rows_with_body_token_truncation": sum(x > 0 for x in truncated_body),
        "zero_body_rows": 0,
        "teacher_keeps_full_sentence": True, "prefix_and_special_tokens_consume_student_budget": True}, records


def save_token_audit(directory, name, rows, tokenizer, max_length):
    summary, records = token_audit(rows, tokenizer, max_length)
    dump(directory / (name + "_TOKENIZATION_AUDIT.json"), summary)
    write_csv(directory / (name + "_TOKENIZATION_PER_SENTENCE.csv"), records)
    return summary


def run_one(role, model_info, seed, args, protocol, output, expected_initial_heads=None):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Verified CUDA required; no silent CPU alternative")
    hp = protocol["training"]; model_dir = model_info["model_dir"]
    train = load_split(args.dataset, "train", 1500); dev = load_split(args.dataset, "dev", 300)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if cfg.model_type != "bert":
        raise ValueError("This paired harness is specifically for the two approved BERT architectures")
    cfg.num_labels = 3; cfg.id2label = dict(enumerate(LABELS)); cfg.label2id = LABEL_ID; cfg.classifier_dropout = 0.1
    model, loading = AutoModelForSequenceClassification.from_pretrained(model_dir, config=cfg,
        use_safetensors=True, local_files_only=True, trust_remote_code=False,
        output_loading_info=True)
    encoder_load = verify_source_encoder(model, model_dir, loading, torch)
    encoder_before, initial_heads = initialize_common_heads(model, seed, torch, expected_initial_heads)
    output.mkdir(parents=True, exist_ok=False); started = timestamp()
    dump(output / "START_RECEIPT.json", {"role": role, "model_id": model_info["model_id"],
        "revision": model_info["revision"], "seed": seed, "utc": started,
        "encoder_state_sha256_before_and_after_head_reset": encoder_before,
        "new_pooler_and_classifier_state_sha256": initial_heads, "sentiment_head_reused": False,
        "pretrained_loading_info": loading,
        "pretrained_encoder_source_verification": encoder_load,
        "pretrained_encoder_hash_excludes_common_new_downstream_pooler": True,
        "source_pooler_intentionally_discarded": encoder_load["source_has_pooler"],
        "gpu": torch.cuda.get_device_name(0), "training": hp, "test_model_evaluations_so_far": 0})
    class Sentences(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            r = self.rows[index]
            value = tokenizer(r["input_text"], max_length=hp["max_length"], padding="max_length", truncation=True, return_tensors="pt")
            return {**{k: v.squeeze(0) for k, v in value.items()}, "labels": torch.tensor(LABEL_ID[r["label"]])}
    def evaluate(rows, model):
        probabilities = []; model.eval()
        with torch.inference_mode():
            for batch in DataLoader(Sentences(rows), batch_size=hp["batch_size"], shuffle=False, num_workers=0):
                inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits
                probabilities.extend(valid_probabilities(logits, next(iter(inputs.values())).shape[0], torch))
        if len(probabilities) != len(rows):
            raise ValueError("Exact evaluation row count required")
        predicted = [LABELS[max(range(3), key=lambda j: p[j])] for p in probabilities]
        records = [{"sentence_id": r["sentence_id"], "corp_code": r["corp_code"], "year": r["year"],
            "rcept_no": r["rcept_no"], "sentence_sha256": r["sentence_sha256"],
            "input_text_sha256": r["input_text_sha256"], "rule_pseudo_label": r["label"],
            "prediction": guess, **{f"probability_{name}": p[j] for j, name in enumerate(LABELS)}}
            for r, p, guess in zip(rows, probabilities, predicted, strict=True)]
        return metrics([r["label"] for r in rows], predicted), records
    save_token_audit(output, "TRAIN", train, tokenizer, hp["max_length"])
    save_token_audit(output, "DEV", dev, tokenizer, hp["max_length"])
    model.to("cuda"); torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Sentences(train), batch_size=hp["batch_size"], shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    updates = math.ceil(len(loader) / hp["gradient_accumulation"]) * hp["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * hp["warmup_fraction"]), updates)
    logs = []; best_score = -1.0; best_dir = None; best_epoch = None
    for epoch in range(1, hp["epochs"] + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0; seen = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
            group_n = accumulation_group_samples(len(train), hp["batch_size"], hp["gradient_accumulation"], step)
            n = batch["labels"].shape[0]; (loss * n / group_n).backward()
            loss_sum += float(loss.detach().cpu()) * n; seen += n
            if (step + 1) % hp["gradient_accumulation"] == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_gradient_norm"], error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        dev_metrics, dev_predictions = evaluate(dev, model)
        write_csv(output / f"dev_predictions_epoch_{epoch:02d}.csv", dev_predictions)
        dump(output / f"dev_metrics_epoch_{epoch:02d}.json", dev_metrics)
        improved = dev_improves(dev_metrics["macro_f1"], best_score)
        if improved:
            best_score = dev_metrics["macro_f1"]; best_epoch = epoch
            best_dir = output / f"checkpoint_epoch_{epoch:02d}"
            model.save_pretrained(best_dir, safe_serialization=True); tokenizer.save_pretrained(best_dir)
        logs.append({"epoch": epoch, "training_examples": seen, "train_loss": loss_sum / seen,
            "dev_macro_f1": dev_metrics["macro_f1"], "selected_as_best_so_far": improved,
            "best_epoch_so_far": best_epoch, "test_evaluations": 0})
        print(json.dumps({"role": role, "seed": seed, **logs[-1]}), flush=True)
    write_csv(output / "training_log.csv", logs)
    checkpoint = {str(p.relative_to(best_dir)): sha(p) for p in best_dir.rglob("*") if p.is_file()}
    best_seal = {"status": "DEV_SELECTED_BEST_CHECKPOINT_SEALED_BEFORE_TEST", "best_epoch": best_epoch,
        "dev_macro_f1": best_score, "tie_policy": "earliest_epoch", "epochs_completed": hp["epochs"],
        "checkpoint_directory": str(best_dir.resolve()), "checkpoint_files": checkpoint, "test_evaluations": 0}
    dump(output / "BEST_MODEL_SEAL.json", best_seal)
    del model, optimizer, scheduler; torch.cuda.empty_cache()
    begin_fixed_test(output, best_dir)
    # The fixed test file is first parsed here, after dev selection is sealed.
    test = load_split(args.dataset, "test", 900)
    save_token_audit(output, "TEST", test, tokenizer, hp["max_length"])
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, use_safetensors=True,
        local_files_only=True, trust_remote_code=False).to("cuda")
    if model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError("Saved checkpoint labels changed")
    test_metrics, test_predictions = evaluate(test, model)
    write_csv(output / "test_predictions.csv", test_predictions); dump(output / "test_metrics.json", test_metrics)
    if any(sha(best_dir / name) != h for name, h in checkpoint.items()):
        raise ValueError("Selected checkpoint changed during fixed test evaluation")
    receipt = {"status": "PAIRED_MODEL_SEED_COMPLETE", "role": role, "model_id": model_info["model_id"],
        "revision": model_info["revision"], "seed": seed, "utc_started": started, "utc_finished": timestamp(),
        "best_epoch": best_epoch, "dev_macro_f1": best_score, "test_evaluation_passes": 1,
        "test_rows": len(test), "test_metrics": test_metrics, "best_model_seal_sha256": sha(output / "BEST_MODEL_SEAL.json"),
        "trained_model_state_sha256": state_hash(model, torch),
        "new_pooler_and_classifier_state_sha256": initial_heads,
        "pretrained_encoder_source_verification": encoder_load,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "old_trained_state_reused": False,
        "checkpoint_files": checkpoint, "network_requests": 0}
    dump(output / "MODEL_SEED_RECEIPT.json", receipt)
    del model; torch.cuda.empty_cache()
    return receipt


def main():
    disable_network()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path(__file__).with_name("PROTOCOL.json"))
    p.add_argument("--training-contract", type=Path, default=Path(__file__).with_name("TRAINING_EXECUTION_CONTRACT.json"))
    p.add_argument("--head-policy", type=Path, default=Path(__file__).with_name("HEAD_POLICY_SUPPLEMENT.json"))
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--models", type=Path, required=True)
    p.add_argument("--mode", choices=("preflight", "tokenizer-preflight", "train"), default="preflight")
    p.add_argument("--output", type=Path); p.add_argument("--approval", type=Path)
    p.add_argument("--tokenizer-preflight-receipt", type=Path)
    args = p.parse_args(); protocol, spec, protected = validate(args)
    if args.mode == "preflight":
        print(json.dumps({"status": "PAIRED_TRAINING_INPUT_PREFLIGHT_PASS_NOT_TRAINED", "protected": protected,
            "model_fits": 0, "network_requests": 0})); return 0
    if args.mode == "tokenizer-preflight":
        if not args.output or args.output.exists():
            raise ValueError("Tokenizer-only preflight also needs a fresh output")
        from transformers import AutoTokenizer
        args.output.mkdir(parents=True, exist_ok=False)
        summaries = {}
        for role, model in spec["models"].items():
            tokenizer = AutoTokenizer.from_pretrained(model["model_dir"], local_files_only=True, trust_remote_code=False)
            for split, n in (("train", 1500), ("dev", 300), ("test", 900)):
                rows = load_split(args.dataset, split, n)
                summaries[role + "/" + split] = save_token_audit(args.output, role.upper() + "_" + split.upper(),
                    rows, tokenizer, protocol["training"]["max_length"])
        if not protected_unchanged(protected):
            raise ValueError("Tokenizer-only preflight changed inputs")
        dump(args.output / "TOKENIZER_PREFLIGHT_RECEIPT.json", {"status": "BOTH_NATIVE_TOKENIZERS_ALL2700_BODY_RETENTION_PASS",
            "script_sha256": sha(__file__),
            "models_spec_sha256": sha(args.models), "dataset_seal_sha256": sha(args.dataset / "DATASET_SEAL.json"),
            "protocol_sha256": sha(args.protocol), "training_contract_sha256": sha(args.training_contract),
            "head_policy_sha256": sha(args.head_policy), "summaries": summaries,
            "rows_per_model": 2700, "model_forwards": 0, "training_runs": 0,
            "test_tokenizer_features_checked_not_test_model_performance": True,
            "files": {str(p.relative_to(args.output)): sha(p) for p in args.output.iterdir() if p.is_file()},
            "protected_unchanged": True, "network_requests": 0})
        return 0
    if not args.output or not args.approval or args.output.exists():
        raise ValueError("Training requires fresh output and exact ROOT approval")
    if not args.tokenizer_preflight_receipt:
        raise ValueError("Completed whole-cohort native-tokenizer body-retention preflight required")
    token_check = json.loads(args.tokenizer_preflight_receipt.read_text(encoding="utf-8"))
    token_expected = {"status": "BOTH_NATIVE_TOKENIZERS_ALL2700_BODY_RETENTION_PASS", "script_sha256": sha(__file__),
        "models_spec_sha256": sha(args.models), "dataset_seal_sha256": sha(args.dataset / "DATASET_SEAL.json"),
        "protocol_sha256": sha(args.protocol), "training_contract_sha256": sha(args.training_contract),
        "head_policy_sha256": sha(args.head_policy), "rows_per_model": 2700,
        "model_forwards": 0, "training_runs": 0, "protected_unchanged": True, "network_requests": 0}
    if any(token_check.get(k) != v for k, v in token_expected.items()):
        raise ValueError("Tokenizer-only preflight does not bind this exact training candidate")
    protected[str(args.tokenizer_preflight_receipt.resolve())] = sha(args.tokenizer_preflight_receipt)
    expected = {"script_sha256": sha(__file__), "protocol_sha256": sha(args.protocol),
        "training_contract_sha256": sha(args.training_contract),
        "head_policy_sha256": sha(args.head_policy),
        "tokenizer_preflight_receipt_sha256": sha(args.tokenizer_preflight_receipt),
        "dataset_seal_sha256": sha(args.dataset / "DATASET_SEAL.json"), "models_spec_sha256": sha(args.models),
        "output": str(args.output.resolve()), "seeds": protocol["training"]["training_seeds"],
        "allow_eight_new_paired_model_runs": True, "local_only_no_publication": True}
    approval = json.loads(args.approval.read_text(encoding="utf-8-sig"))
    if any(approval.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact training approval differs")
    protected[str(args.approval.resolve())] = sha(args.approval)
    args.output.mkdir(parents=True, exist_ok=False); started = timestamp()
    dump(args.output / "START_RECEIPT.json", {**expected, "utc": started, "protected": protected})
    completed = []; active = None
    try:
        for seed in protocol["training"]["training_seeds"]:
            paired_heads = None
            for role in ("bert", "finbert"):
                active = {"role": role, "seed": seed}
                completed.append(run_one(role, spec["models"][role], seed, args, protocol,
                    args.output / role / f"seed_{seed}", expected_initial_heads=paired_heads))
                paired_heads = completed[-1]["new_pooler_and_classifier_state_sha256"]
                if not protected_unchanged(protected):
                    raise ValueError("Immutable input changed; stop subsequent models")
        dump(args.output / "PAIRED_EXECUTION_RECEIPT.json", {"status": "ALL_EIGHT_PAIRED_MODEL_RUNS_COMPLETE",
            "utc_started": started, "utc_finished": timestamp(), "models_completed": len(completed),
            "runs": completed, "protected_unchanged": True, "network_requests": 0,
            "not_human_gold_accuracy": True, "not_current_corpus_random_sample": True,
            "manuscript_or_published_repository_modified": False})
        return 0
    except BaseException as error:
        dump(args.output / "FAILED_PAIRED_EXECUTION_RECEIPT.json", {"status": "FAILED_STOP_REMAINING_MODELS",
            "error_type": type(error).__name__, "active_run": active,
            "completed_runs": [{"role": r["role"], "seed": r["seed"]} for r in completed],
            "all_eight_complete": False, "utc": timestamp(), "network_requests": 0,
            "protected_unchanged": protected_unchanged(protected)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
