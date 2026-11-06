#!/usr/bin/env python3
"""Separate native-encoder, cached OFF-head, and new shared-PG CPU timings.

The native model is loaded once and always runs without an adapter. Timing
inputs are the first32 predeclared old TRAIN rows, never fresh holdout rows.
The PG measurement creates independent new readout-only64-update attempts; it
does not resume or change any measured scientific candidate. No GPU is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import distribution
import json
import math
from pathlib import Path
import resource
import statistics
import time

import torch

from glide_particle_sliders import binding as original
from glide_particle_sliders import binding_normalized_v3 as normalized
from glide_particle_sliders.native import load_glide, sha256
from particle_sliders.inference import open_inference


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def latencies(seconds):
    if not seconds or any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("positive finite timing observations required")
    ordered = sorted(seconds)
    return dict(observations=len(seconds), mean_ms=statistics.mean(seconds) * 1000,
                p50_ms=statistics.median(seconds) * 1000,
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1] * 1000)


def padded_tokens(lengths, budget):
    """Replicate native text-only length-sorted padded-token batching counts."""
    order = sorted(lengths)
    result, start = 0, 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and (stop - start + 1) * order[stop] <= budget:
            stop += 1
        result += (stop - start) * order[stop - 1]
        start = stop
    return result


def source_receipt():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "pyproject.toml", root / "native-model.lock.json",
             *sorted((root / "src/glide_particle_sliders").glob("*.py"))]
    packages = {}
    for name in ("torch", "transformers", "particle-sliders-core", "particlegan", "hypergan", "hndl"):
        dist = distribution(name)
        packages[name] = dict(version=dist.version,
            direct_url=json.loads(dist.read_text("direct_url.json") or "{}"))
    return dict(files={str(path.relative_to(root)): sha256(path) for path in paths}, packages=packages)


def run(model_dir, lock_file, cache_file, output, *, threads=4, pg_batches=(8,), head_repeats=200,
        adapter_file=None, readout_file=None):
    if threads != 4 or not pg_batches or any(type(batch) is not int or not 2 <= batch <= 512 for batch in pg_batches):
        raise ValueError("this bounded benchmark requires4 CPU threads and explicit valid PG batch sizes")
    if len(set(pg_batches)) != len(pg_batches) or head_repeats < 20:
        raise ValueError("unique PG batch sizes and at least20 head timing repeats required")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    cache_file = Path(cache_file).resolve()
    cache = original.load_readout_file(cache_file, cache=True)
    ids = [index for index, split in enumerate(cache["splits"]) if split == "train"][:32]
    if len(ids) != 32:
        raise ValueError("benchmark requires32 predeclared old TRAIN rows")
    rows = [cache["metadata"]["dataset"][index] for index in ids]
    if any(row["split"] != "train" for row in rows):
        raise ValueError("cache metadata contradicts declared TRAIN selection")
    requests = [row["neutral"] for row in rows]
    sources = source_receipt()
    result = dict(schema="glide-cpu-throughput-v1", started=now(), source=sources,
        cache_sha256=sha256(cache_file), selected_row_ids=[cache["row_ids"][index] for index in ids],
        selection="first32 old TRAIN rows in immutable cache order; no quality-dependent selection",
        resource=dict(device="cpu", torch_threads=threads, gpu_launches=0),
        native_model_loaded_once=True, native_adapter_strength=0., status="running",
        native=[], cached_off_head=[], cached_active_adapter=[], pg_attempts=[])
    write_json(root / "progress.json", result)
    started = time.perf_counter()
    model, identity = load_glide(model_dir, lock_file, threads=threads)
    result["native_model_load_seconds"] = time.perf_counter() - started
    result["native_model_identity"] = identity
    print(json.dumps(dict(event="native_model_loaded", seconds=result["native_model_load_seconds"])), flush=True)
    if identity != cache["model_identity"] or not torch.equal(model.network.readout.weight, cache["readout_weight"]):
        raise ValueError("cached features and loaded frozen native model differ")
    if model.network.readout.forward.__func__ is not torch.nn.Linear.forward:
        raise ValueError("the native throughput model must have an unwrapped OFF readout")
    readout_before = model.network.readout.weight.detach().clone()
    texts = []
    for request in requests:
        entries, visual = model._prepare(request)
        if visual or len(entries) != 1 or len(entries[0][2]) != 2:
            raise ValueError("benchmark requires text-only single two-choice requests")
        texts.append(entries[0][3])
    lengths = [len(tokens) for tokens in model.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    result["native_input_tokens"] = dict(min=min(lengths), max=max(lengths), total=sum(lengths),
                                        token_budget=model.token_budget)
    for batch in (1, 2, 4, 8):
        # One excluded API warmup call per arm. Its requests are reported
        # separately and never counted toward timed throughput.
        warmup = model.decide_batch(requests[:batch])
        if any(not isinstance(response, dict) for response in warmup):
            raise RuntimeError("native warmup failed")
        durations, padded = [], 0
        started = time.perf_counter()
        for start in range(0, len(requests), batch):
            chunk = requests[start:start + batch]
            began = time.perf_counter()
            responses = model.decide_batch(chunk)
            durations.append(time.perf_counter() - began)
            if len(responses) != len(chunk) or any(not isinstance(response, dict) for response in responses):
                raise RuntimeError("native timed API call failed")
            padded += padded_tokens(lengths[start:start + batch], model.token_budget)
        elapsed = time.perf_counter() - started
        record = dict(request_batch_size=batch, timed_requests=len(requests), timed_api_calls=len(durations),
            warmup_requests_excluded=batch, seconds=elapsed, requests_per_second=len(requests) / elapsed,
            native_input_tokens=sum(lengths), padded_tokens=padded, padded_tokens_per_second=padded / elapsed,
            api_call_latency=latencies(durations), includes="prompt rendering, tokenization, native encoder, frozen OFF readout, answer postprocessing")
        result["native"].append(record)
        write_json(root / "progress.json", result)
        print(json.dumps(dict(event="native_throughput", batch=batch, requests_per_second=record["requests_per_second"])), flush=True)
    # These genuine frozen native encoder features were captured in the existing
    # old TRAIN cache. This is head-only timing, with no encoder/tokenizer work.
    hidden = cache["neutral_hidden"][ids]
    with torch.inference_mode():
        for batch in (1, 8, 32):
            value = hidden[:batch]
            for _ in range(20):
                torch.nn.functional.linear(value, model.network.readout.weight)[:, :2]
            durations = []
            for _ in range(head_repeats):
                began = time.perf_counter()
                torch.nn.functional.linear(value, model.network.readout.weight)[:, :2]
                durations.append(time.perf_counter() - began)
            result["cached_off_head"].append(dict(batch_size=batch, warmup_forwards_excluded=20,
                timed_forwards=head_repeats, rows_per_second=batch * head_repeats / sum(durations),
                latency=latencies(durations), includes="native frozen Linear1024-to-255 readout and first2 output view",
                excludes="encoder, tokenization, adapter, probability postprocessing"))
    if not torch.equal(model.network.readout.weight, readout_before) or any(p.requires_grad for p in model.network.parameters()):
        raise RuntimeError("the frozen native benchmark model changed")
    result["native_readout_unchanged"] = True
    if adapter_file is not None:
        if readout_file is None:
            raise ValueError("the active cached adapter requires its exact frozen readout_file")
        result["active_adapter_sha256"] = sha256(adapter_file)
        result["active_readout_sha256"] = sha256(readout_file)
        with open_inference(adapter_file, backend=normalized.INFERENCE, operation="response",
                            options=dict(readout_file=str(Path(readout_file).resolve()))) as inference:
            for batch in (1, 8, 32):
                value = hidden[:batch]
                for _ in range(20):
                    inference.response(hidden=value, strength=1.)
                durations = []
                for _ in range(head_repeats):
                    began = time.perf_counter()
                    inference.response(hidden=value, strength=1.)
                    durations.append(time.perf_counter() - began)
                result["cached_active_adapter"].append(dict(batch_size=batch, warmup_forwards_excluded=20,
                    timed_forwards=head_repeats, rows_per_second=batch * head_repeats / sum(durations),
                    latency=latencies(durations), strength=1.,
                    includes="shared native-only inference API, normalized HNDL routed adapter, native readout, observation and detach",
                    excludes="encoder, tokenization, probability postprocessing"))
    result["native_peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(root / "progress.json", result)
    for batch in pg_batches:
        attempt = root / f"pg-batch-{batch}"
        attempt.mkdir()
        cfg = normalized.config(cache_file=str(cache_file), steps=64, batch=batch, seed=7,
            lr=.001, d_lr_mult=18., prior_lr_mult=30., ema_decay=None, device="cpu", critic="residual64")
        record = dict(batch_size=batch, common_schedule_horizon=64, committed_updates=0,
                      config=cfg.as_dict(), extra_training_warmup_updates=0)
        started = time.perf_counter()
        with normalized.BACKEND.open(cfg) as session:
            record["prepare_seconds"] = time.perf_counter() - started
            intervals, updates = [], []
            previous = time.perf_counter()
            def on_step(stats):
                nonlocal previous
                observed = time.perf_counter()
                intervals.append(observed - previous)
                previous = observed
                updates.append(stats)
            started = time.perf_counter()
            session.train(on_step=on_step)
            elapsed = time.perf_counter() - started
            if session.game.completed_steps != 64 or session.ema.decay is not None or session.ema.values is not None:
                raise RuntimeError("PG benchmark budget or no-EMA contract differs")
            session.checkpoint(attempt / "step-64.pt")
            session.export_inference(attempt / "raw-inference.pt", selection="raw")
            record.update(committed_updates=64, physical_updates=64, replayed_prefix=0,
                timed_updates=64, seconds=elapsed, updates_per_second=64 / elapsed,
                example_exposures_per_second=64 * batch / elapsed, update_latency=latencies(intervals),
                steady_steps9_to64_latency=latencies(intervals[8:]), weight_ema=False,
                g_row_exposures=int(session.game.exposures["g"].sum()),
                d_row_exposures=int(session.game.exposures["d"].sum()),
                checkpoint_sha256=sha256(attempt / "step-64.pt"),
                raw_export_sha256=sha256(attempt / "raw-inference.pt"))
        with (attempt / "updates.jsonl").open("w") as file:
            for stats in updates:
                file.write(json.dumps(stats, allow_nan=False) + "\n")
        result["pg_attempts"].append(record)
        write_json(root / "progress.json", result)
        print(json.dumps(dict(event="shared_pg_throughput", batch=batch, updates_per_second=record["updates_per_second"])), flush=True)
    if source_receipt() != sources:
        raise RuntimeError("benchmark source/runtime identity changed")
    result.update(status="completed", finished=now(),
        total_new_pg_unique_updates=64 * len(pg_batches), total_new_pg_physical_updates=64 * len(pg_batches),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=["One CPU timing pass per native batch, without repeated-trial confidence intervals",
                     "Native OFF head timing excludes the routed adapter; active cached API and PG step timing include it",
                     "All native batch arms use the same32 old TRAIN requests",
                     "PG benchmark is a new64-update throughput cohort, not a semantic-quality comparison"])
    write_json(root / "results.json", result)
    write_json(root / "progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/native-model")
    parser.add_argument("--lock", default="native-model.lock.json")
    parser.add_argument("--cache", default="artifacts/label-probe-inputs-v1/synthetic-label-cache.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pg-batches", type=int, nargs="+", default=[8])
    parser.add_argument("--head-repeats", type=int, default=200)
    parser.add_argument("--adapter", default="artifacts/normalized-residual-dev-v3-continuation/raw-inference.pt")
    parser.add_argument("--readout", default="artifacts/readout-v2.pt")
    args = parser.parse_args()
    result = run(args.model_dir, args.lock, args.cache, args.output, threads=args.threads,
                 pg_batches=tuple(args.pg_batches), head_repeats=args.head_repeats,
                 adapter_file=args.adapter, readout_file=args.readout)
    print(json.dumps(dict(native=result["native"], cached_off_head=result["cached_off_head"],
                          cached_active_adapter=result["cached_active_adapter"],
                          pg_attempts=result["pg_attempts"], new_updates=result["total_new_pg_unique_updates"])), flush=True)


if __name__ == "__main__":
    main()
