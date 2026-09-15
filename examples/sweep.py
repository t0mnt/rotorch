"""
Run one example over the whole benchmark matrix and write every result to one csv.

The matrix is five implementations x the batch sizes x cpu and cuda, each repeated so that the
faster run can be kept and so that a compiled run is seen once cold and once warm:

    python examples/sweep.py                        # everything this machine can run
    python examples/sweep.py --devices cuda         # gpu only
    python examples/sweep.py --preset quick         # eager only, small batches, one rep
    python examples/sweep.py --dry-run              # print the plan and stop

Each run is a separate process, so one that dies takes its row down and nothing else. The csv
is appended to as results land and is re-read on startup, so the sweep can be interrupted and
restarted and will pick up where it left off. A configuration that fails at one batch size is
not tried at a larger one, since these failures are failures of size.

The whole thing is a few hours, most of it compiling: every batch size compiles anew, and
:code:`--compile operators` builds 98 operators each time. :code:`--cuda-batches` and
:code:`--cpu-batches` are the dials if that is too much, and :code:`--configs` drops columns.
"""
import argparse
import csv
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

# Implementation -> the flags that select it. The order is the order of the table in the docs,
# and also the order to run them in: the eager ones are cheap and answer most of the question.
CONFIGS = {
    "cgenn": ["--impl", "cgenn"],
    "rotorch": [],
    "cgenn-compiled": ["--impl", "cgenn", "--compile", "model"],
    "rotorch-operators": ["--compile", "operators"],
    "rotorch-model": ["--compile", "model"],
}
EAGER = ("cgenn", "rotorch")

FIELDS = ["timestamp", "host", "device", "config", "batch", "rep", "status", "median_ms",
          "mean_ms", "first_step_ms", "peak_memory_mib", "total_s", "wall_s", "parameters",
          "val_loss", "returncode", "error", "steps", "warmup", "train_samples", "example",
          "python", "torch", "cuda", "gpu", "cpu", "platform"]

SUMMARY = re.compile(r"first step ([\d.]+) ms, then ([\d.]+) ms/step \(mean ([\d.]+), "
                     r"total ([\d.]+) s\)")
VAL_LOSS = re.compile(r"(\d+) parameters, val loss ([-\d.]+)")
MEMORY = re.compile(r"peak memory (\d+) MiB")
# Failures worth telling apart in the csv: the first two are limits of the card, the third of
# the toolchain, and an illegal access has been known to take the whole machine with it.
FAILURES = [("out of memory", "out-of-memory"),
            ("illegal memory access", "illegal-memory-access"),
            ("CUDA error", "cuda-error"),
            ("timed out after", "timeout"),
            ("BackendCompilerFailed", "compile-failed"),
            ("Could not import the cgenn model", "cgenn-path"),
            ("ModuleNotFoundError", "import-error")]


def environment(python):
    """Ask the interpreter that will run the benchmarks what it is about to run them on."""
    code = ("import json, platform, torch;"
            "print(json.dumps({'python': platform.python_version(),"
            "'torch': torch.__version__,"
            "'cuda': torch.version.cuda or '',"
            "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else '',"
            "'cpu': platform.processor(), 'platform': platform.platform()}))")
    probe = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if probe.returncode:
        raise SystemExit(f"Could not import torch with {python}:\n{probe.stderr}")
    return json.loads(probe.stdout)


def classify(output):
    """Name the failure, so that a row that has no timing still says something."""
    for signature, name in FAILURES:
        if signature in output:
            return name
    return "failed"


def run(example, config, device, batch, rep, args, environ):
    """One process, one row. Never raises: a failure is a result too."""
    command = [args.python, example, "--device", device, "--batch-size", str(batch),
               "--steps", str(args.steps), "--warmup", str(args.warmup),
               "--train-samples", str(max(args.train_samples, batch)),
               "--val-samples", str(args.val_samples), "--print-interval", str(args.steps),
               *CONFIGS[config]]
    if args.cgenn_path:
        command += ["--cgenn-path", args.cgenn_path]

    start = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=os.path.dirname(example) or ".", text=True,
                              capture_output=True, timeout=args.timeout)
        output, returncode = done.stdout + done.stderr, done.returncode
    except subprocess.TimeoutExpired as expired:
        output = (expired.stdout or b"").decode(errors="replace")
        output, returncode = output + f"\ntimed out after {args.timeout} s\n", -1
    wall = time.perf_counter() - start

    row = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(),
               device=device, config=config, batch=batch, rep=rep, status="ok",
               returncode=returncode,
               error="", wall_s=round(wall, 1), steps=args.steps, warmup=args.warmup,
               train_samples=max(args.train_samples, batch),
               example=os.path.basename(example), **environ)
    if match := SUMMARY.search(output):
        row.update(first_step_ms=match[1], median_ms=match[2], mean_ms=match[3],
                   total_s=match[4])
    if match := VAL_LOSS.search(output):
        row.update(parameters=match[1], val_loss=match[2])
    if match := MEMORY.search(output):
        row.update(peak_memory_mib=match[1])
    if returncode or "median_ms" not in row:
        row.update(status="failed", error=classify(output))
        with open(f"{args.output}.{config}_{device}_b{batch}.log", "w",
                  encoding="utf-8") as log:  # Keep the whole thing, for the post mortem.
            log.write(" ".join(command) + "\n\n" + output)
    return row


def write(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def completed(path):
    """(config, device, batch, rep) of every run already in the csv, failures included."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        return {(row["config"], row["device"], int(row["batch"]), int(row["rep"] or 1)):
                row["status"] for row in csv.DictReader(handle)}


def plan(args, has_cuda):
    """Cheap and informative first: every eager run, then the compiled ones by batch size."""
    devices = args.devices or (["cpu", "cuda"] if has_cuda else ["cpu"])
    for device in devices:
        batches = args.cpu_batches if device == "cpu" else args.cuda_batches
        configs = [c for c in CONFIGS if c in args.configs]
        for stage in (EAGER, tuple(c for c in configs if c not in EAGER)):
            for batch in batches:
                for config in [c for c in configs if c in stage]:
                    for rep in range(1, args.reps + 1):
                        yield config, device, batch, rep


def summarize(path):
    """The table the docs want: the faster of the reps, and the speedup over cgenn."""
    best = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or not row["median_ms"]:
                continue
            key = (row["device"], row["config"], int(row["batch"]))
            best[key] = min(float(row["median_ms"]), best.get(key, float("inf")))

    for device in dict.fromkeys(key[0] for key in best):
        batches = sorted({key[2] for key in best if key[0] == device})
        configs = [c for c in CONFIGS if any(key[:2] == (device, c) for key in best)]
        print(f"\n{device}: ms/step, and the speedup over cgenn")
        print("batch".rjust(7) + "".join(name.rjust(22) for name in configs))
        for batch in batches:
            cells = []
            for config in configs:
                value = best.get((device, config, batch))
                baseline = best.get((device, "cgenn", batch))
                if value is None:
                    cells.append("-".rjust(22))
                elif baseline and config != "cgenn":
                    cells.append(f"{value:.1f} ({baseline / value:.1f}x)".rjust(22))
                else:
                    cells.append(f"{value:.1f}".rjust(22))
            print(str(batch).rjust(7) + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", default=os.path.join(os.path.dirname(__file__),
                                                          "hulls.py"))
    parser.add_argument("--output", default=f"sweep-{socket.gethostname()}.csv")
    parser.add_argument("--devices", nargs="*", choices=["cpu", "cuda"],
                        help="default: cpu, and cuda if torch can see one")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--cpu-batches", nargs="*", type=int,
                        default=[32, 128, 512, 2048])
    parser.add_argument("--cuda-batches", nargs="*", type=int,
                        default=[32, 128, 512, 2048, 4096, 8192, 16384])
    parser.add_argument("--reps", type=int, default=2,
                        help="the faster one is the timing; the first is also the cold compile")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds, per run")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cgenn-path", default=None)
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    parser.add_argument("--retry-failed", action="store_true",
                        help="rerun rows the csv records as failed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.preset == "quick":  # Twenty minutes, to check the machine before the long night.
        args.configs = list(EAGER)
        args.cpu_batches = args.cuda_batches = [32, 512]
        args.reps = 1

    environ = environment(args.python)
    done = completed(args.output)
    keep = {"ok"} if args.retry_failed else {"ok", "failed", "skipped"}
    runs = [r for r in plan(args, has_cuda=bool(environ["gpu"])) if done.get(r) not in keep]

    print(f"{environ['gpu'] or 'no gpu'}, torch {environ['torch']}, "
          f"cuda {environ['cuda'] or 'none'}, python {environ['python']}")
    print(f"{len(runs)} runs to go, {len(done)} already in {args.output}")
    if args.dry_run:
        for config, device, batch, rep in runs:
            print(f"  {device:5s} {config:18s} batch {batch:5d} rep {rep}")
        return

    blocked = set()  # (config, device) that has already failed, at a smaller batch size.
    for config, device, batch, rep in runs:
        if (config, device) in blocked:
            print(f"[{time.strftime('%H:%M:%S')}] {config} {device} b{batch}: skipped, "
                  f"this configuration already failed at a smaller batch size")
            write(args.output, dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    host=socket.gethostname(), device=device, config=config,
                                    batch=batch, rep=rep, status="skipped",
                                    error="failed at a smaller batch size", **environ))
            continue

        print(f"[{time.strftime('%H:%M:%S')}] {config} {device} b{batch} rep {rep} ...",
              flush=True)
        row = run(args.example, config, device, batch, rep, args, environ)
        write(args.output, row)
        if row["status"] == "ok":
            print(f"    {row['median_ms']} ms/step, first step "
                  f"{float(row['first_step_ms']) / 1000:.1f} s"
                  + (f", peak {row['peak_memory_mib']} MiB" if row.get("peak_memory_mib")
                     else ""), flush=True)
        else:
            print(f"    {row['error']} after {row['wall_s']} s, see "
                  f"{args.output}.{config}_{device}_b{batch}.log", flush=True)
            blocked.add((config, device))

    summarize(args.output)
    print(f"\nresults: {args.output}")


if __name__ == "__main__":
    main()
