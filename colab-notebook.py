# erebus-trainer-mid -- Google Colab (free tier, T4) driver. Paste into one cell.
#
# One binpack per Colab session. Everything that must survive the runtime lives
# on Google Drive under DRIVE_DIR:
#   bin/erebus-trainer-mid-<commit>   trainer built from REPO_URL at that commit
#   checkpoints/erebus-mid-<N>/       newest checkpoint only (~125 MB)
#   checkpoints/erebus-mid.session    resume marker for an unfinished session
#   logs/                             one training log per run
# The checkpoint is copied to Drive every time the trainer saves one (every
# SAVE_RATE superbatches), so a disconnect loses at most that much work. Rerun
# the cell with the same BINPACK_URL to continue an interrupted file; the
# trainer's .session file makes it finish the same window.
#
# Before the first run: push src/, Cargo.toml and Cargo.lock to REPO_URL.
# Runtime -> Change runtime type -> T4 GPU.

import math
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import urllib.request

SCRIPT_START = time.time()

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
# Paste the file's Hugging Face URL from train-order.txt.
BINPACK_URL  = "https://huggingface.co/datasets/vondele/linrock_relabel_1/resolve/main/test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_0.binpack"
BINPACK_NAME = BINPACK_URL.split("/")[-1].split("?")[0]

NET_ID      = "erebus-mid"           # must match NET_ID in src/main.rs
BINARY_NAME = "erebus-trainer-mid"   # [[bin]] name in Cargo.toml

# The trainer is built from this repo and branch (or tag / commit). A private
# repo needs a GITHUB_TOKEN Colab secret (a fine-grained token with read access).
REPO_URL = "https://github.com/jshriver/erebus-trainer-midlevel.git"
REPO_REF = "main"

DRIVE_DIR = pathlib.Path("/content/drive/MyDrive/erebus-mid")
WORK      = pathlib.Path("/content/work")      # trainer runs here (local disk)
BUILD_DIR = pathlib.Path("/content/build")
# Local disk, wiped when the runtime ends. The largest binpacks are ~44 GB.
BINPACK_TMP = pathlib.Path("/content/binpacks")

# Colab free tier ends a runtime at 12h at most, often sooner. Stop the trainer
# before that so the final Drive sync runs. Stopping is safe: the trainer resumes
# from the last checkpoint and its .session file.
MAX_TRAIN_HOURS = 11.5

CKPT_DIR       = WORK / "checkpoints"   # the trainer's OUT_DIR, relative to WORK
DRIVE_CKPT     = DRIVE_DIR / "checkpoints"
DRIVE_LOGS     = DRIVE_DIR / "logs"
DRIVE_BIN      = DRIVE_DIR / "bin"
SESSION_FILE   = f"{NET_ID}.session"
CKPT_RE        = re.compile(rf"^{re.escape(NET_ID)}-(\d+)$")


# ════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════
def strip_ansi(text: str) -> str:
    return re.sub(r'\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|.)', '', text)


def sh(*cmd: str, **kw) -> subprocess.CompletedProcess:
    """Run a command, raising with its output if it fails."""
    p = subprocess.run(list(cmd), capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


def secret(name: str) -> str | None:
    """A Colab secret, or None if it isn't set or the notebook has no access."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def list_checkpoints(d: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    """Every complete <NET_ID>-N checkpoint in d (has optimiser_state/, the same
    test the trainer uses to resume), oldest first."""
    out = []
    if d.exists():
        for p in d.iterdir():
            m = CKPT_RE.match(p.name)
            if m and (p / "optimiser_state").is_dir():
                out.append((int(m.group(1)), p))
    return sorted(out)


# ════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════
def mount_drive() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    for d in (DRIVE_CKPT, DRIVE_LOGS, DRIVE_BIN):
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Drive folder: {DRIVE_DIR}")


def check_gpu() -> None:
    p = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if p.returncode != 0 or "GPU" not in p.stdout:
        raise RuntimeError("no GPU: Runtime -> Change runtime type -> T4 GPU")
    print(f"✓ {p.stdout.strip()}")


def repo_url_with_token() -> str:
    token = secret("GITHUB_TOKEN")
    return REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL


def remote_commit() -> str:
    """Commit that REPO_REF points to, so a cached binary is reused only for the
    exact same source."""
    if re.fullmatch(r"[0-9a-f]{40}", REPO_REF):
        return REPO_REF
    out = sh("git", "ls-remote", repo_url_with_token(), REPO_REF).stdout.split()
    if not out:
        raise RuntimeError(f"{REPO_REF} not found in {REPO_URL}")
    return out[0]


def get_binary() -> pathlib.Path:
    """The trainer for the current commit: from Drive if already built, otherwise
    built here (rustup + cargo, CUDA) and saved to Drive for later runs."""
    commit = remote_commit()
    cached = DRIVE_BIN / f"{BINARY_NAME}-{commit[:12]}"
    dst = WORK / BINARY_NAME
    WORK.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        shutil.copy2(cached, dst)
        dst.chmod(0o755)
        print(f"✓ trainer {commit[:12]} from Drive")
        return dst

    print(f"🔨 building trainer at {commit[:12]} (first run for this commit, ~5-10 min) ...")
    t0 = time.time()
    cargo = pathlib.Path.home() / ".cargo/bin/cargo"
    if not cargo.exists():
        sh("bash", "-c", "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    sh("git", "clone", "-q", repo_url_with_token(), str(BUILD_DIR))
    sh("git", "checkout", "-q", commit, cwd=BUILD_DIR)
    env = dict(os.environ, CUDA_PATH="/usr/local/cuda",
               PATH=f"{cargo.parent}:/usr/local/cuda/bin:{os.environ['PATH']}")
    sh(str(cargo), "build", "--release", "--locked", cwd=BUILD_DIR, env=env)
    built = BUILD_DIR / "target/release" / BINARY_NAME
    shutil.copy2(built, dst)
    dst.chmod(0o755)
    for old in DRIVE_BIN.glob(f"{BINARY_NAME}-*"):
        old.unlink()
    shutil.copy2(built, cached)
    print(f"✓ built in {(time.time() - t0) / 60:.1f} min, saved to {cached}")
    return dst


def restore_checkpoint() -> int | None:
    """Copy the newest Drive checkpoint (and the session marker, if any) to
    CKPT_DIR, where the trainer looks. Returns its superbatch, or None."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    found = list_checkpoints(DRIVE_CKPT)
    if not found:
        print("⚠ no checkpoint on Drive: training starts from scratch")
        return None
    n, src = found[-1]
    if not (CKPT_DIR / src.name).exists():
        shutil.copytree(src, CKPT_DIR / src.name)
    session = DRIVE_CKPT / SESSION_FILE
    if session.exists():
        shutil.copy2(session, CKPT_DIR / SESSION_FILE)
        print(f"  restored session marker: {session.read_text().strip()}")
    print(f"✓ restored {src.name} from Drive")
    return n


def remote_size(url: str) -> int | None:
    """Size in bytes of the file at url (follows Hugging Face's CDN redirect)."""
    try:
        req = urllib.request.Request(url.split("?")[0], method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception as e:
        print(f"  could not get the size of {url}: {e}")
        return None


def ensure_aria2c() -> None:
    if shutil.which("aria2c"):
        return
    print("📦 installing aria2 ...")
    sh("apt-get", "update", "-qq")
    sh("apt-get", "install", "-y", "-qq", "aria2")


def fetch_binpack() -> pathlib.Path:
    """Download the binpack to BINPACK_TMP and return it. Raises unless the file
    is complete (aria2c succeeded, its .aria2 resume file is gone, and the size
    matches Hugging Face), so a truncated binpack never reaches the trainer."""
    expected = remote_size(BINPACK_URL)
    if expected is None:
        raise RuntimeError(f"can't get the size of {BINPACK_URL}")
    print(f"binpack: {BINPACK_NAME} ({expected / 1e9:.2f} GB on Hugging Face)")

    BINPACK_TMP.mkdir(parents=True, exist_ok=True)
    dst = BINPACK_TMP / BINPACK_NAME
    control = dst.with_name(dst.name + ".aria2")   # aria2c's resume state; gone once complete
    for old in [*BINPACK_TMP.glob("*.binpack"), *BINPACK_TMP.glob("*.binpack.aria2")]:
        if old not in (dst, control):
            old.unlink()
    if dst.exists() and not control.exists():
        if dst.stat().st_size == expected:
            print(f"using binpack: {dst} (already downloaded this runtime)")
            return dst
        dst.unlink()

    # aria2c writes each connection's piece at its own offset, so the file can
    # reach full size before it's complete. Measure allocated blocks instead.
    def on_disk() -> int:
        return dst.stat().st_blocks * 512 if dst.exists() else 0

    have = on_disk()
    free = shutil.disk_usage(BINPACK_TMP).free
    if expected - have > free:
        raise RuntimeError(f"not enough local disk: need {(expected - have) / 1e9:.1f} GB, "
                           f"have {free / 1e9:.1f} GB")

    ensure_aria2c()
    print(f"📥 downloading to {dst} with aria2c ...")
    t0 = time.time()
    proc = subprocess.Popen(
        ["aria2c", "-c", "-x", "16", "-s", "16", "--max-tries=20", "--retry-wait=10",
         "--timeout=60", "--file-allocation=none", "--auto-file-renaming=false",
         "--allow-overwrite=true", "--console-log-level=warn", "--summary-interval=0",
         "--download-result=hide", "-d", str(BINPACK_TMP), "-o", dst.name,
         BINPACK_URL.split("?")[0]],
        stdout=subprocess.DEVNULL)
    while proc.poll() is None:
        time.sleep(30)
        got = min(on_disk(), expected)
        rate = (got - have) / max(time.time() - t0, 1)
        eta = f"{(expected - got) / rate / 60:.1f} min left" if rate > 0 else "stalled"
        print(f"   {got / 1e9:6.2f} / {expected / 1e9:.2f} GB  ({got / expected * 100:5.1f}%)  "
              f"{rate / 1e6:.0f} MB/s  {eta}", flush=True)

    size = dst.stat().st_size if dst.exists() else 0
    if proc.returncode != 0 or control.exists() or size != expected:
        raise RuntimeError(f"download incomplete (aria2c exit {proc.returncode}, "
                           f"{min(on_disk(), expected) / 1e9:.2f} of {expected / 1e9:.2f} GB). "
                           f"Rerun the cell to resume.")
    print(f"✓ downloaded {expected / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    return dst


# ════════════════════════════════════════════════════════════════════
# DRIVE SYNC
# ════════════════════════════════════════════════════════════════════
# Held for the whole copy, so syncs never overlap and the final one always runs
# after any periodic one.
_sync_lock = threading.Lock()


def _sync(ckpt_n: int | None, log_path: pathlib.Path | None) -> str:
    """Copy checkpoint ckpt_n to Drive, then drop older ones there. The copy goes
    to a temporary name first, so Drive always holds one complete checkpoint.
    The session marker is mirrored (deleted on Drive once the trainer clears it)."""
    msg = "no checkpoint"
    ckpts = dict(list_checkpoints(CKPT_DIR))
    if ckpt_n in ckpts:
        src = ckpts[ckpt_n]
        tmp = DRIVE_CKPT / f".{src.name}.partial"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        final = DRIVE_CKPT / src.name
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        for n, p in list_checkpoints(DRIVE_CKPT):
            if n != ckpt_n:
                shutil.rmtree(p)
        msg = src.name
    session_local, session_drive = CKPT_DIR / SESSION_FILE, DRIVE_CKPT / SESSION_FILE
    if session_local.exists():
        shutil.copy2(session_local, session_drive)
    elif session_drive.exists():
        session_drive.unlink()
    if log_path and log_path.exists():
        shutil.copy2(log_path, DRIVE_LOGS / log_path.name)
    return f"✓ {msg} on Drive ({time.strftime('%H:%M:%S')})"


def sync_now(ckpt_n: int | None, log_path: pathlib.Path | None) -> str:
    """Blocking sync, waiting for any sync in progress."""
    with _sync_lock:
        try:
            return _sync(ckpt_n, log_path)
        except Exception as e:
            return f"✗ Drive sync of {NET_ID}-{ckpt_n} FAILED: {e}"


def sync_in_background(ckpt_n: int, log_path: pathlib.Path) -> None:
    """Sync on a thread so training isn't held up. Skipped if a sync is still
    running (the next checkpoint or the final sync covers it)."""
    if not _sync_lock.acquire(blocking=False):
        return
    def work():
        try:
            print(f"\n[sync] {_sync(ckpt_n, log_path)}", flush=True)
        except Exception as e:
            print(f"\n[sync] ✗ Drive sync of {NET_ID}-{ckpt_n} failed: {e}", flush=True)
        finally:
            _sync_lock.release()
    threading.Thread(target=work, daemon=True).start()


def cleanup_local_checkpoints(keep: int = 1) -> None:
    """Keep only the newest local checkpoints (Drive has its own copy)."""
    for _, p in list_checkpoints(CKPT_DIR)[:-keep]:
        shutil.rmtree(p)


# ════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════
def run_training(binary: pathlib.Path, binpack: pathlib.Path) -> tuple[int, str, int | None, bool, pathlib.Path]:
    """Run the trainer, streaming its output. Copies each saved checkpoint to
    Drive and stops the trainer when the time budget runs out.
    Returns (exit_code, log_text, last_saved_superbatch, hit_time_limit, log_path)."""
    cmd = [str(binary), str(binpack)]
    print(f"\n▶ running: {' '.join(cmd)}\n{'─' * 60}")
    log_dir = WORK / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"training-{time.strftime('%Y%m%d-%H%M%S')}-{binpack.stem}.log"
    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"$ {' '.join(cmd)}\n")
    saved_re = re.compile(rf"Saved\s+\[{re.escape(NET_ID)}-(\d+)\]")

    proc = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    hit_limit = threading.Event()
    def stop_for_time():
        hit_limit.set()
        print(f"\n⏱ {MAX_TRAIN_HOURS}h budget reached. Stopping the trainer so the checkpoint reaches Drive.")
        proc.terminate()
    timer = threading.Timer(max(0.0, MAX_TRAIN_HOURS * 3600 - (time.time() - SCRIPT_START)), stop_for_time)
    timer.daemon = True
    timer.start()

    lines, last_saved = [], None
    try:
        for line in proc.stdout:
            cleaned = strip_ansi(line)
            print(cleaned, end="", flush=True)
            log_file.write(cleaned)
            lines.append(cleaned)
            m = saved_re.search(cleaned)
            if m:
                last_saved = int(m.group(1))
                cleanup_local_checkpoints()
                sync_in_background(last_saved, log_path)
        proc.wait()
        if hit_limit.is_set():
            log_file.write(f"\n[notebook] stopped at the {MAX_TRAIN_HOURS}h budget\n")
        log_file.write(f"\n[notebook] trainer exit code {proc.returncode}\n")
    finally:
        timer.cancel()
        log_file.close()
    return proc.returncode, "".join(lines), last_saved, hit_limit.is_set(), log_path


# ════════════════════════════════════════════════════════════════════
# LOG PARSING
# ════════════════════════════════════════════════════════════════════
def parse_log(text: str):
    """Per-superbatch loss, saved checkpoints and the LR schedule from the log.
    Returns (superbatches, losses, checkpoints, schedule), where schedule is
    (initial_lr, final_lr, global_end) or None if the LR line is missing."""
    superbatch_loss, checkpoints, schedule = {}, [], None
    sb_loss = re.compile(r"superbatch\s+(\d+)\s+\|.*?running loss\s+([\d.]+)")
    saved   = re.compile(rf"Saved\s+\[{re.escape(NET_ID)}-(\d+)\]")
    cosine  = re.compile(r"start at\s+([\d.eE+-]+)\s+and cosine decay to\s+([\d.eE+-]+)\s+at superbatch\s+(\d+)")
    for line in text.splitlines():
        if m := sb_loss.search(line):
            superbatch_loss[int(m.group(1))] = float(m.group(2))
        if m := saved.search(line):
            checkpoints.append(int(m.group(1)))
        if m := cosine.search(line):
            schedule = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
    superbatches = sorted(superbatch_loss)
    return superbatches, [superbatch_loss[sb] for sb in superbatches], checkpoints, schedule


def cosine_lr(sb: int, initial: float, final: float, global_end: int) -> float:
    """bullet's CosineDecayLR, to report the LR over this run."""
    if sb >= global_end:
        return final
    lam = 1.0 - 0.5 * (1.0 + math.cos(math.pi * sb / global_end))
    return initial + lam * (final - initial)


# The running loss alternates by about ±1e-4 between odd and even superbatches
# (shuffle-buffer cycling), so average an even number of them at each end.
LOSS_END_WINDOW = 4


def loss_ends(losses: list[float]) -> tuple[float, float, float]:
    """(start, end, % change), each end the mean of up to LOSS_END_WINDOW
    superbatches. Short runs use half the readings per end."""
    n = max(1, min(LOSS_END_WINDOW, len(losses) // 2))
    start, end = sum(losses[:n]) / n, sum(losses[-n:]) / n
    return start, end, (end - start) / start * 100 if start else 0.0


def wdl_range(log: str, first_sb: int, last_sb: int) -> tuple[float, float] | None:
    """WDL weight at the first and last superbatch, from the GlobalLinearWDL line.
    It rises across the plan (WDL_START -> WDL_END), which raises the loss floor,
    so a slowly rising loss within a run is expected."""
    m = re.search(r"linear taper\s+([\d.]+)\s*->\s*([\d.]+)\s+over 1\.\.=(\d+)", log)
    if not m:
        return None
    start, end, final = float(m.group(1)), float(m.group(2)), int(m.group(3))
    def lam(sb: int) -> float:
        return start + min(max((sb - 1) / max(final - 1, 1), 0.0), 1.0) * (end - start)
    return lam(first_sb), lam(last_sb)


def loss_summary(log: str) -> str | None:
    sbs, losses, _, _ = parse_log(log)
    if not losses:
        return None
    start, end, change = loss_ends(losses)
    line = (f"Loss: {start:.6f} → {end:.6f} ({change:+.2f}%, "
            f"mean of {max(1, min(LOSS_END_WINDOW, len(losses) // 2))} sb per end) "
            f"over superbatches {sbs[0]}–{sbs[-1]}")
    if lam := wdl_range(log, sbs[0], sbs[-1]):
        line += f"\nWDL λ: {lam[0]:.3f} → {lam[1]:.3f}"
    _, _, _, schedule = parse_log(log)
    if schedule:
        initial, final, global_end = schedule
        a, b = cosine_lr(sbs[0], initial, final, global_end), cosine_lr(sbs[-1], initial, final, global_end)
        line += (f"\nLR: {a:.3e} → {b:.3e} ({b / initial * 100:.1f}% of LR_START; "
                 f"plan {initial:g} → {final:g} over 1..={global_end})")
    return line


def eval_check_lines(log: str) -> str | None:
    """The trainer's end-of-session float evals of fixed positions (cp). The
    engine on the same checkpoint's quantised.bin should give nearly the same."""
    lines = [l.strip() for l in log.splitlines() if l.strip().startswith("eval-check")]
    return "\n".join(lines) if lines else None


def extract_summary(log: str) -> str:
    """Key lines from bullet's header and footer."""
    patterns = [
        ("Resume",     r"^resume:\s*(.+)"),
        ("Start SB",   r"Start Superbatch\s*:\s*(\d+)"),
        ("End SB",     r"End Superbatch\s*:\s*(\d+)"),
        ("Eval Scale", r"Eval Scale\s*:\s*(.+)"),
        ("Save Rate",  r"Save Rate\s*:\s*(.+)"),
        ("WDL",        r"WDL Scheduler\s*:\s*(.+)"),
        ("LR",         r"LR Scheduler\s*:\s*(.+)"),
        ("Total Training Time", r"Total Training Time\s*:\s*(.+)"),
    ]
    found = {}
    for line in log.splitlines():
        line = line.strip()
        for key, pat in patterns:
            if m := re.search(pat, line):
                found[key] = m.group(1).strip()
    out = [f"{k}: {found[k]}" for k, _ in patterns if k in found and k not in ("Start SB", "End SB")]
    if "Start SB" in found and "End SB" in found:
        out.insert(1, f"Start SB: {found['Start SB']} | End SB: {found['End SB']}")
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main() -> None:
    mount_drive()
    check_gpu()
    binary = get_binary()
    resume_n = restore_checkpoint()
    binpack = fetch_binpack()

    t0 = time.time()
    exit_code, log, last_saved, hit_limit, log_path = run_training(binary, binpack)
    elapsed = time.time() - t0

    # Final sync, whether the run finished, failed or ran out of time. It waits
    # for any periodic sync still in progress, and mirrors the session marker.
    sync_result = sync_now(last_saved if last_saved is not None else resume_n, log_path)
    if last_saved is None:
        sync_result = f"⚠ no new checkpoint this run. {sync_result}"
    print(f"[sync] {sync_result}")

    if hit_limit:
        head = f"⏱ Stopped at the {MAX_TRAIN_HOURS}h budget after {elapsed / 3600:.2f}h (rerun to resume)"
    elif exit_code != 0:
        head = f"✗ Training FAILED (exit {exit_code}) after {elapsed / 3600:.2f}h"
    else:
        head = f"✓ Training complete after {elapsed / 3600:.2f}h"
    head = f"[{NET_ID} · Colab] {head}"
    print(f"\n{head}")

    loss_line = loss_summary(log)
    checks = eval_check_lines(log)
    msg = (f"{head}\nFile: {BINPACK_NAME}\n"
           f"Started from: {f'{NET_ID}-{resume_n}' if resume_n else 'scratch'}\n"
           f"Drive: {sync_result}\n"
           + (f"{loss_line}\n" if loss_line else "")
           + "\n" + extract_summary(log)
           + (f"\n\nEval check (float net, cp):\n{checks}" if checks else ""))
    print("\n" + msg)

    from google.colab import drive
    drive.flush_and_unmount()   # make sure every write has reached Drive
    print("✓ Drive flushed")


main()
