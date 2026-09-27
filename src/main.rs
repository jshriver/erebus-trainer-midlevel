//! erebus-trainer-mid -- bullet trainer for the erebus "midlevel" NNUE arch.
//!
//!     (768 x INPUT_BUCKETS -> HIDDEN_SIZE) x 2 -> L2_SIZE -> L3_SIZE -> 1
//!     = (768x8 -> 1024) x 2 -> 16 -> 32 -> 1, SCReLU after every hidden layer
//!
//! King input buckets choose which copy of the input weights each side uses;
//! material-count output buckets choose the weights of the 16, 32 and 1 layers.
//! The accumulator and the 2048 -> 16 layer are saved as i16, the small layers
//! after it as f32.
//!
//! Produces `<OUT_DIR>/<NET_ID>-<N>/quantised.bin`. The engine's
//! `src/nnue.rs::parse_net` must read exactly the layout written by the
//! `save_format` below -- see README.md for the byte layout.
//!
//! Usage:
//!     cargo build --release --features cuda
//!     ./erebus-trainer-mid <data.binpack | data-dir> [more paths...]
//!
//! Every training hyper-parameter lives in the CONFIG block below -- edit it and
//! rebuild. The only runtime argument is where the data is.
//!
//! Resume is automatic: on start it looks for `<OUT_DIR>/<NET_ID>-<N>/` and, if
//! found, loads the highest N and continues from superbatch N+1. Re-run the same
//! command after a Colab disconnect and it picks up where it stopped.

use bullet_lib::{
    game::{
        inputs::{ChessBuckets, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};
use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry};

// ========================= CONFIG -- edit, then `cargo build --release` =========================

/// Hidden / accumulator width per perspective. MUST equal `HL` in the engine's
/// `src/nnue.rs`. Changing this is a recompile of BOTH this trainer and erebus.
const HIDDEN_SIZE: usize = 1024;

/// King bucket for each square of a perspective's own king, seen from that
/// perspective (a1 = 0 .. h8 = 63; bullet flips Black's view). Each bucket gets
/// its own copy of the 768 input weights. The layout is symmetric left/right
/// (same bucket for mirrored files) but the features are NOT mirrored.
/// MUST equal `KING_BUCKETS` in the engine's `src/nnue.rs`.
#[rustfmt::skip]
const KING_BUCKET_LAYOUT: [usize; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
];
const INPUT_BUCKETS: usize = get_num_buckets(&KING_BUCKET_LAYOUT);

/// Output buckets: bullet `MaterialCount<OUTPUT_BUCKETS>`, i.e. bucket =
/// (pieces on board - 2) / 4. Each bucket has its own weights and biases for
/// every layer after the accumulator. MUST equal the engine's value.
const OUTPUT_BUCKETS: usize = 8;

/// Width of the first layer after the accumulator (2 * HIDDEN_SIZE -> L2_SIZE).
/// It is recomputed at every evaluation and is the largest per-eval cost
/// (2048 x 16 = ~33k multiply-adds), so keep it narrow.
const L2_SIZE: usize = 16;
/// Width of the second middle layer (L2_SIZE -> L3_SIZE -> 1). Cheap.
const L3_SIZE: usize = 32;

/// Checkpoint name prefix. Checkpoints land in `<OUT_DIR>/<NET_ID>-<N>/`, and
/// the session marker is `<OUT_DIR>/<NET_ID>.session`. Different from the mini
/// trainer's "erebus" so the two runs never resume into each other.
const NET_ID: &str = "erebus-mid";
/// Where checkpoints are written / resumed from. Put this on persistent storage
/// (colab-notebook.py copies each checkpoint to Google Drive) when on a preemptible VM.
const OUT_DIR: &str = "checkpoints";

/// Training is ONE global schedule over the whole corpus (every file in
/// `POSITION_COUNTS`); individual runs advance along it. The cosine LR and
/// linear WDL both run over `1..=GLOBAL_END`, so they progress smoothly no
/// matter how the run is split into sessions -- no per-file LR restarts.
///
///   GLOBAL_END = round(TOTAL_PASSES * corpus_positions * FILTER_KEEP_FRAC
///                      / (BATCHES_PER_SUPERBATCH * BATCH_SIZE))
///
/// Each invocation resumes from the last checkpoint and trains ONE pass over the
/// file(s) you pass it, then stops:
///
///   session_end = min(resume_point
///                     + round(PASS_FRACTION_PER_FILE * passed_positions
///                             * FILTER_KEEP_FRAC / pos_per_superbatch),
///                     GLOBAL_END)
///
/// `passed_positions` is looked up per file: a `<file>.binpack.count` sidecar
/// (plain integer; `_` / `,` / ws ignored) if present, else the `POSITION_COUNTS`
/// table keyed by basename. Workflow: feed the files one per session in shuffled
/// order (run-all.sh), or all at once if they fit on disk; after TOTAL_PASSES
/// laps the net is done at GLOBAL_END. `EREBUS_END_SUPERBATCH=N` forces a
/// specific session end. Interrupted sessions resume exactly (see .session file).
const TOTAL_PASSES: f64 = 1.0;
/// Fraction of each passed file to consume per session. 1.0 = one full pass.
const PASS_FRACTION_PER_FILE: f64 = 1.0;

/// Known raw position counts, keyed by binpack basename (no directory), from
/// `binpack_counter`. Order doesn't matter. Sum = 218_849_949_380 over 41 files.
const POSITION_COUNTS: &[(&str, u64)] = &[
    ("test60-2021-11-nov-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_452_424_355),
    ("test60-2021-12-dec-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_363_206_227),
    ("test77-2021-12-dec-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 6_286_624_013),
    ("test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 7_419_909_666),
    ("test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_574_170_531),
    ("test79-2022-04-apr-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_930_087_205),
    ("test79-2022-05-may-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_284_179_270),
    ("test80-2022-06-jun-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_488_679_928),
    ("test80-2022-07-jul-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_835_573_847),
    ("test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_801_599_910),
    ("test80-2022-09-sep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_171_904_814),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_0.binpack", 2_030_804_185),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_1.binpack", 2_030_725_513),
    ("test80-2022-11-nov-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_608_318_891),
    ("test80-2023-01-jan-16tb7p.v6-sk20.min.relabel-BT4-tf13tune.binpack", 4_707_093_556),
    ("test80-2023-02-feb-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_626_845_354),
    ("test80-2023-03-mar-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_520_899_664),
    ("test80-2023-04-apr-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_653_619_110),
    ("test80-2023-05-may-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 5_600_480_538),
    ("test80-2023-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_756_356_195),
    ("test80-2023-07-jul-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_212_977_488),
    ("test80-2023-08-aug-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 2_693_519_136),
    ("test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_257_611_143),
    ("test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_012_783_968),
    ("test80-2023-11-nov-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 2_724_311_169),
    ("test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_016_184_922),
    ("leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack", 5_681_356_602),
    ("leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack", 5_679_303_898),
    ("leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack", 5_680_096_474),
    ("leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack", 5_681_363_129),
    ("leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack", 5_680_919_670),
    ("T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack", 9_133_725_682),
    ("T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack", 9_150_872_906),
    ("T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack", 9_136_446_438),
    ("T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack", 9_128_728_654),
    ("T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack", 9_160_229_495),
    ("dfrc_n5000.relabel-BT4-tf13tune.binpack", 12_353_351_142),
    ("fishpack32.relabel-BT4-tf13tune.binpack", 2_555_358_353),
    ("multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack", 9_485_503_089),
    ("nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack", 13_937_427_120),
    ("wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack", 2_344_376_130),
];
/// Fraction of raw positions expected to survive `filter()`. 1.0 treats the raw
/// count as the training-position budget (the loader may wrap slightly at the
/// end); lower it (~0.6) if bullet logs that it looped the data before finishing.
const FILTER_KEEP_FRAC: f64 = 1.0;
/// Safety cap on a single session's superbatch budget -- a bad count can't
/// launch a runaway session. The largest single binpack is ~139 superbatches,
/// so this only trips on a corrupt/huge count. GLOBAL_END is not clamped (it's
/// derived from the trusted compiled table).
const MAX_SUPERBATCH: usize = 5000;
/// Positions per batch.
const BATCH_SIZE: usize = 16_384;
/// Batches per superbatch. 6104 * 16384 ~= 100M positions / superbatch.
const BATCHES_PER_SUPERBATCH: usize = 6104;
/// Save a checkpoint every this many superbatches (also always saves the last).
const SAVE_RATE: usize = 10;

/// Learning rate: cosine decay from LR_START to LR_FINAL over `1..=GLOBAL_END`
/// (the whole TOTAL_PASSES plan, not the individual session).
const LR_START: f32 = 0.001;
const LR_FINAL: f32 = 2.5e-6;

/// WDL lambda: linear taper from WDL_START to WDL_END over `1..=GLOBAL_END`.
/// target = lambda * game_result + (1 - lambda) * sigmoid(score / EVAL_SCALE)
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.6;

/// Eval scale. MUST equal the engine's `const SCALE`. Also bullet's eval_scale.
const EVAL_SCALE: i32 = 400;
/// Feature-transformer quantisation. Engine: `const QA: i64 = 255;`
const QA: i16 = 255;
/// Quantisation of the 2048 -> 16 weights `l1w`. Engine: `const QB: i64 = 64;`
/// AdamW's default +-1.98 clip keeps them within +-127, so the engine can form
/// `clamp(acc, 0, QA) * l1w` in i16 (255 * 127 < 32767).
const QB: i16 = 64;
/// Weight clip for the bucketed input weights `l0w` and the shared factoriser
/// `l0f`. They are summed at save time, so each is held to +-0.99 to keep the
/// saved weight within +-1.98 (x QA = +-505, safely inside i16 accumulators).
const L0_CLIP: f32 = 0.99;

/// Positions evaluated with the trained float net at the end of every session.
/// `erebus --probe-fen "<fen>"` on the exported quantised.bin should give
/// nearly the same centipawns; a large gap means the trainer's save layout and
/// the engine's parser disagree.
const EVAL_CHECK_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/5pp1/7p/8/8/7P/5PP1/3R2K1 w - - 0 1",
    "4k3/8/8/8/8/8/8/QQQQK3 b - - 0 1",
];

/// CPU worker threads for the training loop.
const CPU_THREADS: usize = 4;
/// Threads for decoding SF binpacks.
const DATA_THREADS: usize = 8;
/// Shuffle-buffer size in MiB. Bigger = better local mixing before batching.
const SHUFFLE_BUFFER_MB: usize = 4096;

/// bullet's SF loader has no data cursor: every process start reads the file
/// list from the beginning. When true, the list is rotated left by the resume
/// superbatch so each resumed session begins on a different binpack.
const ROTATE_DATA_EACH_SESSION: bool = true;

// =============================================================================================

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("erebus-trainer-mid: {}", msg.as_ref());
    std::process::exit(1);
}

/// Clean "no work to do" exit (plan already complete, etc.). Exit code 3 so a
/// driver script can tell it apart from an error (1) or a training run (0).
fn nothing_to_do(msg: impl AsRef<str>) -> ! {
    println!("{}", msg.as_ref());
    std::process::exit(3);
}

fn usage() -> ! {
    eprintln!(
        "erebus-trainer-mid -- bullet trainer for erebus (768x{INPUT_BUCKETS} -> {HIDDEN_SIZE})x2 -> {L2_SIZE} -> {L3_SIZE} -> 1 NNUE\n\
         \n\
         usage:  erebus-trainer-mid <PATH>...\n\
         \n\
         each PATH is a .binpack file or a directory (all *.binpack inside it,\n\
         sorted by name). All hyper-parameters are compiled in -- see the CONFIG\n\
         block in src/main.rs. Resume from {OUT_DIR}/{NET_ID}-<N>/ is automatic."
    );
    std::process::exit(2);
}

/// Expand args into a concrete list of .binpack file paths.
fn collect_data_paths(inputs: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for inp in inputs {
        let p = std::path::Path::new(inp);
        if p.is_dir() {
            let mut found: Vec<String> = std::fs::read_dir(p)
                .unwrap_or_else(|e| die(format!("read_dir {inp}: {e}")))
                .flatten()
                .map(|e| e.path())
                .filter(|q| q.extension().map(|x| x == "binpack").unwrap_or(false))
                .map(|q| q.to_string_lossy().into_owned())
                .collect();
            found.sort();
            if found.is_empty() {
                die(format!("no *.binpack files in directory {inp}"));
            }
            out.extend(found);
        } else if p.is_file() {
            out.push(inp.clone());
        } else {
            die(format!("data path does not exist: {inp}"));
        }
    }
    out
}

/// Raw position count for one input file: `<path>.count` sidecar if present,
/// else the `POSITION_COUNTS` table keyed by basename. Neither => fatal.
fn count_for(path: &str) -> u64 {
    let side = format!("{path}.count");
    if let Ok(raw) = std::fs::read_to_string(&side) {
        let digits: String = raw.chars().filter(char::is_ascii_digit).collect();
        return digits.parse().unwrap_or_else(|e| die(format!("{side}: no valid integer ({e})")));
    }

    let base = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());

    match POSITION_COUNTS.iter().find(|(name, _)| *name == base) {
        Some((_, n)) => *n,
        None => die(format!(
            "no position count for '{base}': add it to POSITION_COUNTS in src/main.rs, \
             or drop a '{side}' file next to it, or set EREBUS_END_SUPERBATCH=N"
        )),
    }
}

/// Sum of raw position counts across all input files.
fn count_positions(files: &[String]) -> u64 {
    let mut total: u64 = 0;
    for f in files {
        let n = count_for(f);
        if n == 0 {
            die(format!("{f}: position count is 0"));
        }
        total += n;
    }
    total
}

/// Total positions across every file in the compiled `POSITION_COUNTS` table.
fn corpus_positions() -> u64 {
    POSITION_COUNTS.iter().map(|(_, n)| *n).sum()
}

/// `<OUT_DIR>/<NET_ID>.session` records `<began_at> <stop_at>` for the current
/// session so a preempted + resumed run finishes the same window instead of
/// opening a fresh one from the (later) resume point.
fn session_path() -> String {
    format!("{OUT_DIR}/{NET_ID}.session")
}

fn read_session() -> Option<(usize, usize)> {
    let s = std::fs::read_to_string(session_path()).ok()?;
    let mut it = s.split_whitespace();
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

fn write_session(began: usize, stop: usize) {
    let _ = std::fs::create_dir_all(OUT_DIR);
    let p = session_path();
    if let Err(e) = std::fs::write(&p, format!("{began} {stop}\n")) {
        eprintln!("erebus-trainer-mid: warning: could not write {p}: {e}");
    }
}

/// `LinearWDL` tapers over the `end_superbatch` bullet passes it (the per-session
/// end). This variant tapers over an absolute `1..=final_superbatch` window so
/// the WDL schedule stays global across sessions, matching the LR cosine.
#[derive(Clone, Debug)]
struct GlobalLinearWDL {
    start: f32,
    end: f32,
    final_superbatch: usize,
}

impl wdl::WdlScheduler for GlobalLinearWDL {
    fn blend(&self, _batch: usize, superbatch: usize, _max: usize) -> f32 {
        let denom = self.final_superbatch.saturating_sub(1).max(1) as f32;
        let t = (superbatch.saturating_sub(1) as f32 / denom).clamp(0.0, 1.0);
        self.start + t * (self.end - self.start)
    }

    fn colourful(&self) -> String {
        format!("linear taper {} -> {} over 1..={} (global)", self.start, self.end, self.final_superbatch)
    }
}

/// Highest N such that `<OUT_DIR>/<NET_ID>-<N>/optimiser_state/` exists.
fn latest_checkpoint() -> Option<(String, usize)> {
    let prefix = format!("{NET_ID}-");
    let mut best: Option<(String, usize)> = None;
    for ent in std::fs::read_dir(OUT_DIR).ok()?.flatten() {
        let path = ent.path();
        if !path.is_dir() {
            continue;
        }
        let name = ent.file_name();
        let name = name.to_string_lossy();
        let Some(num) = name.strip_prefix(&prefix) else { continue };
        let Ok(n) = num.parse::<usize>() else { continue };
        if !path.join("optimiser_state").is_dir() {
            continue;
        }
        if best.as_ref().map_or(true, |(_, b)| n > *b) {
            best = Some((path.to_string_lossy().into_owned(), n));
        }
    }
    best
}

/// Standard bullet Stockfish-binpack position filter.
fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    // ---- resume point ----
    let (resume_dir, start_superbatch) = match latest_checkpoint() {
        Some((dir, n)) => {
            println!("resume: {dir} (completed superbatch {n}) -> continuing from {}", n + 1);
            (Some(dir), n + 1)
        }
        None => {
            println!("resume: no '{OUT_DIR}/{NET_ID}-<N>' checkpoint found -> starting fresh");
            (None, 1)
        }
    };

    // ---- data files ----
    let mut files = collect_data_paths(&args);
    if ROTATE_DATA_EACH_SESSION && files.len() > 1 {
        let k = (start_superbatch - 1) % files.len();
        if k != 0 {
            files.rotate_left(k);
            println!("rotated {} data files left by {k} (per-session coverage spread)", files.len());
        }
    }
    let paths_ref: Vec<&str> = files.iter().map(String::as_str).collect();

    // ---- global plan + this session's stop point ----
    let pos_per_superbatch = BATCHES_PER_SUPERBATCH * BATCH_SIZE;
    let global_end: usize = ((TOTAL_PASSES * corpus_positions() as f64 * FILTER_KEEP_FRAC
        / pos_per_superbatch as f64)
        .round() as usize)
        .max(1);

    let end_superbatch: usize = if let Ok(v) = std::env::var("EREBUS_END_SUPERBATCH") {
        let n = v
            .trim()
            .parse::<usize>()
            .unwrap_or_else(|e| die(format!("EREBUS_END_SUPERBATCH not a number: {e}")));
        if start_superbatch > n {
            nothing_to_do(format!(
                "'{NET_ID}': EREBUS_END_SUPERBATCH={n} but already at superbatch {start_superbatch}."
            ));
        }
        write_session(start_superbatch, n);
        println!("session {start_superbatch}..={n}  (EREBUS_END_SUPERBATCH override; global end {global_end})");
        n
    } else if start_superbatch > global_end {
        nothing_to_do(format!(
            "'{NET_ID}' has completed its {TOTAL_PASSES}-pass plan ({global_end} superbatches). \
             Nothing to do -- raise TOTAL_PASSES and rebuild to train longer."
        ));
    } else if let Some((began, stop)) =
        read_session().filter(|&(b, s)| start_superbatch >= b && start_superbatch <= s)
    {
        // resuming a session interrupted mid-window (e.g. Colab preemption)
        let stop = stop.min(global_end);
        println!("session {began}..={stop} resumed at superbatch {start_superbatch}  (global end {global_end})");
        stop
    } else {
        // new session: one pass over the file(s) passed now
        let passed = count_positions(&files);
        let budget = ((PASS_FRACTION_PER_FILE * passed as f64 * FILTER_KEEP_FRAC
            / pos_per_superbatch as f64)
            .round() as usize)
            .clamp(1, MAX_SUPERBATCH);
        let stop = (start_superbatch - 1 + budget).min(global_end);
        write_session(start_superbatch, stop);
        println!(
            "session {start_superbatch}..={stop}  (+{budget} sb = {PASS_FRACTION_PER_FILE} pass over \
             {passed} positions in {} file(s); global end {global_end})",
            files.len()
        );
        stop
    };

    println!("--------------------------------------------------------------");
    println!(
        "net id        : {NET_ID}   arch (768x{INPUT_BUCKETS} -> {HIDDEN_SIZE}) x 2 -> {L2_SIZE} -> {L3_SIZE} -> 1, \
         {OUTPUT_BUCKETS} output buckets, SCReLU"
    );
    println!("quantisation  : QA={QA} QB={QB} eval_scale={EVAL_SCALE}");
    println!("plan          : {TOTAL_PASSES} pass(es) over corpus -> global end {global_end} superbatches");
    println!("this session  : {start_superbatch}..={end_superbatch}  ({BATCHES_PER_SUPERBATCH} x {BATCH_SIZE})");
    println!("lr            : {LR_START} -> {LR_FINAL} cosine over 1..={global_end}");
    println!("wdl lambda    : {WDL_START} -> {WDL_END} linear over 1..={global_end}");
    println!("save rate     : every {SAVE_RATE} superbatches -> {OUT_DIR}/");
    println!("data          : {} files, {DATA_THREADS} decode threads, {SHUFFLE_BUFFER_MB} MiB buffer", files.len());
    println!("--------------------------------------------------------------");

    // ---- build trainer ----
    // Saved layout, little-endian, in this order (the engine's parse_net must
    // read exactly this; see README.md). `ob` = output bucket. Layers after the
    // accumulator are transposed from bullet's input-major storage so each
    // output neuron's weights are contiguous.
    //   i16  l0w  [INPUT_BUCKETS][768][HIDDEN_SIZE]   x QA (factoriser l0f merged in)
    //   i16  l0b  [HIDDEN_SIZE]                       x QA
    //   i16  l1w  [ob][L2_SIZE][2 * HIDDEN_SIZE]      x QB (stm 1024, then ntm 1024)
    //   f32  l1b  [ob][L2_SIZE]
    //   f32  l2w  [ob][L3_SIZE][L2_SIZE]
    //   f32  l2b  [ob][L3_SIZE]
    //   f32  l3w  [ob][L3_SIZE]
    //   f32  l3b  [ob]
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBuckets::new(KING_BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            // merge the shared factoriser into every king bucket's weights
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Factoriser: one 768-input weight set shared by all king buckets.
            // Every position trains it, so the buckets start from common
            // piece-square knowledge and only learn their differences. It is
            // added into l0w at save time, so the engine never sees it.
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // Each layer after the accumulator holds OUTPUT_BUCKETS weight sets
            // stacked; `select` keeps the set for this position's bucket
            // (rows ob*size .. (ob+1)*size).
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);

            let h2 = l1.forward(hidden).select(output_buckets).screlu();
            let h3 = l2.forward(h2).select(output_buckets).screlu();
            l3.forward(h3).select(output_buckets)
        });

    let l0_clip = AdamWParams { max_weight: L0_CLIP, min_weight: -L0_CLIP, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);

    if let Some(dir) = &resume_dir {
        // loads `<dir>/optimiser_state` (weights + Adam moments). Does NOT set
        // the start superbatch -- that's steps.start_superbatch below.
        trainer.load_from_checkpoint(dir);
    }

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: EVAL_SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch,
            end_superbatch,
        },
        // both schedulers run over the ABSOLUTE 1..=global_end window, so they
        // progress smoothly no matter how the plan is split into sessions.
        wdl_scheduler: GlobalLinearWDL { start: WDL_START, end: WDL_END, final_superbatch: global_end },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: LR_START,
            final_lr: LR_FINAL,
            final_superbatch: global_end,
        },
        // never coarser than the session (bullet always saves the last too).
        save_rate: SAVE_RATE.min(end_superbatch).max(1),
    };

    let settings = LocalSettings {
        threads: CPU_THREADS,
        test_set: None, // validation not implemented in current bullet
        output_directory: OUT_DIR,
        batch_queue_size: 64,
    };

    let data_loader =
        SfBinpackLoader::new_concat_multiple(&paths_ref, SHUFFLE_BUFFER_MB, DATA_THREADS, filter);

    trainer.run(&schedule, &settings, &data_loader);

    // bullet's loader threads can panic (e.g. a truncated binpack) and run() still
    // returns normally. The last superbatch is always saved, so a missing
    // checkpoint for it means the session ended early: keep the resume marker and
    // exit non-zero so the driver doesn't record this file as done.
    let reached = latest_checkpoint().map_or(0, |(_, n)| n);
    if reached < end_superbatch {
        die(format!(
            "session ended early: last checkpoint is superbatch {reached}, expected {end_superbatch}. \
             The data loader probably failed (check the log for a panic, e.g. a truncated binpack)."
        ));
    }

    // session finished cleanly -> clear the resume marker
    let _ = std::fs::remove_file(session_path());

    // Float evals of the trained net, to compare against the engine on the
    // exported quantised.bin. Diagnostics only: a failure here must not turn a
    // good session into a failed one.
    println!("eval check (float net, cp; compare with erebus --probe-fen on {NET_ID}-{end_superbatch}/quantised.bin):");
    for fen in EVAL_CHECK_FENS {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trainer.eval(fen))) {
            Ok(v) => println!("  eval-check {:>6.0}  {fen}", v * EVAL_SCALE as f32),
            Err(_) => println!("  eval-check  error  {fen}"),
        }
    }

    if end_superbatch >= global_end {
        println!(
            "done -- {TOTAL_PASSES}-pass plan complete at superbatch {global_end}.\n\
             deploy:  cp {OUT_DIR}/{NET_ID}-{global_end}/quantised.bin  <midlevel engine>/nets/net.nnue"
        );
    } else {
        println!(
            "session done at {end_superbatch}/{global_end}. run the next binpack to continue."
        );
    }
}
