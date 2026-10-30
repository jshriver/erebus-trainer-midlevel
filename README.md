# erebus-trainer-midlevel

bullet NNUE trainer for a stronger but still fast erebus net, forked from
`erebus-trainer-mini`. The session, schedule and resume machinery is the mini's;
the network is:

```
(768 x 8 king buckets -> 1024) x 2 -> 16 -> 32 -> 1,  SCReLU
8 output buckets (material count) choose the 16, 32 and 1 layer weights
```

No threat features and no 2048-wide first layer (those are the big net's), so
evaluation stays fast.

## Status: training complete

The final net is `erebus-mid/checkpoints/erebus-mid-2188/quantised.bin`, the same
file the engine embeds as `nets/net.nnue` in
[erebus-nnue-midlevel](https://github.com/jshriver/erebus-nnue-midlevel)
(SHA-256 `2e58d38d86e9032cc32a73bfea6429f59a25bbe0dd949380b7cc1d7bd86e4da9`).

| | |
|---|---|
| data | one pass over 41 relabelled binpacks, 218.8 billion raw positions |
| length | 2188 superbatches of 6104 x 16,384 positions (about 100M each) |
| learning rate | 0.001 -> 2.5e-6, cosine over the whole run |
| WDL lambda | 0.2 -> 0.6, linear over the whole run |
| final loss | about 0.035 |

| path | contents |
|---|---|
| `docs/INFERENCE.md` | everything needed to evaluate this net in another language, with test vectors |
| `docs/ref_eval.py` | slow, dependency-free reference evaluator for `quantised.bin` |
| `erebus-mid/checkpoints/erebus-mid-2188/` | final net, `raw.bin`, optimiser state and training log |
| `erebus-mid/bin/` | the trainer binary that finished the run |
| `train-order.txt` | shuffled file order; `x` marks a file as trained (all 41 are) |

```
cargo build --release                                   # CUDA; binary: erebus-trainer-mid
./target/release/erebus-trainer-mid data.binpack        # one binpack per session
```

All settings are consts in the CONFIG block of `src/main.rs`.

## Running on Colab (free tier, T4)

`colab-notebook.py` is one Colab cell. Set `BINPACK_URL` to the next file in
`train-order.txt` and run it. To train another net from scratch, start with an
empty Drive folder and clear the `x` marks. Each run:

1. mounts Google Drive (`MyDrive/erebus-mid/`)
2. builds the trainer from this GitHub repo at `REPO_REF`, the first time only for each commit; the binary is kept in `bin/` on Drive
3. restores the newest checkpoint and `.session` marker from Drive
4. downloads the binpack with aria2c, checked against its Hugging Face size
5. trains, copying each saved checkpoint to Drive (Drive keeps only the newest, ~125 MB)
6. saves the log to Drive and prints a summary: loss at each end, WDL λ and LR range, eval check

After a disconnect, rerun the cell with the same `BINPACK_URL`: the `.session`
marker makes the trainer finish that file's window. Push `src/`, `Cargo.toml`
and `Cargo.lock` before the first run. A private repo needs a `GITHUB_TOKEN`
Colab secret.

Kept apart from the mini run (Kaggle): `NET_ID = "erebus-mid"` (checkpoints
`erebus-mid-<N>`, marker `erebus-mid.session`), binary `erebus-trainer-mid`,
and its own Drive folder.

## Architecture

- **King input buckets** (`KING_BUCKET_LAYOUT`): each side reads a separate copy
  of the 768 input weights, chosen by where its own king stands (from its own
  side of the board). Files are paired left/right (a/h, b/g, ...), but the
  features themselves are not mirrored. The same number of inputs change per
  move as in the mini; an accumulator only needs rebuilding when a king moves
  into a different bucket.

  ```
  rank 1   0 1 2 3 3 2 1 0
  rank 2   4 4 5 5 5 5 4 4
  rank 3   6 6 6 6 6 6 6 6
  rank 4-8 7
  ```
- **Factoriser** `l0f`: one extra 768-input weight set shared by all buckets.
  Every position trains it, so a rarely-used bucket still starts from the
  common piece-square values. It is added into `l0w` at save time, so the
  engine never sees it. `l0w` and `l0f` are each clipped to ±`L0_CLIP` (0.99).
- **Output buckets** (`OUTPUT_BUCKETS`, bullet `MaterialCount<8>`):
  bucket = (pieces on board − 2) / 4. Each bucket has its own weights and
  biases for all three layers after the accumulator.
- **Middle layers** `L2_SIZE = 16`, `L3_SIZE = 32`. The 2048 → 16 layer runs at
  every evaluation (~33k multiply-adds) and is the largest per-eval cost; 16 →
  32 → 1 is cheap.

## Byte layout of `quantised.bin`

Little-endian, in this order, then `"bullet"` padding to a 64-byte boundary.
`ob` = output bucket. The layers after the accumulator are transposed from
bullet's storage so each output neuron's weights are contiguous.

| block | type | shape | count | scale |
|---|---|---|---|---|
| `l0w` | i16 | `[8 king bucket][768 feature][1024]` | 6,291,456 | × QA (255), factoriser merged in |
| `l0b` | i16 | `[1024]` | 1,024 | × QA |
| `l1w` | i16 | `[8 ob][16][2048]` (stm 1024, then ntm 1024) | 262,144 | × QB (64) |
| `l1b` | f32 | `[8 ob][16]` | 128 | float |
| `l2w` | f32 | `[8 ob][32][16]` | 4,096 | float |
| `l2b` | f32 | `[8 ob][32]` | 256 | float |
| `l3w` | f32 | `[8 ob][32]` | 256 | float |
| `l3b` | f32 | `[8 ob]` | 8 | float |

13,128,224 bytes of data (13,128,256 on disk). The f32 blocks start at byte
13,109,248, a multiple of 4. The feature index within a king bucket is bullet
`Chess768`, the same as the mini: `384*colour + 64*piece + square`, where from
Black's side colours are swapped and the square is `^ 56`.

## Forward pass (what the engine computes)

Summary only; [docs/INFERENCE.md](docs/INFERENCE.md) has the full details.

```
c_us, c_them = clamp(acc, 0, QA)                      # i16, per side (stm first)
s[o]  = Σ_i c[i]^2 * l1w[ob][o][i]                    # i = 0..2047, integer
x[o]  = s[o] / (QA * QA * QB) + l1b[ob][o]            # to float
h2[o] = clamp(x[o], 0, 1)^2                           # 16
h3[j] = clamp(Σ_o l2w[ob][j][o] * h2[o] + l2b[ob][j], 0, 1)^2   # 32
out   = Σ_j l3w[ob][j] * h3[j] + l3b[ob]
eval_cp = trunc(out * SCALE)                          # SCALE = 400, toward zero
```

`s[o]` can pass the i32 range in the worst case (2048 terms of up to 255² × 127),
so accumulate it in i64, or convert partial i32 sums to i64 or float every few
hundred inputs. Forming `c * l1w` in i16 is safe, as in the mini (255 × 127 < 32767).

## Checking trainer ↔ engine agreement

At the end of each session the trainer prints `eval-check` lines: its float net's
eval of fixed positions, in centipawns. The Colab notebook repeats them in its end-of-run summary.
The engine on the same checkpoint's `quantised.bin` should give nearly the same
numbers (a few cp of quantisation error):

```
erebus --eval-file erebus-mid-<N>/quantised.bin   # then: position fen <fen> / eval
```

A large gap means the save layout and the engine's parser disagree.
