# C33 - Does the oracle asymmetry replicate on a second generator?

**Verdict: DOES NOT REPLICATE.**  On `entropy/gpt2_zinc_87m` the oracle is worth
`0.4162` of the deployed arm's reported gap on `aromatic_rings`, is **not defined** on
`hbd_count`, and is `1.4386` on `qed`.  C27's generator-1 shares were `0.8756` /
`0.8819` / `0.8594`.  Decision rule F1 - "the oracle is worth at least half the
reported gap on every anchor where the share is computable" - **does not fire**,
because `aromatic_rings` returns `0.4162 < 0.50`.

That is the headline and it is not the whole result.  The *direction* of C27's finding
survives intact on all three anchors: equalising the information always makes the
deployed guided arm look better, never worse.  What does not survive is the
**magnitude**, and the reason is measurable rather than mysterious - on generator 2 the
deployed guidance cells sit at a token budget where the two best-of-N curves have barely
separated, so the denominator of the share is a gap of `0.0731` / `0.0664` / `0.1163`
hit-rate points instead of C27's `0.3093` / `0.2180` / `0.3193`.  The share is a ratio
of two quantities that both moved.

**The verdict is accounting-dependent and this is stated up front.**  Under the
pre-registered pessimistic accounting S1 - which charges head scoring a full re-read of
every candidate, an accounting C33.0.3 argues is wrong - F1 *would* fire and the verdict
would be REPLICATES IN DIRECTION, NOT IN MAGNITUDE.  See C33.10.

This section scores `outputs/c33_prereg/C33.0_preregistration.md` verbatim, including
where it fails, and the pre-registration is **not amended**.  Its block from
`## C33.0.1 Why this experiment exists` onward is reproduced byte-identically at the end
of this file (C33.14).

---

## C33.1 What was run

| item | value |
| --- | --- |
| generator | `entropy/gpt2_zinc_87m` @ `f42a5a10e24c0350aeadb50865bd90a714d0b2bf` |
| parameters | 87,331,584, all frozen (`all_parameters_frozen: true`, `training_mode: false`) |
| architecture | GPT-2, full softmax attention, 12 blocks, 13 probe points, 2707-token byte-level BPE |
| anchors | `aromatic_rings`, `hbd_count`, `qed` - C31's three |
| generation seeds | 101 / 202 / 303, pool seed `seed * 1000` |
| pool | 16,384 molecules per seed, `generation.sample_unconditional` with C31's base policy |
| N grid | 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32, all disjoint consecutive groups over the pool |
| arms | `oracle_selected`, `head_selected`, `head_selected_at_75pct` |
| heads | C31 Stage 2's, read off disk; probe point 12, head seed 1234, `frozen_state` |
| guided cells re-priced | all 30 C31 k-sweep cells; **none re-run** |
| accounting | processed generator tokens (`actual`) |
| wall time | 175.5 s for the whole three-anchor sweep (**no wall-clock claim is made**) |

Runner `scripts/27_c33_oracle_asymmetry_gen2.py`; summariser `scripts/27_summarise_c33.py`;
artifacts `outputs/c33_headsel_{prop}/head_selected_metrics.json`,
`outputs/c33_pool/`, `outputs/c33_selections/`, `outputs/c33_summary/c33_metrics.json`.
Nothing outside `outputs/c33_*` was written.  No head was trained, no weight changed, no
interval was re-derived, no bootstrap was computed, nothing was committed.

The pool and every per-arm selection are on disk, as C33.0.2 required, so no future
experiment has to regenerate them: `pool_seed{seed}.npz` carries the token counts, the
content lengths, both read positions, the concatenated sequence ids with offsets, the
per-anchor head probabilities at both positions, the true property values and the true
hit mask; `pool_seed{seed}_smiles.json` carries the decoded SMILES;
`c33_selections/{prop}_seed{seed}.npz` carries the selected index of every arm at every N.

---

## C33.2 Gates, as numeric residuals, before any headline

| gate | blocking | criterion | result |
| --- | --- | --- | --- |
| **G1** pool identity | yes | regenerated `oracle_selected` reproduces `outputs/c31_bestofn_{prop}/n_sweep_metrics.json` at every grid point and seed | max abs hit-rate residual **`0.0`**, max abs token residual **`0.0`**, over 33 cells per anchor (99 total) - **PASSES** |
| **G2** head provenance | no | parameter-level identity plus metadata agreement | **PASSES** on all three anchors, 5/5 C31 deployed k cells agreeing per anchor |
| **G3(a)** cost identity within C33 | yes | tokens per returned molecule identical across arms at every N | max abs residual **`0.0`** - **PASSES** |
| **G3(b)** cost identity against C31 | no | re-priced budgets equal the recorded ones; `tokens mod (k+1) == 0` | max abs residual **`0.0`**; recomputed from per-seed records, residual **`0.0`**; max cost-identity residual **`0`** over 30 cells - **PASSES** |
| **G4** N=1 identity | no | all arms and C31 agree at N=1 | max abs residual **`0.0`** - **PASSES** |
| **G5** head discriminates | no | terminal-position pool AUROC, near-chance iff min over seeds `< 0.55` | `0.9028` / `0.8733` / `0.8067` mean - **PASSES**, no anchor near chance |
| **G6** frozen interval is C31's | yes | `(lo, hi, base_rate)` equals every C31 deployed cell's; files SHA-256'd | **PASSES**, hashes stable |

**All three blocking gates pass**, so the headline is stated rather than withheld.

### C33.2.1 G1 in full

The maximum absolute hit-rate residual is exactly `0.0` and the maximum absolute token
residual is exactly `0.0`, on all three anchors, at all 11 grid points and all 3 seeds.
There is no rounding here to look through: the regenerated pool reproduces C31's
`oracle_selected` curve bit for bit, which is what proves that drawing one pool per seed
and sharing it across three anchors did not alter it.  Prediction Q1 is confirmed.

### C33.2.2 G2 in full

| anchor | checkpoint | probe point | file SHA-256 (first 16) | parameter SHA-256 (first 16) | bytes | bins | mask bins | binner | C31 depth-sweep test AUROC |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |
| aromatic_rings | `head_aromatic_rings_L12_seed1234.pt` | 12 | `6bb5ec17d86019a7` | `88dca3671bdfdef8` | 1066796 | 6 | 1 | categorical | 0.8204 |
| hbd_count | `head_hbd_count_L12_seed1234.pt` | 12 | `bb661e32af9becda` | `7e28a905759183ae` | 1067760 | 7 | 1 | categorical | 0.7836 |
| qed | `head_qed_L12_seed1234.pt` | 12 | `290630bf62e84d74` | `871d0c4c056f64a0` | 1081576 | 20 | 2 | quantile | 0.7284 |

All three checkpoints are `in_dim = 768`, `hidden_dim = 256`, `dropout = 0.1`,
`input = frozen_state`, `head_seed = 1234`, `layer == probe_point == 12`.  The binner and
the interval mask travel inside the checkpoint, so binning is matched to guidance by
construction.  Probe point 12 is `hidden_states[12]`, the final hidden layer of a 12-block
model; C27's generator-1 head was recorded as `layer = -1` for the structurally identical
state.

**The file-hash criterion, scored honestly.**  C27 pre-registered a *file* SHA-256 gate
and it failed by construction.  C33.0.4 declared in advance that C33 would report the file
hash as evidence and use parameter-level identity as the criterion.  C33 demonstrates the
failure directly rather than asserting it: the summariser saves **one identical checkpoint
dict twice, under two different file names, in a temporary directory it then deletes**, and
records what comes out.

| quantity | value |
| --- | --- |
| file SHA-256, save A | `18184e46c16756d64eac0f3d70e081d771f7a15727a59559bbbe8371491bfbae` |
| file SHA-256, save B | `8a210d7161ab001d3d32549bf99aeb767b2bd880fcb72a80dbf55ea861899eba` |
| file SHA-256 equal | **false** |
| file bytes, A and B | 1066540 and 1066540, difference `0` |
| parameter SHA-256, both | `88dca3671bdfdef8fd3f692a2d4352bc6cf3a4e34633847337adec46c3a43314` |
| tensors identical | **true** |

Two saves of the same dict produce files of *identical length* and *different bytes*, with
byte-identical tensors.  A file hash tests serialisation, not content.  This is recorded
under `gates.G2_head_provenance.c27_file_hash_criterion_failed_by_construction` and is
reproducible by re-running the summariser.

### C33.2.3 G5 in full

| anchor | terminal AUROC (101 / 202 / 303) | mean | 75% AUROC mean | terminal > 75% on every seed | pool true hit rate |
| --- | --- | ---: | ---: | --- | ---: |
| aromatic_rings | 0.9047 / 0.9003 / 0.9035 | 0.9028 | 0.8765 | yes | 0.1242 / 0.1230 / 0.1254 |
| hbd_count | 0.8695 / 0.8795 / 0.8710 | 0.8733 | 0.8255 | yes | 0.0815 / 0.0883 / 0.0892 |
| qed | 0.8064 / 0.8048 / 0.8089 | 0.8067 | 0.7783 | yes | 0.1012 / 0.1040 / 0.1005 |

No anchor is near chance, so the head arm measures something on all three and no anchor's
comparison is voided.  Every anchor's pool AUROC exceeds its own C31 held-out depth-sweep
AUROC, which is expected: the pool AUROC is read at the terminal position of a completed
molecule, the depth sweep averages over all four prefix quartiles.

---

## C33.3 The two curves, with every per-seed value

`hit_rate` is over molecules RDKit parsed and scored - the same denominator every C31
guidance cell uses.  `head_selected` does **not** down-rank invalid candidates, because
RDKit validity is oracle information; the consequence is visible in
`hit_rate_over_all_returned`, which sits `0.0001` to `0.0025` below `hit_rate` for the head
arms.  Head-arm validity stays in [`0.9954`, `0.9995`].  The oracle arm reaches validity
`1.0000` at every N >= 2 and `0.9982` at N=1, where there is nothing to select and the
oracle arm cannot avoid an invalid draw either; its `hit_rate_over_all_returned` deficit is
therefore `0.0` for N >= 2 and at most `0.00022` at N=1.  Uniqueness is `1.0000` for every
arm at every N.

### aromatic_rings

| N | tok/mol | oracle mean | oracle 101 / 202 / 303 | head mean | head 101 / 202 / 303 | 75% mean | 75% 101 / 202 / 303 | gap |
| ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | 36.0 | 0.1244 | 0.1244 / 0.1232 / 0.1256 | 0.1244 | 0.1244 / 0.1232 / 0.1256 | 0.1244 | 0.1244 / 0.1232 / 0.1256 | 0.0000 |
| 2 | 72.1 | 0.2325 | 0.2350 / 0.2281 / 0.2343 | 0.2124 | 0.2150 / 0.2112 / 0.2112 | 0.2074 | 0.2086 / 0.2044 / 0.2091 | 0.0200 |
| 3 | 108.1 | 0.3275 | 0.3303 / 0.3221 / 0.3302 | 0.2770 | 0.2762 / 0.2739 / 0.2809 | 0.2681 | 0.2687 / 0.2635 / 0.2720 | 0.0505 |
| 4 | 144.2 | 0.4119 | 0.4170 / 0.4053 / 0.4136 | 0.3265 | 0.3310 / 0.3275 / 0.3209 | 0.3113 | 0.3164 / 0.3058 / 0.3117 | 0.0855 |
| 6 | 216.3 | 0.5458 | 0.5564 / 0.5366 / 0.5443 | 0.3980 | 0.4068 / 0.3891 / 0.3979 | 0.3748 | 0.3828 / 0.3695 / 0.3719 | 0.1478 |
| 8 | 288.3 | 0.6514 | 0.6558 / 0.6470 / 0.6514 | 0.4476 | 0.4477 / 0.4521 / 0.4430 | 0.4127 | 0.4166 / 0.4082 / 0.4133 | 0.2038 |
| 9 | 324.4 | 0.6881 | 0.6907 / 0.6802 / 0.6934 | 0.4599 | 0.4606 / 0.4631 / 0.4559 | 0.4255 | 0.4282 / 0.4207 / 0.4275 | 0.2282 |
| 12 | 432.5 | 0.7958 | 0.7993 / 0.7868 / 0.8015 | 0.5070 | 0.5143 / 0.5029 / 0.5037 | 0.4696 | 0.4728 / 0.4666 / 0.4695 | 0.2889 |
| 16 | 576.7 | 0.8809 | 0.8877 / 0.8672 / 0.8877 | 0.5367 | 0.5436 / 0.5235 / 0.5431 | 0.5018 | 0.5132 / 0.4843 / 0.5078 | 0.3442 |
| 24 | 865.1 | 0.9536 | 0.9531 / 0.9501 / 0.9575 | 0.5814 | 0.6003 / 0.5727 / 0.5712 | 0.5468 | 0.5544 / 0.5279 / 0.5582 | 0.3722 |
| 32 | 1153.4 | 0.9811 | 0.9863 / 0.9707 / 0.9863 | 0.6047 | 0.6314 / 0.5879 / 0.5949 | 0.5546 | 0.5745 / 0.5333 / 0.5560 | 0.3764 |

### hbd_count

| N | tok/mol | oracle mean | oracle 101 / 202 / 303 | head mean | head 101 / 202 / 303 | 75% mean | 75% 101 / 202 / 303 | gap |
| ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | 36.0 | 0.0865 | 0.0816 / 0.0885 / 0.0893 | 0.0865 | 0.0816 / 0.0885 / 0.0893 | 0.0865 | 0.0816 / 0.0885 / 0.0893 | 0.0000 |
| 2 | 72.1 | 0.1654 | 0.1560 / 0.1689 / 0.1711 | 0.1467 | 0.1378 / 0.1506 / 0.1518 | 0.1375 | 0.1304 / 0.1406 / 0.1414 | 0.0186 |
| 3 | 108.1 | 0.2366 | 0.2256 / 0.2395 / 0.2446 | 0.1890 | 0.1821 / 0.1940 / 0.1908 | 0.1732 | 0.1646 / 0.1771 / 0.1779 | 0.0476 |
| 4 | 144.2 | 0.3018 | 0.2878 / 0.3064 / 0.3113 | 0.2243 | 0.2129 / 0.2334 / 0.2268 | 0.2015 | 0.1946 / 0.2042 / 0.2056 | 0.0775 |
| 6 | 216.3 | 0.4151 | 0.4000 / 0.4176 / 0.4278 | 0.2739 | 0.2621 / 0.2861 / 0.2735 | 0.2400 | 0.2388 / 0.2445 / 0.2366 | 0.1412 |
| 8 | 288.3 | 0.5151 | 0.4985 / 0.5229 / 0.5239 | 0.3168 | 0.3000 / 0.3431 / 0.3074 | 0.2715 | 0.2594 / 0.2827 / 0.2724 | 0.1983 |
| 9 | 324.4 | 0.5551 | 0.5423 / 0.5555 / 0.5676 | 0.3262 | 0.3102 / 0.3469 / 0.3214 | 0.2844 | 0.2754 / 0.3001 / 0.2775 | 0.2289 |
| 12 | 432.5 | 0.6623 | 0.6520 / 0.6593 / 0.6755 | 0.3658 | 0.3446 / 0.4010 / 0.3517 | 0.3206 | 0.3191 / 0.3248 / 0.3179 | 0.2965 |
| 16 | 576.7 | 0.7585 | 0.7441 / 0.7490 / 0.7822 | 0.3994 | 0.3803 / 0.4338 / 0.3842 | 0.3407 | 0.3320 / 0.3497 / 0.3405 | 0.3591 |
| 24 | 865.1 | 0.8939 | 0.8856 / 0.8798 / 0.9164 | 0.4437 | 0.4200 / 0.4845 / 0.4267 | 0.3851 | 0.3695 / 0.4021 / 0.3838 | 0.4502 |
| 32 | 1153.4 | 0.9453 | 0.9453 / 0.9316 / 0.9590 | 0.4703 | 0.4364 / 0.5226 / 0.4521 | 0.3957 | 0.3691 / 0.4227 / 0.3953 | 0.4750 |

### qed

| N | tok/mol | oracle mean | oracle 101 / 202 / 303 | head mean | head 101 / 202 / 303 | 75% mean | 75% 101 / 202 / 303 | gap |
| ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | 36.0 | 0.1021 | 0.1014 / 0.1042 / 0.1006 | 0.1021 | 0.1014 / 0.1042 / 0.1006 | 0.1021 | 0.1014 / 0.1042 / 0.1006 | 0.0000 |
| 2 | 72.1 | 0.1934 | 0.1921 / 0.1971 / 0.1908 | 0.1590 | 0.1585 / 0.1629 / 0.1554 | 0.1539 | 0.1518 / 0.1592 / 0.1506 | 0.0344 |
| 3 | 108.1 | 0.2749 | 0.2723 / 0.2794 / 0.2728 | 0.1903 | 0.1888 / 0.1938 / 0.1881 | 0.1833 | 0.1805 / 0.1885 / 0.1808 | 0.0846 |
| 4 | 144.2 | 0.3503 | 0.3464 / 0.3572 / 0.3472 | 0.2173 | 0.2117 / 0.2244 / 0.2158 | 0.2048 | 0.2006 / 0.2123 / 0.2015 | 0.1329 |
| 6 | 216.3 | 0.4752 | 0.4700 / 0.4832 / 0.4725 | 0.2460 | 0.2441 / 0.2565 / 0.2373 | 0.2338 | 0.2373 / 0.2357 / 0.2285 | 0.2292 |
| 8 | 288.3 | 0.5776 | 0.5703 / 0.5869 / 0.5757 | 0.2601 | 0.2598 / 0.2653 / 0.2553 | 0.2478 | 0.2454 / 0.2477 / 0.2502 | 0.3175 |
| 9 | 324.4 | 0.6229 | 0.6165 / 0.6341 / 0.6181 | 0.2599 | 0.2604 / 0.2573 / 0.2620 | 0.2537 | 0.2523 / 0.2490 / 0.2598 | 0.3630 |
| 12 | 432.5 | 0.7294 | 0.7128 / 0.7509 / 0.7245 | 0.2739 | 0.2769 / 0.2801 / 0.2647 | 0.2648 | 0.2634 / 0.2720 / 0.2590 | 0.4555 |
| 16 | 576.7 | 0.8167 | 0.8057 / 0.8379 / 0.8066 | 0.2840 | 0.2920 / 0.2903 / 0.2698 | 0.2844 | 0.2845 / 0.2852 / 0.2835 | 0.5327 |
| 24 | 865.1 | 0.9247 | 0.9150 / 0.9501 / 0.9091 | 0.3019 | 0.2947 / 0.3186 / 0.2922 | 0.2886 | 0.2947 / 0.2819 / 0.2893 | 0.6229 |
| 32 | 1153.4 | 0.9648 | 0.9668 / 0.9746 / 0.9531 | 0.3051 | 0.2891 / 0.3307 / 0.2955 | 0.3075 | 0.3145 / 0.2891 / 0.3190 | 0.6597 |

**The single most important structural fact about generator 2 is in these tables.**  The
head arm tracks the oracle arm closely at small N and then falls away: on `qed` the head
arm is essentially flat from N=8 (`0.2601`) to N=32 (`0.3051`) while the oracle arm climbs
from `0.5776` to `0.9648`.  The head can rank two or three candidates but cannot find the
best of thirty-two.  On generator 1 the two arms stayed much closer, which is precisely
why C27's shares were large and stable.

---

## C33.4 Budget-matched gaps

`gap(N) = oracle_curve(N) - head_curve(N)`, well defined everywhere, is the last column of
each table above.  Maxima over the grid: `aromatic_rings` `0.3764` at N=32, `hbd_count`
`0.4750` at N=32, `qed` `0.6597` at N=32.

`gap(b) = oracle_curve(b) - head_curve(b)` at each guided cell's own measured budget is
tabulated per cell in C33.5.  At the **deployed k=2 budget** - the pre-registered headline
point - it is:

| anchor | deployed k=2 budget (tok/mol) | bracketing N | gap(b) on generator 2 | C27's generator-1 gap |
| --- | ---: | --- | ---: | ---: |
| aromatic_rings | 131.37 | [3, 4] | **0.0731** | 0.3093 |
| hbd_count | 130.78 | [3, 4] | **0.0664** | 0.2180 |
| qed | 131.79 | [3, 4] | **0.1163** | 0.3193 |

C31's deployed cells cost about 131 tokens per molecule, which on generator 2's frontier
lands between N=3 and N=4 - the region where the oracle and head curves have separated by
only a few hit-rate points.  This is the arithmetic reason the share behaves differently,
and it is a fact about *where C31's cells sit on the token axis*, not about the head.

---

## C33.5 The oracle share

Computed **only where `adv_oracle(c) < 0`**, per C33.0.6 rule 1.  Cells whose advantage
against the oracle curve is not negative are in the separate table C33.5.2, with both raw
advantages, and their share cell is left explicitly undefined - never filled, never
imputed, never sign-flipped.  Shares above 1 are reported as reversals and **not clipped**.

### C33.5.1 The headline: the deployed arm at k = 2, one cell per anchor

Fixed in the pre-registration (C33.0.6 rule 4) and not selected from the 30 afterwards.

| anchor | guided hit rate (101 / 202 / 303) | adv vs oracle curve | 95% t interval, 2 df | adv vs head curve | 95% t interval, 2 df | oracle share | gen-1 share |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| aromatic_rings | 0.2064 (0.2344 / 0.2129 / 0.1719) | **-0.1756** | [-0.2621, -0.0890] | **-0.1025** | [-0.1800, -0.0248] | **0.4162** | 0.8756 |
| hbd_count | 0.3092 (0.3125 / 0.3047 / 0.3105) | **+0.0317** | [-0.0045, +0.0678] | **+0.0980** | [+0.0657, +0.1303] | **not defined (advantage vs oracle curve is not negative)** | 0.8819 |
| qed | 0.2435 (0.2422 / 0.2285 / 0.2598) | **-0.0809** | [-0.1297, -0.0321] | **+0.0355** | [-0.0128, +0.0837] | **1.4386** | 0.8594 |

Per-seed advantages, each interpolating that seed's own curve at that seed's own budget:

| anchor | adv vs oracle, 101 / 202 / 303 | sd | adv vs head, 101 / 202 / 303 | sd |
| --- | --- | ---: | --- | ---: |
| aromatic_rings | -0.1470 / -0.1652 / -0.2143 | 0.0348 | -0.0741 / -0.0972 / -0.1359 | 0.0312 |
| hbd_count | +0.0484 / +0.0222 / +0.0244 | 0.0145 | +0.1114 / +0.0854 / +0.0973 | 0.0130 |
| qed | -0.0765 / -0.1023 / -0.0638 | 0.0196 | +0.0390 / +0.0144 / +0.0528 | 0.0194 |

Read plainly:

- **`aromatic_rings`: the oracle accounts for `0.4162` of the deployed gap, not `0.8756`.**
  Removing the oracle from best-of-N moves the deployed arm from `-0.1756` to `-0.1025`.
  It still loses, and it still loses by a margin whose t interval excludes zero.  This is
  the anchor that kills F1.
- **`hbd_count`: not computable, exactly as pre-registered.**  C33.0.6 rule 5 and
  prediction Q7 committed in advance that this cell's advantage against the oracle curve
  is `+0.0317` and that the share would therefore not exist.  It is `+0.0317`, its t
  interval `[-0.0045, +0.0678]` spans zero, and the cell is reported as **unresolved at
  three generation seeds** under F6.  The empty cell is a scored prediction, not a
  suppressed result.
- **`qed`: share `1.4386`, a reversal.**  `adv_head = +0.0355 > 0`: equalising the
  information does not shrink this gap, it flips its sign.  The value is not clipped.
  **But its t interval `[-0.0128, +0.0837]` spans zero**, so the reversal itself is *not
  resolved at three generation seeds*, and the share inherits that.  A share of `1.4386`
  whose numerator's sign is unresolved should not be quoted as "the oracle was worth 144%
  of the gap"; it should be quoted as "on `qed` the deployed arm is at or slightly above
  the equal-information curve, and three seeds cannot say which".

### C33.5.2 Cells whose advantage against the oracle curve is not negative

Seven of 30 cells.  Share undefined by construction; both raw advantages given.

| cell | adv vs oracle curve | adv vs head curve | gap(b) | share |
| --- | ---: | ---: | ---: | --- |
| `aromatic_rings_mid_L2_lam2_k2` | +0.1724 | +0.2408 | 0.0683 | not defined |
| `aromatic_rings_mid_L2_lam2_k4` | +0.2473 | +0.3728 | 0.1255 | not defined |
| `aromatic_rings_mid_L2_lam2_k8` | +0.0595 | +0.2860 | 0.2266 | not defined |
| `hbd_count_deployed_L12_lam1_k2` | +0.0317 | +0.0980 | 0.0664 | not defined |
| `hbd_count_mid_L2_lam2_k2` | +0.2295 | +0.2918 | 0.0622 | not defined |
| `hbd_count_mid_L2_lam2_k4` | +0.1696 | +0.2889 | 0.1193 | not defined |
| `hbd_count_mid_L2_lam2_k8` | +0.0159 | +0.2454 | 0.2294 | not defined |

The gap carries over where the share does not, and it is uniformly positive: at every one
of these budgets the equal-information comparator is `0.0622` to `0.2294` hit-rate points
below the oracle comparator.

### C33.5.3 All 30 cells

Budgets and hit rates are C31's, unchanged; the two advantages, the gap and the share are
C33's.  `--` marks a share that is not defined; `!` marks a share carrying the
"not resolved at three generation seeds" flag of C33.0.6 rule 3.

| cell | hit | tok/mol | adv vs oracle | adv vs head | gap(b) | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aromatic_rings_deployed_L12_lam1_k2` | 0.2064 | 131.37 | -0.1756 | -0.1025 | 0.0731 | 0.4162 |
| `aromatic_rings_deployed_L12_lam1_k4` | 0.3781 | 200.74 | -0.1389 | -0.0045 | 0.1344 | 0.9678 |
| `aromatic_rings_deployed_L12_lam1_k8` | 0.4234 | 338.05 | -0.2783 | -0.0425 | 0.2359 | 0.8474 |
| `aromatic_rings_deployed_L12_lam1_k16` | 0.4326 | 613.15 | -0.4575 | -0.1098 | 0.3477 | 0.7600 |
| `aromatic_rings_deployed_L12_lam1_k32` | 0.4410 | 1162.86 | -0.5401 | -0.1637 | 0.3764 | 0.6969 |
| `aromatic_rings_mid_L2_lam2_k2` | 0.5430 | 126.49 | +0.1724 | +0.2408 | 0.0683 | -- |
| `aromatic_rings_mid_L2_lam2_k4` | 0.7451 | 190.41 | +0.2473 | +0.3728 | 0.1255 | -- |
| `aromatic_rings_mid_L2_lam2_k8` | 0.7451 | 321.98 | +0.0595 | +0.2860 | 0.2266 | -- |
| `aromatic_rings_mid_L2_lam2_k16` | 0.7438 | 585.48 | -0.1392 | +0.2058 | 0.3450 | 2.4776 |
| `aromatic_rings_mid_L2_lam2_k32` | 0.7271 | 1118.78 | -0.2507 | +0.1252 | 0.3759 | 1.4992 |
| `hbd_count_deployed_L12_lam1_k2` | 0.3092 | 130.78 | +0.0317 | +0.0980 | 0.0664 | -- |
| `hbd_count_deployed_L12_lam1_k4` | 0.3837 | 200.28 | -0.0063 | +0.1208 | 0.1271 | 20.1802 ! |
| `hbd_count_deployed_L12_lam1_k8` | 0.3424 | 340.49 | -0.2286 | +0.0104 | 0.2390 | 1.0453 |
| `hbd_count_deployed_L12_lam1_k16` | 0.3496 | 617.87 | -0.4282 | -0.0561 | 0.3721 | 0.8690 |
| `hbd_count_deployed_L12_lam1_k32` | 0.3418 | 1163.57 | -0.6035 | -0.1286 | 0.4750 | 0.7870 |
| `hbd_count_mid_L2_lam2_k2` | 0.4980 | 125.78 | +0.2295 | +0.2918 | 0.0622 | -- |
| `hbd_count_mid_L2_lam2_k4` | 0.5457 | 191.42 | +0.1696 | +0.2889 | 0.1193 | -- |
| `hbd_count_mid_L2_lam2_k8` | 0.5719 | 325.21 | +0.0159 | +0.2454 | 0.2294 | -- |
| `hbd_count_mid_L2_lam2_k16` | 0.5614 | 593.05 | -0.2047 | +0.1595 | 0.3642 | 1.7793 |
| `hbd_count_mid_L2_lam2_k32` | 0.5592 | 1099.48 | -0.3765 | +0.0938 | 0.4703 | 1.2491 |
| `qed_deployed_L12_lam1_k2` | 0.2435 | 131.79 | -0.0809 | +0.0355 | 0.1163 | 1.4386 |
| `qed_deployed_L12_lam1_k4` | 0.2376 | 200.86 | -0.2109 | -0.0022 | 0.2087 | 0.9895 |
| `qed_deployed_L12_lam1_k8` | 0.2129 | 338.26 | -0.4237 | -0.0488 | 0.3749 | 0.8848 |
| `qed_deployed_L12_lam1_k16` | 0.2332 | 612.61 | -0.5970 | -0.0530 | 0.5439 | 0.9112 |
| `qed_deployed_L12_lam1_k32` | 0.2249 | 1161.34 | -0.7399 | -0.0802 | 0.6597 | 0.8916 |
| `qed_mid_L6_lam2_k2` | 0.3055 | 129.32 | -0.0137 | +0.0994 | 0.1130 | 8.2766 ! |
| `qed_mid_L6_lam2_k4` | 0.3112 | 201.19 | -0.1379 | +0.0712 | 0.2091 | 1.5165 |
| `qed_mid_L6_lam2_k8` | 0.3212 | 339.68 | -0.3168 | +0.0593 | 0.3761 | 1.1873 |
| `qed_mid_L6_lam2_k16` | 0.3032 | 620.87 | -0.5300 | +0.0165 | 0.5465 | 1.0311 |
| `qed_mid_L6_lam2_k32` | 0.2808 | 1171.61 | -0.6841 | -0.0243 | 0.6597 | 0.9644 |

Four cells are flagged `extrapolated_beyond_grid` because their budget exceeds the N=32
point of the curve and they are therefore compared against the curve's terminal value, as
C28 and C31 pre-registered: `aromatic_rings_deployed_L12_lam1_k32`,
`hbd_count_deployed_L12_lam1_k32`, `qed_deployed_L12_lam1_k32` and `qed_mid_L6_lam2_k32`.
They are the same four C31 flagged against its own oracle curve.

**Note what happens at large k on the deployed arm.**  `aromatic_rings` k=4 through k=32
gives shares `0.9678`, `0.8474`, `0.7600`, `0.6969` and `qed` gives `0.9895`, `0.8848`,
`0.9112`, `0.8916` - squarely in C27's band.  The anchor that fails F1 fails it **only at
k=2**, the single cheapest cell, which is the one the pre-registration fixed as the
headline.  This is reported rather than repaired: the headline was chosen in advance for a
reason (it is the closest analogue of C27's deployed comparison and the cheapest point of
C31's grid) and moving it after seeing the numbers would be exactly the sin the
pre-registration exists to prevent.

---

## C33.6 Arm counts

Counted on the point-estimate advantage, as C27 counted them.

| curve the cells are priced against | generator 2 (30 cells) | generator 1, C27 (46 cells) |
| --- | ---: | ---: |
| above `oracle_selected` | **7** | 1 |
| above `head_selected` | **18** | 15 |

Per anchor on generator 2:

| anchor | cells | above oracle curve | above head curve |
| --- | ---: | ---: | ---: |
| aromatic_rings | 10 | 3 | 5 |
| hbd_count | 10 | 4 | 8 |
| qed | 10 | 0 | 5 |

F3 fires: `18 > 7`.  Equalising the information more than doubles the number of guided
configurations that sit above the best-of-N curve at their own budget, on generator 2 as
on generator 1.  This is the part of C27 that replicates cleanly.

---

## C33.7 The diagnostic arm

`head_selected_at_75pct` is the same head read at content position `max(1, floor(3n/4))`.
It was fixed in C33.0.2 arm 3, before any number was seen, so it cannot be produced
afterwards as a rescue.

F5 requires the 75% arm to lie strictly between the N=1 base rate and `head_selected` at
every N >= 2.  **It fires on `aromatic_rings` and `hbd_count` and fails on `qed`**, at
N=16 (`0.2844` vs the terminal arm's `0.2840`) and N=32 (`0.3075` vs `0.3051`): on `qed`
the 75%-position head is, at large N, indistinguishable from or marginally better than the
terminal-position head.  Prediction Q8 committed in advance that F5 would fire on at most
two anchors, and it fires on exactly two.

The reading: on `aromatic_rings` and `hbd_count`, a real part of the head arm's ranking
power comes from the molecule being finished - the 75% arm reaches `0.5546` and `0.3957`
at N=32 where the terminal arm reaches `0.6047` and `0.4703`.  On `qed` the terminal
advantage has already evaporated by N=16, which is consistent with `qed` having the
weakest head (pool AUROC `0.8067`, a 20-bin quantile binner with a 2-bin interval mask)
and with `qed`'s head arm being nearly flat past N=8.

Agreement with the oracle arm's own selection falls from `0.7696` / `0.7344` / `0.7349`
at N=2 to `0.1484` / `0.1797` / `0.1029` at N=32, so the head arm is genuinely selecting
different molecules, not shadowing the oracle.

---

## C33.8 Decision rules, scored as written

| rule | fires | evidence |
| --- | --- | --- |
| **F1** the oracle asymmetry replicates | **NO** | `aromatic_rings` share `0.4162 < 0.50`; `qed` `1.4386 >= 0.50`; `hbd_count` not computable |
| **F2** it replicates at C27's magnitude | **NO** | neither computable share is in [0.75, 1.00]: `0.4162` below, `1.4386` above |
| **F3** equalising information changes the arm count | **YES** | 18 above the head curve vs 7 above the oracle curve, of 30 |
| **F4** the head arm is not degenerate | **YES** | N=1 to N=32 gains `0.4803` / `0.3839` / `0.2030`, all >= 0.02 |
| **F5** the diagnostic arm behaves as C27's did | **NO** | fires on 2 of 3 anchors; `qed` violates at N=16 and N=32 |
| **F6** the honesty rule | **YES** (fires) | 2 cells whose seed sd exceeds their own mean advantage; `hbd_count` deployed k=2's `adv_oracle` t interval spans zero |

**F6 in detail.**  Cells reported as *not resolved at three generation seeds* rather than
as a win or a loss: `hbd_count_deployed_L12_lam1_k4` (adv vs oracle `-0.0063`, sd
`0.0283`) and `hbd_count_mid_L2_lam2_k8` (adv `+0.0159`, sd `0.0378`).  Anchor
`hbd_count`'s deployed k=2 `adv_oracle` t interval `[-0.0045, +0.0678]` spans zero, so
that anchor is unresolved on top of having no computable share.

**Verdict, by the rule fixed in C33.0.8:** REPLICATES iff F1 and F2; REPLICATES IN
DIRECTION, NOT IN MAGNITUDE iff F1 and not F2; **DOES NOT REPLICATE iff not F1**;
PARTIALLY UNINTERPRETABLE if a blocking gate fails or fewer than two anchors have a
computable share.  All blocking gates pass and two anchors are computable, so the
uninterpretable branch does not apply.  F1 does not fire.

> **DOES NOT REPLICATE** - on generator 2 the oracle is not worth at least half of the
> deployed arm's reported gap on every anchor where the share is computable.

---

## C33.9 Predictions, scored including the falsified ones

| # | prediction | outcome | measured |
| --- | --- | --- | --- |
| Q1 | G1's max abs hit-rate residual is exactly 0.0 on all three anchors | **CONFIRMED** | 0.0 / 0.0 / 0.0 |
| Q2 | terminal pool AUROC > 0.70 on all three, and exceeds the 75% AUROC on all three | **CONFIRMED** | 0.9028 / 0.8733 / 0.8067 vs 0.8765 / 0.8255 / 0.7783 |
| Q3 | F4 fires: head arm gains >= 0.02 from N=1 to N=32 on all three | **CONFIRMED** | 0.4803 / 0.3839 / 0.2030 |
| Q4 | F1 fires on every anchor with a computable deployed-k2 share | **FALSIFIED** | `aromatic_rings` 0.4162 |
| Q5 | F2 fires: every computable share is in [0.75, 1.00] | **FALSIFIED** | 0.4162 and 1.4386 |
| Q6 | F3 fires and `n_arms_above_head_selected_curve` >= 15 of 30 | **CONFIRMED** | 18 of 30, vs 7 above the oracle curve |
| Q7 | the `hbd_count` deployed-k2 share is not computable | **CONFIRMED** | adv vs oracle `+0.0317`, share undefined |
| Q8 | F5 fires on at most two of three anchors | **CONFIRMED** | fires on 2 (`qed` fails) |
| Q9 | gap(b) at the deployed k=2 budget is smaller than C27's on at least two anchors | **CONFIRMED** | smaller on all three: 0.0731 / 0.0664 / 0.1163 vs 0.3093 / 0.2180 / 0.3193 |
| Q10 | no arm of C33 changes any C31 number | **CONFIRMED** | newest C31 artifact mtime `2026-08-03T09:08:01Z`, before the pre-registration freeze at `2026-08-03T13:41:02Z` |

**8 confirmed, 2 falsified.**  The two falsified are Q4 and Q5 - the two that carry the
replication claim.  Q5 was flagged in the pre-registration as the one held most weakly,
for reasons (different depth index, different tokenisation, five cells already crossing)
that turned out to be the right reasons.  Q4 was not flagged as weak, and it failed.

---

## C33.10 Sensitivity S1 - the pessimistic accounting

C33.0.3 argues that head scoring costs zero generator tokens: the head reads the
final-layer state at a position the generator has already computed, the same state the LM
head reads to emit the next position's logits.  Our implementation recomputes those states
in one extra forward pass only because `transformers.generate()` does not expose hidden
states.  That recompute was measured: **`36.0437` processed tokens per pool molecule**,
which is exactly a full re-read of the pool (the pool itself costs `36.0208` / `36.0103` /
`36.1000` tokens per molecule at seeds 101 / 202 / 303, and the recompute costs the same
number at every seed).

S1 charges it in full: the head curve's budget at N becomes
`tokens(N) + N * 36.0437`, i.e. the head curve costs twice what the oracle curve costs.

| quantity | free accounting (pre-registered as correct) | S1 pessimistic |
| --- | ---: | ---: |
| deployed k=2 share, aromatic_rings | 0.4162 | **1.0545** |
| deployed k=2 share, hbd_count | not defined | not defined |
| deployed k=2 share, qed | 1.4386 | **2.1660** |
| arms above the head curve, of 30 | 18 | **23** |
| F1 | does not fire | **fires** |
| F2 | does not fire | does not fire |
| **verdict** | **DOES NOT REPLICATE** | REPLICATES IN DIRECTION, NOT IN MAGNITUDE |

**The C33 verdict survives only under the free accounting, and is labelled as such**, as
C33.0.7 requires.  A reader who rejects the zero-token argument gets a weaker conclusion -
direction replicates, magnitude does not - not the C27 result.  Under neither accounting
does F2 fire, so **no accounting reproduces C27's 0.75-1.00 magnitude on generator 2**.

---

## C33.11 Where the pre-registration failed or was imprecise

Disclosed here.  **The pre-registration is not amended.**

1. **§C33.0.6 miscounts the crossing cells.**  It states that "5 of its 30 cells already
   sit above the oracle-selected curve (all mid-network λ=2)".  On the point-estimate
   criterion that C27 used to count arms, **7** of C31's 30 cells have a positive
   advantage against the oracle curve, and one of them -
   `hbd_count_deployed_L12_lam1_k2` - is a **deployed** arm, not a mid-network one.  The
   figure "5, all mid-network" is C31's *crossing* count, which additionally requires the
   seed t interval to exclude zero and the validity floor to be cleared.  The
   pre-registration conflates two different counts.  Nothing downstream breaks - rule 5 of
   the same section names the `hbd_count` deployed cell's `+0.0317` explicitly, so the
   document contradicts its own parenthetical two paragraphs later - but the arm-count
   comparison in C33.6 uses the point-estimate criterion, and readers comparing "7" here
   against "5" there should know why.

2. **Rule 1 guards the sign of the denominator but not its magnitude.**  The share is
   computed whenever `adv_oracle < 0`, including when `adv_oracle` is `-0.0063`
   (`hbd_count_deployed_k4`, share `20.1802`) or `-0.0137` (`qed_mid_L6_lam2_k2`, share
   `8.2766`).  These numbers are arithmetically correct and substantively meaningless: a
   ratio whose denominator is a hair below zero explodes.  Rule 3 catches both, because
   both cells' `adv_oracle` t intervals span zero and both shares therefore carry the
   "not resolved at three generation seeds" flag - so the pre-registration is
   *self-repairing* here rather than broken.  A future pre-registration should state the
   magnitude guard directly instead of relying on the interval to catch it.

3. **The share is fragile in exactly the way C33.0.9's second attack anticipated.**  The
   two computable generator-2 shares straddle C27's band from both sides (`0.4162` and
   `1.4386`), and their denominators are 3 to 5 times smaller than generator 1's.  The
   pre-registration was right to require the raw gap alongside the share; the gap is what
   is comparable across generators and it is reported everywhere the share is (C33.4,
   C33.5.3).

4. **Two defects in the runner were found and fixed before the measurement run**, and are
   disclosed because "the code compiled" is not the same as "the code was correct".
   (a) The G2 cross-check against `outputs/c31_heads/depth_{prop}.json` indexed a `rows`
   key that in C31's artifact is a dict of split names, not a list of per-probe-point
   rows; it raised `AttributeError` and was repointed at `by_probe_point[str(L)]
   ["per_seed"][...]["test"]["intervals"]["target"]["auroc"]`, which is where the number
   actually lives.  (b) The single shared hidden-state pass used
   `props[0]`'s probe point for all anchors with no check that the anchors agree; all
   three C31 deployed heads read probe point 12, so no number was affected, but a guard
   was added that stops C33 rather than mis-scoring if a future config disagrees.  Both
   fixes preceded every measurement artifact; the pre-registration's mtime
   (`2026-08-03T13:41:02Z`) still strictly precedes the earliest C33 artifact
   (`2026-08-03T15:52:18Z`), which `tests/test_oracle_asymmetry_gen2.py` asserts.

5. **No deviation was necessary from the pre-registered design itself.**  Every anchor,
   seed, grid point, arm, gate, statistic and threshold is as written.  The only
   substantive judgement C33 had to make - what to do with two near-zero negative
   denominators - was already answered by rule 3.

---

## C33.12 Limitations

- **Three generation seeds.**  Every interval here is a Student t interval on 2 df with
  `t = 4.302653`.  At n = 3 that interval is wide and the per-seed values are published
  beside every mean so a reader can see exactly how wide.  **No bootstrap is computed
  anywhere**: at n = 3 the percentile bootstrap of a mean is identically `[min, max]`,
  since `P(all three resamples hit the minimum) = 1/27 = 0.037 > 0.025`.
- **The share compares two frontiers that differ in base rate, interval, tokenisation and
  in what the guidance arms are.**  C33.0.9 raised this before the run and it is
  unrepairable without re-running generator 1's whole pipeline under generator 2's
  intervals.  The share is the comparable quantity only to the extent that the
  denominators are comparable, and C33.4 shows they are not: generator 2's denominators
  are 3 to 5 times smaller at the headline point.  **The raw gap, not the share, is the
  quantity that travels.**
- **The headline is one cell per anchor**, fixed in advance.  C33.5.3 shows that
  `aromatic_rings` would pass F1 comfortably at k = 4, 8, 16 and 32.  Reporting the k=2
  cell as the headline is what the pre-registration required; reporting the whole grid
  beside it is what stops that requirement from being misleading.
- **The head is scored at the terminal content token**, which is the easiest question the
  head can be asked.  C33.0.9's answer stands: guidance's head also sees the near-complete
  prefix at its last guided step, so nothing is withheld from guidance that arm 2 is
  given; the residual asymmetry is that selection may act *after* the outcome exists while
  steering must commit *before* it.  On generator 2 that near-oracle worry does not even
  arise - the head arm is far *below* the oracle arm at every N >= 2.
- **Two anchors, not three, have a computable share.**  The verdict rests on
  `aromatic_rings` and `qed`.
- **No wall-clock claim is made anywhere**; cost is processed generator tokens throughout.
- **Nothing here revisits generator 1.**  C33 does not re-run C27 and cannot say whether
  C27's own numbers would move under a larger seed count.

---

## C33.13 What C33 changes elsewhere - for the reviewer to merge

C33 edits no existing report, config, test or `outputs/` directory.  The following are
disagreements for the owner to merge, not edits.

1. **The C27 headline must be scoped to generator 1.**  "The oracle is worth ~85-88% of
   the reported gap" is a generator-1 statement.  On generator 2, at the matching deployed
   configuration, it is `0.4162` on one anchor, undefined on a second and `1.4386` on a
   third.  Any merged text stating the share without naming the generator is now wrong.
2. **The portable form of C27's claim is the arm count and the direction, not the share.**
   Both replicate: equalising information moves guided cells above the comparator (15/46
   on generator 1, 18/30 on generator 2, against 1/46 and 7/30) and the deployed arm's
   deficit shrinks on every anchor of both generators.  That is what should be promoted.
3. **The gap should be reported wherever the share is.**  `gap(b)` is defined for every
   cell including the crossing ones; the share is not.
4. **Q10 holds**: no C31 number moved.  C33 re-priced C31's 30 cells and re-ran none of
   them; the newest C31 artifact predates the C33 pre-registration freeze.
5. `reports/pilot_report.md`, `README.md`, `reports/PAPER_WORKSHOP_DRAFT.md` and every
   other section file are untouched by C33 by design.

---

## C33.14 The pre-registration, copied verbatim

The block below is byte-identical to
`outputs/c33_prereg/C33.0_preregistration.md` from `## C33.0.1 Why this experiment exists`
to end of file - 25,884 characters, SHA-256
`55e9b536638ca945e49f7ad3daa649d5610e8574fbdfaa01bbd85285aad15b1e`, as recorded in
`outputs/c33_prereg/prereg_lock.json`.  `tests/test_oracle_asymmetry_gen2.py` asserts the
copy is byte-identical and that the pre-registration's mtime strictly precedes every C33
artifact.  It is reproduced here unchanged, including the passages this section scores as
failures.

---

## C33.0.1 Why this experiment exists

C27 removed a scoping defect that had been in every headline this project produced. Best-of-N
selects its winner with `bestofn.selection_key` evaluated on the **true RDKit property of the
finished molecule** — the ground-truth oracle. Guided decoding only ever sees a learned probe.
"Best-of-N dominates guidance at matched processed tokens" was therefore, in part, a
restatement of "ground truth beats an estimate of it at equal token cost".

C27 re-ran the frontier with best-of-N restricted to the **same head, same probe point, same
target interval, same binning** that guidance steers with — an equal-information comparator —
and the deployed arm's gap collapsed:

| anchor | vs oracle-selected | vs head-selected | oracle share of the gap |
| --- | ---: | ---: | ---: |
| aromatic_rings | -0.3532 | -0.0439 | 0.876 |
| hbd_count | -0.2472 | -0.0292 | 0.882 |
| qed | -0.3715 | -0.0522 | 0.859 |

That is the most portable methodological claim this project has produced: it is about how
these comparisons are *scoped*, not about any one molecule, property or λ. And it currently
rests on **one generator** — GP-MoLFormer-Uniq, 46.8M parameters, linear attention, an
atom-level SMILES vocabulary, one corpus.

C33 replicates it on **generator 2**: `entropy/gpt2_zinc_87m`, revision
`f42a5a10e24c0350aeadb50865bd90a714d0b2bf`, 87,331,584 parameters, GPT-2 with full softmax
attention, byte-level BPE, 12 blocks and therefore 13 probe points — the generator C31 and C32
used. **The question is: on a second, architecturally different generator, is the oracle still
worth ~85–88% of the reported gap?**

Both answers are publishable and neither is to be forced. If the share is again ~0.85–0.88,
the claim is a claim about the *method of comparison* and survives a generator change. If it
is materially different, that is the result and C33 says so. **C33 does not exist to agree
with C27.**

This project has retracted its own numbers three times. The prior that any single-generator
number replicates should be treated as weak.

## C33.0.2 What is run — fixed here, in full

**Anchors** `aromatic_rings`, `hbd_count`, `qed` — C31's three properties, so every C33 number
is directly comparable to a C31 number. **Generation seeds** 101 / 202 / 303, C31's.
**Accounting** `actual` processed generator tokens (`compute.ComputeMeter`);
`processed_tokens_full_recompute` is also recorded. **No wall-clock claim is made anywhere.**

**The generator is frozen.** `eval()`, `requires_grad_(False)` on every parameter. No
fine-tuning, no LoRA, no RL, no activation edit, no weight change of any kind. **No head is
trained for C33**: every head it loads was trained by C31 Stage 2 and is read off disk.

**Windows and target intervals are inherited from C31 and never re-derived.** They are read
from `outputs/c31_zinc50k/target_intervals.json` and `outputs/c31_zinc50k/windows.json` and
SHA-256'd into every C33 artefact. Re-deriving an interval on a fresh sample would silently
move the comparator and void the comparison with C31.

**The pool.** Regenerated with C31's exact unconditional-sampling call and C31's seeds:
`generation.sample_unconditional(gen, cfg["base_policy"], 32 * 512, seed=seed * 1000)` — note
`seed * 1000`, `scripts/06_best_of_n.py`'s convention, which is why C26's and C27's pool gates
were identities. 16,384 molecules per generation seed. Regeneration is necessary because C31
saved no per-molecule best-of-N data. **This time the pool and the per-arm selections are
written to disk** (`outputs/c33_pool/`) so that no future experiment has to regenerate them:
per seed, the sequences, the decoded SMILES, the token cost, the true property values, the
per-candidate selection key components and the head probabilities; and per (property, arm, N),
the selected indices.

The pool depends only on (seed, base policy, generator), so it is identical across the three
anchors — it is drawn **once per seed and shared**, which is a compute optimisation and not a
design change, and G1 is what proves the sharing did not alter it.

**N grid, fixed here:** 1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32 — C26/C27/C31's grid unchanged.
Best-of-N is evaluated over **all disjoint consecutive groups of N over the whole 16,384-
molecule pool** (C26's corrected estimator), not the first N of a slot.

**The arms.** Three, over the same groups:

1. `oracle_selected` — `bestofn.selection_key` on the true RDKit property, invalid and
   property-unavailable candidates ranked worst. C26/C31's arm **verbatim**. It exists as a
   **gate** (G1), not as a new measurement.
2. `head_selected` — the group member with the **highest head-predicted probability that the
   finished molecule lands in the target interval**, `argmax_j TargetScorer(h_j)`, read at the
   **last content token**, ties broken by the lowest index (the first-wins rule `min` gives
   the oracle arm). **No oracle information of any kind**: invalid candidates are **not**
   down-ranked, because RDKit validity is oracle information. A head arm may therefore return
   an unparseable molecule the oracle arm would have rejected. That is a real cost of not
   having ground truth: it is **reported, not repaired**. Validity, uniqueness and
   `hit_rate_over_all_returned` are published for every arm at every grid point.
3. `head_selected_at_75pct` — **secondary/diagnostic**, defined here so that it cannot be
   added afterwards to rescue a verdict. Arm 2 with the head read at content position
   `max(1, floor(3n/4))`. It bounds how much of arm 2 comes from the head seeing a *complete*
   molecule.

**Evaluation is identical for all three arms**: `bestofn.summarise` on the selected molecule,
true RDKit property, the frozen C31 interval, `hit_rate` over molecules RDKit parsed and
scored — the same denominator every C31 guidance cell uses.

**Why the terminal content token, and not `<eos>`.** The head predicts, from the state at a
*partial* molecule, whether the *finished* molecule lands in the interval.
`prefixes.select_quartile_prefixes` draws one prefix per quartile of `[1, n]` and quartile 4's
upper bound is exactly `n`, so the last content token is the `relative_position = 1.0`
endpoint of the head's own training distribution rather than an extrapolation off it. `<eos>`
is deliberately **not** used: no training prefix ever ended there, so scoring at `<eos>` would
be out of distribution. Empty-content sequences fall back to index 0.

The consequence is stated here rather than discovered later: at the terminal position the head
is asked its easiest possible question, so `head_selected` may come close to `oracle_selected`.
If it does, the honest reading is **not** "the head is secretly the oracle" but "the head has
the property information once the molecule exists; what guidance lacks is that information
*early enough to act on*". C33 does not claim to match the *timing* of information, only its
source. §C33.0.9.

## C33.0.3 Which head — resolved from artefacts, or C33 stops

The head must be the one the **C31 deployed arm** used, or the information is not matched. It
is resolved from the C31 artefacts and never assumed:
`outputs/c31_ksweep_{prop}_deployed_L12_lam1_k{k}/k_cell_metrics.json` records `arm:
"deployed"`, `head_input: "frozen_state"`, `head_checkpoint`, an absolute `head_file`,
`head_seed`, `probe_point` and `layer`. C33 reads **all five k cells** of the deployed arm per
property and requires them to agree on all of those fields; the agreed `head_file` is the
checkpoint loaded. The binner and the interval mask travel **inside** the checkpoint, so
binning is matched by construction.

**If provenance does not resolve unambiguously — a missing cell, a disagreement between k
cells, a missing file, a `head_input` that is not `frozen_state`, or a `layer` that does not
equal the `probe_point` — C33 stops and reports that rather than guessing.**

Note the indexing, recorded here so it cannot be confused later: `probe_point = 12` is
`hidden_states[12]`, and this generator has 12 blocks and 13 probe points, so probe point 12
is the final hidden layer and `hidden_states[12] is hidden_states[-1]`. C27's deployed head on
generator 1 was recorded as `layer = -1` for the same state.

**Token accounting for head scoring is zero generator tokens, and this is a claim, not an
oversight.** The head reads the final-layer state at a position the generator has *already*
computed — the same state the LM head reads to emit the next position's logits — so a deployed
head-selecting sampler pays nothing extra. Our implementation nevertheless recomputes those
states in one extra forward pass, because `transformers.generate()` does not expose hidden
states. That is an engineering artefact, not a property of the method. It is measured anyway
and published as `head_scoring_recompute_tokens_per_pool_molecule`, and the pessimistic
sensitivity that charges it is pre-registered as S1 below.

## C33.0.4 Gates — run and reported BEFORE any headline

**G1 — pool identity.** The regenerated `oracle_selected` curve must reproduce
`outputs/c31_bestofn_{prop}/n_sweep_metrics.json` at **every grid point and every generation
seed**, in hit rate and in tokens per returned molecule. The **maximum absolute residual per
property** is reported as a number, never as the word "matches". This is what proves the
regenerated pool is C31's pool. **If it is not exactly 0.0, C33 says so and quantifies it
rather than rounding it away**, and the section states what a nonzero residual does to every
downstream number.

**G2 — head provenance.** For each head actually loaded: the resolved `head_file`, the file
SHA-256, the **parameter-level SHA-256** (an order-stable hash over the sorted `state_dict`
keys and their raw tensor bytes), the checkpoint metadata (`property`, `input`, `head_seed`,
`in_dim`, `hidden_dim`, `n_bins`, `dropout`), the binner kind and parameters, and the number
of bins the interval mask selects. Cross-checked against `outputs/c31_heads/depth_{prop}.json`
for the same probe point and head seed.

C27 pre-registered a **file**-hash criterion for its head gate and that criterion **failed by
construction**: `torch.save` writes a zip whose internal archive directory is named after the
output file, so two saves of one dict under two names differ in bytes while holding identical
tensors. C33 records that finding, does not repeat the mistake, and **scores the file hash as
reported evidence rather than as a pass/fail criterion** — the pass/fail criterion is
parameter-level identity and metadata agreement. This is a change of criterion relative to
C27 and it is declared here, in advance, not discovered afterwards.

**G3 — cost identity.** Two parts, both reported as residuals:
(a) **within C33**, all three arms select from one pool, so tokens per returned molecule must
be identical across arms at every grid point — expected `0.0` by construction;
(b) **against C31**, the re-priced guided cells' `tokens_per_molecule_actual` must equal the
value recorded in each `k_cell_metrics.json` exactly, because C33 re-prices those cells and
does not re-run them, and each cell's `processed_tokens_actual mod (k+1) == 0` identity (C31's
G4) is re-checked from the stored numbers.

**G4 — N=1 is an identity across arms (secondary).** At N=1 there is one candidate and nothing
to select, so all three arms must agree exactly with each other and with C31. Reported as a
residual; a nonzero value is a bug alarm.

**G5 — the head discriminates something at all (secondary).** Per anchor and per seed, the
AUROC of the head's terminal-position `P(y_final ∈ I)` for discriminating true hits among the
16,384 pool molecules, and the same at the 75% position. **If it is near 0.5 the head arm is
measuring nothing, the section must say so, and no comparison against guidance may be drawn
from it on that anchor.** "Near chance" is fixed here as `min over seeds < 0.55`, C27's
threshold reused rather than re-chosen.

**G6 — the frozen interval is C31's.** `target_intervals.json` and `windows.json` in
`outputs/c31_zinc50k/` are SHA-256'd and the `(lo, hi, base_rate)` triple used by C33 is
required to equal the triple recorded in each C31 deployed k cell.

**G1, G3(a) and G6 are blocking**: if any fails, C33 reports the failure and the oracle-share
headline is **not** stated. G2, G4 and G5 are reported and inform interpretation; a G5 failure
voids the head arm on the affected anchor only.

## C33.0.5 The guided arms that get re-priced

**All 30 C31 k-sweep cells** (`outputs/c31_ksweep_*`): 3 properties × 2 arms (deployed
L12 λ=1; mid λ=2, probe point as C31 selected it) × k ∈ {2, 4, 8, 16, 32}. **No new guidance
run is generated and no C31 cell is re-run.** Each cell is interpolated onto **both** curves —
oracle-selected and head-selected — at its **own measured token budget**, linearly in processed
tokens per molecule, using `scripts/21_summarise_c26.py::interp` **imported unmodified**, the
same function that produced C26's, C27's, C28's, C30's and C31's numbers. Per-seed advantages
interpolate each generation seed's own curve at that seed's own budget, so the t interval is
over genuinely paired differences.

Reported: `n_arms_above_head_selected_curve` and `n_arms_above_oracle_selected_curve`, per
property and in total. Any cell whose budget falls outside the measured grid is flagged
`extrapolated_beyond_grid` and compared against the curve's terminal value, exactly as C28 and
C31 pre-registered.

## C33.0.6 The oracle-share statistic — defined here, before any number is seen

For a guided cell `c` with measured budget `b`:

```
adv_oracle(c) = hit_rate(c) - oracle_selected_curve_interpolated_at(b)
adv_head(c)   = hit_rate(c) - head_selected_curve_interpolated_at(b)
oracle_share(c) = 1 - adv_head(c) / adv_oracle(c)
```

which is algebraically `gap(b) / |adv_oracle(c)|` when `adv_oracle(c) < 0`, where
`gap(b) = oracle_curve(b) - head_curve(b) >= 0` is the price of ground truth at that budget.
On C27's generator-1 deployed arms this reproduces 0.876 / 0.882 / 0.859.

**C33 has a problem C27 did not have.** On generator 2, C31 reports that 5 of its 30 cells
already sit **above** the oracle-selected curve (all mid-network λ=2), so some advantages are
*positive*. "The fraction of the gap that was the oracle" is only meaningful where there is a
gap — where the advantage against the oracle curve is **negative**. Dividing by a positive or
near-zero `adv_oracle` produces a number with no interpretation, and dividing by a
sign-flipped one produces a number that looks like a share and is not. The rule, fixed now:

1. **`oracle_share` is computed only when `adv_oracle(c) < 0`.** Where `adv_oracle(c) >= 0`
   the cell is reported in a **separate table** with **both raw advantages** and the share
   cell is left explicitly **"not defined (advantage vs oracle curve is not negative)"** —
   never filled, never imputed, never replaced by a signed variant.
2. A share **> 1** means `adv_head(c) > 0`: equalising the information did not merely shrink
   the gap, it **reversed** it. That is reported as such, in words, and the value is **not
   clipped to 1**.
3. Where the seed-level t interval on `adv_oracle(c)` spans zero, the share is reported with
   the flag **"not resolved at three generation seeds"**, because a share computed on a mean
   whose sign is unresolved inherits that unresolvedness.
4. The **headline** oracle-share table is the **deployed arm at k = 2** — one cell per anchor,
   fixed now, chosen because it is the C31 configuration closest to C27's deployed comparison
   and the cheapest point of C31's grid. All 30 cells are reported too; the headline is not
   selected from them afterwards.
5. C31's own deployed-arm-vs-oracle numbers are `hbd_count +0.0317` (t interval spans zero),
   `aromatic_rings -0.1756` (excludes zero, negative), `qed -0.0809`. **So on the deployed arm
   the share is expected to be computable for `aromatic_rings` and `qed` and NOT for
   `hbd_count`.** The section must say that plainly rather than filling the cell. This is
   written before the C33 run, from C31's published numbers, precisely so that the empty cell
   cannot be read as a result that was hidden.
6. A **budget-matched gap** is also reported, and it is well defined everywhere:
   `gap(b) = oracle_curve(b) - head_curve(b)` at each cell's own budget, and
   `gap(N) = oracle_curve(N) - head_curve(N)` at every grid point. Where the share is
   undefined, the gap is still reported. **The gap, not the share, is the quantity that
   carries over to positive-advantage cells.**

## C33.0.7 Statistics

Three generation seeds. Uncertainty is a **seed-level Student t interval on 2 df**
(`t₀.₉₇₅,₂ = 4.302653`), from `scripts/21_summarise_c26.py::t_interval`, with the **raw
per-seed values published alongside every mean**.

**No bootstrap is computed anywhere in C33.** At n = 3 the percentile bootstrap of a mean is
identically `[min, max]`: the smallest attainable bootstrap mean is the minimum, attained when
all three resampled indices hit it, with probability 1/27 = 0.037 > 0.025. Such an interval
carries exactly the information of a three-way sign test at null probability 0.25 and cannot
reject anything.

11 grid points × 3 anchors × 3 arms is descriptive; no per-point test is claimed and no
correction is applied to the curves. The headline is **three** pre-specified comparisons, one
per anchor (C33.0.6 rule 4); all three are reported, so there is no selection to correct for,
and no anchor's result may be quoted without the other two. Where n = 3 cannot support an
inference, the section **says so plainly** instead of decorating the number.

**Sensitivity S1, pre-registered.** If head scoring is charged a full re-read of every
candidate — the pessimistic accounting, which we argue is wrong — the head-selected curve's
budget at N rises by `N × head_scoring_recompute_tokens_per_pool_molecule`. The head curve is
re-priced under that accounting and reported as a secondary curve, with the arm counts
recomputed. Any C33 verdict that survives only under the free accounting is labelled as such.

## C33.0.8 Decision rules — scored as written

| # | rule | fires iff |
| --- | --- | --- |
| **F1** | **the oracle asymmetry replicates on generator 2** | for every anchor whose deployed-k2 cell has `adv_oracle < 0`, `oracle_share ≥ 0.50` — i.e. the oracle is worth at least half the reported gap |
| **F2** | **it replicates at C27's magnitude** | every computable deployed-k2 `oracle_share` lies in **[0.75, 1.00]**, C27's 0.859–0.882 with a deliberately generous margin |
| **F3** | **equalising information changes the arm count** | `n_arms_above_head_selected_curve` > `n_arms_above_oracle_selected_curve` over the 30 cells |
| **F4** | **the head arm is not degenerate** | on every anchor, `head_selected` gains ≥ **0.02** absolute hit rate from N=1 to N=32 (C27's E3 threshold, reused not re-chosen). An anchor failing F4 is reported as **degenerate** and no share is read from it |
| **F5** | **the diagnostic arm behaves as C27's did** | on every anchor, `head_selected_at_75pct` at every N ≥ 2 lies strictly between the N=1 base rate and `head_selected` at the same N |

**The verdict rule, fixed now:**

- **REPLICATES** iff F1 and F2 both fire on every anchor where the share is computable.
- **REPLICATES IN DIRECTION, NOT IN MAGNITUDE** iff F1 fires and F2 does not.
- **DOES NOT REPLICATE** iff F1 does not fire.
- **PARTIALLY UNINTERPRETABLE** if a blocking gate fails, or if fewer than two anchors have a
  computable share. In that case the verdict is stated as such and the headline is withheld.

F3, F4 and F5 are scored independently of the verdict and reported whichever way F1/F2 go.

**F6 — the honesty rule, which fires regardless of the others.** Any cell whose
between-generation-seed sd exceeds the absolute value of its own mean advantage is reported as
**"not resolved at three generation seeds"** rather than as a win or a loss. Any anchor whose
deployed-k2 `adv_oracle` t interval spans zero is reported as unresolved even where the point
estimate is negative.

## C33.0.9 The attack this design invites, stated before the result

A sceptical reviewer will say: *scoring the head at the terminal state gives best-of-N a
near-oracle, so you have not matched information at all — guidance only ever sees partial
prefixes.* The pre-registered answer, which the design must be judged against, is C27's and is
restated rather than reinvented:

1. Guidance's head also sees the near-complete prefix, at the last guided step. Nothing is
   withheld from guidance that arm 2 is given.
2. The residual asymmetry is therefore **not** about the property oracle. It is that selection
   may act *after* the outcome exists while steering must commit *before* it.
3. Arm 3 (`head_selected_at_75pct`) quantifies how much of arm 2 depends on the molecule being
   finished, using the same head and no extra compute, and is registered here before any
   number is seen so that it cannot be produced afterwards as a rescue.

A second attack is specific to C33: *you are comparing a share computed on generator 2's
frontier against a share computed on generator 1's, and the two frontiers differ in base rate,
in interval and in what the guidance arms are.* True, and unrepairable without re-running
generator 1's whole pipeline under generator 2's intervals. C33 therefore reports the share
**and** the raw gap in hit-rate points, and states in the section that the share is the
comparable quantity only to the extent that the denominators are comparable. §Limitations.

## C33.0.10 Predictions, committed before the run

Scored in the section. Falsified predictions are reported as falsified, in the same table.

1. **Q1** — G1's maximum absolute hit-rate residual is exactly `0.0` on all three anchors.
   *(The pool depends only on (seed, policy, generator) and all three are pinned; anything else
   means the sampler is not reproducible, which would be a much bigger finding than C33.)*
2. **Q2** — G5's terminal-position pool AUROC exceeds 0.70 on all three anchors, and exceeds
   the 75%-position AUROC on all three.
3. **Q3** — F4 fires: `head_selected` gains ≥ 0.02 from N=1 to N=32 on all three anchors.
4. **Q4** — F1 fires on every anchor with a computable deployed-k2 share.
5. **Q5** — F2 fires: every computable deployed-k2 share is in [0.75, 1.00]. *This is the
   prediction that decides whether C33 is a replication or a qualification, and it is the one
   I hold most weakly:* generator 2's heads were trained by C31 at a different depth index on
   a different tokenisation, and generator 2's guidance already crosses the oracle curve on
   five cells, which is direct evidence that its frontier behaves differently.
6. **Q6** — F3 fires, and `n_arms_above_head_selected_curve` ≥ 15 of 30.
7. **Q7** — the `hbd_count` deployed-k2 share is **not computable**, because C31 records that
   cell's advantage against the oracle curve as +0.0317. *(Committed here so that its absence
   in the results table is a scored prediction rather than a gap.)*
8. **Q8** — F5 fires on at most two of the three anchors. *(On generator 1 it fired on all
   three; on a byte-level BPE tokenisation the 75% content position is a different fraction of
   the molecule and the arm may not be so cleanly ordered.)*
9. **Q9** — the budget-matched gap `oracle_curve(b) - head_curve(b)` at the deployed k=2
   budget is **smaller** on generator 2 than C27's generator-1 gaps (0.3093 / 0.2180 / 0.3193
   in hit-rate points, the C27 `adv_oracle - adv_head` differences), on at least two anchors.
   *(Generator 2's k-sweep budgets are much larger multiples of its base cost than C27's
   deployed budget was, and the two curves converge as N grows only if the head is good; this
   is the prediction most likely to be falsified by the curves' shape rather than by the head.)*
10. **Q10** — no arm of C33 changes any C31 number. C33 re-prices C31's cells; it re-runs
    none of them, and every C31 artefact is read-only.

## C33.0.11 What C33 will NOT do

Stated so its absence is not read as an omission.

- **No weight of the generator changes**, and **no head is trained**. C31's heads are read off
  disk.
- **No new guidance run.** The 30 C31 cells are re-priced, not re-run.
- **No third generator**, no alternative serialization, no λ sweep, no k outside C31's grid,
  no probe point other than the deployed one, no calibration variant.
- **No interval or window is re-derived.** C31's frozen values are read and hashed.
- **No existing artefact, report, config or `outputs/` directory is edited or overwritten.**
  Every C33 output is namespaced `outputs/c33_*`. Conflicts with the merged report go in a
  "What C33 changes elsewhere" section for the owner to merge, exactly as C23–C32 did.
- **No wall-clock claim anywhere.** Cost is processed generator tokens.
- **No bootstrap anywhere.**
- **Nothing is committed to git.**

## C33.0.12 The reporting rule

Whatever comes out, the section states it in this order: what was run, then the gates as
numeric residuals, then the two curves with **every per-seed value in full**, then the
budget-matched gaps, then the oracle-share table with the crossing cells in their own separate
table, then the arm counts, then the diagnostic arm, then the decision rules scored as
written, then the predictions scored including the falsified ones, then the sensitivity, then
limitations, then what it changes elsewhere. A result that disagrees with C27 is written up at
the same length and with the same care as one that agrees. **Machine-derived numbers are
printed with the ASCII hyphen `-`, never the Unicode minus U+2212**, because the binding tests
assert on ASCII. If a deviation from this pre-registration proves necessary, the deviation is
reported and **this document is not amended**.
