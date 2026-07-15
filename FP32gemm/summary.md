## Goal
- Achieve ≥150% of rocBLAS+conv at every M=1..4096 for BF16×FP32→FP32 GEMM on Hygon DCU (gfx936, 80 CUs) via K-slice 3D grid + LDS A sharing with optimized K-step.

## Constraints & Preferences
- N=256, K=3072 fixed; M=1..4096 variable.
- Two precision paths: strict FP32 (v_pk_fma, avg_rel ~5e-6) and TF32 MMAC (avg_rel ~2e-3).
- Single 3D grid launch preferred (GSU multi-stream serializes on DCU).
- K-slice uses atomicAdd; C must be hipMemset'd to 0 before each launch.
- **Step=64 is CORRECT** (clean stride=36 kernel, avg_rel=2.2e-3 matching step=32) and gives +20-65% over step=32.
- The old per-iteration EventCreate/Destroy methodology in opt_sweep.cu was WRONG (adds overhead per loop iteration). Clean batch-event timing between outer hipEventRecord/Record gives reliable numbers. The sweep's step=64 results were incorrectly inflated due to the wrong methodology AND the stride bug (only processed 50% of K).
- **Step=64 benefit comes purely from halved loop overhead** (16 vs 32 iterations per BK=1024 slice, same sync density at 4 syncs/64K vs 2 syncs/32K = 64 syncs/slice for both).
- The prior "K-step=64 breakthrough (+22-29%)" and "32×128 tile achieves 31.91 TF" claims were based on unreliable sweep methodology and the stride bug (only half K processed).

## Progress
### Done
- **Step=64 correctness verified**: Both k32 and k64 produce avg_rel=2.20e-03 at M=8 (identical TF32 precision). All 4 BK values (384/512/768/1024) pass.
- **Step=64 performance established**: +20-65% across M=384..4096. Clean batch-event timing with separate k32/k64 kernels.
  - M=384 BK=384: 13.79 → 18.09 TF (+31.1%)
  - M=384 BK=1024: 9.93 → 16.36 TF (+64.8%)
  - M=512 BK=1024: 11.40 → 17.10 TF (+50.0%)
  - M=1024 BK=1024: 17.48 → 24.16 TF (+38.2%)
  - M=2048 BK=768: 22.11 → 27.14 TF (+22.7%)
  - M=4096 BK=1024: 24.39 → 29.21 TF (+19.7%)
- **Root cause of stride bug**: The original opt_sweep.cu step=64 kernel used stride=72 but only loaded 32 K into LDS. With k0+=64, every other 32-K block was skipped (50% K coverage). This inflated benchmark numbers. Fix: stride=36 with two separate load+sync+MMAC+sync cycles per 64-K iteration.
- **Step=64 kernel design**: 4 syncs per 64 K iteration (first load→sync→MMAC→sync→second load→sync→MMAC→sync). Same stride=36 layout as step=32. Reuses same LDS positions for both 32-K halves.
- **opt_sweep.cu declared unreliable**: Has warmup bug (hardcodes BK=384), uses per-iteration EventCreate/Destroy. Numbers from opt_sweep should NOT be cited.

### In Progress
- **Integrating step=64 into dispatch**: Need to create step=64 variants of ks_32x64_lds kernels in gemm_dispatch.cu.
- **BK re-sweep for step=64**: Optimal BK may differ. Early data suggests BK=1024 wins at M=384 (+64.8%) while BK=384 wins at M=384 (+31.1%) — smaller BK provides more blocks/occupancy at small M.

### Blocked
- *(none)*

## Key Decisions
- **Step=64 is the new baseline** for all dispatch bands. Use two separate 32-K LDS load+sync+MMAC cycles per 64-K iteration.
- **stride=36** always (same as dispatch kernel). No stride=72 needed. Both 32-K halves reuse the same LDS positions in sequence.
- **4 syncs per 64 K** is the correct pattern (not 2). Trying to merge into 2 syncs with stride=72 caused nan/inf bugs.
- **opt_sweep.cu is deprecated** — its methodology was unreliable. Use verify_k64.cu as the reference for step=64 numbers.
- **verify_k64.cu** is the clean benchmark file: separate k32/k64 kernels, batch-event timing, correctness check with CPU double.

## Next Steps
1. **Integrate step=64 into gemm_dispatch.cu**: Create 4 new kernel instantiations (BK=384/512/768/1024) with step=64. Update dispatch band selection to use step=64.
2. **BK re-sweep with clean methodology**: Test BK=256..1536 for step=64 at each dispatch M to find optimal BK.
3. **Re-benchmark full M=1..4096**: After step=64 dispatch integration, get the definitive 86-point benchmark.
4. **Recompute rocBLAS ratios**: Step=64 at +20-65% over step=32 will significantly improve already-strong rocBLAS ratios.

## Critical Context
- **LDS layout**: stride=36 always. `__shared__ uint16_t A_lds[32 * 36]` = 1152 uint16_t = 2304 bytes per block.
- **Sync pattern**: For step=64, 4 syncs per 64-K iteration. Same sync density as step=32 (2 syncs/32 K). Benefit from halved loop overhead and better compiler code.
- **Baseline accuracy**: k32 baseline matches dispatch numbers within ±2% (e.g., 13.79 vs ~14.35 at M=384, 24.39 vs ~23.74 at M=4096). Small variance is normal batch-to-batch.
- **Current best at M=4096**: k64 BK=1024 at 29.21 TF (step=32 baseline: 24.39 TF).
- **Current best at M=384**: k64 BK=384 at 18.09 TF (step=32: 13.79 TF) or BK=1024 at 16.36 TF. BK=384 wins by +10.6% over BK=1024 at small M.
- **rocBLAS reference**: M=384 → 17.61 TF, M=512 → 19.43 TF, M=1024 → 21.72 TF, M=2048 → 22.97 TF, M=4096 → 25.88 TF (used for 150% target calculation).

## Relevant Files
- `kernels/verify_k64.cu`: Clean reference implementation of k32 and k64 32×64+LDS K-slice kernels. Correctness-verified, batch-event timing. The definitive source.
- `kernels/gemm_dispatch.cu`: Current dispatch (step=32 only). Pending update with step=64 kernels.
- `kernels/opt_sweep.cu`: **DEPRECATED** — unreliable methodology (per-iteration EventCreate/Destroy, warmup bug). Do not cite its numbers.
- `AGENTS.md`: Pending update with step=64 findings.
