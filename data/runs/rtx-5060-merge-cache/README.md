# Stable-shape merge-plan witness

These paired traces use the checked-in research Perm scale probe at commit
`8ab32714` with Axis supplied as a path dependency. Both commands used the same
RTX 5060, batch 48, 32x32 images, 32 convolution features, three ordered Adam
steps, three exact-set Adam steps, and `AXIS_PROFILE=1`.

```text
baseline Axis: 7a14357
candidate implementation tree: bc8f87b

                         baseline     cached
complete wall time         85.55 s     25.07 s
ordered run                47.770 s    17.317 s
set run                    37.554 s     7.529 s
GPU utilization mean       10.27%      13.23%
GPU utilization peak       31%         97%
one-second samples          86          26
```

Both runs printed the same results:

```text
ordered initial=0.285610 final=0.240767 set_mse=0.23245905 ordered_mse=0.24444176 min_center_distance=0.014384
set     initial=0.280453 final=0.224598 set_mse=0.22758923 ordered_mse=0.24550579 min_center_distance=0.039436
```

`baseline-profile.log` and `cached-profile.log` retain stdout, `AXIS_PROFILE`,
GNU `time -v`, and successful process-exit receipts. `baseline-gpu.csv` and
`cached-gpu.csv` are raw one-second `nvidia-smi` samples with timestamp,
utilization percentage, used MiB, and power draw in watts. This is a same-host
regression witness, not a framework benchmark. Both executable receipts used:

```sh
AXIS_PROFILE=1 CUDA_TOOLKIT_PATH="$TOOLKIT" \
  /usr/bin/time -v "$TARGET/release/perm" --conv-scale --smoke
```

The executable was first built from Perm `8ab32714`, with its Axis path
dependency directed at the named baseline or candidate checkout. Sampling used:

```sh
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 1000
```

`ordered-1000.log` records one complete ordered `run()` after the following
temporary consumer patch. The summary-loop change did not affect the run; the
main loop still began with ordered and then exact-set.

```diff
-const SCALE_STEPS: usize = 20;
+const SCALE_STEPS: usize = 1_000;
-    for method in [Method::Ordered, Method::Set] {
+    for method in [Method::Ordered] {
-        (full_steps, full_samples, &SEEDS)
+        (full_steps, full_samples, &SEEDS[..1])
```

The ordered timer includes training and evaluation and reported 120.820 seconds;
loss moved from `0.285610` to `0.036525`. The subsequent exact-set run was
intentionally stopped, so this log has no `PASS` footer or successful process
exit. It establishes only that one ordered 1,000-step workload was observed
inside 30 minutes.
