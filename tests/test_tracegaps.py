from byteprof.tracegaps import analyze_gaps


def k(ts, dur):
    return {"ph": "X", "cat": "kernel", "name": "k", "ts": ts, "dur": dur}


def cpu(name, ts, dur):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur}


def test_no_kernels_returns_none():
    assert analyze_gaps([cpu("cudaStreamSynchronize", 0, 5)]) is None


def test_launch_gaps_and_sync_bubbles_on_synthetic_trace():
    events = [
        k(0, 10),                 # 0-10
        k(15, 5),                 # gap 10-15 (5 us): nothing overlaps -> launch gap
        cpu("cudaLaunchKernel", 11, 2),
        k(40, 10),                # gap 20-40 (20 us): stream sync overlaps -> sync bubble
        cpu("cudaStreamSynchronize", 22, 15),
        k(45, 15),                # overlaps previous kernel: no gap
        k(70, 10),                # gap 60-70 (10 us): memcpy starting inside the gap -> sync bubble
        cpu("cudaMemcpyAsync", 65, 3),
        k(85, 5),                 # gap 80-85 (5 us): sync ended exactly where the gap starts -> launch gap
        cpu("cudaEventSynchronize", 70, 10),
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 100},  # ignored
        {"ph": "M", "name": "meta"},                                            # ignored
    ]
    r = analyze_gaps(events)
    assert r["kernels"] == 6 and r["gaps"] == 4
    assert r["window_us"] == 90
    assert r["sync_bubbles"] == 2 and r["launch_gaps"] == 2
    assert abs(r["sync_bubble_frac"] - 30 / 90) < 1e-9
    assert abs(r["launch_gap_frac"] - 10 / 90) < 1e-9
    assert abs(r["idle_frac"] - 40 / 90) < 1e-9
    assert r["gap_median_us"] == 7.5   # gaps 5, 5, 10, 20
    assert abs(r["gap_p90_us"] - 17.0) < 1e-9
