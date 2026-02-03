import time
import numpy as np
import torch
from beampred.config import N_WIDE_BEAMS


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_fc(model):
    flops = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv1d):
            flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0] * N_WIDE_BEAMS
    return flops


def measure_latency(model, n_runs=1000, input_dim=N_WIDE_BEAMS):
    model.eval()
    model.cpu()
    x = torch.randn(1, input_dim)

    for _ in range(100):
        with torch.no_grad():
            model(x)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append(time.perf_counter() - start)

    return np.mean(times) * 1e6, np.std(times) * 1e6


def analyze_complexity(models_dict):
    results = {}
    for name, model in models_dict.items():
        n_params = count_params(model)
        flops = estimate_flops_fc(model)
        latency_mean, latency_std = measure_latency(model)
        results[name] = {
            "params": n_params,
            "flops": flops,
            "latency_us": latency_mean,
            "latency_std_us": latency_std,
        }
    return results


def print_complexity_table(results):
    print(f"\n{'Model':<18} {'Params':>10} {'FLOPs':>12} {'Latency (us)':>14}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<18} {r['params']:>10,} {r['flops']:>12,} {r['latency_us']:>11.1f} ± {r['latency_std_us']:.1f}")
