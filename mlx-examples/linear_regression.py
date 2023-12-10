# based on https://github.com/ml-explore/mlx/blob/main/examples/python/linear_regression.py
# Copyright © 2023 Apple Inc.

import time

import mlx.core as mx


def main(device):
    mx.default_stream(getattr(mx, device))

    num_features = 1000
    num_examples = 1_000
    num_iters = 10_000
    lr = 0.01
    key = mx.random.key(0)

    # True parameters
    key, key0 = mx.random.split(key)
    w_star = mx.random.normal((num_features,), key=key0)

    # Input examples (design matrix)
    key, key0 = mx.random.split(key)
    X = mx.random.normal((num_examples, num_features), key=key0)

    # Noisy labels
    key, key0 = mx.random.split(key)
    eps = 1e-2 * mx.random.normal((num_examples,), key=key0)
    y = X @ w_star + eps

    # Initialize random parameters
    key, key0 = mx.random.split(key)
    w = 1e-2 * mx.random.normal((num_features,), key=key0)

    def loss_fn(w):
        return 0.5 * mx.mean(mx.square(X @ w - y))

    grad_fn = mx.value_and_grad(loss_fn)

    tic = time.perf_counter()
    for _ in range(num_iters):
        loss, grad = grad_fn(w)
        w = w - lr * grad
        # this evaluation drastically degenerates the performance even though the final results are identical
        # but is necessary to avoid graph explosion
        mx.eval(w)
    toc = time.perf_counter()

    loss = loss_fn(w)
    error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
    throughput = num_iters / (toc - tic)

    print(f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, " f"Throughput {throughput:.5f} (it/s)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = p.parse_args()
    main(args.device)
