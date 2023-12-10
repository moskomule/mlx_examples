# based on https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py
# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mnist


def permutation(size: int, key: mx.array = None) -> mx.array:
    noise = mx.random.uniform(shape=(size,), key=key)
    return mx.argsort(noise)


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [nn.Linear(idim, odim) for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y, key):
    perm = mx.array(permutation(y.size, key=key))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def main():
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    key = mx.random.key(0)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

    # Load the model
    model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    _tic = time.perf_counter()
    for e in range(num_epochs):
        tic = time.perf_counter()
        key, key0 = mx.random.split(key)
        for X, y in batch_iterate(batch_size, train_images, train_labels, key=key0):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            # I don't know this eval is necessary...?
            # mx.eval(model.parameters(), optimizer.state)
        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()
        print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}," f" Time {toc - tic:.3f} (s)")
    _toc = time.perf_counter()
    print(f"{_toc-_tic:.5f} (s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple MLP on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main()
