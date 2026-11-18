"""Small trainable NumPy TCN for offline, participant-isolated research only.

Inputs are standardized TRAIN-FOLD features [examples, time, features], not raw
EEG. The caller owns splitting, fold-only scaling, eligibility and evaluation.
Two causal residual Conv1D blocks (kernel 3, dilations 1 and 2) are trained with
analytic backpropagation and full-batch Adam. Every layer is masked at invalid
time slots, including its biases. Missing intervals retain their original clock
positions and never become learned synthetic EEG. No trained production model,
recording, dependency or preprocessing contract is changed by this module.

Run this file directly for finite-difference, masking, causality and independent
synthetic chronology checks. They validate the implementation, not human EEG
classification accuracy or efficacy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np


def _inputs(x, mask, *, features=None):
    values = np.asarray(x, dtype=np.float64)
    valid = np.asarray(mask)
    if (values.ndim != 3 or any(size < 1 for size in values.shape)
            or valid.shape != values.shape[:2] or valid.dtype != np.bool_):
        raise ValueError("TCN needs x [N,T,F] and a bool validity mask [N,T]")
    if features is not None and values.shape[2] != features:
        raise ValueError("TCN input feature count differs from its training fold")
    if np.any(np.sum(valid, axis=1) == 0):
        raise ValueError("TCN cannot score a sequence with no valid EEG intervals")
    if not np.isfinite(values[valid]).all():
        raise ValueError("TCN valid EEG features must all be finite")
    # np.where also permits NaN placeholders in excluded intervals. No fitting
    # statistic is calculated here; invalid slots are zero after TRAIN scaling.
    return np.where(valid[:, :, None], values, 0.), valid


def _sigmoid(values):
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def _conv_forward(values, weight, bias, dilation):
    """Kernel index k always reads t-k*dilation, with zero left context only."""
    result = np.broadcast_to(bias, (*values.shape[:2], len(bias))).copy()
    time = values.shape[1]
    for kernel in range(weight.shape[0]):
        lag = kernel * dilation
        if lag < time:
            result[:, lag:] += values[:, :time-lag] @ weight[kernel]
    return result


def _conv_backward(values, weight, upstream, dilation):
    d_values = np.zeros_like(values)
    d_weight = np.zeros_like(weight)
    time = values.shape[1]
    for kernel in range(weight.shape[0]):
        lag = kernel * dilation
        if lag < time:
            source, gradient = values[:, :time-lag], upstream[:, lag:]
            d_values[:, :time-lag] += gradient @ weight[kernel].T
            d_weight[kernel] = np.einsum("nti,nto->io", source, gradient)
    return d_values, d_weight, np.sum(upstream, axis=(0, 1))


@dataclass
class NumpyTCN:
    """Causal residual encoder and a masked-average binary logit head.

    predict_logits returns [N]. predict_proba returns [N] probability of label 1.
    Eligibility still belongs to the caller; no value is returned for an empty
    validity mask. The pooling head uses the whole observed sequence, while the
    hidden state at t depends only on positions <=t.
    """

    input_features: int
    channels: int
    _parameters: dict[str, np.ndarray]
    dilations: tuple[int, ...] = (1, 2)
    kernel_size: int = 3
    training_history: tuple[float, ...] = ()

    @property
    def parameter_count(self):
        return sum(int(value.size) for value in self._parameters.values())

    def _forward(self, values, valid):
        p = self._parameters
        gate = valid[:, :, None]
        projected = values @ p["input.weight"] + p["input.bias"]
        hidden = np.maximum(projected, 0.) * gate
        blocks = []
        for index, dilation in enumerate(self.dilations):
            incoming = hidden
            before_relu = incoming + _conv_forward(incoming, p[f"block{index}.weight"],
                                                    p[f"block{index}.bias"], dilation)
            hidden = np.maximum(before_relu, 0.) * gate
            blocks.append((incoming, before_relu))
        counts = np.sum(valid, axis=1)
        pooled = np.sum(hidden, axis=1) / counts[:, None]
        logits = pooled @ p["head.weight"] + p["head.bias"][0]
        cache = (values, valid, projected, blocks, hidden, pooled, counts)
        return logits, cache

    def predict_logits(self, x, mask):
        values, valid = _inputs(x, mask, features=self.input_features)
        result, _ = self._forward(values, valid)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite TCN output; inspect train-fold scaling")
        return result

    def predict_proba(self, x, mask):
        return _sigmoid(self.predict_logits(x, mask))

    def _loss_and_gradients(self, values, valid, labels, weights, l2):
        logits, cache = self._forward(values, valid)
        normalizer = np.sum(weights)
        loss = float(np.sum(weights * (np.logaddexp(0., logits) - labels * logits)) / normalizer)
        gradient = weights * (_sigmoid(logits) - labels) / normalizer
        values, valid, projected, blocks, hidden, pooled, counts = cache
        p = self._parameters
        grads = {"head.weight": pooled.T @ gradient,
                 "head.bias": np.array([np.sum(gradient)])}
        upstream = (gradient[:, None, None] * p["head.weight"][None, None, :]
                    / counts[:, None, None]) * valid[:, :, None]
        # The broadcasted head gradient must cover every original time position.
        upstream = np.broadcast_to(upstream, hidden.shape).copy()
        for index in reversed(range(len(self.dilations))):
            incoming, before_relu = blocks[index]
            d_pre = upstream * (before_relu > 0.) * valid[:, :, None]
            d_conv, g_weight, g_bias = _conv_backward(
                incoming, p[f"block{index}.weight"], d_pre, self.dilations[index])
            grads[f"block{index}.weight"] = g_weight
            grads[f"block{index}.bias"] = g_bias
            upstream = d_pre + d_conv  # Identity residual and the causal conv branch.
        d_projected = upstream * (projected > 0.) * valid[:, :, None]
        grads["input.weight"] = np.einsum("ntf,ntc->fc", values, d_projected)
        grads["input.bias"] = np.sum(d_projected, axis=(0, 1))
        for name, parameter in p.items():
            if name.endswith(".weight"):
                loss += .5 * l2 * float(np.sum(parameter * parameter))
                grads[name] += l2 * parameter
        return loss, grads


def _initialize(features, channels, seed):
    rng = np.random.default_rng(seed)
    parameters = {
        "input.weight": rng.normal(0., math.sqrt(2 / features), (features, channels)),
        "input.bias": np.zeros(channels),
        "head.weight": rng.normal(0., math.sqrt(1 / channels), channels),
        "head.bias": np.zeros(1),
    }
    for index in range(2):
        parameters[f"block{index}.weight"] = rng.normal(0., .35 / math.sqrt(3 * channels),
                                                        (3, channels, channels))
        parameters[f"block{index}.bias"] = np.zeros(channels)
    return NumpyTCN(features, channels, parameters)


def fit_tcn(x, mask, y, sample_weight=None, *, seed, epochs=120, channels=8,
            lr=.01, l2=1e-3):
    """Fit one fold with fixed hyperparameters; no split/scaling/selection occurs here.

    Full-batch weighted binary cross-entropy plus weight-only L2 is optimized by
    Adam (beta1=.9, beta2=.999, epsilon=1e-8), clipping the total gradient norm at
    5. No validation/test data are accepted or consulted by the fit function.
    The returned training_history contains objective values before each update.
    """
    values, valid = _inputs(x, mask)
    labels = np.asarray(y, dtype=np.float64)
    weights = np.ones(len(values)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if (labels.shape != (len(values),) or not np.isfinite(labels).all()
            or not np.isin(labels, (0., 1.)).all()):
        raise ValueError("TCN requires one binary label per sequence")
    if (weights.shape != labels.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(np.sum(weights)) or np.sum(weights) <= 0):
        raise ValueError("TCN requires finite nonnegative sample weights with positive total")
    if len(np.unique(labels[weights > 0])) < 2:
        raise ValueError("TCN training fold must contain both labels with positive weight")
    if (type(channels) is not int or not 1 <= channels <= 128
            or type(epochs) is not int or not 1 <= epochs <= 10000):
        raise ValueError("TCN channels/epochs must be bounded positive integers")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("TCN learning rate must be positive and L2 nonnegative")
    model = _initialize(values.shape[2], channels, seed)
    momentum = {name: np.zeros_like(p) for name, p in model._parameters.items()}
    variance = {name: np.zeros_like(p) for name, p in model._parameters.items()}
    history = []
    for step in range(1, epochs+1):
        loss, gradients = model._loss_and_gradients(values, valid, labels, weights, l2)
        if not math.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("Non-finite TCN objective/gradient; inspect train-fold scaling")
        history.append(loss)
        norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
        clipping = min(1., 5. / max(norm, 1e-30))
        for name, parameter in model._parameters.items():
            gradient = gradients[name] * clipping
            momentum[name] = .9 * momentum[name] + .1 * gradient
            variance[name] = .999 * variance[name] + .001 * gradient * gradient
            m_hat, v_hat = momentum[name] / (1-.9**step), variance[name] / (1-.999**step)
            parameter -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    model.training_history = tuple(history)
    return model


def _chronology_fixture(count, seed):
    """Same marginal feature distribution, opposite order of two brief events."""
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    rng.shuffle(labels)
    valid = np.ones((count, 10), dtype=bool)
    valid[:, (1, 8)] = False
    values = rng.normal(0., .08, (count, 10, 1))
    for index, label in enumerate(labels):
        onset = int(rng.integers(2, 5))
        polarity = 1. if label else -1.
        values[index, onset:onset+2, 0] += (polarity, -polarity)
    values *= rng.uniform(.8, 1.2, (count, 1, 1))
    values[~valid] = 0.
    return values, valid, labels.astype(float)


def self_check():
    """Fail on incorrect gradients, causal indexing, masks or absent order learning."""
    rng = np.random.default_rng(6419)
    values = rng.normal(0., .5, (3, 6, 2))
    valid = np.array([[True, True, False, True, True, True],
                      [True, False, True, True, False, True],
                      [True, True, True, True, True, True]])
    values, valid = _inputs(values, valid)
    labels, weights, l2 = np.array([0., 1., 1.]), np.array([.4, 1.7, .9]), .007
    model = _initialize(2, 3, 391)
    # Move biases off exact ReLU kinks for two-sided numerical derivatives.
    for name, parameter in model._parameters.items():
        if name.endswith(".bias"):
            parameter[:] = rng.normal(.1, .03, parameter.shape)
    _, analytic = model._loss_and_gradients(values, valid, labels, weights, l2)
    worst_absolute = 0.
    checked = 0
    epsilon = 1e-6
    for name, parameter in model._parameters.items():
        for position in np.ndindex(parameter.shape):
            original = parameter[position]
            parameter[position] = original+epsilon
            above, _ = model._loss_and_gradients(values, valid, labels, weights, l2)
            parameter[position] = original-epsilon
            below, _ = model._loss_and_gradients(values, valid, labels, weights, l2)
            parameter[position] = original
            error = abs((above-below)/(2*epsilon) - analytic[name][position])
            worst_absolute = max(worst_absolute, float(error))
            checked += 1
    if worst_absolute > 2e-6:
        raise AssertionError(f"TCN gradient mismatch: {worst_absolute:g}")

    # Hand-calculated dilated causal indexing, not a second implementation.
    toy = np.arange(1., 6.)[None, :, None]
    convolution = _conv_forward(toy, np.array([1., 2., 4.])[:, None, None], np.zeros(1), 2)
    np.testing.assert_allclose(convolution.ravel(), [1., 2., 5., 8., 15.], atol=0, rtol=0)

    altered = values.copy()
    altered[~valid] = np.nan
    np.testing.assert_array_equal(model.predict_logits(values, valid), model.predict_logits(altered, valid))
    earlier = model._forward(values, valid)[1][4]
    altered = values.copy()
    altered[:, 4:] += 100.
    later = model._forward(altered, valid)[1][4]
    np.testing.assert_array_equal(earlier[:, :4], later[:, :4])

    train, train_mask, train_y = _chronology_fixture(160, 1001)
    test, test_mask, test_y = _chronology_fixture(160, 2002)
    temporal = fit_tcn(train, train_mask, train_y, seed=19, channels=4, epochs=140)
    accuracy = float(np.mean((temporal.predict_logits(test, test_mask) >= 0) == test_y))
    shuffled = test.copy()
    for index in range(len(test)):
        positions = np.flatnonzero(test_mask[index])
        shuffled[index, positions] = shuffled[index, rng.permutation(positions)]
    shuffled_accuracy = float(np.mean((temporal.predict_logits(shuffled, test_mask) >= 0) == test_y))
    if accuracy < .9 or shuffled_accuracy > .65:
        raise AssertionError(f"Synthetic chronology check failed: ordered={accuracy}, shuffled={shuffled_accuracy}")
    repeated = fit_tcn(train, train_mask, train_y, seed=19, channels=4, epochs=140)
    np.testing.assert_array_equal(temporal.predict_logits(test, test_mask), repeated.predict_logits(test, test_mask))
    if temporal.training_history[-1] >= temporal.training_history[0]:
        raise AssertionError("TCN optimizer failed to reduce its training objective")
    return {"finite_difference_parameters_checked": checked,
            "gradient_max_absolute_error": worst_absolute,
            "causal_dilation_and_future_invariance": True,
            "invalid_value_invariance": True, "seed_determinism": True,
            "chronology_test_accuracy": accuracy, "valid_slot_shuffle_accuracy": shuffled_accuracy,
            "chronology_fixture_only_not_human_accuracy": True,
            "fixture_parameter_count": temporal.parameter_count}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2, allow_nan=False))
