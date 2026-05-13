# Marabou verification query for mnist_small_fc.onnx
# Property: for sample input classified as digit 7,
# all inputs within L-inf ball of epsilon=0.01 are also classified as digit 7
# Approach: for each class j != label, run a separate query asking
# "does there exist an input in the epsilon-ball where output[j] >= output[label]?"

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.expanduser("~/TrustworthyAI_hw3/Marabou"))
sys.path.insert(0, os.path.expanduser("~/TrustworthyAI_hw3/Marabou/build"))

from maraboupy import Marabou

EPSILON = 0.01
MODEL_PATH = os.path.expanduser("~/TrustworthyAI_hw3/mnist_small_fc.onnx")
SAMPLE_PATH = os.path.expanduser("~/TrustworthyAI_hw3/sample_input.npy")
LABEL_PATH  = os.path.expanduser("~/TrustworthyAI_hw3/sample_label.npy")

sample = np.load(SAMPLE_PATH).flatten().astype(np.float64)
label  = int(np.load(LABEL_PATH))
print(f"Sample label: {label}, epsilon: {EPSILON}")
print(f"Checking robustness: is every input in the epsilon-ball classified as {label}?\n")

adversarial_found = False
start_total = time.time()

for j in range(10):
    if j == label:
        continue

    # Fresh network for each query
    network = Marabou.read_onnx(MODEL_PATH)
    inputs  = network.inputVars[0].flatten()
    outputs = network.outputVars[0].flatten()

    # Input constraints: L-inf ball
    for i, var in enumerate(inputs):
        lo = float(np.clip(sample[i] - EPSILON, 0.0, 1.0))
        hi = float(np.clip(sample[i] + EPSILON, 0.0, 1.0))
        network.setLowerBound(var, lo)
        network.setUpperBound(var, hi)

    # Output constraint: output[j] >= output[label]  (adversarial condition)
    # i.e., output[label] - output[j] <= 0
    network.addInequality([outputs[label], outputs[j]], [1, -1], 0)

    print(f"  Query: class {j} >= class {label} ?", end=" ", flush=True)
    t0 = time.time()
    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=120)
    result = network.solve(options=options)
    elapsed = time.time() - t0

    exit_code = result[0]
    vals      = result[1]
    print(f"-> {exit_code.upper()} ({elapsed:.1f}s)")

    if exit_code == "sat":
        adversarial_found = True
        adv = np.array([vals[inputs[i]] for i in range(len(inputs))])
        max_diff = np.max(np.abs(adv - sample))
        print(f"    Adversarial example found! Max perturbation: {max_diff:.6f}")
        print(f"    The model misclassifies a perturbed input as class {j} instead of {label}")
        break

total_time = time.time() - start_total
print(f"\nTotal verification time: {total_time:.1f}s")

if adversarial_found:
    print(f"Result: SAT — property NOT verified. Adversarial example exists within epsilon={EPSILON}.")
else:
    print(f"Result: UNSAT — property VERIFIED. All inputs within epsilon={EPSILON} are classified as {label}.")
