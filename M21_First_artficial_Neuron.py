# Build Your First Artificial Neuron

### Step 1: Import Libraries

import matplotlib.pyplot as plt

### Step 2: Set Up the AND Gate Truth Table

gate_inputs  = [(0, 0), (0, 1), (1, 0), (1, 1)]
gate_outputs = [0, 0, 0, 1]

for inputs, output in zip(gate_inputs, gate_outputs):
    print(inputs, '->', output)

### Step 3: Define a Simple Neuron

def neuron(input1, input2, weight1, weight2, threshold):
    total = (input1 * weight1) + (input2 * weight2)
    if total >= threshold:
        return 1
    else:
        return 0

### Step 4: Give the Neuron Weights and a Threshold

weight1 = 1
weight2 = 1
threshold = 2

### Step 5: Test the Neuron on Every Input

for input1, input2 in gate_inputs:
    result = neuron(input1, input2, weight1, weight2, threshold)
    print((input1, input2), '-> neuron says:', result)

### Step 6: Plot the Inputs and What the Neuron Decided

plt.figure(figsize=(5, 5))

for inputs, output in zip(gate_inputs, gate_outputs):
    plt.scatter(inputs[0], inputs[1], c='green' if output == 1 else 'gray', s=200)

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Single neuron learning the AND gate')
plt.show()