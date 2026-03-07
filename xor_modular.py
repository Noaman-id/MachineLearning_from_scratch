import math
import random

XOR_dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def neuron_forward(weights, inputs, bias):
    z = 0
    for w, x in zip(weights, inputs):
        z += w * x
    z += bias
    a = sigmoid(z)
    return a

def layer_forward(layer_weights, inputs, layer_biases):
    outputs = []

    for weights, bias in zip(layer_weights, layer_biases):
        neuron_output = neuron_forward(weights, inputs, bias)
        outputs.append(neuron_output)

    return outputs

def network_forward(hidden_weights, hidden_biases, output_weights, output_bias, x):
    hidden_outputs = layer_forward(hidden_weights, x, hidden_biases)
    pred = neuron_forward(output_weights, hidden_outputs, output_bias)
    return hidden_outputs, pred

def sigmoid_derivative(a):
    return a * (1 - a)

def train_network(XOR_dataset,
                  hidden_weights,
                  hidden_biases,
                  output_weights,
                  output_bias,
                  lr,
                  epochs):
    
    for epoch in range(epochs):
        for x, target in XOR_dataset:
            hidden_outputs = layer_forward(hidden_weights, x, hidden_biases)
            pred = neuron_forward(output_weights, hidden_outputs, output_bias)
            g = pred - target

            old_output_weights = output_weights[:]

            for i in range(len(output_weights)):
                output_weights[i] = output_weights[i] - lr * g * hidden_outputs[i]
            output_bias -= lr * g

            # Backprop dans la couche cachée
            hidden_deltas = []
            for i in range(len(hidden_outputs)):
                h = hidden_outputs[i]
                v = old_output_weights[i]

                delta = g * v * sigmoid_derivative(h)
                hidden_deltas.append(delta)

            # Mise à jour des poids cachés
            for j in range(len(hidden_weights)):      # neuron
                for i in range(len(x)):               # input
                    hidden_weights[j][i] = hidden_weights[j][i] - lr * hidden_deltas[j] * x[i]

                hidden_biases[j] = hidden_biases[j] - lr * hidden_deltas[j]
        
    return hidden_weights, hidden_biases, output_weights, output_bias

if __name__ == "__main__":

    # hyperparameters
    lr = 0.1
    epochs = 5000

    # hidden layer initialization
    hidden_weights = [
        [random.uniform(-1,1), random.uniform(-1,1)],
        [random.uniform(-1,1), random.uniform(-1,1)]
    ]

    hidden_biases = [
        random.uniform(-1,1),
        random.uniform(-1,1)
    ]

    # output neuron
    output_weights = [random.uniform(-1,1), random.uniform(-1,1)]
    output_bias = random.uniform(-1,1)

    # train
    hidden_weights, hidden_biases, output_weights, output_bias = train_network(
        XOR_dataset,
        hidden_weights,
        hidden_biases,
        output_weights,
        output_bias,
        lr,
        epochs
    )

    print("\nTesting network\n")

    for x, target in XOR_dataset:

        hidden_outputs = layer_forward(hidden_weights, x, hidden_biases)
        pred = neuron_forward(output_weights, hidden_outputs, output_bias)

        predicted_class = 1 if pred >= 0.5 else 0

        print(f"Input: {x}")
        print(f"Prediction: {pred:.4f}")
        print(f"Predicted class: {predicted_class} | Target: {target}")
        print()