import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def visualize_neural_network(num_layers, neurons_per_layer, weights, biases, activations, display_weights=False, display_biases=False, color_stroke_weights=True):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.axis('off')

    # Parameters for the layout
    layer_vertical_spacing = 3  # Increased vertical spacing between layers
    neuron_horizontal_spacing = 1.5  # Increased horizontal spacing between neurons

    # Determine the maximum number of neurons in any layer
    max_neurons = max(neurons_per_layer)

    # Calculate scaling factors for circles
    x_scale = 1.0  # Adjust as needed based on your plot dimensions
    y_scale = 1.0  # Adjust as needed based on your plot dimensions

    # Set the plot limits
    ax.set_xlim(-1, num_layers * layer_vertical_spacing * x_scale)
    ax.set_ylim(-(max_neurons - 1) * neuron_horizontal_spacing * y_scale - 1, (max_neurons - 1) * neuron_horizontal_spacing * y_scale + 1)



    # Plot the neurons and the connections
    for layer in range(num_layers):
        num_neurons = neurons_per_layer[layer]

        # Calculate neuron positions
        y_positions = np.linspace(-(num_neurons - 1), num_neurons - 1, num=num_neurons) * neuron_horizontal_spacing * y_scale
        x_position = layer * layer_vertical_spacing * x_scale

        # Draw connections and weights
        if layer < num_layers - 1:
            next_num_neurons = neurons_per_layer[layer + 1]
            next_y_positions = np.linspace(-(next_num_neurons - 1), next_num_neurons - 1,
                                           num=next_num_neurons) * neuron_horizontal_spacing * y_scale

            for neuron_idx, y in enumerate(y_positions):
                for next_neuron_idx, next_y in enumerate(next_y_positions):
                    weight = weights[layer][next_neuron_idx][neuron_idx]
                    line_color = ('green' if weight > 0 else 'red') if color_stroke_weights else 'black'
                    line_width = (0.2 + 3 * abs(weight)) if color_stroke_weights else 1
                    line = plt.Line2D([x_position, x_position + layer_vertical_spacing * x_scale], [y, next_y],
                                      color=line_color,
                                      linewidth=line_width, zorder=1)  # Set zorder to 1
                    ax.add_artist(line)

                    # Draw weight values
                    if display_weights:
                        mid_x = (x_position + (x_position + layer_vertical_spacing * x_scale)) / 2
                        mid_y = (y + next_y) / 2
                        plt.text(mid_x, mid_y, f'w={weight:.2f}', fontsize=8, ha='center')

        # Draw neurons
        for neuron_idx, y in enumerate(y_positions):
            neuron_size = 0.2 * np.sqrt(max_neurons / np.max(neurons_per_layer)) * 6000
            # Draw the neuron as a circle
            ax.scatter(x_position, y, s=neuron_size, edgecolor="k", facecolor="black")


            # Draw activation values
            if activations and layer < len(activations):
                activation = activations[layer][neuron_idx]
                plt.text(x_position, y, f'{activation:.2f}', fontsize=8, ha='center', va='center', color='white')

            # Draw bias values
            if biases and layer < len(biases) and display_biases:
                bias = biases[layer][neuron_idx]
                plt.text(x_position, y + 0.3, f'b={bias:.2f}', fontsize=8, ha='center')





    plt.show()

# Example usage:
num_layers = 4
neurons_per_layer = [10, 8, 8, 6]
weights = [
    np.random.uniform(-1, 1, (neurons_per_layer[i + 1], neurons_per_layer[i])) for i in range(num_layers - 1)
]
biases = [
    np.random.uniform(-1, 1, neurons_per_layer[i]) for i in range(num_layers)
]
activations = [
    np.random.uniform(0, 1, neurons_per_layer[i]) for i in range(num_layers)
]

visualize_neural_network(num_layers, neurons_per_layer, weights, biases, activations, color_stroke_weights=True)
