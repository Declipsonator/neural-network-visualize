import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def visualize_neural_network(num_layers, neurons_per_layer, weights, biases, activations, display_weights=False,
                             display_biases=False, color_stroke_weights=True, display_activations=False):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.axis('off')

    # Parameters for the layout
    layer_vertical_spacing = 3  # Increased vertical spacing between layers
    neuron_horizontal_spacing = 1.5  # Increased horizontal spacing between neurons

    # Determine the maximum number of neurons in any layer
    max_neurons = max(neurons_per_layer)
    if max_neurons > 16:
        max_neurons = 16

    # Calculate scaling factors for circles
    x_scale = 1.0  # Adjust as needed based on your plot dimensions
    y_scale = 1.0  # Adjust as needed based on your plot dimensions

    # Set the plot limits
    ax.set_xlim(-1, num_layers * layer_vertical_spacing * x_scale - 1.7)
    ax.set_ylim(-(max_neurons - 1) * neuron_horizontal_spacing * y_scale - 2,
                (max_neurons - 1) * neuron_horizontal_spacing * y_scale + 2)

    # Plot the neurons and the connections
    for layer in range(num_layers):
        num_neurons = neurons_per_layer[layer]

        # Calculate neuron positions
        y_positions = (np.linspace(-(num_neurons - 1), num_neurons - 1,
                                   num=num_neurons) * neuron_horizontal_spacing * y_scale) if num_neurons <= 16 \
             else (np.linspace(-(14 - 1), 14 - 1, num=14) * neuron_horizontal_spacing * y_scale)
        x_position = layer * layer_vertical_spacing * x_scale

        # If the layer has more than 16 neurons, limit the display to 12 neurons and add an ellipsis in the middle
        if num_neurons > 16:
            y_positions = np.concatenate(
                (y_positions[:6], y_positions[-6:]))  # Keep only the first 6 and last 6 neurons
            ax.text(x_position, 0, "... × {}".format(num_neurons - 12), ha='center', va='center', fontsize=20,
                    fontdict={'weight': 800}, )  # Display the ellipsis

        # Draw connections and weights
        if layer < num_layers - 1:
            next_num_neurons = neurons_per_layer[layer + 1]
            next_y_positions = (np.linspace(-(next_num_neurons - 1), next_num_neurons - 1,
                                           num=next_num_neurons) * neuron_horizontal_spacing * y_scale) if next_num_neurons <= 16 \
                else (np.linspace(-(14 - 1), 14 - 1, num=14) * neuron_horizontal_spacing * y_scale)

            # If the next layer has more than 12 neurons, limit the display to the first 6 and last 6 neurons
            if next_num_neurons > 16:
                next_y_positions = np.concatenate((next_y_positions[:6], next_y_positions[-6:]))

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
            neuron_size = 0.2 * np.sqrt(max_neurons / np.max(neurons_per_layer)) * 4000 if 0.2 * np.sqrt(max_neurons / np.max(neurons_per_layer)) * 4000 > 500 else 500
            # Draw the neuron as a circle
            ax.scatter(x_position, y, s=neuron_size, edgecolor="k", facecolor="black", zorder=3)  # Set zorder to 3

            # Draw activation values
            if activations and layer < len(activations) and display_activations:
                activation = activations[layer][neuron_idx]
                plt.text(x_position, y, f'{activation:.2f}', fontsize=8, ha='center', va='center', color='white')

            # Draw bias values
            if biases and layer < len(biases) and display_biases:
                bias = biases[layer][neuron_idx]
                plt.text(x_position, y + 0.3, f'b={bias:.2f}', fontsize=8, ha='center')
    plt.tight_layout()
    plt.show()


# make an image representing the weights going into a single neuron
def visualize_single_neuron_weights(weights, shape, min=None, max=None):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.axis('off')
    ax.aspect = 'equal'


    in_format = np.reshape(weights, shape)

    if min is None:
        min = np.min(in_format)
    if max is None:
        max = np.max(in_format)

    if len(shape) > 2:
        print("Please provide a 1D or 2D array.")


    for i in in_format:
        if(len(in_format) == 1):
            ax.imshow(np.array([i]), cmap='coolwarm', vmin=min, vmax=max)
        else:
            ax.imshow(in_format, cmap='coolwarm', vmin=min, vmax=max)

    plt.tight_layout()
    plt.show()






# Example usage:
num_layers = 4
neurons_per_layer = [784, 12, 12, 26]
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
visualize_single_neuron_weights(weights[0][1], (28, 28))