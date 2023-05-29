import matplotlib.pyplot as plt

# Data points
values = [74.44, 72.18, 77.44, 79.70, 82.71, 79.70, 74.44, 82.71, 81.95, 72.93, 81.95, 79.70, 81.20]

# Set the index where the red dot should be plotted
red_index = 4

# X-axis labels
x_labels = [
    'Beluga', 'Bowhead', 'False Killer', 'Fin-Finback', 'Humpback', 'Killer',
    'Long-Finned Pilot', 'Melon Headed', 'Minke', 'Northern Right', 'Short-Finned (Pacific)',
    'Southern Right', 'Sperm'
]

# Plot the graph
plt.figure()

# Plot bullet points
for i in range(len(values)):
    color = 'r' if i == red_index else 'g'
    plt.plot(i, values[i], 'o', markersize=10, color=color)

    # Calculate y-coordinate for the text
    y_offset = 0.3
    y = values[i] + y_offset

    plt.text(i, y, str(values[i]), ha='center', va='top', color='black')

# Plot line connecting the points
plt.plot(range(len(values)), values, 'k-')

# Set x-axis tick labels
plt.xticks(range(len(values)), x_labels, rotation=45)

# Set labels and title
plt.xlabel('Species')
plt.ylabel('Accuracy Scores')
plt.title('Testing Results')

# Show the graph
plt.show()
