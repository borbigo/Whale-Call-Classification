import matplotlib.pyplot as plt

# Data points
values = [74.44, 72.18, 77.44, 79.70, 82.71, 79.70, 74.44, 82.71, 81.95, 72.93, 81.95, 79.70, 81.20]

# Set the index where the red dot should be plotted
red_index = 5

# X-axis labels
x_labels = [
    'Beluga', 'Bowhead', 'False Killer', 'Fin-Finback', 'Humpback', 'Killer',
    'Long-Finned Pilot', 'Melon Headed', 'Minke', 'Northern Right', 'Short-Finned (Pacific)',
    'Southern Right', 'Sperm'
]

# Set the figure and axis color
fig = plt.figure(facecolor='#4d6680')
ax = fig.add_subplot(111, facecolor='#4d6680')

# Plot the graph
ax.plot(range(len(values)), values, color='white')

# Plot bullet points
for i in range(len(values)):
    color = '#990000' if i == red_index else '#1b7e1b'
    ax.plot(i, values[i], 'o', markersize=10, color=color)
    ax.text(i, values[i], str(values[i]), ha='center', va='bottom', color='white')

# Set x-axis tick labels
ax.set_xticks(range(len(values)))
ax.set_xticklabels(x_labels, rotation=45, color='white')

# Set labels and title
ax.set_xlabel('Species', color='white')
ax.set_ylabel('Accuracy Score', color='white')
ax.set_title('Testing Results', color='white')

# Set the spines color to white
ax.spines['top'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')


# Set the background color
ax.set_facecolor('#4d6680')

# Show the graph
plt.show()
