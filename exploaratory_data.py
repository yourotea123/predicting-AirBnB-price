import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocessing import preprocess_data
import os
import matplotlib.pyplot as plt
import seaborn as sns

data=preprocess_data('train')


# #--------Geolocation ------------------
plt.figure(figsize=(10, 6)) # Create a scatter plot
scatter = plt.scatter(data.longitude, data.latitude, c=data.price, cmap='viridis', marker='o', edgecolor='k', s=50) # Scatter plot with color based on 'class_label'
plt.xlabel('Longitude') # Label the axes
plt.ylabel('Latitude')
plt.title('Latitude and Longitude Scatter Plot by Price')
plt.grid(True)
plt.colorbar(scatter, label='Price')# Add a color bar to indicate the class labels
plt.show()


# --- Feature and Target Separation ---
# Create the results directory if it doesn't exist
os.makedirs('./Results', exist_ok=True)

# Feature and target separation
data_X = data.drop('price', axis=1)
data_Y = data['price']

# List of features to plot against the label
features = list(data_X.columns)
label = 'price'

n_cols = 5
n_rows = 5
subplots_per_fig = n_cols * n_rows

# Plot and save histograms
for page, start in enumerate(range(0, len(features), subplots_per_fig), start=1):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    # Plot each feature in the current figure page
    for i, feature in enumerate(features[start:start + subplots_per_fig]):
        sns.histplot(data[feature], bins=30, kde=True, ax=axes[i])
        axes[i].set_xlabel(feature, fontsize=8)  # Smaller x-axis label font
        axes[i].set_ylabel('Frequency', fontsize=8)  # Smaller y-axis label font

    # Hide any unused subplots
    for i in range(len(features[start:start + subplots_per_fig]), subplots_per_fig):
        axes[i].axis('off')

    # Adjust spacing and save the figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase spacing between rows and columns
    plt.savefig(f'./results/histogram{page}.png')
    plt.close(fig)

# Plot and save scatter plots
for page, start in enumerate(range(0, len(features), subplots_per_fig), start=1):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    axes = axes.flatten()

    # Scatter plots for each feature against the label
    for i, feature in enumerate(features[start:start + subplots_per_fig]):
        axes[i].scatter(data[feature], data[label], alpha=0.5)
        axes[i].set_xlabel(feature, fontsize=8)  # Smaller x-axis label font
        axes[i].set_ylabel(label, fontsize=8)  # Smaller y-axis label font

    # Hide any unused subplots
    for i in range(len(features[start:start + subplots_per_fig]), subplots_per_fig):
        axes[i].axis('off')

    # Adjust spacing and save the figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Increase spacing between rows and columns
    plt.savefig(f'./results/scatter{page}.png')
    plt.close(fig)