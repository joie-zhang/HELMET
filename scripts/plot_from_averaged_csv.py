import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Set style for professional appearance
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Load the pre-computed averaged data
df = pd.read_csv('/scratch/gpfs/DANQIC/jz4391/HELMET/results/plot_data_averaged_across_cache_sizes.csv')

print("Loaded averaged data with shape:", df.shape)
print("\nTechniques:", df['technique'].unique())
print("Models:", df['model'].unique())

# Define color palette for models
model_palette = {
    'Llama-3.1-8B-Instruct': '#AB63FA',
    'Qwen2.5-7B-Instruct': '#00CC96',
    'DeepSeek-R1-Distill-Llama-8B': '#EF553B',
    'DeepSeek-R1-Distill-Qwen-7B': '#FFA15A',
}

# Cleaner model names for legend display
model_display_names = {
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'DeepSeek-R1-Distill-Llama-8B': 'R1-Distill-Llama-8B',
    'DeepSeek-R1-Distill-Qwen-7B': 'R1-Distill-Qwen-7B',
}

# Define marker shapes for techniques
marker_dict = {
    'baseline': 'o',
    'INT8': 's',
    'INT4': '^',
    'pyramidkv': 'P',
    'snapkv': 'X',
    'streamingllm': '*',
    'duoattn': 'v',
}

# Define display labels for techniques
technique_labels = {
    'baseline': 'Baseline',
    'INT8': 'Int8',
    'INT4': 'NF4',
    'pyramidkv': 'PyramidKV',
    'snapkv': 'SnapKV',
    'streamingllm': 'StreamingLLM',
    'duoattn': 'DuoAttn',
}

# Marker sizes - increased for better visibility
marker_size_dict = {
    'o': 450,
    's': 450,
    '^': 450,
    'P': 550,
    'X': 550,
    '*': 850,
    'v': 270,
}

# Define technique connection order (with duoattn between INT8 and baseline)
technique_connection_order_with_duo = ['INT4', 'INT8', 'duoattn', 'baseline', 'snapkv', 'pyramidkv', 'streamingllm']

# Define custom legend order (with SnapKV before PyramidKV)
legend_order = ['baseline', 'INT4', 'INT8', 'snapkv', 'pyramidkv', 'streamingllm', 'duoattn']

# Create output directory
output_dir = '/scratch/gpfs/DANQIC/jz4391/HELMET/results/plots'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PLOT: All Techniques WITH Connection Lines (Including DuoAttn)
# Using pre-computed averaged data from CSV
# ============================================================================
print("\nCreating plot from averaged CSV data...")

# Create single subplot with larger figure for better readability
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Store points for each model/technique to enable connecting them
model_points = {}  # {model: {technique: (x, y)}}

# Plot each point from the CSV
for _, row in df.iterrows():
    model = row['model']
    technique = row['technique']
    x = row['avg_memory_gb']
    y = row['avg_performance_score']

    # Skip if model or technique not in our palettes
    if model not in model_palette or technique not in marker_dict:
        continue

    # Initialize model_points if needed
    if model not in model_points:
        model_points[model] = {}

    # Store the point for connecting
    model_points[model][technique] = (x, y)

    # Plot the point
    ax.scatter(
        [x], [y],
        color=model_palette[model],
        marker=marker_dict[technique],
        s=marker_size_dict[marker_dict[technique]],
        alpha=0.8,
        zorder=3
    )

# Now connect points for each model in the specified order
for model in model_points.keys():
    points_to_connect = []
    for tech in technique_connection_order_with_duo:
        if tech in model_points[model]:
            x, y = model_points[model][tech]
            points_to_connect.append((x, y))

    # Draw dashed line connecting the points
    if len(points_to_connect) > 1:
        x_coords = [p[0] for p in points_to_connect]
        y_coords = [p[1] for p in points_to_connect]
        ax.plot(x_coords, y_coords, color=model_palette[model], linestyle='--',
               alpha=0.4, linewidth=3.0, zorder=2)

ax.set_xlabel('Memory Usage (GB)', fontweight='bold', fontsize=24)
ax.set_ylabel('Average Performance Score', fontweight='bold', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set y-axis to start at 0
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1] * 1.05)

# Create legend in horizontal layout at bottom - two separate rows
model_elements = []
for model in sorted(model_palette.keys()):
    display_name = model_display_names.get(model, model)
    model_elements.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=model_palette[model], markersize=16,
                              markeredgewidth=0,
                              label=display_name))

technique_elements = []
for tech in legend_order:
    if tech in marker_dict:
        # Use larger marker size for StreamingLLM star
        ms = 20 if tech == 'streamingllm' else 16
        technique_elements.append(Line2D([0], [0], marker=marker_dict[tech], color='gray',
                                      linestyle='None', markersize=ms,
                                      markeredgewidth=0,
                                      label=technique_labels[tech]))

# Create first legend for models (top row)
leg1 = fig.legend(handles=model_elements,
          title='Models',
          title_fontsize=22,
          loc='lower center',
          bbox_to_anchor=(0.5, 0.01),
          ncol=len(model_elements),
          frameon=False,
          fancybox=False,
          shadow=False,
          fontsize=20,
          columnspacing=1.5,
          handletextpad=0.5)

# Add second legend for techniques (bottom row)
leg2 = fig.legend(handles=technique_elements,
              title='Techniques',
              title_fontsize=22,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.07),
              ncol=len(technique_elements),
              frameon=False,
              fancybox=False,
              shadow=False,
              fontsize=20,
              columnspacing=1.5,
              handletextpad=0.5)

# Add first legend back as artist (since second legend call replaces it)
fig.add_artist(leg1)

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

# Save with the same filename as the original
output_path = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections_incl_duo.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path}")

output_path_pdf = os.path.join(output_dir, 'averages_comparison_1x1_all_techniques_with_connections_incl_duo.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"  Saved: {output_path_pdf}")

# Display plot inline (useful for Jupyter notebooks)
try:
    from IPython.display import display, Image as IPImage
    print("\nDisplaying plot:")
    display(IPImage(filename=output_path))
except ImportError:
    print("\nNote: Run in Jupyter notebook to see inline plot preview.")

plt.close()

print("\nPlot created successfully from averaged CSV data!")
print("\nSummary:")
print(f"  - Total data points plotted: {len(df)}")
print(f"  - Models: {len(model_palette)}")
print(f"  - Techniques: {len(df['technique'].unique())}")
print("\nNote: SnapKV and PyramidKV points represent averages across both")
print("      w256_c2048 (small) and w2048_c8192 (large) cache configurations.")
