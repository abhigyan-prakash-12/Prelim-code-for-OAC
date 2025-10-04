import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
#Defining scales
complexity_levels = ['High', 'Mid', 'Low']
dimensions = ['0D', '1D', '2D', '3D']
cycles = ['None', 'Partial', 'Complex']

def cat_to_num(category, category_list):
    return category_list.index(category)
#Models to be plotted
models = [
    ('FaIR', 'Low', '0D', 'Partial'),
    ('OpenAirClim', 'Mid', '2D', 'Complex'),
    ('E39/C', 'High', '3D', 'Complex'),
    ('EBM', 'Low', '0D', 'None'),
    ('ESM', 'High', '3D', 'Complex'),
    ('GCM', 'High', '3D', 'None'),
    ('CICERO', 'Low', '0D', 'Partial')
]

models_group_1 = [m for m in models if m[0] in ['ESM', 'GCM', 'EBM']]
models_group_2 = [m for m in models if m[0] not in ['ESM', 'GCM', 'EBM']]
#Function to make the 3D classification plot
def plot_models(models, title, elev=20, azim=45):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['skyblue', 'lightcoral', 'palegreen', 'khaki', 'plum', 'lightgray', 'orange']

    pos_dict = defaultdict(list)
    for model in models:
        x = cat_to_num(model[1], complexity_levels)
        y = cat_to_num(model[2], dimensions)
        z = cat_to_num(model[3], cycles)
        if model[0] == 'OpenAirClim':
            z = 1.5

        pos_dict[(x, y, z)].append(model)

    for i, ((x, y, z), models_at_pos) in enumerate(pos_dict.items()):
        count = len(models_at_pos)
        for j, model in enumerate(models_at_pos):
            label = model[0]
            angle = 2 * np.pi * j / count
            radius = 0.15 
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            dz = 0  

            color = colors[(i + j) % len(colors)]
            ax.scatter(x + dx, y + dy, z + dz, color=color, s=80, depthshade=True)
            ax.plot([x + dx, x + dx], [y + dy, y + dy], [0, z + dz], linestyle='dotted', color='gray', linewidth=1)
            ax.plot([x + dx, x + dx], [0, y + dy], [z + dz, z + dz], linestyle='dotted', color='gray', linewidth=1)
            ax.text(x + dx + 0.2, y + dy + 0.2, z + dz + 0.15, label,
                    ha='left', va='bottom', fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', edgecolor='black', pad=0.2))

    ax.set_xticks(range(len(complexity_levels)))
    ax.set_xticklabels(complexity_levels, fontweight='bold', fontsize=8, rotation=15, ha='right')
    ax.set_xlabel('Complexity', fontweight='bold', labelpad=10)

    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimensions, fontweight='bold', fontsize=8)
    ax.set_ylabel('Dimensions', fontweight='bold', labelpad=10)

    ax.set_zticks(range(len(cycles)))
    ax.set_zticklabels(cycles, fontweight='bold', fontsize=8)
    ax.set_zlabel('Cycles', fontweight='bold', labelpad=10)

    ax.set_title(title, pad=20, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()

plot_models(models_group_1, "Climate Models: ESM, GCM, EBM")
plot_models(models_group_2, "Climate Models: FaIR, OpenAirClim, E39/C, CICERO")
