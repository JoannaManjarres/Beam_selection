from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy import ndimage

def plot_3D_scene(data_lidar_process_all, data_position_rx, data_position_tx, sample_for_plot):
    sample_for_plot = sample_for_plot
    #rx_as_cube = position_of_rx_as_cube [sample_for_plot, :, :, :]
    rx = data_position_rx [sample_for_plot, :, :, :]
    tx = data_position_tx [sample_for_plot, :, :, :]
    scenario_complet = data_lidar_process_all [sample_for_plot, :, :, :]
    fig = plt.figure()

    plt.rcParams.update ({'font.size': 14})
    #title = 'Scene ' + str (sample_for_plot) + ' of dataset ' + data_name
    #plt.title (title)

    #ax = fig.add_subplot (1, 2, 1, projection='3d')
    ax = plt.axes(projection='3d')

    #ax.voxels (rx_as_cube, alpha=0.12, edgecolor=None, shade=True, color='red')  # Voxel visualization
    #ax.voxels (scenario_complet, alpha=0.12, edgecolor=None, shade=True, color='red')  # Voxel visualization
    #ax.set_title ('Rx')
    #ax.set_xlabel ('x', labelpad=10)
    #ax.set_ylabel ('y', labelpad=10)
    #ax.set_zlabel ('z', labelpad=10)
    #plt.tight_layout ()

    objects = scenario_complet
    objects = np.array (objects, dtype=bool)
    rx = np.array (rx, dtype=bool)
    tx = np.array (tx, dtype=bool)

    voxelarray = objects | rx | tx

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)

    color_object = '#cccccc90'
    color_rx = 'red'
    color_tx = 'blue'

    colors[objects] = color_object
    colors[rx] = color_rx
    colors[tx] = color_tx

    # and plot everything
    # ax = plt.figure().add_subplot(projection='3d')

    # Set axes label
    ax.set_xlabel ('x', labelpad=10)
    ax.set_ylabel ('y', labelpad=10)
    ax.set_zlabel ('z', labelpad=10)

    # set predefine rotation
    # ax.view_init(elev=49, azim=115)

    #ax = fig.add_subplot (1, 2, 2, projection='3d')
    # ax.voxels(voxelarray, alpha=0.5, edgecolor=None, shade=True, antialiased=False,
    #         color='#cccccc90')  # Voxel visualization
    ax.voxels (voxelarray, facecolors=colors, edgecolor=None, antialiased=False)
    ax.set_title ('Full scenario of scene ' + str(sample_for_plot))
    ax.set_xlabel ('x', labelpad=10)
    ax.set_ylabel ('y', labelpad=10)
    ax.set_zlabel ('z', labelpad=10)
    plt.tight_layout ()

    c1 = mpatches.Patch (color=color_object, label='Objects')
    c2 = mpatches.Patch (color=color_rx, label='Rx')
    c3 = mpatches.Patch (color=color_tx, label='Tx')

    ax.legend (handles=[c1, c2, c3], loc='center left', bbox_to_anchor=(-0.1, 0.9))


def plot_2D_scene(data_lidar_2D, sample_for_plot):

    plt.imshow(data_lidar_2D[sample_for_plot], cmap='Greys', origin='lower', extent=[0, 200, 0, 20])
    plt.title("2D Scenario of Scene " + str(sample_for_plot))
    plt.tight_layout(h_pad=0.5)