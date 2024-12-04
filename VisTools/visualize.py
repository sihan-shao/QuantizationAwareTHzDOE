from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
import os

def visualize(args, save_path, surface_path):
    result_file_path = os.path.join(save_path, '2D_images/')
    if not os.path.isdir(result_file_path):
        os.makedirs(result_file_path)
    surf_name = args.surf_name

    with h5py.File(surface_path,'r') as f:

        Z_LIMIT = 10

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        X, Y = np.meshgrid(x, y)
        
        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        else:
            print ('%s is not found in %s' % (surf_name, surface_path))
        
        Z = np.array(f[surf_name][:])
        print(Z)
        #Z[Z > Z_LIMIT] = Z_LIMIT
        #Z = np.log(Z)  # logscale

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap = 'summer', levels=np.arange(args.vmin, args.vmax, args.vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + surf_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        fig = plt.figure()
        CS = plt.contourf(X, Y, Z, levels=np.arange(args.vmin, args.vmax, args.vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')
        plt.show()

        # Save 2D heatmaps image
        plt.figure()
        sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=args.vmin, vmax=args.vmax,
                               xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(result_file_path + surf_name + '_2dheat.pdf',
                                      dpi=300, bbox_inches='tight', format='pdf')

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        fig.savefig(result_file_path + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')


def visualize_notebook(args, save_path, surface_path):
    result_file_path = os.path.join(save_path, '2D_images/')
    if not os.path.isdir(result_file_path):
        os.makedirs(result_file_path)
    surf_name = args.surf_name

    with h5py.File(surface_path, 'r') as f:
        Z_LIMIT = 10

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])
        
        X, Y = np.meshgrid(x, y)

        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        else:
            print(f'{surf_name} is not found in {surface_path}')
            Z = np.zeros_like(X)  # Or handle the case appropriately
        print(Z.min())

        # Ensure Z is limited and/or transformed if needed
        # Z[Z > Z_LIMIT] = Z_LIMIT
        # Z = np.log(Z)  # logscale

        # Save 2D contours image
        plt.figure()
        CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(Z.min(), Z.max(), (Z.max() - Z.min()) / 10))
        plt.clabel(CS, inline=1, fontsize=8)
        plt.title('Contour Plot')
        plt.colorbar(CS)
        plt.show()

        plt.figure()
        CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(Z.min(), Z.max(), (Z.max() - Z.min()) / 10))
        plt.colorbar(CS)
        plt.title('Filled Contour Plot')
        plt.show()

        # Save 2D heatmaps image
        plt.figure()
        sns.heatmap(Z, cmap='viridis', cbar=True,
                    xticklabels=False, yticklabels=False)
        plt.title('Heatmap')
        plt.show()

        # Different viewing angles
        angles = [(30, 30), (45, 30), (60, 30),(60, 30)]
        
        # Plot in different views
        for elev, azim in angles:
            # Save 3D surface image
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
            ax.set_title('3D Surface Plot')
            ax.view_init(elev=elev, azim=azim)
            plt.title(f'Elevation: {elev}, Azimuth: {azim}')
            plt.show()


def visualize_trajectory(proj_file, dir_file, show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""
    assert exists(proj_file), 'Projection file does not exist.'
    f = h5py.File(proj_file, 'r')
    fig = plt.figure()
    plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')
    f.close()

    if exists(dir_file):
        f2 = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f2.close()

    fig.savefig(proj_file + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    if show: plt.show()

def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])

    fig = plt.figure()
    CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

    # plot trajectories
    pf = h5py.File(proj_file, 'r')
    plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pf.close()
    if show: plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_path', default='results', help='Base path of logging')
    parser.add_argument('--ex_name', default='resnet56_base', help='Type of Experiments')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    args = parser.parse_args()
    save_path = os.path.join(args.logging_path, args.ex_name) 
    surface_path = f"{save_path}/3d_surface_file.h5"
    visualize(args, save_path, surface_path)