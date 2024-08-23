import torch
import numpy as np
import os
import copy
import h5py


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)

    return [x_direction, y_direction]


def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)

    return direction


def get_weights(model):
    return [p.data for p in model.parameters()]


def get_random_weights(weights):
    return [torch.randn(w.size()).to(device) for w in weights]


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights): 
        d.mul_(w.norm() / (d.norm() + 1e-10)) 


def normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.fill_(0) 
        normalize_direction(d, w)


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y

def setup_PCA_directions(args, model_files, w, s):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """
    pass

def project_trajectory(dir_file, w, s, dataset, model_name, model_files, dir_type='weight', proj_method='cos'):
    """
        Project the optimization trajectory onto the given two directions.

        Args:
          dir_file: the h5 file that contains the directions
          w: weights of the final model
          s: states of the final model
          model_name: the name of the model
          model_files: the checkpoint files
          dir_type: the type of the direction, weights or states
          proj_method: cosine projection

        Returns:
          proj_file: the projection filename
    """
