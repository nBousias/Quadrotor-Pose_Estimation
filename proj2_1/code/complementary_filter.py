# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    #TODO Your code here - replace the return value with one you compute

    em = np.abs(norm(linear_acceleration/9.8)-1)

    alpha = np.piecewise(em, [em < 0.1, 0.1<=em<=0.2, em>0.2], [1, lambda em: -10*em+2 ,0])

    omega_hat = skew_sym_matrix(angular_velocity)
    R = initial_rotation.as_matrix() @ expm(omega_hat*dt)

    g_prime = R @ normalize(linear_acceleration/9.8)

    Dq_acc = np.array([0, g_prime[2]/np.sqrt(2*(1+g_prime[0])), -g_prime[1]/np.sqrt(2*(1+g_prime[0])), np.sqrt((g_prime[0]+1)/2)])
    dq = (1-alpha)*np.array([0,0,0,1]) + alpha*Dq_acc

    R = Rotation.from_quat(dq).as_matrix() @ R

    return Rotation.from_matrix(R)


def normalize(a):
    return a/norm(a)

def skew_sym_matrix(dw):
    return np.array([[0,-dw[2],dw[1]],
                     [dw[2],0,-dw[0]],
                     [-dw[1],dw[0],0]])