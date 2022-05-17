import numpy as np
import matplotlib.pyplot as plt

# Coefficients taken from 
#  [1] https://en.wikipedia.org/wiki/Finite_difference_coefficient
#  [2] "Finite Difference Coefficients Calculator", Taylor,
#       Cameron R. (2016). URL: https://web.media.mit.edu/~crtaylor/calculator.html
FD_COEFFICIENTS = { # [1]
    1 : {  2 : np.array([-1/2, 0, 1/2]),
        4 : np.array([1/12, -2/3, 0, 2/3, -1/12]),
        6 : np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
        8 : np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280 ])
    },
    2 : {   2 : np.array([1, -2, 1]),
        4 : np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
        6 : np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
        8 : np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),
        10: np.array([8 ,-125 ,1000 ,-6000 ,42000 ,-73766 ,42000 ,-6000 ,1000 ,-125 ,8])/(25200) # [2]
        }
    }


def get_fd_coefficients(derivative, accuracy):
    return FD_COEFFICIENTS[derivative][accuracy]

def get_laplacian_matrix(derivative, accuracy):
    # We build the laplacian matrix according to the 2009 ARD Paper by Raghuvanshi and Mehra
    # This is the 'K' Matrix appearing in the update rule of the FDTD wave equation
    coefs = get_fd_coefficients(derivative, accuracy)
    nr_pts = int(accuracy / 2)
    k = np.zeros(shape=nr_pts)
    k = coefs[0:nr_pts]
    K = np.zeros(shape=(nr_pts, nr_pts))

    for z in range(nr_pts):
        line = k[0:(z+1)]
        dest = np.zeros(shape=nr_pts)
        dest = line[::-1]
        K[z][0:(z+1)] = dest

    K = (np.vstack([K,-np.flipud(K)]))
    K = (np.hstack([-np.fliplr(K),K]))
    return K


if __name__ == '__main__':
    plt.close('all')
    
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('First Order')
    for ind, current_ax in enumerate(FD_COEFFICIENTS[1]):
        print(ind)
        current_ax = ax.flatten()[ind]
        curren_fdc = list(FD_COEFFICIENTS[1].values())[ind]
        num_pts = len(curren_fdc)
        current_ax.set_title(f"{num_pts} point stencil")
        current_ax.plot((np.arange(num_pts) - int(num_pts/2)), curren_fdc, '-o')
        current_ax.set_xlabel('dx')
        current_ax.set_ylabel('weight')
        current_ax.set_xlim(-8, 8)
        current_ax.set_ylim(-1, 1)
        
    fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(10, 5))
    fig.suptitle('Laplacian')
    for ind, current_ax in enumerate(FD_COEFFICIENTS[2]):
        print(ind)
        current_ax = ax.flatten()[ind]
        curren_fdc = list(FD_COEFFICIENTS[2].values())[ind]
        num_pts = len(curren_fdc)
        current_ax.set_title(f"{num_pts} point stencil")
        current_ax.plot((np.arange(num_pts) - int(num_pts/2)), curren_fdc, '-o')
        current_ax.set_xlabel('dx')
        current_ax.set_ylabel('weight')
        current_ax.set_xlim(-8, 8)
        current_ax.set_ylim(min(curren_fdc)-0.5, max(curren_fdc)+0.5)
    ax[2,1].axis('off')
    
    plt.figure(3)
    K = get_laplacian_matrix(2, 6)
    plt.imshow(K)
    plt.show()
    print(K)

