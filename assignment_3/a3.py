import numpy as np

def complex_iter(xmin=-2, xmax=2, ymin=-2, ymax=2, resolution=1000, max_iter=100):
    """
    Compute the complex iteration over a 2x2 grid.
    Parameters:
        xmin : float or int
        first value for x-axis boundary
        
        xmax : float or int
        second value for x-axis boundary
        
        ymin : float or int
        first value for y-axis boundary
        
        ymax : float or int
        second value for y-axis boundary
        
        resolution : float or int
        the discreteness of each axis, i.e the incremental values of the grid
        
        max_iter : float or int
        maximum number of iterations allowed for each point
    
    Returns:
        C : 2D array of complex points
        diverge_iter : 2D array of iteration number at which each point diverged
                      (points that stay bounded get value max_iter)
    """
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    Z = np.zeros_like(C)
    diverge_iter = np.full(C.shape, max_iter)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        diverged = (np.abs(Z) > 2) & (diverge_iter == max_iter)
        diverge_iter[diverged] = i

    return C, diverge_iter