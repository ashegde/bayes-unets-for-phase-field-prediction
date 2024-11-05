import numpy as np
import scipy
import scipy.fftpack


def dct2(a: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Discrete Cosine Transform (DCT) of a given array.

    This function performs a two-dimensional DCT by applying the DCT
    along the rows and then along the columns. The normalization is set
    to 'ortho'.

    Parameters:
    -----------
    a : np.ndarray
        A 2D numpy array on which the DCT is to be computed.

    Returns:
    --------
    np.ndarray
        A 2D array containing the DCT coefficients of the input array.
    
    References:
    -----------
    - https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    """
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a: np.ndarray) -> np.ndarray:
    """
    Compute the 2D Inverse Discrete Cosine Transform (IDCT) of a given array.

    This function performs a two-dimensional IDCT by applying the IDCT
    along the rows and then along the columns. The normalization is set
    to 'ortho'.

    Parameters:
    -----------
    a : np.ndarray
        A 2D numpy array containing DCT coefficients to be transformed back.

    Returns:
    --------
    np.ndarray
        A 2D array containing the original data after applying the IDCT.
    
    References:
    -----------
    - https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
    """
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


class CahnHilliardSimulator:
    """
    A class to simulate the Cahn-Hilliard equation for phase separation.

    The implementation is based on the work by Lee et al. (2014) which
    provides the mathematical and numerical derivations of the Cahn-Hilliard 
    equation.

    Attributes:
    -----------
    u : np.ndarray
        The current state of the system (concentration field or order parameter).
    x_res : int
        The resolution of the grid in the x-direction.
    y_res : int
        The resolution of the grid in the y-direction.
    t : float
        The current time of the simulation.
    dt : float
        The time step for the simulation.
    X : np.ndarray
        Meshgrid for x-coordinates.
    Y : np.ndarray
        Meshgrid for y-coordinates.
    Leid : np.ndarray
        Eigenvalues for the linear operator.
    CHeig : np.ndarray
        Coefficients for the Cahn-Hilliard update.

    References:
    -----------
    Lee, D., Huh, J. Y., Jeong, D., Shin, J., Yun, A., & Kim, J. (2014).
    Physical, mathematical, and numerical derivations of the Cahn–Hilliard 
    equation. Computational Materials Science, 81, 216-225.
    """

    def __init__(self, dt: float):
        """
        Initialize the Cahn-Hilliard simulator with a concentration field.

        Parameters:
        -----------
        dt : float
            Time step for the simulation.

        Raises:
        -------
        ValueError: If the initial concentration field has an invalid shape.
        """
        # Problem setup
        x_right = 2
        y_right = 2
        x_res = 32
        y_res = 32
        x = np.linspace(0.5 * x_right / x_res, x_right - 0.5 * x_right / x_res, x_res)
        y = np.linspace(0.5 * y_right / y_res, y_right - 0.5 * y_right / y_res, y_res)
        h = x[1] - x[0]
        epsilon = 4 * h / (2 * np.sqrt(2) * np.arctanh(0.9))
        xp = np.linspace(0, (x_res - 1) / x_right, x_res)
        yq = np.linspace(0, (y_res - 1) / y_right, y_res)

        # Stored variables
        self.u = None
        self.x_res = x_res
        self.y_res = y_res
        self.t = 0.0
        self.dt = dt
        self.f = lambda a: a**3 - 3 * a
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')  # X, Y are (M, N)
        self.Leig = -(np.tile((xp**2), (y_res, 1)).T + np.tile(yq**2, (x_res, 1))) * (np.pi**2)
        self.CHeig = np.ones((x_res, y_res)) - 2 * self.dt * self.Leig + self.dt * (epsilon**2) * (self.Leig**2)

    def free_energy_deriv(self, u: np.ndarray) -> np.ndarray:
        """
        Derivative of the free energy functional.

        Parameters:
        -----------
        u : np.ndarray
            Concentration field

        Returns:
        --------
        np.ndarray
            Free energy functional derivative 
        """
        return u**3 - 3*u

    def initialize(self, u: np.ndarray):
        """
        Initialize the concentration field for the simulation.

        Parameters:
        -----------
        u : np.ndarray
            Initial concentration field to be set.

        Raises:
        -------
        ValueError: If the shape of the initial concentration field is not
                    consistent with the simulator's resolution.
        """
        self.u = u

    def step(self) -> np.ndarray:
        """
        Perform a single time step of the simulation.

        This method updates the concentration field using the Cahn-Hilliard 
        equation and returns the updated field.

        Returns:
        --------
        np.ndarray
            The updated concentration field after one time step.

        Raises:
        -------
        ValueError: If the concentration field has not been initialized.
        """
        if self.u is None:
            raise ValueError('Field u not initialized!')
        fu = self.f(self.u)
        hat_u = dct2(np.real(self.u))
        df = dct2(np.real(fu))
        hat_u = (hat_u + self.dt * self.Leig * df) / self.CHeig
        self.u = idct2(hat_u)
        self.t += self.dt
        return self.u
