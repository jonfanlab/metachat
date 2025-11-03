import numpy as np
import torch
import consts

def Hz_to_Ex(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
             eps_0: float = consts.eps_0) -> torch.Tensor:
    '''
    This function performs finite difference on Hz to obtain corresponding Ex field.
    Note that ceviche implements the finite differnce slightly different from the
    conventional setup. Hence, the difference between Hz[:, 1] and Hz[:, 0] produces
    Ex[:, 0] instead of Ex[:, 1]. That's why x[:, 0:-1] is divided instead of x[:, 1:].
    Due to the need of averaging, the required dimension of the material grid is 1
    dimension larger from the bottom than fields.
    ----------
    Parameters
    ----------
    Hz : np.array:
        Hz fields (-Hy).
    dL : float:
        step size.
    omega : float:
        angular frequency.
    eps_grid : np.array:
        material grid.
    eps_0 : float
        vaccum permittivity.
    -------
    Returns
    -------
    Ex : np.array:
        finite-differenced Ex fields.
    '''
    # Material averaging
    x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :])
    Ex_R = (Hz_I[:, 1:, :] - Hz_I[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]
    Ex_I = -(Hz_R[:, 1:, :] - Hz_R[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]

    return torch.stack((Ex_R, Ex_I), axis = 1)

def Hz_to_Ey(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
             eps_0: float = consts.eps_0) -> torch.Tensor:
    '''
    This function performs finite difference on Hz to obtain corresponding Ey field.
    Note that ceviche implements the finite differnce slightly different from the
    conventional setup. Hence, the difference between Hz[1, :] and Hz[0, :] produces
    Ey[0, :] instead of Ex[1, :]. That's why y[0:-1, :] is divided instead of y[1:, :].
    Due to the periodic structure, the required dimension of the material grid is same as the fields.
    ----------
    Parameters
    ----------
    Hz : np.array:
        Hz fields (-Hy).
    dL : float:
        step size.
    omega : float:
        angular frequency.
    eps_grid : np.array:
        material grid.
    eps_0 : float
        vaccum permittivity.
    -------
    Returns
    -------
    Ey : np.array:
        finite-differenced Ey (Ez) fields.
    '''

    y = 1 / 2 * (eps_grid[:, :, 1:, :] + torch.roll(eps_grid[:, :, 1:, :], 1, dims = 3))

    Ey_R = -(torch.roll(Hz_I, -1, dims = 2) - Hz_I)/dL/omega/eps_0/y[:, 0, :, :]

    Ey_I = (torch.roll(Hz_R, -1, dims = 2) - Hz_R)/dL/omega/eps_0/y[:, 0, :, :]

    return torch.stack((Ey_R, Ey_I), axis = 1)

def E_to_Hz(Ey_R: torch.Tensor, Ey_I: torch.Tensor, Ex_R: torch.Tensor, Ex_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, \
            omega: torch.Tensor, mu_0: float = consts.mu_0) -> torch.Tensor:
    '''
    This function performs finite difference on Ey, Ex to obtain corresponding Hz field.
    The -1j in the denominator has been absorbed for Hz -> -Hy
    ----------
    Parameters
    ----------
    Ey : np.array:
        Ey fields (Ez).
    Ex : np.array:
        Ex fields.
    dL : float:
        step size.
    omega : float:
        angular frequency.
    mu_0 : float
        vaccum permeability.
    -------
    Returns
    -------
    Hz : np.array:
        finite-differenced Hz (-Hy) fields.
    '''

    # omega = omega.view(-1, 1, 1)  # Reshape omega for broadcasting
    Hz_R = ((Ey_I[:, 1:] - torch.roll(Ey_I[:, 1:], 1, dims = 2)) - (Ex_I[:, 1:] - \
             Ex_I[:, 0:-1]))/dL/omega/mu_0
    Hz_I = -((Ey_R[:, 1:] - torch.roll(Ey_R[:, 1:], 1, dims = 2)) - (Ex_R[:, 1:] - \
              Ex_R[:, 0:-1]))/dL/omega/mu_0
    return torch.stack((Hz_R, Hz_I), axis = 1)

def H_to_H(Hz_R: torch.Tensor, Hz_I: torch.Tensor, Mz_R: torch.Tensor, Mz_I: torch.Tensor, dL: float, omega: torch.Tensor, eps_grid: torch.Tensor, \
           eps_0: float = consts.eps_0, mu_0: float = consts.mu_0) -> torch.Tensor:

    '''
    This function calls FD_Ex, FD_Ez, and subsequently FD_H to implement the
    Helmholtz equation for the H field.
    '''

    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, Mz_R, Mz_I, dL, omega, eps_grid, eps_0)
    FD_Ey = Hz_to_Ey(Hz_R, Hz_I, Mz_R, Mz_I, dL, omega, eps_grid, eps_0)
    FD_H = -E_to_Hz(-FD_Ey[:, 0, :-1], -FD_Ey[:, 1, :-1], -FD_Ex[:, 0], -FD_Ex[:, 1], Mz_R, Mz_I, dL, omega, mu_0)

    source_vector = (1/(mu_0)) * torch.stack((Mz_I, -Mz_R), axis = 1)[:,:,1:-1,:]

    FD_H -= source_vector
    
    return FD_H