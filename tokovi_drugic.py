import numpy as np
import os, scipy, mpmath
from numba import njit, prange

os.chdir('/Users/ana/Desktop/ta2nise5/parameters')

def kinetic(file='parametri-kinetic.txt'):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            seznam.append([x, y, orb1, orb2, t])
    return np.array(seznam)

def interaction(file='parametri-interaction.txt'):
    seznam = []
    with open(file, 'r') as f:
        for line in f:
            [x, y, orb1, orb2, utez] = list(map(float, line.split()))
            seznam.append([x, y, orb1, orb2, utez])
    return np.array(seznam)

def positions(a, b, b2):
    return np.array([[-a/4, b/2 - b2],
                    [-a/4,b2],
                    [a/4,-b2],
                    [a/4, -b/2 + b2],
                    [a/4,b/4],
                    [-a/4, -b/4]])

''' matrix for number density operator '''
def j_tok(Kymesh, Kxmesh, a, b, b2, file='parametri-kinetic.txt'):
    pos = positions(a, b, b2)
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)

    for line in file:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        if orb1 == orb2 and (x,y) == (0,0): pass # this is onsite energy, does not contribute to j
        else:
            osnova = 1j * t * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b))
            lega = pos[orb2] - pos[orb1] - np.array([x*a, y*b])
            ad_x = osnova * lega[0]
            ad_y = osnova * lega[1]

            jx[orb1 - 1, orb2 - 1] += ad_x
            if orb1 != orb2:
                jx[orb2 - 1, orb1 - 1] += ad_x.conjugate() 
            jy[orb1 - 1, orb2 - 1] += ad_y
            if orb1 != orb2:
                jy[orb2 - 1, orb1 - 1] += ad_y.conjugate()
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype='complex')
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

''' matrix for energy density operator -- contribution from hop-hop '''
def j_1(Kymesh, Kxmesh, a, b, b2, kinetic, mu):
    pos = positions(a, b, b2)
    Ny, Nx = Kymesh.shape
    jx = np.zeros((6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)
    for line in kinetic:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        if orb1 == orb2 and (x,y) == (0,0):
            t += - mu
        for line_ in kinetic:
            x_, y_, orb1_, orb2_, t_ = line_
            x_, y_, orb1_, orb2_, t_ = float(x_), float(y_), int(orb1_), int(orb2_), float(t_)
            if orb1_ == orb2_ and (x_,y_) == (0,0):
                t_ += -mu
            if orb2 == orb1_:
                osnova = - 1j * t * t_ * 0.5 * np.exp(-1j * (Kxmesh * (x + x_) * a + Kymesh * (y + y_) * b))
                lega = pos[orb1 - 1] - pos[orb2 - 1] + np.array([x + x_, y + y_])
                ad_x = osnova * lega[0]
                ad_y = osnova * lega[1]
                jx[orb1 - 1, orb2_ - 1] += ad_x
                jy[orb1 - 1, orb2_ - 1] += ad_y
    jmatrix = np.zeros((2,6,6,Ny,Nx), dtype='complex')
    jmatrix[0] = jx
    jmatrix[1] = jy
    return jmatrix

def Delta(Kxmesh, Kymesh, Nk, rho, i, j, x): 
    if type(x) == np.ndarray:
        return np.array([np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x1[0] - 1j * Kymesh * x1[1])) for x1 in x]) / Nk
    else: return np.sum(rho[i, j] * np.exp(-1j * Kxmesh * x[0] - 1j * Kymesh * x[1])) / Nk

''' tensor for energy density operator -- contribution from hop-interaction'''
def j_2_kq(Kymesh, Kxmesh, qy, qx, V, a, b, b2, file1='hop.txt', file2='int.txt'):
    pos = {1: np.array([-a/4, b/2 - b2]), 2: np.array([-a/4,b2]), 3 : np.array([a/4,-b2]), 4: np.array([a/4, -b/2 + b2]), 5: np.array([a/4,b/4]), 6: np.array([-a/4, -b/4])}
    Ny, Nx = Kymesh.shape
    Nk = Ny * Nx
    jx = np.zeros((6, 6, 6, Ny, Nx), dtype='complex')
    jy = np.copy(jx)
    with open(file1, 'r') as f1:
        for line in f1:
            [x, y, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)

            with open(file2, 'r') as f2:
                for line in f2:
                    [x_, y_, orb1_, orb2_, utez] = list(map(float, line.split()))
                    orb1_, orb2_ = int(orb1_), int(orb2_)

                    if orb2 == orb2_:
                        osnova = - 1j * t * utez * V * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b)) * np.exp(1j * (qx * x_ * a + qy * y_ * b)) * np.exp(-1j*(qx * x * a + qy * y * b))
                        position = pos[orb2] - pos[orb1_] - np.array([x_*a, y_*b])
                        jx[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[0]
                        jy[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[1]

                    if orb1 == orb2_:
                        osnova = 1j * t * utez * V * np.exp(-1j * (Kxmesh * x * a + Kymesh * y * b)) * np.exp(1j * (qx * x_ * a + qy * y_ * b))
                        position = pos[orb1] - pos[orb1_] - np.array([x_*a, y_*b])
                        jx[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[0]
                        jy[orb1 - 1, orb1_ - 1, orb2 - 1] += osnova * position[1]
    return jx / Nk, jy / Nk

@njit(cache=True)
def spektralna_k(omega, mu, energije_k, Gamma):
    N_orbitals = energije_k.shape[0]
    A = np.zeros((N_orbitals, N_orbitals), dtype=np.complex128)
    for i in range(N_orbitals):
        A[i,i] = -1/np.pi * Gamma / ((omega - energije_k[i] + mu)**2 + Gamma**2)
    return A

''' df/domega, f je Fermi-Diracova porazdelitvena funkcija '''
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2


def helper_phi(omegas, j_tilde, energije, mu, Gamma):
    transportna = np.zeros((omegas.shape[0]), dtype='complex')
    for q, omega in enumerate(omegas):
        A = spektralna_k(omega, mu, energije, Gamma)
        for nu in range(2):
            transportna[nu][q] += 2 * np.trace(j_tilde @ A @ j_tilde @ A)
    return transportna

@njit(parallel=True, cache=True)
def transportna_phi(Kymesh, vecs, energije, j_matrix, mu, omegas, Gamma):
    Ny, Nx = Kymesh.shape
    transportna = np.zeros((2, omegas.shape[0]), dtype=np.complex128)

    for n in prange(Nx):
        for ind in [0, Ny//2]:
            vec = vecs[:,:,ind,n]
            for nu in range(2):
                j_tok = vec @ j_matrix[nu][:,:,ind,n] @ vec.conj().T
                transportna[nu] += helper_phi(omegas, j_tok, energije[:,ind,n], mu, Gamma)

    for m in prange(Ny):
        for ind in [0, Nx//2]:
            vec = vecs[:,:,m,ind]
            for nu in range(2):
                j_tok = vec @ j_matrix[nu][:,:,m,ind] @ vec.conj().T
                transportna[nu] += helper_phi(omegas, j_tok, energije[:,m,ind], mu, Gamma)

    for m in range(1,Ny):
        if m == Ny//2: pass
        else:
            for n in range(1,Nx//2):
                vec = vecs[:,:,m,n]
                for nu in range(2):
                    j_tok = vec @ j_matrix[nu][:,:,m,n] @ vec.conj().T
                    transportna[nu] += 2 * helper_phi(omegas, j_tok, energije[:,m,n], mu, Gamma)
    
    return transportna.real

def transportna_phi_Boltzmann(Kymesh, Kxmesh, energije, mu, omegas, faktor=1):

    energije_new = np.copy(energije)

    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_new_y, velocity_new_x = np.zeros(energije.shape), np.zeros(energije.shape)

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[1] - energije[0] < 0.005) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
        energije_new[1,ind1[i], ind2[i]] = energije[0,ind1[i], ind2[i]]
        energije_new[0,ind1[i], ind2[i]] = energije[1,ind1[i], ind2[i]]
    for j in range(3,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[1,:enice[-1],j] = energije[0,:enice[-1],j]
                energije_new[0,:enice[-1],j] = energije[1,:enice[-1],j]
    for j in range(Nx//2+1, Nx-5):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[1,enice[0]:,j] = energije[0,enice[0]:,j]
                energije_new[0,enice[0]:,j] = energije[1,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[3] - energije[2] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[3,:enice[-1],j] = energije[2,:enice[-1],j]
                energije_new[2,:enice[-1],j] = energije[3,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[3,enice[0]:,j] = energije[2,enice[0]:,j]
                energije_new[2,enice[0]:,j] = energije[3,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[5] - energije[4] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[5,:enice[-1],j] = energije[4,:enice[-1],j]
                energije_new[4,:enice[-1],j] = energije[5,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[5,enice[0]:,j] = energije[4,enice[0]:,j]
                energije_new[4,enice[0]:,j] = energije[5,enice[0]:,j]

    maksimumi_y, maksimumi_x = [], []
    for i in range(6):
        gr = np.gradient(energije[i])
        velocity_new_y[i, :, :], velocity_new_x[i, :, :] = gr[0]/dKy, gr[1]/dKx
        maksimumi_y.append(np.max(np.abs(velocity_new_y[i,:,:])))
        maksimumi_x.append(np.max(np.abs(velocity_new_x[i,:,:])))

    domega = omegas[1] - omegas[0]
    transportna = np.zeros((2, omegas.shape[0]))
    v_max = np.array([np.max(maksimumi_y), np.max(maksimumi_x)])
    sigma = faktor * np.array([np.sqrt(v_max[0] * domega * dKy), np.sqrt(v_max[1] * domega * dKx)])
    for i in range(Ny):
        for j in range(Nx//2):
            for m in range(6):
                ''' factor 2 is for spin degeneracy,
                 second factor 2 if due to inversion symmetry of the Brillouin zone '''
                transportna[0] += 2 * 2 *  1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[m,i,j] + mu)**2/(2*sigma[0]**2)) * velocity_new_x[m,i,j]**2
                transportna[1] += 2 * 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[m,i,j] + mu)**2/(2*sigma[1]**2)) * velocity_new_y[m,i,j]**2
    jnew = np.zeros((2, 6, Ny, Nx))
    jnew[0] = velocity_new_x
    jnew[1] = velocity_new_y
    return transportna, jnew

def L_Boltzmann(Kymesh, Kxmesh, energije, mu, T):
    energije_new = np.copy(energije)

    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_y, velocity_x = np.zeros(energije.shape), np.zeros(energije.shape)
    velocity_new_y, velocity_new_x = np.copy(velocity_y), np.copy(velocity_x)

    for i in range(6):
        gr = np.gradient(energije[i])
        velocity_y[i, :, :], velocity_x[i, :, :] = gr[0]/dKy, gr[1]/dKx

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[1] - energije[0] < 0.005) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
        energije_new[1,ind1[i], ind2[i]] = energije[0,ind1[i], ind2[i]]
        energije_new[0,ind1[i], ind2[i]] = energije[1,ind1[i], ind2[i]]
    for j in range(3,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[1,:enice[-1],j] = energije[0,:enice[-1],j]
                energije_new[0,:enice[-1],j] = energije[1,:enice[-1],j]
    for j in range(Nx//2+1, Nx-5):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[1,enice[0]:,j] = energije[0,enice[0]:,j]
                energije_new[0,enice[0]:,j] = energije[1,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[3] - energije[2] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[3,:enice[-1],j] = energije[2,:enice[-1],j]
                energije_new[2,:enice[-1],j] = energije[3,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[3,enice[0]:,j] = energije[2,enice[0]:,j]
                energije_new[2,enice[0]:,j] = energije[3,enice[0]:,j]

    mask = np.zeros(energije[1].shape)
    ind1, ind2 = np.where(energije[5] - energije[4] < 0.001) 
    for i in range(len(ind1)):
        mask[ind1[i], ind2[i]] = 1
    for j in range(Nx//4,Nx//2):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[0] != 0:
                energije_new[5,:enice[-1],j] = energije[4,:enice[-1],j]
                energije_new[4,:enice[-1],j] = energije[5,:enice[-1],j]
    for j in range(Nx//2, Nx - Nx//4):
        indices = mask[:,j]
        nicle = np.where(indices == 0)[0]
        enice = np.where(indices == 1)[0]
        if len(nicle) * len(enice) > 0:
            if enice[-1] != 0:
                energije_new[5,enice[0]:,j] = energije[4,enice[0]:,j]
                energije_new[4,enice[0]:,j] = energije[5,enice[0]:,j]

    for i in range(6):
        gr = np.gradient(energije_new[i])
        velocity_new_y[i, :, :], velocity_new_x[i, :, :] = gr[0]/dKy, gr[1]/dKx

    L11 = np.zeros(2)
    L12 = np.copy(L11)
    L11_new, L12_new = np.copy(L11), np.copy(L11)
    for i in range(Ny):
        for j in range(Nx//2):
            for m in range(6):
                L11[0] += 2 * 2 * velocity_x[m,i,j]**2 * (-fd_1(energije[m,i,j] - mu, T) ) 
                L11[1] += 2 * 2 * velocity_y[m,i,j]**2 * (-fd_1(energije[m,i,j] - mu, T) )
                L12[0] += 2 * 2 * velocity_x[m,i,j]**2 * (energije[m,i,j] - mu) * (-fd_1(energije[m,i,j] - mu, T))
                L12[1] += 2 * 2 * velocity_y[m,i,j]**2 * (energije[m,i,j] - mu) * (-fd_1(energije[m,i,j] - mu, T))

                L11_new[0] += 2 * 2 * velocity_new_x[m,i,j]**2 * (-fd_1(energije_new[m,i,j] - mu, T) ) 
                L11_new[1] += 2 * 2 * velocity_new_y[m,i,j]**2 * (-fd_1(energije_new[m,i,j] - mu, T) )
                L12_new[0] += 2 * 2 * velocity_new_x[m,i,j]**2 * (energije_new[m,i,j] - mu) * (-fd_1(energije_new[m,i,j] - mu, T))
                L12_new[1] += 2 * 2 * velocity_new_y[m,i,j]**2 * (energije_new[m,i,j] - mu) * (-fd_1(energije_new[m,i,j] - mu, T))
    return L11, L12, L11_new, L12_new