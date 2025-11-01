import numpy as np
import scipy.linalg as LA
import os
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import seaborn as sns
from numba import njit, prange

def parametri_kinetic(epsilon, t1x, t1y, t2x, t2y):
    kinetic = np.array([[0, 0, 1, 1, -epsilon],
                        [1, 0, 1, 1, t1x],
                        [0, 1, 1, 1, t1y],
                        [1, 0, 2, 2, -t2x],
                        [0, 1, 2, 2, -t2y],
                        [0, 0, 2, 2, epsilon]])
    return kinetic

def Energy(Kymesh, Kxmesh, epsilon, t1x, t1y, t2x, t2y, a, b):
    Ny, Nx = Kymesh.shape
    energy = np.zeros((2, Ny, Nx))
    energy[0] = -epsilon + 2*t1x*np.cos(Kxmesh * a) + 2*t1y*np.cos(Kymesh * b)
    energy[1] = epsilon - 2*t2x*np.cos(Kxmesh * a) - 2*t2y*np.cos(Kymesh * b)
    return energy

def Rho(Ny, Nx, energije, mu, T):
    rho0 = np.zeros((2, 2, Ny, Nx))
    for orb in range(2):
        rho0[orb, orb, :, :] = 1/(np.exp((energije[orb] - mu)/T) + 1)
    return rho0

def H_hopping(Kymesh, Kxmesh, a, b, parametri_kinetic):
    Ny, Nx = Kymesh.shape
    hop = np.zeros((2, 2, Ny, Nx), dtype='complex')
    for line in parametri_kinetic:
        x, y, orb1, orb2, t = line
        x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
        ad = t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
        hop[orb1 - 1, orb2 - 1] += ad
        if orb1 != orb2: hop[orb2 - 1, orb1 - 1] += ad.conjugate()
        if orb1 == orb2 and (x,y) != (0,0): hop[orb1 - 1, orb2 - 1] += ad.conjugate()
    return hop

def j_tok(Kymesh, Kxmesh, a, b, parametri_kinetic, positions):
    Ny, Nx = Kxmesh.shape
    tok = np.zeros((2, 2, 2, Ny, Nx), dtype='complex')
    for line in parametri_kinetic:
        x, y, orb1, orb2, t = line
        if orb1 == orb2 and (x,y) == (0,0): pass
        else:
            x, y, orb1, orb2, t = float(x), float(y), int(orb1), int(orb2), float(t)
            pos = positions[orb2 - 1] - positions[orb1 - 1] - np.array([x*a, y*b])
            ad = 1j * t * np.exp(-1j*(Kxmesh * x * a + Kymesh * y * b))
            for nu in range(2):
                tok[nu, orb1 - 1, orb2 - 1] += ad * pos[nu]
                tok[nu, orb2 - 1, orb1 - 1] += ad.conj() * pos[nu]
    return tok
    
@njit(cache=True)
def fd_1(omega, T): return -1/(4*T)/np.cosh(omega/(2*T))**2

def H_diagonalize2(Ny, Nx, Hamiltonian):
    energije, vecs = np.zeros((2, Ny, Nx)), np.zeros((2, 2, Ny, Nx), dtype='complex')

    for n in range(Nx):
        en, v = LA.eigh(Hamiltonian[:, :, 0, n])
        energije[:, 0, n] = en
        vecs[:, :, 0, n] = v

        en, v = LA.eigh(Hamiltonian[:, :, Ny//2, n])
        energije[:, Ny//2, n] = en
        vecs[:, :, Ny//2, n] = v

    for m in range(Ny):
        en, v = LA.eigh(Hamiltonian[:, :, m, 0])
        energije[:, m, 0] = en
        vecs[:, :, m, 0] = v

        en, v = LA.eigh(Hamiltonian[:, :, m, Nx//2])
        energije[:, m, Nx//2] = en
        vecs[:, :, m, Nx//2] = v

    for m in range(1,Ny//2):
        for n in range(1,Nx//2):
            en, v = LA.eigh(Hamiltonian[:, :, m, n])
            energije[:, m, n] = en
            energije[:, -m, -n] = en
            vecs[:, :, m, n] = v
            vecs[:, :, -m, -n] = v.conj()

            en, v = LA.eigh(Hamiltonian[:, :, m, n + Nx//2])
            energije[:, m, n + Nx//2] = en
            energije[:, -m, Nx//2 - n] = en
            vecs[:, :, m, n + Nx//2] = v
            vecs[:, :, -m, Nx//2 - n] = v.conj()
    return energije, vecs

''' Boltzmann's transport function and group velocity '''
def phi_B(Kymesh, Kxmesh, energije, omegas, mu, faktor):
    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_y, velocity_x = np.zeros(energije.shape), np.zeros(energije.shape)

    maksimumi_y, maksimumi_x = [], []
    for i in range(2):
        gr = np.gradient(energije[i])
        velocity_y[i, :, :], velocity_x[i, :, :] = gr[0]/dKy, gr[1]/dKx
        maksimumi_y.append(np.max(np.abs(velocity_y[i,:,:])))
        maksimumi_x.append(np.max(np.abs(velocity_x[i,:,:])))

    domega = omegas[1] - omegas[0]
    transportna = np.zeros((2, omegas.shape[0]))
    v_max = np.array([np.max(maksimumi_y), np.max(maksimumi_x)])
    sigma = faktor * np.array([np.sqrt(v_max[0] * domega * dKy), np.sqrt(v_max[1] * domega * dKx)])
    
    for m in [0, Ny//2]:
        for n in prange(Nx):
            for orb in range(2):
                transportna[0] += 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[1]**2)) * velocity_x[orb,m,n]**2
                transportna[1] += 2 *  1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[0]**2)) * velocity_y[orb,m,n]**2

    for n in [0, Nx//2]:
        for m in prange(Ny):
            for orb in range(2):
                transportna[0] += 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[1]**2)) * velocity_x[orb,m,n]**2
                transportna[1] += 2 *  1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[0]**2)) * velocity_y[orb,m,n]**2
    
    for m in prange(Ny):
        for n in range(Nx//2):
            if m not in [0, Ny//2]:
                print(m,n)
                for orb in range(2):
                    transportna[0] += 2 * 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[1]**2)) * velocity_x[orb,m,n]**2
                    transportna[1] += 2 * 2 *  1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[0]**2)) * velocity_y[orb,m,n]**2

    velocity = np.zeros((2, 2, Ny, Nx))
    velocity[0] = velocity_x
    velocity[1] = velocity_y

    return transportna, velocity

def phi_BB(Kymesh, Kxmesh, energije, omegas, mu, faktor):
    Ny, Nx = Kymesh.shape
    dKy, dKx = np.diff(Kymesh[:,0])[0], np.diff(Kxmesh[0])[0]
    velocity_y, velocity_x = np.zeros(energije.shape), np.zeros(energije.shape)

    maksimumi_y, maksimumi_x = [], []
    for i in range(2):
        gr = np.gradient(energije[i])
        velocity_y[i, :, :], velocity_x[i, :, :] = gr[0]/dKy, gr[1]/dKx
        maksimumi_y.append(np.max(np.abs(velocity_y[i,:,:])))
        maksimumi_x.append(np.max(np.abs(velocity_x[i,:,:])))

    domega = omegas[1] - omegas[0]
    transportna = np.zeros((2, omegas.shape[0]))
    v_max = np.array([np.max(maksimumi_y), np.max(maksimumi_x)])
    sigma = faktor * np.array([np.sqrt(v_max[0] * domega * dKy), np.sqrt(v_max[1] * domega * dKx)])

    for m in prange(Ny):
        for n in range(Nx):
            print(m,n)
            for orb in range(2):
                transportna[0] += 2 * 1/(2*np.pi*sigma[1]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[1]**2)) * velocity_x[orb,m,n]**2
                transportna[1] += 2 * 1/(2*np.pi*sigma[0]**2)**0.5 * np.exp(-(omegas - energije[orb,m,n] + mu)**2/(2*sigma[0]**2)) * velocity_y[orb,m,n]**2

    velocity = np.zeros((2, 2, Ny, Nx))
    velocity[0] = velocity_x
    velocity[1] = velocity_y

    return transportna, velocity

@njit(cache=True)
def Spectral_k(omega, mu, energy_k, Gamma):
    A = np.zeros((2,2), dtype='complex')
    for i in range(2):
        A[i,i] = -1/np.pi * Gamma / ((omega - energy_k[i] + mu)**2 + Gamma**2)
    return A

''' Kubo's transport function '''
@njit(parallel=True, cache=True)
def phi_K(Kymesh, Kxmesh, tok, energy, omegas, mu, Gamma):
    Ny, Nx = Kxmesh.shape
    transportna = np.zeros((2, len(omegas)), dtype='complex')

    for m in [0, Ny//2]:
        for n in prange(Nx):
            for i, omega in enumerate(omegas):
                A = Spectral_k(omega, mu, energy[:, m, n], Gamma)
                for nu in range(2):
                    tok_k = tok[nu,:,:,m,n]
                    for a in range(2):
                        for b in range(2):
                            transportna[nu,i] += 2. * tok_k[a, b] * A[b, b] * tok_k[b, a] * A[a, a]

    for n in [0, Nx//2]:
        for m in prange(Ny):
            for i, omega in enumerate(omegas):
                A = Spectral_k(omega, mu, energy[:, m, n], Gamma)
                for nu in range(2):
                    tok_k = tok[nu,:,:,m,n]
                    for a in range(2):
                        for b in range(2):
                            transportna[nu,i] += 2. * tok_k[a, b] * A[b, b] * tok_k[b, a] * A[a, a]

    for m in prange(Ny):
        for n in prange(1,Nx//2):
            if m not in [0, Ny//2]:
                print(m,n)
                for i, omega in enumerate(omegas):
                    A = Spectral_k(omega, mu, energy[:, m, n], Gamma)
                    for nu in range(2):
                        tok_k = tok[nu,:,:,m,n]
                        for a in range(2):
                            for b in range(2):
                                transportna[nu,i] += 2. * 2. * (tok_k[a, b] * A[b, b] * tok_k[b, a] * A[a, a]).real

    return transportna

class model:
    def __init__(self, a, b, Ny, Nx, phys_parameters, parameters, positions):
        epsilon, t1x, t1y, t2x, t2y = phys_parameters
        self.pos = positions
        self.Nx, self.Ny = Nx, Ny
        self.Nk = Ny * Nx
        Ky = 2*np.pi/b * np.arange(-Ny/2, Ny/2) / Ny
        Kx = 2*np.pi/a * np.arange(-Nx/2, Nx/2) / Nx
        Kxmesh, Kymesh = np.meshgrid(Kx, Ky)
        self.kxmesh = Kxmesh
        self.kymesh = Kymesh

        self.Hamiltonian = H_hopping(self.kymesh, self.kxmesh, a, b, parameters)
        self.energy, self.vecs = H_diagonalize2(Ny, Nx, self.Hamiltonian)
        self.tok = j_tok(self.kymesh, self.kxmesh, a, b, parameters, positions)
    
    def occupation(self, mu, T):
        fs = np.zeros((2, self.Ny, self.Nx))
        for n in range(Nx):
            for m in [0, Ny//2]:
                if T == 0: 
                    fs[:, m, n] = self.energy[:,m,n] < mu
                else: 
                    fs[:, m, n] = 1/(1 + np.exp((self.energy[:,m,n] - mu)/T))

        for m in range(Ny):
            for n in [0, Ny//2]:
                if T == 0:
                    fs[:, m, n] = self.energy[:,m,n] < mu
                else:
                    fs[:, m, n] = 1/(1 + np.exp((self.energy[:,m,n] - mu)/T))

        for m in range(1,Ny//2):
            for n in range(1,Nx//2):
                if T == 0:
                    fs[:, m, n] = self.energy[:,m,n] < mu
                    fs[:, m, n+Nx//2] = self.energy[:,m,n+Nx//2] < mu
                    fs[:, -m, -n] = self.energy[:,m,n] < mu
                    fs[:, -m, Nx//2-n] = self.energy[:,m,n+Nx//2] < mu
                else:
                    fs[:, m, n] = 1/(1 + np.exp((self.energy[:,m,n] - mu)/T))
                    fs[:, m, n+Nx//2] = 1/(1 + np.exp((self.energy[:,m,n+Nx//2] - mu)/T))
                    fs[:, -m, -n] = 1/(1 + np.exp((self.energy[:,m,n] - mu)/T))
                    fs[:, -m, Nx//2-n] = 1/(1 + np.exp((self.energy[:,m,n+Nx//2] - mu)/T))
        return np.sum(fs) / np.prod(fs.shape)

    def occupation_error(self, mu, occ_target, T):
        return self.occupation(mu, T) - occ_target

    def find_mu(self, mu1, mu2, occ_target, T) -> None:
        sign1 = np.sign(m.occupation(mu1, T) - occ_target)
        sign2 = np.sign(m.occupation(mu2, T) - occ_target)

        while sign1 != -1:
            mu1 -= 0.5
            sign1 = np.sign(m.occupation(mu1, T) - occ_target)

        while sign2 != +1:
            mu2 += 0.5
            sign2 = np.sign(m.occupation(mu2, T) - occ_target)
        m.mu = optimize.brentq(self.occupation_error, mu1, mu2, args=(occ_target, T))


epsilon = 8.5
t1x, t1y = 3., 3.
t2x, t2y = 1., 1.

phys_parameters = [epsilon, t1x, t1y, t2x, t2y]
parameters = parametri_kinetic(epsilon, t1x, t1y, t2x, t2y)


a, b = 1., 1.
pos = np.array([[0.,0.], [a/2, b/2]])

Ny, Nx = 400, 400
m = model(a, b, Ny, Nx, phys_parameters, parameters, pos)
print(m.energy.shape)
T = 1/100
eps = 1e-6
omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
print(omega_max)
mu1 = -5
mu2 = 1
occ_target = 0.503
#rho, energije, fs, vecs, err, n = Rho_next(m.kxmesh, m.rho, m.Hamiltonian, T, m.mu, maxiter, mix, epsilon)

m.find_mu(mu1, mu2, occ_target, T)
plt.plot(m.energy[0,Ny//2], '.-')
plt.plot(m.energy[1,Ny//2], '.-')

print(m.occupation(m.mu, T))
print(np.min(m.energy[1]) - np.max(m.energy[0]))
plt.axhline(m.mu)
plt.show()

omegas = np.linspace(-0.5,0.5,1201)
transportna, velocity = phi_BB(m.kymesh, m.kxmesh, m.energy, omegas, m.mu, 1)
Gamma = 0.015

plt.plot(m.tok[0,0,0,Ny//2].real)
plt.plot(velocity[0,0,Ny//2].real)
plt.show()
print('here')
transportna_K = phi_K(m.kymesh, m.kxmesh, m.tok, m.energy, omegas, m.mu, Gamma)

plt.plot(omegas, transportna[1] / (2*Gamma))

plt.plot(omegas, transportna_K[1].real * np.pi, '.')
#plt.yscale('log')
plt.show()

np.save('phi_Kubo.npy', transportna_K)
np.save('phi_Boltzmann.npy', transportna)

print(m.mu)
print(np.min(m.energy[1]))