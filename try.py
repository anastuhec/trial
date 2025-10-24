import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as LA
plt.rcParams.update({'savefig.dpi':300, 'axes.labelweight':'normal', 'axes.linewidth':0.8})
from matplotlib import rc
preamble = r'''
\usepackage{physics} \usepackage{upgreek} \usepackage{mhchem} \usepackage{bm}
'''
plt.rc('text.latex', preamble=preamble)
rc('text', usetex=True)



def rho0(Nk):
    rho = np.zeros((2,2,Nk))
    rho[0,0,:] = 1
    return rho

def hoping_file(t1, t2, eps1, eps2, name='parametri-kinetic.txt'):
    with open(name, 'w') as f:
        # x orb1 orb2 t
        f.write(f'1 1 1 {t1}')
        f.write('\n')
        f.write(f'-1 1 1 {t1}')
        f.write('\n')

        f.write(f'1 2 2 {t2}')
        f.write('\n')
        f.write(f'-1 2 2 {t2}')
        f.write('\n')

        f.write(f'0 1 1 {eps1}')
        f.write('\n')
        f.write(f'0 2 2 {eps2}')

def interaction_file(Vb, Vc, U, name='parametri-interaction.txt'):
    with open(name, 'w') as f:
        # x orb1 orb2 utez amplituda
        f.write(f'0 1 2 0.5 {Vb}')
        f.write('\n')
        f.write(f'0 2 1 0.5 {Vb}')
        f.write('\n')
        f.write(f'1 1 2 0.5 {Vc}')
        f.write('\n')
        f.write(f'-1 2 1 0.5 {Vc}')

def H_hopping(K, file='parametri-kinetic.txt'):
    Nk = len(K)
    hop = np.zeros((2,2,Nk), dtype='complex')
    with open(file, 'r') as f:
        for line in f:
            [x, orb1, orb2, t] = list(map(float, line.split()))
            orb1, orb2 = int(orb1), int(orb2)
            ad = t * np.exp(-1j*K*x)
            hop[orb1-1, orb2-1] += ad
            if orb1 != orb2: hop[orb2-1, orb1-1] += ad.conj()
    return hop

def H_hartree(K, Vb, Vc, rho):
    Nk = len(K)
    hartree = np.zeros((2,2), dtype='complex')
    hartree[0,0] += (Vb + Vc)*np.sum(rho[1,1,:])/Nk
    hartree[1,1] += (Vb + Vc)*np.sum(rho[0,0,:])/Nk
    return hartree

def H_fock(K, Vb, Vc, rho):
    Nk = len(K)
    fock = np.zeros((2,2,Nk), dtype='complex')
    ad = -1/Nk * Vb * np.sum(rho[1,0,:]) - 1/Nk * Vc * np.sum(rho[1,0,:] * np.exp(-1j*K)) * np.exp(-1j*K)
    fock[0,1] += ad
    fock[1,0] =+ ad.conj()
    return fock

def H_perturb(K, eps0):
    Nk = len(K)
    perturb = np.zeros((2,2,Nk), dtype='complex')
    ad = 1j * 2*eps0*np.sin(K)
    perturb[0,1] += ad
    perturb[1,0] += ad.conj()
    return perturb

def diagonalize(rho, H_hop, K, T, mu, phys_parameters, eps0):
    _, _, _, Vb, Vc = phys_parameters
    Nk = len(K)
    energije, vecs = np.zeros((2,Nk)), np.zeros((2,2,Nk), dtype='complex')
    fs = np.zeros((2,2,Nk))
    
    H = H_hop + H_hartree(K, Vb, Vc, rho) + H_fock(K, Vb, Vc, rho) + H_perturb(K, eps0)

    for i in range(Nk):
        en, v = LA.eigh(H[i])
        energije[:,i] = en
        vecs[:,:,i] = v
        if T == 0: np.fill_diagonal(fs[:,:,i], np.array([1,0]))
        else: np.fill_diagonal(fs[:,:,i], 1/(1 + np.exp((en-mu)/T)))
    return energije, vecs, fs

def F(rho, H_hop, K, T, mu, phys_parameters, eps0):
    _, vecs, fs = diagonalize(rho, H_hop, K, T, mu, phys_parameters, eps0)
    rho_new = np.einsum('ijk,jmk,mnk->ink', vecs, fs, np.swapaxes(vecs.conj(),0,1))
    return rho_new, np.max(np.abs(rho - rho_new))

def zasedenost(rho):
    return (np.sum(np.diag(np.einsum('ijk->ij', rho))) / rho.shape[-1]).real

def Rho_next(rho, H_hop, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter, mix=0.5):
    err, N_iters = 1, 0
    while err < epsilon_threshold and N_iters < maxiter:
        if N_iters < N_epsilon: eps = eps0
        else: eps = 0
        rho_new, err = F(rho, H_hop, K, T, mu, phys_parameters, eps)
        rho = mix*rho_new + (1-mix)*rho 
        N_iters += 1
    rho, _ = F(rho, H_hop, K, T, mu, phys_parameters, 0)
    energije, vecs, fs = diagonalize(rho, H_hop, K, T, mu, phys_parameters, 0)
    return rho, err, energije, vecs, fs, zasedenost(rho)

def Phi(K, rho):
    Nk = len(K)
    return [1/Nk * np.sum(rho[0,1,:]*np.exp(-1j*K*delta)).real for delta in [0,1]]

def NewMu(K, rho, H_hop, T, mu, dmu, phys_parameters, eps0,
          epsilon_threshold, N_epsilon, maxiter, n_pass=1e-4, mix2=0.001, mix3=1.5, max_trials=30):
    rho_a, err_a, energije_a, vecs_a, fs_a, n_a = Rho_next(rho, H_hop, K, T, mu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter)
    if np.abs(n_a - 1) < n_pass and err_a < epsilon_threshold:
        return rho_a, err_a, energije_a, vecs_a, fs_a, n_a, mu
    n_b = Rho_next(rho, H_hop, K, T, mu + dmu, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter)[-1]
    chi = (n_b - n_a)/dmu
    if chi != 0: mu = mu - mix2 * (n_a - 1)/np.abs(chi)

    pogoj = False
    koraki = 0
    if np.abs(chi) > 0: faktor = (n_a - 1)/chi * mix3
    else: faktor = 0.1
    if chi >= 0:
        if n_a >= 1:
            sign = -1
        elif n_a < 1: sign = +1
    elif chi < 0:
        if n_a > 1: sign = +1
        elif n_a < 1: sign = -1
        
    sgns = np.ones(2) * np.sign(n_a - 1)
    ns = np.array([0, n_a])
    mus = [0, mu]
    enough = False
    while sgns[0] == sgns[1]:
        if np.abs(n_a - 1) < n_pass and err_a < epsilon_threshold:
            enough = True
            break
        rho_b, err_b, energije_b, vecs_b, fs_b, n_b = Rho_next(rho, H_hop, K, T, mu + faktor*koraki*sign, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter)
        if np.abs(n_b - 1) < n_pass and err_b < epsilon_threshold:
            return rho_b, err_b, energije_b, vecs_b, fs_b, n_b, mu + faktor*koraki*sign
        ns[0] = n_b
        mus[0] = mu + faktor*koraki*sign
        sgns[1] = np.sign(n_b - 1)
        if sgns[0] != sgns[1]: break
        if n_b < 1 and n_b < ns[1]: sign *= -1
        if n_b > 1 and n_b > ns[1]: sign *= -1
        ns = np.roll(ns, 1)
        mus = np.roll(mus, 1)
        sgns[1] = np.sign(n_b - 1)
        koraki += 1
        if np.abs(n_b - 1) < n_pass and err_b < epsilon_threshold:
            enough = True
            mu_mid = mu + faktor*koraki*sign
            break
        
    mus = np.sort(np.array([mu + faktor*koraki*sign, mu + faktor*(koraki-1)*sign]))
    ns = np.sort(np.array(ns))

    trials = 0
    while pogoj == False:
        if enough == True:
            break   
        mu_mid = (mus[0] + mus[1])/2
        n_mid = Rho_next(rho, H_hop, K, T, mu_mid, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter)[-1]
        if n_mid > 1: mus[1] = mu_mid
        elif n_mid < 1: mus[0] = mu_mid
        if np.abs(n_mid - 1) < n_pass: break
        trials += 1 
        if trials > max_trials: break
    rho, err, energije, vecs, fs, n = Rho_next(rho, H_hop, K, T, mu_mid, phys_parameters, eps0,
             epsilon_threshold, N_epsilon, maxiter)
    return rho, err, energije, vecs, fs, n, mu_mid

def fd_1(omega, T):
    return -1/(4*T)/np.cosh(omega/(2*T))**2

def parameters(phys_parameters, Nk, mu=0):
    t, t_, epsilon, Vb, Vc = phys_parameters




t, t_, epsilon, Vb, Vc = phys_parameters