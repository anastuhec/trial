import numpy as np
import os 
from takarada_funkcije import *

parameters1 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-4,
'N_epsilon' : 5,
'maxiter' : 50,
'eps_last' : 1e-4,
'dmu' : 0.1,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 5,
'eps0' : 0.03,
}

parameters2 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-4,
'N_epsilon' : 5,
'maxiter' : 20,
'eps_last' : 1e-4,
'dmu' : 0.1,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 5,
'eps0' : 0.03,
}

Nk = 3000


'''
 non-interacting case
    verifying convergences
    1) Kubo --> Boltzmann in the limit Gamma --> 0
    2) phi_K --> omega * phi in the limit Gamma --> 0
a = 1
b = 0
t = 3
t_ = 1
t12 = 0.
epsilon = 4.1
Vb = 0
Vc = 0

phys_parameters = {'b': b,
                   't': t,
                    't_': t_,
                    't12': t12,
                    'epsilon': epsilon,
                    'Vb': Vb,
                    'Vc': Vc}


print('starting')

scale = 1.02
beta0 = 230
betas = np.hstack([np.linspace(30,220,80)[::-1], np.linspace(15,30,80)[::-1]])
Ts = 1/np.array(betas)
ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]

Gamma = 0.01
eps = 1e-9
Nomega = 150

collect = np.zeros((2, 3, len(Ts)))

mu0 = 0.
include_hartree = False
mu0 = find_GS_mu(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, beta0=50, beta1=40)
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()

os.chdir('/Users/ana/Desktop/takarada/podatki')
np.save('disperzija.npy', np.vstack([m.K, m.energije[0] - m.mu, m.energije[1] - m.mu]))

omegas = np.linspace(-0.7,0.7,100)
Gammas = np.linspace(0.005,0.05,10)

phis = np.zeros((len(Gammas), len(omegas)))
phisK = np.copy(phis)
for i, Gamma in enumerate(Gammas):
    phis[i] = phi_Kubo(m.K, m.vecs, m.energije, j_tok(m.K, phys_parameters, m.mu), m.mu, omegas, Gamma)
    phisK[i] = phi_K(m.K, m.vecs, m.energije, j_tok(m.K, phys_parameters, m.mu), j_1(m.K, phys_parameters, m.mu), m.mu, omegas, Gamma)
os.chdir('/Users/ana/Desktop/takarada/podatki')
np.save('phis_Kubo.npy', phis)
np.save('phis_K.npy', phisK)
print('calculated Kubo')

faktors = np.array([0.1,0.2,0.4,0.6])
phis = np.zeros((2, len(faktors), len(omegas)))
for i, faktor in enumerate(faktors):
    phis[0,i] = phi_Boltzmann(m.K, m.rho, phys_parameters, m.energije, m.mu, omegas, shape='Gaussian', faktor=faktor)
    phis[1,i] = phi_Boltzmann(m.K, m.rho, phys_parameters, m.energije, m.mu, omegas, shape='Lorentzian', faktor=faktor)
np.save('phis_Boltzmann.npy', phis)
print('done')
'''
parameters1 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-4,
'N_epsilon' : 5,
'maxiter' : 50,
'eps_last' : 1e-4,
'dmu' : 0.1,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 5,
'eps0' : 0.03,
}

parameters2 = {'n_pass' : 1e-3,
'epsilon_threshold' : 1e-4,
'N_epsilon' : 5,
'maxiter' : 20,
'eps_last' : 1e-4,
'dmu' : 0.1,
'mix2' : 0.001,
'mix3' : 1.5,
'faktor1' : 0.1,
'max_trials' : 5,
'eps0' : 0.03,
}

Nk = 300

# example (c)
a = 1.
b = 0.
t = 5.
t_ = 1.
t12 = 0.
epsilon = 5.75
Vb = 2.
Vc = 2.

phys_parameters = {'b': b,
                   't': t,
                    't_': t_,
                    't12': t12,
                    'epsilon': epsilon,
                    'Vb': Vb,
                    'Vc': Vc}

phys_parameters = [b, t, t_, t12, epsilon, Vb, Vc]

scale = 1.02
beta0 = 110
betas = [100 - 5*i for i in range(20)]
Ts = 1/np.array(betas)
ends = [int(np.emath.logn(scale, beta0/beta)) for beta in betas]

Gamma = 0.01
eps = 1e-9
Nomega = 150

mu0 = 0.
include_hartree = False
mu0 = find_GS_mu(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree, beta0=50, beta1=40)
m = model(Nk, mu0, phys_parameters, parameters1, parameters2, include_hartree)
m.GS()

os.chdir('/Users/ana/Desktop/takarada/podatki')
np.save('disperzija_EI.npy', np.vstack([m.K, m.energije[0] - m.mu, m.energije[1] - m.mu]))

omegas = np.linspace(-0.5,0.5,100)
Gammas = np.linspace(0.005,0.05,10)

phis = np.zeros((len(Gammas), len(omegas)), dtype=np.complex128)
phisK = np.copy(phis)
phisQ = np.zeros((len(Gammas), 4, len(omegas)), dtype=np.complex128)
tok = j_tok(m.K, phys_parameters, m.mu)
tokK = j_1(m.K, phys_parameters, m.mu)
for i, Gamma in enumerate(Gammas):
    print(i)
    phis[i] = phi_Kubo(m.K, m.vecs, m.energije, tok, m.mu, omegas, Gamma)
    phisK[i] = phi_K(m.K, m.vecs, m.energije, tok, tokK, m.mu, omegas, Gamma)

    phi3, phi4, phi5, phi6 = phi_Q(m.K, m.rho, m.vecs, m.energije, phys_parameters, tok, m.mu, omegas, Gamma)

    phisQ[i,0] = phi3
    phisQ[i,1] = phi4
    phisQ[i,2] = phi5
    phisQ[i,3] = phi6
 

os.chdir('/Users/ana/Desktop/takarada/podatki')
np.save('phis_Kubo_EI.npy', phis)
np.save('phis_K_EI.npy', phisK)
print('calculated Kubo')

faktors = np.array([0.1,0.2,0.4,0.6])
phis = np.zeros((2, len(faktors), len(omegas)))
for i, faktor in enumerate(faktors):
    phis[0,i] = phi_Boltzmann(m.K, m.rho, phys_parameters, m.energije, m.mu, omegas, shape='Gaussian', faktor=faktor)
    phis[1,i] = phi_Boltzmann(m.K, m.rho, phys_parameters, m.energije, m.mu, omegas, shape='Lorentzian', faktor=faktor)
np.save('phis_Boltzmann_EI.npy', phis)
print('done')