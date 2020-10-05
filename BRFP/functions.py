import numpy as np
from scipy.optimize import fmin_l_bfgs_b as fmin
import time


def DFT_matrix(N, t = None, freq = None):
    if t is None:
        spacing = np.arange(N)
    else:
        spacing = t

    if freq is None:
        freq = spacing/N

    i, j = np.meshgrid(freq, spacing)
    omega = np.exp( - 1j )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return freq/(2*np.pi), W

def SE_kernel(x1, x2 = None, alpha = 1, sigma = 1):
    if x2 is None:
        x2 = x1
    return sigma**2*np.exp(-alpha*((x1-x2) ** 2))

#negative log-likelihood when using SE kernel
def like_SE(X, y, t):

    #entrenamiento con respecto al logaritmo de los hyperparámetros, para no imponer restricciones de positividad
    sigma_noise, alpha_1, sigma_1 = np.exp(X)
    Gram=gram_matrix(t, alpha=alpha_1,sigma=sigma_1)+sigma_noise**2*np.identity(len(t))
    cGg=np.linalg.cholesky(Gram)
    invGram=np.dot(np.linalg.inv(cGg.T),np.linalg.inv(cGg))
    nll=np.dot(y,np.dot(invGram,y)) + 2*np.sum(np.log(np.diag(cGg)))
    return 0.5*nll+0.5*len(y)*np.log(2*np.pi)

#derivative of negative log-likelihood when using SE kernel
def like_SE_Df(X, y, t, verbose=True):
    sigma_noise, alpha_1, sigma_1 = np.exp(X)
    Gram0=gram_matrix(t, alpha=alpha_1, sigma=sigma_1)
    Gram=Gram0+sigma_noise**2*np.identity(len(t))
    cGg=np.linalg.cholesky(Gram)
    invGram=np.dot(np.linalg.inv(cGg.T),np.linalg.inv(cGg))
    A=np.dot(invGram,y)
    outer_sub=outersum(t,-t)
    D1=2*sigma_noise**2*np.identity(len(t)) #sigma_noise
    D2=-Gram0*outer_sub**2*alpha_1 #alpha_1
    D3=2*Gram0 #sig_1

    B=np.outer(A,A)-invGram
    d1=  -np.trace(np.dot(B,D1))
    d2=  -np.trace(np.dot(B,D2))
    d3=  -np.trace(np.dot(B,D3))
    if verbose:
        print(['Derivatives: ', d1, d2, d3])
    return np.asarray([0.5*d1, 0.5*d2, 0.5*d3])

# "suma" externa
def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def gram_matrix(X, X2 = None, alpha = 1, sigma = 1):
    if X2 is None:
        X2 = X
#    return np.asarray([[SpectralMixture(x1, x2, alpha, sigma) for x2 in X2] for x1 in X]).reshape((X.shape[0],X2.shape[0]))
    return np.asarray([[SE_kernel(x1, x2, alpha, sigma) for x2 in X2] for x1 in X]).reshape((X.shape[0],X2.shape[0]))

class BRFP:

    def __init__(self, t, w=None):
        self.t = t
        self.n = t.shape[0]

        if w is None:
            self.w = np.linspace(0,0.5,self.n)
        else:
            self.w = w*2*np.pi

        self.w, self.W = DFT_matrix(self.n, t, w)

        # Model variables
        self.alpha = 1
        self.sigma = 1
        self.sigma_noise = 1
        self.opt_res = None

        # Observed variables
        self.x_obs = None
        self.t_obs = None
        self.Ht    = None

        self.res = {'x': np.nan,
                    'x_var': np.nan,
                    'X': np.nan,
                    'X_var': np.nan}
    def observation_matrix(self, xi):
        linear_index, other_index = np.asarray(np.where(self.t.reshape(-1,1) == xi.reshape(-1)))
        n_obs = linear_index.shape[0]
        H = np.zeros([self.n, n_obs])
        for i in range(n_obs):
            p = linear_index[i]
            q = other_index[i]
            H[p,q] = 1
        return H

    def spectrum_covariance_without_noise(self, X, X2 = None):
        H_left = self.observation_matrix(X)

        if X2 is None:
            X2 = X
            H_right = H_left
        else:
            H_right = self.observation_matrix(X2)

        # n = self.t.shape[0]
        Sigma = gram_matrix(X, alpha = self.alpha, sigma=self.sigma)

        covRR = np.matmul(self.W.real.T, np.matmul(Sigma, self.W.real))
        covRR = np.matmul(H_left.T, np.matmul(covRR, H_right))

        covII = np.matmul(self.W.imag.T, np.matmul(Sigma, self.W.imag))
        covII = np.matmul(H_left.T, np.matmul(covII, H_right))

        covRI = np.matmul(self.W.real.T, np.matmul(Sigma, self.W.imag))
        covRI = np.matmul(H_left.T, np.matmul(covRI, H_right))

        covIR = np.matmul(self.W.imag.T, np.matmul(Sigma, self.W.real))
        covIR = np.matmul(H_left.T, np.matmul(covIR, H_right))

        return np.block([[covRR, covRI], [covIR, covII]])

    def spectrum_time_covariance(self, X, X2 = None):
        H_left = self.observation_matrix(X)

        if X2 is None:
            X2 = X
            H_right = H_left
        else:
            H_right = self.observation_matrix(X2)

        Sigma = gram_matrix(X,alpha = self.alpha, sigma=self.sigma)

        covTR = np.matmul(self.W.real.T, Sigma)
        covTR = np.matmul(covTR, H_right)

        covTI = np.matmul(self.W.imag.T, Sigma)
        covTI = np.matmul(covTI, H_right)


        return np.block([[covTR], [covTI]])
    def spectrum_time_covariance(self, X, X2 = None):
        H_left = self.observation_matrix(X)

        if X2 is None:
            X2 = X
            H_right = H_left
        else:
            H_right = self.observation_matrix(X2)

        Sigma = gram_matrix(X,alpha = self.alpha, sigma=self.sigma)

        covTR = np.matmul(self.W.real.T, Sigma)
        covTR = np.matmul(covTR, H_right)

        covTI = np.matmul(self.W.imag.T, Sigma)
        covTI = np.matmul(covTI, H_right)


        return np.block([[covTR], [covTI]])
    def train(self, x_obs,t_obs, params0=None):
        self.x_obs = x_obs
        self.t_obs = t_obs
        self.Ht = self.observation_matrix(t_obs)
        #entrenamiento del GP, es decir, encontrar los parámetros del kernel y del ruido
        args=(x_obs,t_obs)
        time_SE=0

        if params0 is None:
            params0=np.asarray([2,  .1,  1])
        X0=np.log(params0);
        print('Condicion inicial optimizador: ',params0)
        time_GP=time.time()
        X_opt, f_GP, data=fmin(like_SE,X0,like_SE_Df, args,disp=1,factr=0.00000001/(2.22E-12),maxiter=50)
        # X_opt, f_GP, data=fmin(like_SM,X0,like_SM_Df, args,disp=1,factr=0.00000001/(2.22E-12),maxiter=50)
        time_GP=time.time()-time_GP
        self.sigma_noise, self.alpha, self.sigma = np.exp(X_opt)
        print('Hiperparametros encontrados: ', self.sigma_noise, self.alpha, self.sigma)
        print('Negative log-likelihood para hiperámetros optimizados: ', f_GP)

        self.opt_res = (X_opt, f_GP, data)
        return True

    def predict(self, verbose=False):
        print('Computing BRFP covariances matrixes...', end='')
        K_X = self.spectrum_covariance_without_noise(self.t, self.t)
        K_x = gram_matrix(self.t, alpha = self.alpha, sigma=self.sigma)
        K_star_spectrum = self.spectrum_time_covariance(self.t, self.t_obs)
        K_star_time = np.matmul(K_x,self.Ht)
        K_obs = gram_matrix(self.t_obs, alpha=self.alpha, sigma=self.sigma) + self.sigma_noise**2*np.eye(self.x_obs.shape[0])
        invKobs = np.linalg.inv(K_obs)
        print('done.')

        #time
        mu_time = np.matmul(K_star_time, np.matmul(invKobs, self.x_obs))
        cov_time = K_x - np.matmul(K_star_time, np.matmul(invKobs, K_star_time.T))
        var_time = np.diag(cov_time)
        #spectrum
        mu_spectrum = np.matmul(K_star_spectrum, np.matmul(invKobs, self.x_obs))
        cov_spectrum = K_X - np.matmul(K_star_spectrum, np.matmul(invKobs, K_star_spectrum.T))
        var_spectrum = np.diag(cov_spectrum)
        var_spectrum_real = var_spectrum[:self.n]
        var_spectrum_imag  = var_spectrum[self.n:]

        # Assignations
        ## time assign
        self.res['x'] = mu_time
        self.res['x_var'] = var_time
        ## spectrum assign
        self.res['X'] = mu_spectrum[:self.n] + 1j*mu_spectrum[self.n:]
        self.res['X_var'] = var_spectrum_real + 1j*var_spectrum_imag
        self.res['w'] = self.w
        return self.res
