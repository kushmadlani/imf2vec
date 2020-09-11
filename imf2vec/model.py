import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter 
from evaluation import evaluate_model
from sklearn.metrics.pairwise import cosine_similarity

import wandb
import os
# os.environ['WANDB_API_KEY'] = # INSERT weights&biases key here for experiment tracking

class Imf2Vec():
    def __init__(self, counts, Item2Vec, full_items, k, num_factors=40, num_iterations=30, learning_rate={'u':0.1,'i':0.01},
                 alpha_sim_param=0.5, reg_param=0.8, eta=0.9):
        self.counts = counts.toarray()
        self.p = self.counts.copy()
        self.p[np.where(self.p != 0)] = 1.0
        self.counts[np.where(self.counts==0)] = 1

        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.N = self.num_items
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.learning_rate_u = learning_rate['u']
        self.learning_rate_i = learning_rate['i']
        self.eta = eta

        # similarity
        self.alpha = alpha_sim_param
        self.k = k
        self.full_items = full_items
        self.valid_items = Item2Vec.valid_items
        self.S = Item2Vec.S
        self.top_k = Item2Vec.top_k
        self.reverse_lookup = Item2Vec.reverse_lookup
        self.check_valid = Item2Vec.check_valid

    def train_model(self, val_mat, masked_val_mat, eval_n=1):
        self.user_vectors = np.random.normal(scale=1./self.num_factors, 
                                             size=(self.num_users,self.num_factors))
        self.item_vectors = np.random.normal(scale=1./self.num_factors, 
                                             size=(self.num_items,self.num_factors))

        # initialise grad at t-1 for momentum
        grad_u_t = 0
        grad_i_t = 0

        best_mpr = 1 
        
        for _ in range(self.num_iterations):
            # calculate gradients
            grad_u, grad_i = self.gradient_step()

            # momentum adjustment
            adjusted_grad_u = self.eta*grad_u_t + self.learning_rate_u*grad_u
            adjusted_grad_i = self.eta*grad_i_t + self.learning_rate_i*grad_i

            # take gradient step
            self.user_vectors = self.user_vectors - adjusted_grad_u
            self.item_vectors = self.item_vectors - adjusted_grad_i

            # loss function 
            loss = self.loss_fn() 

            grad_u_t = grad_u
            grad_i_t = grad_i

            MAP, rec_at_k, mpr_all, mpr_mask = self.evaluate(val_mat, masked_val_mat, 5)
            wandb.log({
                'Loss': loss,
                'MAP@N': MAP,
                'Recall@N': rec_at_k,
                'MPR (all)': mpr_all,
                'MPR (new)': mpr_mask
            })
            if mpr_mask<best_mpr:
                wandb.run.summary["best_mpr"] = mpr_mask
                best_mpr = mpr_mask

    def gradient_step(self):
        Y = self.item_vectors
        X = self.user_vectors
        r_pred = X@Y.T 

        print('Solving for user vectors...')
        u_indices = np.arange(self.num_users)
        np.random.shuffle(u_indices)
        grad_u = np.zeros_like(X)
        
        for i in u_indices:
            # calculate gradient
            c_true = self.counts[i]
            r_true = self.p[i]
            delta = np.multiply(c_true.T, r_pred[i]-r_true.T)[:,None]

            grad_u[i] = (np.sum(np.multiply(np.repeat(delta, self.num_factors, axis=1),Y),axis=0) + self.reg_param*X[i])/self.N


        print('Solving for item vectors...')
        i_indices = np.arange(self.num_items)
        np.random.shuffle(i_indices)
        grad_i = np.zeros_like(Y)
        
        for i in i_indices:
            # calculate similarity contribution
            if self.check_valid[i]:
                j = self.valid_items.index(self.full_items[i])
                # indices in knn space
                indices = self.top_k[j,0:self.k]
                mapped_indices = [self.reverse_lookup[i] for i in indices]
                S = self.S[j,indices]
                Y_j = self.item_vectors[mapped_indices] + 1e-08
                YTY = cosine_similarity(self.item_vectors[i][None,:], Y_j)
                assert S.shape == np.squeeze(YTY,0).shape 
                scalars = S-np.squeeze(YTY,0)
                sim_i = np.sum(np.multiply(np.repeat(scalars[:,None],self.num_factors,axis=1),Y_j),axis=0)
            else:
                sim_i = 0
            
            # calculate gradient
            c_true = self.counts[:, i]
            r_true = self.p[:, i]
            delta = np.multiply(c_true, r_pred[:,i]-r_true)[:,None]

            grad_i[i] = (np.sum(np.multiply(np.repeat(delta, self.num_factors, axis=1),X),axis=0) + self.reg_param*Y[i] - self.alpha*sim_i)/self.N

        return grad_u, grad_i

    def loss_fn(self):
        r_pred = self.user_vectors@self.item_vectors.T
        MSE = np.sum(np.multiply(self.counts, (r_pred-self.p)**2))/(self.num_items+self.num_users)
        L2 = self.reg_param*(np.sum(self.user_vectors**2)/self.num_users + np.sum(self.item_vectors**2))/self.num_items
        
        item_sim=0
        i_indices = np.arange(self.num_items)
        for i in i_indices:
            # calculate similarity contribution
            if self.check_valid[i]:
                j = self.valid_items.index(self.full_items[i])
                # indices in knn space
                indices = self.top_k[j,0:self.k]
                mapped_indices = [self.reverse_lookup[i] for i in indices]
                S = self.S[j,indices]
                Y_j = self.item_vectors[mapped_indices] + 1e-08
                YTY = cosine_similarity(self.item_vectors[i][None,:], Y_j)
                assert S.shape == np.squeeze(YTY,0).shape 
                scalars = S-np.squeeze(YTY,0)
                item_sim += np.sum(scalars**2)

        return MSE + L2 + item_sim/self.k

    def evaluate(self, mat, masked_mat, top_n):
        R_hat = self.user_vectors@self.item_vectors.T
        MAP, rec_at_k, mpr_all, mpr_mask = evaluate_model(R_hat, mat, masked_mat, top_n)
        return MAP, rec_at_k, mpr_all, mpr_mask





