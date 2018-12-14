# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time

# import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
# from tensorflow.python.keras.optimizers import SGD


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

import argparse
import random
import numpy as np
import time
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import cupy as cp
import multiprocessing
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


##Word embedding Dimension Reduction
def compressing(x_train,original_dim,target_dim):

    X_train = np.asarray(x_train)
    pca_embeddings = {}

    # PCA to get Top Components
    pca =  PCA(n_components = original_dim)
    X_train = X_train - np.mean(X_train)
    #print("Working: ",X_train.dtype)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_

    z = []

    # Removing Projections on Top Components
    for i in range(len(X_train)):
        x = X_train[i]
        for u in U1[0:7]:        
            x = x - np.dot(u.transpose(),x) * u 
        z.append(x)

    z = np.asarray(z)
    #print("Z: ",z.dtype)
    # PCA Dim Reduction
    pca =  PCA(n_components = target_dim)
    X_train = z - np.mean(z)
    #print("Not Working: ",X_train.dtype)
    X_new_final = pca.fit_transform(X_train)


    # PCA to do Post-Processing Again
    pca =  PCA(n_components = target_dim)
    X_new = X_new_final - np.mean(X_new_final)
    X_new = pca.fit_transform(X_new)
    Ufit = pca.components_

    X_new_final = X_new_final - np.mean(X_new_final)

    final_pca_embeddings = []

    for i in range(len(X_new_final)):
        final_pca_embeddings.append(X_new_final[i])
        for u in Ufit[0:7]:
            final_pca_embeddings[i] = final_pca_embeddings[i] - np.dot(u.transpose(),final_pca_embeddings[i]) * u 
    return np.asarray(final_pca_embeddings)
            
def Adam(g_t,w,t,lr):
    alpha = lr
    beta_1 = 0.9
    beta_2 = 0.999                      
    epsilon = 1e-8
    m_t = 0 
    v_t = 0 
    m_t = beta_1*m_t + (1-beta_1)*g_t   #updates the moving averages of the gradient
    v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) #updates the moving averages of the squared gradient
    m_cap = m_t/(1-(beta_1**t))     #calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))     #calculates the bias-corrected estimates                            
    w = w - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon) #updates the parameters
    return w   
def kernel(X,Y,sigma_list):
    final = 0
    for sigma in sigma_list: 
        coefficient = 1/(2* sigma * sigma *  math.pi)
        power = -(np.matmul(X,X) + np.matmul(Y,Y))/(2 * sigma* sigma)
        #print("Check pow:",  power)
        final += coefficient * (math.e)**power
    #print("Kernel: ",final)
    return final


def kermat(W, X, Y, i, batch_size):
    first_term = 0
    second_term = 0
    third_term = 0
    #print("reach here: ")
    #print(xp.matmul(W,X[i]))
    sigma = [1,5,10,40,80]
    for j in range(batch_size):
        #print(X)
        #xp.matmul(W,X[:,i])
        first_term  += kernel(np.matmul(W,X[:,i]),np.matmul(W,X[:,j]),sigma)
        second_term += kernel(np.matmul(W,X[:,i]),Y[:,j],sigma)
        third_term += kernel(Y[:,i],Y[:,j],sigma)
    #print(first_term)
    return first_term,second_term,third_term

#The MMD part that will be used as objective(loss) during training
#Assume both W,X,Y are numpy array
def compute_MMD(batch_size,W,X,Y):
    norm = 1/(batch_size* batch_size)
    first_term = np.zeros(batch_size)
    second_term = np.zeros(batch_size)
    third_term = np.zeros(batch_size)

    p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    #for i in range(batch_size):
        #kermat(W,X,Y,i,first_term,second_term,third_term,batch_size)
    results = [p.apply_async(kermat, args=(W, X, Y, i,batch_size)) for i in range(batch_size)]
    p.close()
    p.join()

    first = 0
    second = 0
    third = 0
    for p in results:
        f, s, t = p.get()
        first += f
        second += s 
        third += t
    objective = norm*(first - 2* second + third)
    return objective

def kernel_grad(X,Y,sigma_list):
    final = 0
    for sigma in sigma_list: 
        coefficient = 1/(2* sigma * sigma *  math.pi)*(-X/(sigma*sigma))
        power = -(np.matmul(X,X) + np.matmul(Y,Y))/(2 * sigma* sigma)
        final += coefficient * (math.e)**power

    return final

def kermat_grad(W, X, Y, i,batch_size):
    first_term = 0
    second_term = 0
    third_term = 0
    sigma = [1,5,10,40,80]
    for j in range(batch_size):
        first_term  += kernel_grad(np.matmul(W,X[:,i]),np.matmul(W,X[:,j]),sigma)
        second_term += kernel_grad(np.matmul(W,X[:,i]),Y[:,j],sigma)
        third_term += kernel_grad(np.matmul(W,X[:,j]),np.matmul(W,X[:,i]),sigma)
    return first_term, second_term , third_term


def grad_MMD(batch_size,W,X,Y,emb_dim):
    norm = 1/(batch_size* batch_size)
    grad_w = np.zeros((emb_dim,batch_size))
    p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
    
    results = [p.apply_async(kermat_grad,args=(W,X,Y,i,batch_size)) for i in range(batch_size)] 
        #print(i, " of ", batch_size)
    p.close()
    p.join()
    for i in range(len(results)):
        p = results[i]
        f,s, t= p.get()
        grad_w[:,i] = f + s +t
        #print("Check ",f)
    #print("Check Grad_W: ",grad_w)
    return grad_w

def abs_sum(X):
    final = 0
    for x in X:
        final += abs(x)
    return final


#Need to implement derivative of MMD respect to WX
def compute_grad_w(MMD,W,X,Y,batch_size,emb_dim):
    temp_grad = grad_MMD(batch_size,W,X,Y,emb_dim)
    #print("First ",temp_grad)
    temp_grad = temp_grad/(2*math.sqrt(MMD))
    return temp_grad



def update_w(grad_w,emb_dim,W_X,index,batch_size,lr):
    for i in range(emb_dim):
        for j in range(batch_size):
            W_X[i,j] = Adam(grad_w[i,j],W_X[i,j],index,lr)
    return W_X

def Orthoganal_W(W,beta):
    W = (1+ beta) * W - beta*(np.matmul(np.matmul(W,W), W))
    return W
def assign(sim):
    print("Reach Assign!")
    count = []
    sim_cp = np.argsort(sim,axis=1)
    for i in range(len(sim_cp)):
        #print("Current assign: ",i)
        j = len(sim_cp)-1
        pick = sim_cp[i,j]
        while pick in count:
            j = j - 1
            #print("Current J: ",j)
            pick = sim_cp[i,j]
        count.append(pick)
    print("Finish Assign")
    return np.asarray(count)


def data_init(X,Y,knn= 0):
    sim_size = min(X.shape[0], Y.shape[0])
    u, s, vt = np.linalg.svd(X[:sim_size], full_matrices=False)
    xsim = (u*s).dot(u.T)
    u, s, vt = np.linalg.svd(Y[:sim_size], full_matrices=False)
    ysim = (u*s).dot(u.T)
    del u, s, vt
    xsim.sort(axis=1)
    ysim.sort(axis=1)
    embeddings.normalize(xsim, 'unit')
    embeddings.normalize(ysim, 'unit')
    sim = xsim.dot(ysim.T)
    if knn > 0:
        knn_sim_fwd = topk_mean(sim, k=knn)
        knn_sim_bwd = topk_mean(sim.T, k=knn)
        sim -= knn_sim_fwd[:, np.newaxis]/2 + knn_sim_bwd/2

    #Forward translation direction
    src_indices = np.arange(sim_size)

    trg_indices = assign(sim)
    X_sort = X[src_indices]
    #print("Check src_indices: ",sim)

    #print("Check trg_indices: ",trg_indices)
    Y_sort = Y[trg_indices]
    print("Finish data_init!")
    #print("Check X: ",X_sort)
    #print("Check Y : ", Y_sort)
    #exit()
    return X_sort.T,Y_sort.T


def train_W(X_ori,Y_ori,base_lr,base_num_batch,emb_dim,precision,max_iter):
    if_end = True
    attempt = 0
    beta = 0.01
    while if_end:
        #X_total = xp.copy(X_ori)
        #Y_total = xp.copy(Y_ori)
        X_total = X_ori
        Y_total = Y_ori
        if_max = False
        attempt += 1
        lr = base_lr
        num_batch = base_num_batch 
        batch_size = int(len(X_total[0])/num_batch)
        total_MMD = 1
        print(X_total.shape)
        #exit()
        if attempt == 1:
            X_test, Y_test = data_init(X_total.T[:10000],Y_total.T[:10000],10)
        else:
            np.random.shuffle(X_total)


        u, s, vt = np.linalg.svd(Y_test.dot(X_test.T))
        W_init = vt.T.dot(u.T)


        index = 1
        W = W_init
        while total_MMD > precision:
            total_MMD = 0
            for i in range(num_batch):
                X =  X_total[:,i*batch_size:(i+1)*batch_size,]
                Y = Y_total[:,i*batch_size:(i+1)*batch_size,]
                W = Orthoganal_W(W,beta)
                MMD = compute_MMD(batch_size,W,X,Y)
                total_MMD += MMD
                grad_w = compute_grad_w(MMD,W,X,Y,batch_size,emb_dim)
                W_X = np.matmul(W,X)
                W_X = update_w(grad_w,emb_dim,W_X,index,batch_size,lr)
                W = np.matmul(W_X, np.linalg.pinv(X))
                W = Orthoganal_W(W,beta)
                #print("Current X: ", X)
                #print("Current Y: ", Y)
                print("CUrrent W: ",W)
                print("Batch Loss: ",MMD)
            total_MMD = total_MMD / num_batch
            print("MMD Loss: ",total_MMD)
            print("W: ",W)
            if math.isnan(W[0,0]):
                print("NOT CONVGERGE!!!!!")
                break
            if index > max_iter:
                ("Reach Max Iteration: ",index)
                if_max = True
                break
            #print("W: ")
            #print(W)
            #print("Current Result: ")
            #print(xp.matmul(W,X_total))
            index += 1
            print("Iteration:", index)
        if not if_max:
            print("Found solution!")
            print("W: ",W)
            return(W)
        exit()


#The refienment step to improve performance after training
def refinement_step():



    return 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()

    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.aaai2018:
        parser.set_defaults(init_dictionary=args.aaai2018, normalize=['unit', 'center'], whiten=True, trg_reweight=1, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.acl2017:
        parser.set_defaults(init_numerals=True, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.acl2017_seed:
        parser.set_defaults(init_dictionary=args.acl2017_seed, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.emnlp2016:
        parser.set_defaults(init_dictionary=args.emnlp2016, orthogonal=True, normalize=['unit', 'center'], batch_size=1000)
    args = parser.parse_args()

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'






    emb_dim = 100

    
    print("reading embeddings and compressing...")
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)

    x = compressing(x,len(x[0]),emb_dim)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)


    z = compressing(z,len(z[0]),emb_dim)
    print("finished reading embeddings and compressing")

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)
    print("Reach Here: ")
    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    print("normalizing embeddings....")
    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    print("normalized embeddings")

    print("building seed dictionary....")

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        a = 1
        '''
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], args.unsupervised_vocab)
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, args.normalize)
        embeddings.normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)
        if args.csls_neighborhood > 0:
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
        '''
    elif args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:  
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    print("seed dictionary built")
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    




    lr = 0.0001
    num_batch = 2000
    precision = 0.00000001
    max_iter = 1

    W = train_W(xp.asnumpy(x.T),xp.asnumpy(z.T),lr,num_batch,emb_dim,precision,max_iter)


    W = xp.asnumpy(W)
    np.save("W_En_FER.txt",W)
    #W = np.load("W_En_IT.txt.npy")
    wx = np.matmul(W,xp.asnumpy(x).T)


    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words,wx.T , srcfile)
    embeddings.write(trg_words, z, trgfile)
    srcfile.close()
    trgfile.close()


if __name__ == '__main__':
    main()
