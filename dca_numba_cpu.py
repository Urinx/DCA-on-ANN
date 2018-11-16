#!/usr/bin/env python
# coding: utf-8

# ====================================
from utils import *
# ------------------------------------
import sys
import itertools
import numpy as np
import numba
from numba import void, uint8, uint32, uint64, int32, int64, float32, float64
from scipy.spatial.distance import pdist, squareform
# ====================================

# @timeit
def DCA(input, output):
    '''
    Direct Coupling Analysis (DCA)
    
    input  - file containing the FASTA alignment
    output - file for dca results. The file is composed by N(N-1)/2 
            (N = length of the sequences) rows and 4 columns: 
            residue i (column 1), residue j (column 2),
            MI(i,j) (Mutual Information between i and j), and 
            DI(i,j) (Direct Information between i and j).
            Note: all insert columns are removed from the alignment.
   
    SOME RELEVANT VARIABLES:
    M       number of sequences in the alignment
    N       number of residues in each sequence (no insert)
    align   M x N matrix containing the alignmnent
    q       equal to 5 (4 ribonucleic acids + 1 gap)
    Meff    effective number of sequences after reweighting
    fij     N x N x q x q matrix containing the reweigthed frequency
            counts
    Pij     N x N x q x q matrix containing the reweighted frequency 
            counts with pseudo counts
    C       N(q-1) x N(q-1) matrix containing the covariance matrix
    '''
    PSEUDO_COUNT_WEIGHT = 0.5
    THETA = 0.2
    STDOUT_WIDTH = 50

    print('=' * STDOUT_WIDTH)
    print('|%s|' % 'Direct Coupling Analysis (DCA)'.center(STDOUT_WIDTH - 2))
    print('=' * STDOUT_WIDTH)

    print('[*] Load mutilseqs alignment data: %s' % input)
    # [M, N, q, align] = load_fasta_data(input)
    align = np.load(input)
    M, N = align.shape
    q = 5

    print('[*] Alignment data matrix:')
    print('[*] \t{M * N, q}: {%d * %d, %d}' % (M, N, q))
    print('[*] \tSpace: %s' % get_size_of(align))
    print('[*] \tTime cost: %s' % time_cost())
    
    print('[*] Calculate Meff, fi, fij:')
    [fi, fij, Meff] = calculate_f(align, THETA)
    print('[*] \tMeff: %f' % Meff)
    print('[*] \tfi: %s, %s' % (
        ' * '.join(map(str, fi.shape)), get_size_of(fi)
    ))
    print('[*] \tfij: %s, %s' % (
        ' * '.join(map(str, fij.shape)), get_size_of(fij)
    ))
    print('[*] \tTime cost: %s' % time_cost())

    print('[*] Calculate Pi, Pij:')
    [Pi, Pij] = calculate_P(fi, fij, PSEUDO_COUNT_WEIGHT)
    print('[*] \tPi: %s, %s' % (
        ' * '.join(map(str, Pi.shape)), get_size_of(Pi)
    ))
    print('[*] \tPij: %s, %s' % (
        ' * '.join(map(str, Pij.shape)), get_size_of(Pij)
    ))
    print('[*] \tTime cost: %s' % time_cost())

    print('[*] Calculate C & invC:')
    C = calculate_C_numba(Pi, Pij)
    invC = np.linalg.inv(C)
    print('[*] \tC: %s, %s' % (
        ' * '.join(map(str, C.shape)), get_size_of(C)
    ))
    print('[*] \tinvC: %s, %s' % (
        ' * '.join(map(str, invC.shape)), get_size_of(invC)
    ))
    print('[*] \tTime cost: %s' % time_cost())

    print('[*] Calculate results & save:')
    print('[*] \tOutput: %s' % output)
    calculate_results(Pij, Pi, fij, fi, invC, output)
    print('[*] \tFile size: %s' % get_file_size_of(output))
    print('[*] \tTime cost: %s' % time_cost())
    print('[*] DCA Done')

def load_fasta_data(rfam_align_file):
    data = fasta_read(rfam_align_file)
    rna_map = {'-': 1, 'A': 2, 'U': 3, 'C': 4, 'G': 5}
    Q = max(rna_map.values())
    N = len(data['seq'][0])
    N_half = N >> 1
    
    # this array take too much spaces
    # need to be optimized!
    numeric_seqs_matrix = np.array([
        [rna_map.get(r, 1) for r in s]
        for s in data['seq'] if s.count('-') < N_half
    ])

    del data
    M = numeric_seqs_matrix.shape[0]

    # reshape
    numeric_seqs_matrix = np.hstack(numeric_seqs_matrix)
    numeric_seqs_matrix.resize((M, N))
    
    return [M, N, Q, numeric_seqs_matrix]

@numba.jit(float64(float64[:], float64[:], float64[:]))
def calculate_tmp_fij_numba(mat_w, mat_a, mat_b):
    t = np.sum(mat_w * mat_a.T * mat_b.T)
    return t

@numba.jit(void(float64[:,:,:,:], uint8, uint8, float64[:], float64[:,:,:]))
def calculate_fij_numba(fij, N, q, W, residue_table):
    for (i,j,A,B) in half_loop_through_fij(N, q):
        t = calculate_tmp_fij_numba(W, residue_table[A][i], residue_table[B][j])
        fij[i,j,A,B] = t
        fij[j,i,B,A] = t

def calculate_f(align, theta):
    (M, N) = align.shape
    q = align.max()

    # W: 1*M
    W = 1 / (
        1 + np.sum(squareform(pdist(align,'hamming')<theta),0)
    )
    Meff = np.sum(W)

    # cache a align residue table: q*N*M
    # use space reduce time
    residue_table = np.zeros((q, N, M))
    for i in range(q):
        residue_table[i] = align.T == i+1

    # fi: N*q
    fi = np.array([np.sum(W * residue_table[i], 1) for i in range(q)]).T / Meff

    # this cost most time!
    fij = np.empty((N, N, q, q))
    calculate_fij_numba(fij, N, q, W, residue_table)
    fij /= Meff
    for i in range(N):
        fij[i,i] = np.eye(q) * fi[i]

    del residue_table
    return [fi, fij, Meff]

def calculate_P(fi, fij, pseudocount_weight):
    (N, q) = fi.shape
    Pi = (1 - pseudocount_weight) * fi + pseudocount_weight/q * np.ones((N,q))
    Pij = (1 - pseudocount_weight) * fij + pseudocount_weight/q/q * np.ones((N,N,q,q))

    for i in range(N):
        Pij[i,i] = 0
        for j in range(q):
            Pij[i,i,j,j] = (1 - pseudocount_weight) * fij[i,i,j,j] + pseudocount_weight / q

    return [Pi, Pij]

@numba.jit(float64[:,:](float64[:,:], float64[:,:,:,:]))
def calculate_C_numba(Pi, Pij):
    (N, q) = Pi.shape
    C = np.zeros(( N*(q-1), N*(q-1) ))
    for (i,j,A,B) in itertools.product(range(N), range(N), range(q-1), range(q-1)):
        C[ (q-1)*i + A, (q-1)*j + B ] = Pij[i,j,A+1,B+1] - Pi[i,A+1] * Pi[j,B+1]
    return C

def calculate_results(Pij, Pi, fij, fi, invC, output):
    (N, q) = Pi.shape
    with open(output, 'w') as f:
        for (i, j) in itertools.combinations(range(N),2):
            W = calculate_W(i,j,q,invC)
            MI = calculate_MI(i,j,fi,fij)
            DI = calculate_DI(i,j,q,W,Pi)
            f.write('{0} {1} {2} {3}\n'.format(i+1, j+1, MI, DI))

def calculate_MI(i, j, fi, fij):
    MI, q = 0, fi.shape[1]
    for (A, B) in itertools.product(range(q), range(q)):
        if fij[i,j,A,B] > 0:
            MI += fij[i,j,A,B] * np.log( fij[i,j,A,B] / fi[i,A] / fi[j,B] )
    return MI

def calculate_W(i, j, q, C):
    W = np.ones((q,q))
    a, b = (q-1)*i, (q-1)*(i+1)
    c, d = (q-1)*j, (q-1)*(j+1)
    W[1:q, 1:q] = np.exp( -C[a:b,c:d] )
    return W

def calculate_DI(i, j, q, W, Pi):
    [mu1, mu2] = compute_mu(i,j,q,W,Pi)
    tiny = 1e-100
    Pdir = W * mu1.T.dot(mu2)
    Pdir /= np.sum(Pdir)
    Pfac = Pi[i,:].reshape(Pi[i,:].shape[0],1) * Pi[j,:]
    DI = np.trace( Pdir.T.dot( np.log( (Pdir+tiny) / (Pfac+tiny) ) ) )
    return DI

def compute_mu(i, j, q, W, Pi):
    epsilon, diff = 1e-4, 1.0
    mu1, mu2 = np.ones((1,q))/q, np.ones((1,q))/q
    pi, pj = Pi[i, :], Pi[j, :]

    while diff > epsilon:
        scra1 = mu2.dot(W.T)
        scra2 = mu1.dot(W)
        new1 = pi / scra1
        new1 /= np.sum(new1)
        new2 = pj / scra2
        new2 /= np.sum(new2)
        diff = max( np.abs(new1-mu1).max(), np.abs(new2-mu2).max() )
        mu1, mu2 = new1, new2

    return [mu1, mu2]

if __name__ == '__main__':
    DCA('node_dca.npy', 'node_dca.di')
