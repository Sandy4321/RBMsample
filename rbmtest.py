#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from RBMlib import RBM
from tqdm import tqdm
import matplotlib.pylab as plt

nvis = 8
nhid = 2
nsmpl = 5000
k = 10
batchsz = 100
epochs = 100

hbias = np.zeros( nhid )
vbias = np.zeros( nvis )
W = np.random.uniform( low=-1, high=+1, size=(nhid, nvis) )

showweight = 2**( np.arange(nvis) )

# Create True distribution from an RBM and sample from it
truerbm = RBM( nvis=nvis, nhid=nhid, hbias=hbias, vbias=vbias, W=W )

v0 = np.random.uniform( size=(nvis, batchsz) )
vsamp0 = truerbm.Sampling( v0, nsmpl )
valhist0 = showweight.dot( vsamp0 )

# Create Test (student) RBM for training
testrbm = RBM( nvis=nvis, nhid=nhid )

# sample from the RBM before training
v1 = np.random.uniform( size=(nvis, batchsz) )
vsamp1 = testrbm.Sampling( v1, nsmpl )
valhist1 = showweight.dot( vsamp1 )

# Train the Test RBM

Fdev = np.zeros( epochs )
mondev = np.zeros( epochs )

print 'Start training'
for ep in tqdm( range( epochs ) ):
    for n in range( nsmpl/batchsz ):
        beg = n * batchsz
        end = beg + batchsz
        v0 = vsamp0[:, beg:end]
        mon, Fnorm = testrbm.UpdateParams( v0, k=k )  # train with CD-k

        mondev[ep] += mon
        Fdev[ep] += Fnorm
print 'End training'

# sample from the trained RBM
v2 = np.random.uniform( size=(nvis, batchsz) )
vsamp2 = testrbm.Sampling( v2, nsmpl )
valhist2 = showweight.dot( vsamp2 )

# Show the result

plt.figure()
nbins = 2**nvis

plt.subplot( 3, 1, 1 )
plt.hist( valhist0, bins=nbins, normed=True )
plt.grid()
plt.title( 'True Distribution from a RBM sampling' )
plt.xlim( (0,nbins) )

plt.subplot( 3, 1, 2 )
plt.hist( valhist1, bins=nbins, normed=True )
plt.grid()
plt.title( 'Test RBM (untrained) distribution' )
plt.xlim( (0,nbins) )

plt.subplot( 3, 1, 3 )
plt.hist( valhist2, bins=nbins, normed=True )
plt.grid()
plt.title( 'Test RBM (trained) distribution' )
plt.xlim( (0,nbins) )
plt.show()

plt.figure()
plt.subplot( 2, 1, 1 )
plt.plot( mondev, 'b-' )
plt.title( 'Monitor value (self Cross Ent.)' )
plt.grid()
plt.subplot( 2, 1, 2 )
plt.plot( Fdev, 'r-' )
plt.title( 'Norm of update vectors' )
plt.grid()
plt.show()


