# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pylab as plt

class RBM( object ):
    """RBM class with full scrach. Maybe, it's slow"""
    """W: parameter (nhidden, nvisual)"""
    """hbias: bias for hidden units"""
    """vbias: bias for visual units"""

    def __init__( self, nvis=128, nhid=256, W=None, hbias=None, vbias=None ):
        self.nvis = nvis
        self.vbias = vbias
        self.nhid = nhid
        self.hbias = hbias
        self.W = W

        if self.W is None:
            lowb = -4 * np.sqrt( 6. / (nvis+nhid) )
            highb = 4 * np.sqrt( 6. / (nvis+nhid) )
            sz = ( nhid, nvis )
            self.W = np.random.uniform( low=lowb, high=highb, size=sz )

        if self.hbias is None:
            self.hbias = np.zeros( nhid )

        if self.vbias is None:
            self.vbias = np.zeros( nvis )

    def Activate( self, x ):
        return( 1./(1.+np.exp(-x)) )

    def Energy( self, hsamp, vsamp ):
        """Calculate Energy H(v,h) from both visible and hidden samples"""
        """vsamp: visible samples of shape (nvis, p) matrix for p visible samples"""
        """hsamp: hidden samples of shape (nhid, p) matrix for p corresponding samples"""
        ah = self.hbias.dot( hsamp )
        bv = self.vbias.dot( vsamp )
        hWv = np.sum( self.W.dot( vsamp ) * hsamp, axis=0 )
        return hWv + ah + bv

    def propup( self, vsamp ):
        """Calculate hidden prob. from visible sample"""
        insum = self.W.dot(vsamp) + self.hbias.reshape( (self.nhid,1) )
        return( self.Activate( insum ) )

    def sample_hid( self, v0samp ):
        probh = self.propup( v0samp )
        h1samp = np.random.binomial( n=1, p=probh )
        return( [probh, h1samp] )

    def propdw( self, hsamp ):
        """Calculate visible prob. from hidden sample"""
        insum = self.W.T.dot(hsamp) + self.vbias.reshape( (self.nvis,1) )
        return( self.Activate( insum ) )

    def sample_vis( self, h0samp ):
        probv = self.propdw( h0samp )
        v1samp = np.random.binomial( n=1, p=probv )
        return( [probv, v1samp] )

    def Gibbs_vhv( self, v0samp ):
        """Oneshot Gibbs sampling from visble sample v0samp"""
        probh, h1samp = self.sample_hid( v0samp )
        probv, v1samp = self.sample_vis( h1samp )
        return( [probh, h1samp, probv, v1samp] )

    def Gibbs_hvh( self, h0samp ):
        """Oneshot Gibbs sampling from hidden sample h0samp"""
        probv, v1samp = self.sample_vis( h0samp )
        probh, h1samp = self.sample_hid( v1samp )
        return( [probv, v1samp, probh, h1samp] )


    def GetGrad( self, v0, k=1, persistent=None ):
        """Calculate Update components of W"""
        N = v0.shape[1]  # column num means sample number
        probh, h1samp = self.sample_hid( v0 )
        posW = probh.dot( v0.T )

        # MCMC loop
        if persistent is None:
            hsamp = h1samp
        else:
            hsamp = persistent

        for kk in range(k):
            [pv, vsamp, ph, hsamp] = self.Gibbs_hvh( hsamp )

        negW = ph.dot( vsamp.T ) 
        # negW = hvmean / k

        dW = (posW - negW)/N
        dh = probh.mean(axis=1) - ph.mean(axis=1)
        dv = v0.mean(axis=1) - vsamp.mean(axis=1)

        monitor = self.CrossEntropy( v0, pv )

        return( [dW, dh, dv, monitor] )
 

    def UpdateParams( self, v0samp, k=1, lr=0.01, persistent=None ):
        """Update params( W, hbias, vbias )"""
        dW, dh, dv, mon = self.GetGrad( v0samp, k=k, persistent=persistent )

        self.W = self.W + lr * dW
        self.hbias = self.hbias + lr * dh
        self.vbias = self.vbias + lr * dv
        Fnorm = np.sum(dW**2) + np.sum(dh**2) + np.sum(dv**2)


#        print "F_norm: ", Fnorm, "CrossEntropy: ", mon

        return( (mon, Fnorm) ) # reurn values are mon:(Cross entropy) and Total Squared norm


    def CrossEntropy( self, v0, pv ):
        """Evaluation of Cross Entropy between v0 and estimated pv"""
        evals = np.sum( v0 * np.log( pv ) + (1-v0) * np.log( 1-pv ), axis=1 )
        return( np.mean( evals ) )


    def MeanCoincidence( self, v0, k=1 ):
        """Evaluation of mean coincidence between generated result"""
        v1 = v0
        for kk in range(k):
            [ph, h1, pv, v1] = self.Gibbs_vhv( v1 )
        evals = np.mean(v1 - v0)**2 
        return( evals )

    def Sampling( self, vinit, nsample, k=10, burnin=100, tqdm=None ):
        """
        Sampling over given RBM:
        `vinit`: initial value, which shape must be (nvis, batchsize)
        `nsample`: number of samples
        `k`: iterations for contrastive divergence (CD-k)
        `burnin`: the number of burnin(warmup) before sampling
        `tqdm`: for display the progress bar
        """
        # Burn in RBM
        nv, batchsz = vinit.shape
        if nv != self.nvis:
            raise ValueError("vinit size is wrong")
        v0 = np.array( vinit )
        vsamp = np.zeros( (self.nvis, nsample) )
        for n in range(burnin):
            out = self.Gibbs_vhv( v0 )
            v0 = out[-1]

        if tqdm is not None:
            prog = tqdm( range(nsample/batchsz) )
        else:
            prog = range( nsample/batchsz )
        # print "Sampling from RBM"
        for n in prog:
            beg = n * batchsz
            end = beg + batchsz
            for kk in range(k):
                out = self.Gibbs_vhv( v0 )
                v0 = out[-1]
            vsamp[:, beg:end] = v0
        return vsamp

    def SaveCoef( self ):
        pass



if __name__ == '__main__':

    nvis = 8
    nhid = 2
    nsmpl = 5000
    k = 10
    batchsz = 10
    burnin = 100
    epochs = 100
    #    hbias = np.random.uniform( low=-1, high=+1, size=nhid )
    #    vbias = np.random.uniform( low=-1, high=+1, size=nvis )
    hbias = np.zeros( nhid )
    vbias = np.zeros( nvis )
    W = np.random.uniform( low=-1, high=+1, size=(nhid, nvis) )

    truerbm = RBM( nvis=nvis, nhid=nhid, hbias=hbias, vbias=vbias, W=W )

    weight = 2**(np.arange( nvis ))

    v0 = np.random.uniform( size=(nvis, batchsz) )
    vsamp0 = truerbm.Sampling( v0, nsmpl )
    VV0 = np.cov( vsamp0 )

    valhist0 = weight.dot( vsamp0 )


    testrbm = RBM( nvis=nvis, nhid=nhid )

    v1 = np.random.uniform( size=(nvis, batchsz) )
    vsamp1 = testrbm.Sampling( v1, nsmpl )
    VV1 = np.cov( vsamp1 )
    valhist1 = weight.dot( vsamp1 )

    # Train the RBM

    Fdev = np.zeros( epochs )
    mondev = np.zeros( epochs )
    for ep in range( epochs ):
        for n in range( nsmpl/batchsz ):
            beg = n * batchsz
            end = beg + batchsz
            v0 = vsamp0[:, beg:end]
            mon, Fnorm = testrbm.UpdateParams( v0, k=k )

            mondev[ep] = mondev[ep] + mon
            Fdev[ep] = Fdev[ep] + Fnorm

    v2 = np.random.uniform( size=(nvis, batchsz) )
    vsamp2 = testrbm.Sampling( v2, nsmpl )
    VV2 = np.cov( vsamp2 )
    valhist2 = weight.dot( vsamp2 )


    plt.figure()
    plt.subplot( 3, 1, 1 )
    nbins = 2**nvis
    plt.hist( valhist0, bins=nbins )
    plt.grid()
    plt.title( 'True Distribution from RBM samping' )

    plt.subplot( 3, 1, 2 )
    plt.hist( valhist1, bins=nbins )
    plt.grid()
    plt.title( 'Test RBM (untrained) distribution' )

    plt.subplot( 3, 1, 3 )
    plt.hist( valhist2, bins=nbins )
    plt.grid()
    plt.title( 'Test RBM(trained) distribution' )

    plt.show()

    plt.figure()
    plt.subplot( 2, 1, 1 )
    plt.plot( mondev, 'b-' )
    plt.title( 'Monitor value (self Cross Ent.)' )
    plt.grid()
    plt.subplot( 2, 1, 2 )
    plt.plot( Fdev, 'r-' )
    plt.title( 'Norm of update quantities' )
    plt.grid()
    plt.show()
