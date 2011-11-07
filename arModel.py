import numpy as np
import csv
from utils import *


class AR(object):
    def __init__(self, p):
        """
        p : int
            order of model
        """
        self.p = p
    
    def learn(self, Y):
        """
        learns the parameters of the AR model using least squares
        
        Y : list of list
            list of time series
            
        Here Y is a list of iterables such that each element of Y is 
        a full time series. The parameters will be learned to fit all the
        time series as best they can. 
        
        This is super important: ALL THE TIME SERIES MUST BE SAMPLED AT THE 
        SAME RATE.
        """
        
        # number of experimental records
        M = len(Y)
        
        # data matrix
        phi = np.vstack([toeplitz(Y[i], self.p) for i in range(M)])
                
        # target data
        z = np.hstack([Y[i][self.p+1:] for i in range(M)])
        
        # least-squares parameter estimate
        theta, residue, rank, s = np.linalg.lstsq(phi,z)
        
        # store parameters
        self.theta = theta
        
        # k step-ahead prediction error
        kstep = 1
        if kstep == 1:
            e = z-np.dot(phi,theta) # one-step-ahead
        elif kstep > 1:
            yhat = np.hstack([self.sim(Y[i],kstep) for i in range(M)])
            y = np.hstack([Y[i][:kstep] for i in range(M)])
            e = y-yhat
        
        # error variance
        eVar = np.var(e)
        
        # parameter covariance
        thetaCov = eVar*np.linalg.inv(np.dot(phi.T,phi))
        
        # store parameter covariance
        self.thetaCov = thetaCov
        
        # model in state-space form 
        def make_state_matrix(a):
            I = np.eye(p-1)
            O = np.zeros((p-1,1))
            return np.vstack([np.hstack([O,I]), a[-1::-1]])
        
        self.A = make_state_matrix(theta)
        self.C = [0 for i in range(p-1)] + [1]
        self.G = np.vstack([np.zeros((p-1,1)),1])
    
    def kf_predict_step(self, x, P, Q):
            x = np.dot(self.A, x)
            P = np.dot(self.A, np.dot(P, self.A.T)) + np.dot(self.G, np.dot(Q, self.G.T))
            yhat = np.dot(self.C, x)
            yVar = np.dot(self.C, np.dot(P, self.C))
            yStd = yVar**0.5
            return x, P, yhat, yVar, yStd
    
    def kf_predict(self, y0, T):
        
        # initialise
        x = list(y0[:self.p])
        P = 0.00000001*y0[1]*np.eye(self.p)
        yhat = y0 #np.zeros((T))
        yStd = np.zeros((T))
        
        # loop for each time-step
        for t in range(T-self.p):
            yt = yhat[-1:-(self.p+1):-1]            
            Q = np.dot(yt,np.dot(self.thetaCov,yt))
            x, P, yhatt, yVar, yStdt = self.kf_predict_step(x, P, Q)
            yhat[t+p] = yhatt
            yStd[t+p] = yStdt
            
        return np.array(yhat), np.array(yStd) 
 
    
    def sim(self, y0, T):
        """
        simulates the AR process
        
        y0 : list
            initial condition
        T : int
            number of time points to simulate
        
        This will simulate the AR process T time points into the future. The
        p values of the initial condition list will be chosen as the initial 
        condition, and they will be the first p elements of the returned 
        sequence. 
        """
        if len(y0) < self.p:
            raise ValueError('initial condition must contain at least %s values'%self.p)
        yhat = list(y0[:self.p])
        for t in range(T-self.p):
            yhat.append(np.inner(yhat[-1:-(self.p+1):-1], self.theta))
        
        return yhat
    
    def rsquared(self, y):
        yhat = self.sim(y,len(y))
        r2 = 1 - np.corrcoef(y,yhat)[0,1]**2
        return r2


if __name__ == "__main__":
    
    import pylab as pb
    
    reader = csv.reader(open('sean.csv'))
    Y = []
    for line in reader:
        Y.append([float(y) for y in line if y !=''])
    
    train = Y[0::2]
    test  = Y[1::2]
    
    p = 4
    
    # new style
    model = AR(p)
    model.learn(train)
    pb.figure()
    for i,y in enumerate(test):
        pb.subplot(len(test),1,i+1)
        pb.plot(y,'b-')
        #pb.plot(model.sim(y,len(y)), 'b-')
    
        yhat, yStd = model.kf_predict(y, len(y))
        x = np.linspace(0,len(yhat)-1,len(yhat))
        pb.plot(yhat,'r-')
        pb.fill_between(x,yhat+yStd,yhat-yStd,color='gray',alpha=0.6)
    
    pb.show()
