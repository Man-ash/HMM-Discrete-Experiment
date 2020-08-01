#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


A = np.array([[0.1,0.5,0.4],
              [0.6,0.3,0.1],
              [0.3,0.6,0.1]])

B = np.array([[0.3,0.4,0.3],
              [0.4,0.2,0.4],
              [0.6,0.2,0.2]])

p = np.array([0.3,0.4,0.3])

O = [1,2,0,0,1,1,2,0,1] 


# In[3]:


def forward_var(A,B,p,O):
    T = len(O) 
    N = A.shape[0]
    alpha = np.zeros((T,N))
    for t in range(alpha.shape[0]):
        for i in range(alpha.shape[1]):
            if t==0:
                alpha[t,i] = B[i,O[0]]*p[i]
            else:
                sum1=0
                for k in range(A.shape[0]):
                    sum1 += alpha[t-1,k]*A[k,i]    
                alpha[t,i] =  sum1*B[i,O[t]]
    return(alpha)


# In[4]:


alpha = forward_var(A,B,p,O)
print(alpha)


# In[5]:


def backward_var(A,B,p,O):
    T = len(O)
    N = A.shape[0]
    beta = np.zeros((T,N))
    for t in range(beta.shape[0]-1,-1,-1):
        for i in range(beta.shape[1]):
            if t == beta.shape[0]-1:
                beta[t,i] = 1
            else:
                sum1=0
                for j in range(A.shape[0]):
                    sum1 += A[i,j]*B[j,O[t+1]]*beta[t+1,j]    
                beta[t,i] =  sum1
    return(beta)


# In[6]:


beta = backward_var(A,B,p,O)
print(beta)


# In[7]:


def forward_eval(alpha):
    prob = np.sum(alpha[-1,:])
    return(prob)


# In[8]:


forward_prob = forward_eval(alpha)
print(forward_prob)


# In[9]:


def backward_eval(B,beta,O):
    vec1 = np.ravel(B[:,O[0]])
    vec2 = np.ravel(beta[0,:])
    prob = np.dot(vec1,vec2)
    return(prob)


# In[10]:


backward_prob = backward_eval(B,beta,O)
print(backward_prob)


# In[11]:


def gammaEval(alpha,beta):
    gamma = np.multiply(alpha,beta) ## Element-wise product
    statewise_sum = np.sum(gamma,axis=1)
    statewise_sum = statewise_sum.reshape(len(statewise_sum),1)
    gamma = gamma/statewise_sum
    return(gamma)


# In[12]:


gamma = gammaEval(alpha,beta)
print(gamma)


# In[13]:


def Viterbi(A,B,p,O):
    T = len(O)
    N = A.shape[0]
    delta = np.zeros((T,N))
    psi = np.zeros((T,N))
    optim_state = []
    for t in range(delta.shape[0]):
        for i in range(delta.shape[1]):
            if t==0:
                delta[t,i] = B[i,O[0]]*p[i]
                psi[t,i] = 0
            else:
                ls = []
                for k in range(N):
                    val = delta[t-1,k]*A[k,i]*B[i,O[t]]
                    ls.append(val)
                arr = np.array(ls)
                max_val = np.max(arr)
                max_val_idx = np.argmax(arr)
                delta[t,i] = max_val
                psi[t,i] = max_val_idx
    final_state_val = np.ravel(delta[-1,:])
    idx = np.argmax(final_state_val)
    optim_state.append(idx)
    for t in range(T-2,-1,-1):
        idx = psi[t,idx]
        idx = int(idx)
        optim_state.append(idx)
    optim_state = optim_state[::-1] 
    return(optim_state)


# In[14]:


optim_state = Viterbi(A,B,p,O)
print(optim_state)


# In[15]:


def joint_state_measure(A,B,alpha,beta,O):
    N = A.shape[0]
    T = len(O)
    eta = np.zeros((T-1,N,N))
    
    for t in range(eta.shape[0]):
        sum1 = 0
        for i in range(eta.shape[1]):
            for j in range(eta.shape[2]):
                eta[t,i,j] = alpha[t,i]*A[i,j]*B[j,O[t+1]]*beta[t+1,j]
                sum1 += eta[t,i,j]
        eta[t,:,:] = eta[t,:,:]/sum1
    return(eta)


# In[16]:


eta = joint_state_measure(A,B,alpha,beta,O)
print(eta)


# In[17]:


def Baum_Welch(eta,gamma,A,B,p,O):
    N = A.shape[0]
    M = B.shape[1]
    mod_A = np.zeros((N,N))
    mod_B = np.zeros((N,M))
    mod_p = np.zeros(N)
    
    for i in range(len(mod_p)):
        mod_p[i] = gamma[0,i]
    
    for i in range(mod_A.shape[0]):
        for j in range(mod_A.shape[1]):
            num_sum = 0
            denom_sum = 0
            for t in range(eta.shape[0]):
                num_sum += eta[t,i,j]
                denom_sum += gamma[t,i]
            a = num_sum/denom_sum
            mod_A[i,j] = a 
    
    for j in range(B.shape[0]):
        for k in range(B.shape[1]):
            num_sum = 0
            denom_sum = 0
            for t in range(gamma.shape[0]):
                if O[t] == k:
                    num_sum += gamma[t,j]
                denom_sum += gamma[t,j]
            b = num_sum/denom_sum
            mod_B[j,k] = b
            
    return(mod_A,mod_B,mod_p)


# In[18]:


Baum_Welch(eta,gamma,A,B,p,O)


# In[1]:


## MANASH PRATIM KAKATI
## PG CERTIFICATION IN AI & ML
## E&ICT ACADAMY, IIT GUWAHATI


# In[ ]:




