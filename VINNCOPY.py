#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Sat May  2 14:56:20 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import autograd.numpy as np
import seaborn as sns
from sklearn import datasets
from autograd import elementwise_grad
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
import pymc3 as pm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score


num_cores = int(mp.cpu_count())
pool=ProcessPool(num_cores)



def sigmod(x):
    return 1 / (1 + np.exp(-x))

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def bayes_glm_init(network_structures,x,y):
    
    n_sample=len(y)
    batchs=[j for j in batch(range(0, n_sample), int(n_sample/np.prod(network_structures))+10)] 
    
    batch_index=0 
    weights_list=[]
    for layer_index in range(len(network_structures)-1):
        init_weights=np.zeros([network_structures[layer_index],network_structures[layer_index+1]])
        for out_features in range(network_structures[layer_index+1]):
            if layer_index==0:
                x_sample=x[batchs[batch_index]]
            y_sample=y[batchs[batch_index]]
            
            init_data=pd.DataFrame(x_sample)
            columns_names=['x'+str(i) for i in range(network_structures[layer_index])]
            
            init_data.columns=columns_names
            formula='Y~ -1 + '+'+'.join(columns_names)

            init_data['Y']=y_sample

            with pm.Model(): #as logistic_model:
                pm.glm.GLM.from_formula(formula,
                                    init_data,
                                    family=pm.glm.families.Binomial())
                start=pm.find_MAP()

            init_weights[:,out_features]=np.array(list(start.values()))
            
            batch_index+=1
    
        weights_list.append(init_weights)
        x_sample=x[batchs[batch_index]]
        for temp in weights_list:
            x_sample=sigmod(np.dot(x_sample,temp))
        
    return weights_list


def plot_traces(traces, retain=0):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:],
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')


    

def loss(pred_prob,y):    
    return np.sum(y*np.log(pred_prob+0.001)+(1-y)*np.log(1-pred_prob+0.001))

def nn_out1(weights1,weights2,weights3,x,y):
    layer1 = np.dot(x, weights1)
    o_layer1=sigmod(layer1)
    layer2 = np.dot(o_layer1,weights2)    
    o_layer2=sigmod(layer2)    
    layer3 = np.dot(o_layer2,weights3)
    o_layer3=sigmod(layer3)
    return loss(o_layer3,y)

def nn_out2(weights2,weights1,weights3,x,y):
    layer1 = np.dot(x, weights1)
    o_layer1=sigmod(layer1)
    layer2 = np.dot(o_layer1,weights2)    
    o_layer2=sigmod(layer2)    
    layer3 = np.dot(o_layer2,weights3)
    o_layer3=sigmod(layer3)
    return loss(o_layer3,y)

def nn_out3(weights3,weights1,weights2,x,y):
    layer1 = np.dot(x, weights1)
    o_layer1=sigmod(layer1)
    layer2 = np.dot(o_layer1,weights2)    
    o_layer2=sigmod(layer2)    
    layer3 = np.dot(o_layer2,weights3)
    o_layer3=sigmod(layer3)
    return loss(o_layer3,y)

def nn_predict(weights1,weights2,weights3,x,y):
    layer1 = np.dot(x, weights1)
    o_layer1=sigmod(layer1)
    layer2 = np.dot(o_layer1,weights2)    
    o_layer2=sigmod(layer2)    
    layer3 = np.dot(o_layer2,weights3)
    o_layer3=sigmod(layer3)
    return o_layer3


gradw1=elementwise_grad(nn_out1)
gradw2=elementwise_grad(nn_out2)
gradw3=elementwise_grad(nn_out3)

ggradw1=elementwise_grad(gradw1)
ggradw2=elementwise_grad(gradw2)
ggradw3=elementwise_grad(gradw3)



def grad_func(args,lr=1,lr_s=0.02):
    np.random.seed()
    mu_vector1=args[1]
    mu_vector2=args[2]
    mu_vector3=args[3]   
    sigma_vector1=args[4]
    sigma_vector2=args[5]
    sigma_vector3=args[6]
    
    x=args[7]
    y=args[8]
    
    para_num=np.shape(x)[1]
    M1=np.shape(mu_vector1)[1]
    M2=np.shape(mu_vector2)[1]
    M3=np.shape(mu_vector3)[1]
    # Weights Sampling
    weights1=np.random.normal(mu_vector1,sigma_vector1,size=[para_num,M1])
    weights2=np.random.normal(mu_vector2,sigma_vector2,size=[M1,M2])
    weights3=np.random.normal(mu_vector3,sigma_vector3,size=[M2,M3])

    # Grad of mu calculation    
    grad1=gradw1(weights1,weights2,weights3,x,y)
    grad2=gradw2(weights2,weights1,weights3,x,y)
    grad3=gradw3(weights3,weights1,weights2,x,y)

    # Grad of sigma calculation    
    sgrad1=ggradw1(weights1,weights2,weights3,x,y)
    sgrad2=ggradw2(weights2,weights1,weights3,x,y)
    sgrad3=ggradw3(weights3,weights1,weights2,x,y)
    
    loss=nn_out3(weights3,weights1,weights2,x,y)
    yhat=nn_predict(weights1,weights2,weights3,x,y)
    return grad1,grad2,grad3,sgrad1,sgrad2,sgrad3,yhat,loss










def batch_trainning(x_train,y_train,x_test,y_test,B,S,network_structures,epislon=3,mu=0,sigma=1.5,mu_p=0.0,sigma_p=0.1,lr=1,lr_s=0.02,verbose=False, bayesian_init=True):
    elbo=[]    
    deviance=[]
    
    para_num=network_structures[0]
    M1=network_structures[1]
    M2=network_structures[2]
    M3=network_structures[3]
    
    
    G_mu1=np.zeros(shape=[para_num,M1])
    G_mu2=np.zeros(shape=[M1,M2])
    G_mu3=np.zeros(shape=[M2,M3])
    G_sigma1=np.zeros(shape=[para_num,M1])
    G_sigma2=np.zeros(shape=[M1,M2])
    G_sigma3=np.zeros(shape=[M2,M3])
    sigma_vector1=np.ones(shape=[para_num,M1])*sigma
    sigma_vector2=np.ones(shape=[M1,M2])*sigma
    sigma_vector3=np.ones(shape=[M2,M3])*sigma

    # Bayesian Regression Initialization
    if bayesian_init==True:
        mu_vector1,mu_vector2,mu_vector3=bayes_glm_init(network_structures,x_train,y_train)
    else:
        mu_vector1=np.zeros(shape=[para_num,M1])
        mu_vector2=np.zeros(shape=[M1,M2])
        mu_vector3=np.zeros(shape=[M2,M3])
    for j in batch(range(0, len(y_train)), B):
    
        x=x_train[j]
        
        y=y_train[j]
        
        args = list(zip(list(range(S)),[mu_vector1]*S,[mu_vector2]*S,[mu_vector3]*S,[sigma_vector1]*S,[sigma_vector2]*S,[sigma_vector3]*S, [x]*S,[y]*S))

        pool=ProcessPool(num_cores)#
        result = pool.map_async(grad_func, args)
        result.wait()
        pool.close()
        pool.join()
        
        grad_total=np.array(result.get())
        grad_mean_total=np.mean(grad_total,0)
        
    
        grad_mu1=grad_mean_total[0]
        grad_mu2=grad_mean_total[1]
        grad_mu3=grad_mean_total[2]
    
    
        Le=grad_mean_total[-1]
    
        G_mu1+=grad_mu1**2
        G_mu2+=grad_mu2**2
        G_mu3+=grad_mu3**2
                
        ada_mu1=lr/np.sqrt(epislon+G_mu1)*grad_mu1
        ada_mu2=lr/np.sqrt(epislon+G_mu2)*grad_mu2
        ada_mu3=lr/np.sqrt(epislon+G_mu3)*grad_mu3

    
        lc_mu1=lc_mu2=lc_mu3=0
        lc_mu1=ada_mu1*(mu_vector1 -mu_p)/sigma_p**2/B# 
        lc_mu2=ada_mu2*(mu_vector2 -mu_p)/sigma_p**2/B#
        lc_mu3=ada_mu3*(mu_vector3 -mu_p)/sigma_p**2/B#
        
    
        mu_vector1 = mu_vector1- lc_mu1+ada_mu1
        mu_vector2 = mu_vector2- lc_mu2+ada_mu2
        mu_vector3 = mu_vector3- lc_mu3+ada_mu3
        
        
        grad_sigma1=grad_mean_total[3]/2
        grad_sigma2=grad_mean_total[4]/2
        grad_sigma3=grad_mean_total[5]/2
        
        G_sigma1+=grad_sigma1**2
        G_sigma2+=grad_sigma2**2
        G_sigma3+=grad_sigma3**2    
        
        ada_sigma1=lr_s/np.sqrt(epislon+G_sigma1)*grad_sigma1
        ada_sigma2=lr_s/np.sqrt(epislon+G_sigma2)*grad_sigma2
        ada_sigma3=lr_s/np.sqrt(epislon+G_sigma3)*grad_sigma3
        
        lc_sigma1=lc_sigma2=lc_sigma3=0
        lc_sigma1=ada_sigma1*0.5/B*(1/sigma_vector1**2-1/sigma_p**2)#
        lc_sigma2=ada_sigma2*0.5/B*(1/sigma_vector2**2-1/sigma_p**2)#
        lc_sigma3=ada_sigma3*0.5/B*(1/sigma_vector3**2-1/sigma_p**2)#
        
        
        sigma_vector1 =np.sqrt(sigma_vector1**2-lc_sigma1+ada_sigma1)
        sigma_vector2 =np.sqrt(sigma_vector2**2-lc_sigma2+ada_sigma2)
        sigma_vector3 =np.sqrt(sigma_vector3**2-lc_sigma3+ada_sigma3)
        
    
        #if min(np.min(sigma_vector1) ,np.min(sigma_vector2), np.min(sigma_vector3))>sigma_p:
       #     print ('Entered!!!')
        
        mu_p=np.mean(np.append(np.append(mu_vector1.ravel(),mu_vector2.ravel()),mu_vector3.ravel()))
    
    # sigma_p update    
        #sigma_p=np.sqrt(np.mean(np.append(np.append(sigma_vector1.ravel()**2,sigma_vector2.ravel()**2),sigma_vector3.ravel())**2))    
        
        sigma_p1=np.sum((mu_vector1-mu_p)**2+sigma_vector1**2)
        sigma_p2=np.sum((mu_vector2-mu_p)**2+sigma_vector2**2)
        sigma_p3=np.sum((mu_vector3-mu_p)**2+sigma_vector3**2)
        
        sigma_p= np.sqrt((sigma_p1+sigma_p2+sigma_p3)/(para_num*M1+M1*M2+M2*1))
        
        Lc1=np.sum(np.log(sigma_vector1/sigma_p1)+0.5/sigma_p**2*((mu_vector1-mu_p)**2+sigma_vector1**2-sigma_p**2))
        Lc2=np.sum(np.log(sigma_vector2/sigma_p2)+0.5/sigma_p**2*((mu_vector2-mu_p)**2+sigma_vector2**2-sigma_p**2))
        Lc3=np.sum(np.log(sigma_vector3/sigma_p3)+0.5/sigma_p**2*((mu_vector3-mu_p)**2+sigma_vector3**2-sigma_p**2))
      
        Lc=Lc1+Lc2+Lc3
     
        elbo.append(np.mean(-Lc/B+Le))
        deviance.append(Le)
        if np.mean(-Lc/B+Le)>=max(elbo):#
            best_arg = list(zip(list(range(S)),[mu_vector1]*S,[mu_vector2]*S,[mu_vector3]*S,[sigma_vector1]*S,[sigma_vector2]*S,[sigma_vector3]*S, [x_test]*S,[y_test]*S))
        if verbose:
            print('ada_mu1 is' +str(ada_mu1[-1]))    
            print('ada_mu2 is' +str(ada_mu2[-1]))
            print('ada_sigma1 is' +str(sigma_vector2[-1]))
            print('ada_sigma2 is '+str(sigma_vector1[-1]))
            print('sigma_p is '+str(sigma_p) )
            print(elbo[-1])
    plt.plot(elbo)
    plt.title('Elbo Func')
    plt.show()
    
    plt.plot(deviance)
    plt.title('Deviance Func')
    plt.show()

    return best_arg,elbo
    

def results_plot(best_arg):
    
    y_test=best_arg[-1][-1]
    # CI plot
    pool=ProcessPool(num_cores)#
    result = pool.map_async(grad_func, best_arg)
    result.wait()
    pool.close()
    pool.join()
    
    grad_total=np.array(result.get())
    grad_mu_total=np.mean(grad_total,0)
    pred_prob=grad_mu_total[-2]
  
    std_sample=[j[-2].ravel() for j in grad_total]
    prob_ub=np.quantile(std_sample,0.975,axis=0)
    prob_lb=np.quantile(std_sample,0.025,axis=0)
    plt.plot(range(len(pred_prob)),pred_prob)
    plt.title('Probability Confidence Interval')
    plt.fill_between(range(len(pred_prob)), prob_ub, prob_lb, color='b', alpha=.1)
    plt.show()
    
    """
    auc_list=[]
    f_p_r_list=[]
    t_p_r_list=[]
    for sample_index in range(S):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,grad_total[sample_index][-2].ravel())        
        roc_auc=auc(false_positive_rate, true_positive_rate)
        auc_list.append(roc_auc)
        f_p_r_list.append(false_positive_rate)
        t_p_r_list.append(true_positive_rate)

    
    up_idx = int(0.975 * (len(auc_list) - 1))
    np.argpartition(f_p_r_list, up_idx)
    
    
    np.quantile(auc_list,0.025)
    
    plt.title('Receiver Operating Characteristic')
    1.96*np.std(auc_list)
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% np.mean(auc_list))
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% np.quantile(auc_list,0.975))
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% np.quantile(auc_list,0.025))
    
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    """
    # AUC plot    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    accuracy=accuracy_score(pred_prob>0.5, y_test)
    print("accuracy is "+str(accuracy)+'' )
    return pred_prob,roc_auc


#sns.distplot(pred_prob)
#plt.show()

if __name__ == '__main__':    


    
    S=200 # number of sampled W
    iteration=30
  
    
    # cancer data
    x_sample,y_sample=datasets.load_breast_cancer(return_X_y=True)
    x_sample=x_sample[:,0:10]

  
    # artifical data
#    n_sample=2000
 #   para_num=30
  #  x_sample, y_sample = datasets.make_classification(n_samples=n_sample, n_features=para_num, n_redundant=0, n_informative=para_num,
    #                         n_clusters_per_class=4)
   # y_sample=np.reshape(y_sample,[n_sample,1])
    #x_sample+=np.random.uniform(-0.1,0.1,np.shape(x_sample))
    # polish bankcruptcy data    
#    data=pd.read_csv('bankruptcy.csv')
#    y_sample=data.iloc[:,-1].values.reshape([len(data),1])
#    y_sample=y_sample.astype('float64')
#    x_sample=data.iloc[:,:20].values
    
    
    x_sample=(x_sample-np.mean(x_sample,0))/np.std(x_sample,axis=0)
    n_sample=np.shape(x_sample)[0]
    
    
    B=int(n_sample/iteration) # number of observation per batch
    
    x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=50/n_sample)
    para_num=np.shape(x_train)[1]
    
    Mrule=(para_num+1)/2
    
    M1=int(Mrule/2)
    M2=int(Mrule/2)
#    M1=M2=4
    M3=1
    network_structures=[para_num,M1,M2,M3]
    
    best_arg=batch_trainning(x_train,y_train,x_test,y_test,B,S,network_structures,epislon=3,mu=0,sigma=1.5,mu_p=0.0,sigma_p=1.6,lr=1,lr_s=0.01)
    pred_prob=results_plot(best_arg)

    #pool.terminate()
   # pool.join()
    
"""
data=pd.DataFrame(x_sample) 
data['Y']=y_sample       
color_dic = {1:'red', 0:'blue'}
colors = data['Y'].map(color_dic)
sm = pd.plotting.scatter_matrix(data.iloc[:,:-1], c=colors, alpha=0.3, figsize=((15,15)));
plt.show()

    
"""