
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import pycombo

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plotdataset(G:nx.Graph):

    nx.draw(G)
    plt.show()


def Q2(G1:nx.Graph):
    #calculate matrix Q with diag set to 0
    A=np.array(nx.adjacency_matrix(G1).todense())
    T=A.sum(axis=(0,1))
    Q=A*0
    w_in=A.sum(axis=1)
    w_out=w_in.reshape(w_in.shape[0],1)
    K=w_in*w_out/T
    Q=(A-K)/T
    #set Qii to zero for every i
    for i in range(Q.shape[0]):
        Q[i][i]=0
    return Q

def Optimizer(method="SGD",constants={'learning_rate':5e-4,'epsilon':1e-7}):
    learning_rate=constants['learning_rate']

    if method =="SGD":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

def Constraint(norms):

    return lambda x: tf.clip_by_norm(x,clip_norm=norms,axes=0)


def Loss(C,Q,regulerization=0):

    return -tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(C),Q),C))

def GD_dev2(Q):
    #Hyperparams
    epsilon=1e-7
    learning_rate=5e-3
    num_epoch = 100000
    display_ratio = 1/100

    #Variable Inits
    np.random.seed(3)
    C_init=Q[0:2]*0
    C_init[0]=np.random.randint(2, size=(1,Q.shape[0]))
    C_init[1]=1-C_init[0]
    C_init=C_init.T

    #Constant Initialization
    Q = tf.constant(Q,dtype=tf.float64)

    #variable definition
    C=tf.Variable(initial_value=tf.constant(C_init),dtype=tf.float64,constraint=Constraint(1))
    variables=[C]

    #buffers for visualization
    L=[]

    #optimization
    optimizer = Optimizer(constants={'learning_rate':learning_rate})
    for e in range(num_epoch):
        with tf.GradientTape() as tape:
            # loss is for minimizaing modularity with a regularization
            loss = Loss(C,Q)
        if -loss>0:
            L.append(-loss.numpy())
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if e%(int(num_epoch*display_ratio))==0:
            loss_values=Loss(C,Q)
            print('modularity ',-loss_values.numpy(),' on iteration',e)

        #update rate small then 0.001% then break
        if len(L)>2 and abs(1-L[-1]/L[-2])<0.00001:
            print('convergence')
            break

    #assign C to a binary attachment result
    C_numoy=C.numpy()
    result= {}
    for i in range(C.shape[0]):
        if C[i][0]>C[i][1]:
            result[i]=1
        else:
            result[i]=0



    return L,result

def Visual(L):
    #Loss plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(L)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Modularity", fontsize=14)
    plt.legend()
    plt.show()
def GD_dev1(Q):
    #Constant Initialization
    np.random.seed(2)
    C_init=Q[0:2]*0
    C_init[0]=np.random.randint(2, size=(1,Q.shape[0]))
    C_init[1]=1-C_init[0]
    epsilon=1e-7
    learning_rate=5e-4
    #tensorflow variable definition
    N=C_init.shape[1]
    l=tf.Variable(initial_value=np.array([[epsilon],[epsilon]]))
    C=tf.Variable(initial_value=tf.constant(C_init),dtype=tf.float64,constraint=Constraint(N))
    variables=[C,l]
    Q = tf.constant(Q,dtype=tf.float64)

    num_epoch = 10000
    display_ratio = 1/100
    optimizer = Optimizer(constants={'learning_rate':learning_rate})
    for e in range(num_epoch):
        # tape=tf.GradientTape()is to record gradient of loss

        with tf.GradientTape() as tape:
        # loss is for minimizaing modularity with a regularization
            temp1=tf.reduce_sum(C,axis=1,keepdims=True)-N/2
            regularization=tf.matmul(tf.transpose(l),tf.reduce_sum(C,axis=1,keepdims=True)-N/2)
            loss=-tf.linalg.trace(tf.matmul(tf.matmul(C,Q),tf.transpose(C)))
        # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
        grads = tape.gradient(loss, variables)
        # TensorFlow自动根据梯度更新参数
        temp=list(zip(grads, variables))
        #print(grads.shape,C.shape,temp)
        optimizer.apply_gradients(zip(grads, variables))
        #C.assign_sub((5e-4)*grads)
        if e%(int(num_epoch*display_ratio))==0:
            print(-loss.numpy(),C.numpy().mean(),l.numpy(),regularization.numpy(),e)



    print("NED")

if __name__ == '__main__':
    G1=nx.karate_club_graph()
    G1 = nx.karate_club_graph()
    G2=nx.les_miserables_graph()
    L, result = (GD_dev2(Q2(G2)))
    Visual(L)
    print(result,L[-1])
    m = pycombo.execute(G2, max_communities=2)
    print(m)

