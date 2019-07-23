import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd



#loading processed data
start_time = time.clock()
train_files=np.load('Nx&y_train_lessthan100m.npz')
train_x=train_files['x']
train_y=train_files['y']
dim=train_x.shape[1]


#create placeholders
def placeholder(nx, ny):
    '''
    the function receives a scalar input size, of nx, ie. number of parameters
    similarly the function receives the ny, size of output. 
    
    returns tensors
    '''  
    X=tf.placeholder(tf.float32,[nx,None],name='X')
    Y=tf.placeholder(tf.float32,[ny,None],name='Y')
    return X,Y


#intialise parameters
def init_parameters(xparameters):
    '''
    initialise parameters to build neural network
    
    returns  a dictionary of values
    '''

    W1=tf.get_variable("W1",[600,xparameters],initializer=tf.contrib.layers.xavier_initializer())
    b1=tf.get_variable("b1",[600,1],initializer=tf.zeros_initializer())
    
    W2=tf.get_variable("W2",[300,600],initializer=tf.contrib.layers.xavier_initializer())
    b2=tf.get_variable("b2",[300,1],initializer=tf.zeros_initializer())
    
    W3=tf.get_variable("W3",[100,300],initializer=tf.contrib.layers.xavier_initializer())
    b3=tf.get_variable("b3",[100,1],initializer=tf.zeros_initializer())
    
    W4=tf.get_variable("W4",[25,100],initializer=tf.contrib.layers.xavier_initializer())
    b4=tf.get_variable("b4",[25,1],initializer=tf.zeros_initializer())
    
    W5=tf.get_variable("W5",[12,25],initializer=tf.contrib.layers.xavier_initializer())
    b5=tf.get_variable("b5",[12,1],initializer=tf.zeros_initializer())
    
    W6=tf.get_variable("W6",[1,12],initializer=tf.contrib.layers.xavier_initializer())
    b6=tf.get_variable("b6",[1,1],initializer=tf.zeros_initializer())
    
    
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2,
                "W3":W3,
                "b3":b3,
                "W4":W4,
                "b4":b4,
                "W5":W5,
                "b5":b5,
                "W6":W6,
                "b6":b6
                }
    return parameters

#forward propogation
def forward_propagation(X,parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]
    W4=parameters["W4"]
    b4=parameters["b4"]
    W5=parameters["W5"]
    b5=parameters["b5"]
    W6=parameters["W6"]
    b6=parameters["b6"]
    
    Z1=tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    
    Z2=tf.add(tf.matmul(W2,A1),b2)
    A2=tf.nn.relu(Z2)
    
    Z3=tf.add(tf.matmul(W3,A2),b3)
    A3=tf.nn.relu(Z3)
    
    Z4=tf.add(tf.matmul(W4,A3),b4)
    A4=tf.nn.relu(Z4)
    
    Z5=tf.add(tf.matmul(W5,A4),b5)
    A5=tf.nn.relu(Z5)
    
    Z6=tf.add(tf.matmul(W6,A5),b6)
    
    return Z6

#cost function
def compute_cost(Z6,Y):
    #need to tranpose the matrix to fit into the 
    #tf.tranpose(Z3)
    #using tf.reduce_mean(tf.square(prediction-y))
    cost=tf.reduce_mean(tf.square(Z6-Y))
    return cost


#start neural network not use minibatch_size=32, X_test, Y_test
def model(train_x, train_y,learning_rate=0.0001, num_epochs=10000, print_cost=True):
    
    (n_x,m)=train_x.shape
    n_y=train_y.shape[0]
    costs=[]
    
    X,Y=placeholder(n_x,n_y)
    parameters=init_parameters(n_x)
    Z6=forward_propagation(X,parameters)
    cost=compute_cost(Z6,Y)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost=0.
            _,epoch_cost=sess.run([optimizer,cost],feed_dict={X:train_x,Y:train_y})
            
            if print_cost==True and epoch%100==0:
                print("cost after epoch %i: %f" % (epoch,epoch_cost))
            if print_cost==True and epoch%200==0 and num_epochs>1000:
                costs.append(epoch_cost)

       
        return parameters

#%% setting up X_train and Y_train vector to go into the neural network. 
train_x=train_x.reshape(train_x.shape[1],train_x.shape[0])
train_y=train_y.reshape(train_y.shape[1],train_y.shape[0])

parameters=model(train_x,train_y)
