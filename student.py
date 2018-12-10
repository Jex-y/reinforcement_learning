import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.warnings.filterwarnings('ignore')
class Q_agent(object):
    def __init__(self,env,mem,input_dim,output_space,layers,hyper=None,seed=0):
        self.mem = mem
        if hyper == None:
            self.hyper = {
            "gamma" : 1,
            "epsilon" : 1,
            "epsilon_decay" : 0.995,
            "epsilon_min" : 0.01,
            "learning_rate" : 0.001,
            "train_batch_size" : 64
            }
        else:
            self.hyper = hyper
        self.input_dim = input_dim
        self.output_space = output_space
        self.SGD = keras.optimizers.SGD
        self.MSE = keras.losses.mean_squared_error
        self.model = self.create_model(
            layers=layers,
            input_dim = input_dim,
            output_space = output_space,
            loss = self.MSE,
            learning_rate=self.hyper["learning_rate"],
            optimiser = self.SGD)
        np.random.seed(seed)
        return None

    def create_model(self,layers,input_dim,output_space,loss,learning_rate,optimiser,load_file=None):
        Dense = keras.layers.Dense
        relu = keras.activations.relu
        linear = keras.activations.linear
        model = keras.Sequential()
        model.add(Dense(layers[0],input_dim=input_dim,activation=relu))
        for l in layers[1:]:
            model.add(Dense(l,activation=relu))
        model.add(Dense(len(output_space),activation=linear))
        model.compile(loss=loss,optimizer=optimiser(lr=learning_rate))
        if load_file != None:
            model.load_weights(load_file)
        return model
    
    def select_action(self,state):
        state = np.array([state],dtype=np.float32)
        if np.random.rand() <= self.hyper["epsilon"]:
            return np.random.choice(self.output_space)
        else:
            values = self.model.predict(state)
            return np.argmax(values[0])

    def retrain(self,batch_size=None,epochs=1):
        if batch_size == None:
            batch_size = self.hyper["train_batch_size"]
        batch = self.mem.recall(batch_size)
        for state, action, reward, next_state, done in batch:
            Q_target = reward
            if not done:
                Q_target = reward + (self.hyper["gamma"] * np.amax(self.model.predict(np.array([next_state]))))
            Q = self.model.predict(state)
            Q[0][action] = Q_target
            self.model.fit(state,Q,epochs=epochs,verbose=0)
        if self.hyper["epsilon"] > self.hyper["epsilon_min"]:
            self.hyper["epsilon"] *= self.hyper["epsilon_decay"]