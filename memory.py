import numpy as np
class short_term(object):
    def __init__(self,mem_size=100000,seed=0): 
        self.mem_size = mem_size
        self.data = []
        np.random.seed(seed)
    
    def save(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.data.append(experience)
        if len(self.data) > self.mem_size:
            self.data = self.data[1:]
        return None

    def recall(self,batch_size=64):
        data_size = len(self.data)
        if data_size < batch_size:
            batch_size = data_size
        indexes = np.random.randint(data_size, size=batch_size)
        return [self.data[i] for i in indexes]