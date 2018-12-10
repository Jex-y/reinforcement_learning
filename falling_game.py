import numpy as np
class falling_game(object):
    def __init__(self,size=(8,8),view=None,gravity=1,seed=0):
        if view == None:
            self.view = size
        else:
            self.view = view

        self.size = size
        self.gravity = gravity
        self.action_space = [-1,0,1]
        self.score_zone = (
            (self.size[0]//2)-1,
            (self.size[0]//2)+1)
        self.reward_multi = 10
        self.punish_multi = -10
        self.level_up = 2
        self.reset()
        self.total_move = 0
        self.seed = seed
        self.input_shape = self.game_state().shape
        return None
       
    
    def reset(self):
        self.grid = np.zeros(shape=self.size[::-1],dtype=np.int16)
        self.score = 100
        self.last_reward = None
        self.last_action = None
        self.done = False
        self.level = 1
        state = self.game_state()
        return state


    def tstep(self,action=0):
        self.action(action)
        for y in range(self.size[1])[::-1]:
            for x in range(self.size[0]):
                if self.grid[y][x] >= 1:
                    value = self.grid[y][x]
                    self.grid[y][x] = 0
                    if y < self.size[1]-1:
                        self.grid[y+self.gravity][x] = value
                    else:
                        if self.score_zone[0] <= x <= self.score_zone[1]:
                            if value > 0:
                                self.score += value * self.reward_multi
                            elif value < 0:
                                self.score += value * self.punish_multi

        for _ in range(np.random.randint(self.level+1)):
            self.place(value=np.random.randint(self.level*-1,self.level+1))
        if self.score <= 0:
            #self.done=True
            pass
        if self.score-100 > self.level**self.level_up:
            self.level += 1

        return self.game_state(), self.score, self.done

    def place(self,value=1,x=None):
        np.random.seed(self.seed) 
        if x == None:
            x = (np.random.randint(0,self.size[0]-1) + self.total_move)%self.size[0]
        elif x > self.size[0]:
            raise IndexError("X value out of range")

        self.grid[0][x] = value
    
    def print_grid(self):
        for row in self.grid:
            for char in row:
                print(char,end=" ")
            print("\n",end="")
        print("* "*self.size[0])
    
    def action(self,action):
        action = int(round(action))
        if action > 1:
            action = 1
        elif action < -1:
            action = -1
        self.total_move += action
        if action != 0:
            for y in range(self.size[1])[::-1]:
                line_buffer = np.zeros(shape=self.size[0],dtype=np.int8)
                for x in range(self.size[0]):
                    line_buffer[(x + action)% self.size[0]] = self.grid[y][x]
                self.grid[y] = line_buffer
    
    def game_state(self):
        state = self.grid
        state = state.flatten()
        return state