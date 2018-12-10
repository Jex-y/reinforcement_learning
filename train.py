from falling_game import falling_game
from memory import short_term
from student import Q_agent
import numpy as np
import time
import os

seed = 42
mem = short_term(mem_size=2**20)
size = (16,16)
env = falling_game(size=size,seed=seed) 
eposides = 2000
folder = "training{}".format(int(time.time()))
os.mkdir(folder)
os.chdir(folder)
save_path = "model"
end_of_ep_scores = np.array([])

hyper = {
        "gamma" : 1,
        "epsilon" : 1,
        "epsilon_decay" : 0.995,
        "epsilon_min" : 0.01,
        "learning_rate" : 0.001,
        "train_batch_size" : 256
     }

DQN = Q_agent(
    env,
    mem,
    input_dim=env.input_shape[0],
    output_space=env.action_space,
    layers=[24,56,24],
    hyper=hyper,
    seed=seed)

print("Starting training {} episodes".format(eposides))
for e in range(eposides):
    state = env.reset()
    for t in range(1000):
        env.seed = int(str(seed)+str(e)+str(t))
        action = DQN.select_action(state)
        next_state, score, done = env.tstep(action=action)
        state = np.array([state],dtype=np.float32)
        mem.save(state,action,score,next_state,done)
        state = next_state
        if done or t == 999:
            print("\rTraining {:.2f}% complete, score: {:.1f}, epsilon: {:.3f}".format(
                (e/eposides)*100,
                score,
                DQN.hyper["epsilon"]
                )
                ,end="\t")
            break
    DQN.retrain()
    end_of_ep_scores = np.append(end_of_ep_scores,score)
    if (e-1)%(eposides//10)== 0:
        path = save_path + "-ep{}".format(e) + ".h5"
        DQN.model.save(path)
        print("\r\nModel saved to: {}{}".format(path,"\t"*10),end="\n")

print("\rTraining Completed")
DQN.model.save(save_path+".h5")
print("Model saved to {}".format(save_path +".h5"))
    
def eval_DQN(DQN,env,seed):
    np.random.seed(seed)
    env.reset()
    env.punish_multi = -10
    env.reward_multi = 10
    DQN.hyper["epsilon"] = 0
    t = 0
    while not env.done and t < 5000:
        state = env.game_state()
        action = DQN.select_action(state)
        env.tstep(action)
        t += 1
    return env.score

print("After training, model score = {}".format(eval_DQN(DQN,env,seed)))
print("saved scores")
np.save("scores.npy",end_of_ep_scores)
print("saved scores to scores.npy")