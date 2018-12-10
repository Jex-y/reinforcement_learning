from tensorflow import keras
from time import sleep
import numpy as np
from falling_game import falling_game

env = falling_game(size=(8,16))
save_path = "model.h5"
model = keras.models.load_model(save_path)
env.reset()
for t in range(100):
    state = np.array([env.game_state()])
    action = np.amax(model.predict(state))
    print(model.predict(state))
    env.tstep(action=action)
    env.print_grid()
    sleep(1)