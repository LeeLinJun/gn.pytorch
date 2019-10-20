import numpy as np
from scipy import interpolate
from tqdm import tqdm
import sys
from sparse_rrt.systems import Pendulum

# Load one task:
# env = swimmer.swimmer6(time_limit=4)
# suite.load(domain_name="swimmer", task_name="random")

# Iterate over a task set:
#  for domain_name, task_name in suite.BENCHMARKING:
#  env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.

max_frame = 200
max_episodes = int(sys.argv[1])
dim_state = 2  # theta and omega
dim_control = 1  # torque
dim_pose = 2  # x and y

model = Pendulum()
#  only action is torque from -1 to 1
actions = np.zeros((201, dim_control))
#  before interp, step 20
x = np.arange(0, 201, 20)
dataset = np.zeros((max_episodes, max_frame+1, dim_control+dim_state+dim_pose))
len_dataset = dataset.shape[0]

for idx in tqdm(range(len_dataset)):
    # for j in range(5):
    y = np.random.uniform(model.MIN_TORQUE, model.MAX_TORQUE, x.shape)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 201)
    ynew = interpolate.splev(xnew, tck, der=0)

    actions[:, 0] = ynew

    actions = np.clip(actions, model.MIN_TORQUE, model.MAX_TORQUE)
    start_state = np.array([np.random.uniform(low=model.MIN_ANGLE,
                                              high=model.MAX_ANGLE),
                            np.random.uniform(low=model.MIN_W,
                                              high=model.MAX_W)])

    for i in range(max_frame+1):
        action = actions[i]
        # from IPython import embed; embed()
        new_state = model.propagate(start_state, action,
                                    1,  # np.random.randint(low=20, high=200),
                                    0.002)
        dataset[idx, i, :dim_control] = action
        dataset[idx, i, dim_control:dim_control+dim_state] = new_state
        dataset[idx, i, dim_control+dim_state:dim_control+dim_state+dim_pose] \
            = [np.cos(start_state[0]), np.sin(start_state[0])]

        start_state = new_state


np.save(sys.argv[2], dataset)
