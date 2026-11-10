**Course:** UCS760 – EDGE AI AND ROBOTICS: REINFORCEMENT LEARNING & CONVERSATIONAL AI

A Reinforcement Learning-based robotic manipulation project that explores **Soft Actor-Critic (SAC)** for learning end-effector positioning in a simulated UR5 robotic arm environment.

The project focuses on training an autonomous RL agent to learn target-reaching behaviour through repeated interactions with a simulated robotic environment. The implementation uses PyBullet for physics simulation and Stable-Baselines3 for reinforcement learning.

## 1. Project Overview

Reinforcement Learning enables an autonomous agent to learn behaviour by interacting with an environment and receiving feedback in the form of rewards.

In this project, the problem is formulated as a continuous-action reinforcement learning task. A SAC agent learns to predict suitable horizontal end-effector coordinates to approach a target cube in a simulated UR5 robotic environment.

Instead of relying on manually specified end-effector target coordinates for every movement, the agent learns a policy through environment interaction and reward feedback.

The project investigates how an off-policy deep reinforcement learning algorithm can be applied to robotic positioning and object grasping tasks.

## 2. Reinforcement Learning Framework

The problem is modelled using an agent–environment interaction framework.

**Agent:** Soft Actor-Critic (SAC) policy that learns end-effector positioning.

**Environment:** Custom Gymnasium-compatible UR5 robotic simulation developed using PyBullet.

**Learning Objective:** Learn a policy that selects horizontal end-effector target positions to reach the target cube efficiently.

### Reinforcement Learning Workflow

1. **State Observation:** The agent receives the target cube's horizontal position.
2. **Action Selection:** The SAC policy predicts continuous X and Y coordinates for the desired end-effector position.
3. **Environment Interaction:** The action is passed to the PyBullet environment.
4. **Robot Movement:** Inverse kinematics converts the target position into the corresponding robot joint configuration.
5. **Reward Feedback:** The environment calculates a reward based on target-reaching performance and episode progression.
6. **Policy Optimization:** During training, SAC uses collected experience to update its policy and value networks.
7. **Learning Iteration:** The process repeats over multiple episodes, allowing the agent to improve its decision-making policy.

## 3. Soft Actor-Critic (SAC)

The project uses **Soft Actor-Critic**, an off-policy deep reinforcement learning algorithm based on the maximum-entropy reinforcement learning framework.

SAC is particularly suitable for continuous action spaces, making it relevant to robotic control problems where actions involve real-valued coordinates.

The algorithm optimizes a stochastic policy while considering both expected cumulative reward and policy entropy.

### Why SAC?

- **Off-policy Learning:** Learns from previously collected interaction experiences.
- **Continuous Action Space:** Supports continuous-valued control actions.
- **Maximum-Entropy Objective:** Encourages exploration while optimizing task performance.
- **Actor-Critic Architecture:** Uses an actor to select actions and critic networks to estimate their value.
- **Experience Replay:** Reuses stored transitions during training.

### SAC Components

| Component | Role |
|---|---|
| Actor Network | Learns the policy for selecting actions |
| Critic Networks | Estimate action values |
| Entropy Regularization | Encourages exploration |
| Replay Buffer | Stores past environment interactions |
| Target Networks | Support stable value estimation |

## 4. Reinforcement Learning Problem Formulation

### Observation Space (State)

The observation consists of the target cube's horizontal position:

- X-coordinate
- Y-coordinate

The agent uses this information to determine the desired end-effector target position.

### Action Space

The action space is continuous and consists of two values:

\[
a_t = [x_t, y_t]
\]

Where:

- \(x_t\): Desired horizontal end-effector X-coordinate.
- \(y_t\): Desired horizontal end-effector Y-coordinate.

The target Z-coordinate is fixed by the environment.

### Reward Function

The reward function provides feedback to guide the agent towards the target.

The current implementation includes:

- Positive reward when the end-effector reaches the target region.
- Additional reward based on the remaining episode steps.
- Distance-based penalties when the target is not reached.
- Penalties when the maximum episode length is reached.

The reward is designed to encourage efficient target-reaching behaviour.

### Episode Termination

An episode terminates when the end-effector reaches the target region or the maximum number of steps is reached.

The environment also executes predefined gripper-closing and lifting routines after the target-reaching condition.

**Important implementation detail:** The SAC policy learns horizontal end-effector positioning. Gripper closing and object lifting are scripted environment operations, not independently learned SAC actions. Reaching the target region also does not necessarily mean that a successful grasp has occurred.

## 5. Training the Reinforcement Learning Agent

The SAC agent is trained through repeated interactions with the custom robotic environment.

### Training Configuration

| Parameter | Configuration |
|---|---|
| RL Algorithm | Soft Actor-Critic (SAC) |
| Policy Architecture | MlpPolicy |
| Environment | Custom UR5 PyBullet Environment |
| Observation Space | 2-dimensional continuous space |
| Action Space | 2-dimensional continuous space |
| Training Timesteps | 10,000 |
| Checkpoint Interval | 1,000 timesteps |
| RL Framework | Stable-Baselines3 |

### Training Implementation

The training procedure is implemented in `main_rl.py`.

```python
model = SAC(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=10000,
    callback=checkpoint_callback
)

model.save("./models/ur_robot_sac")
```

During training:

- The agent interacts with the simulated environment.
- Experiences are collected for off-policy learning.
- Actor and critic networks are optimized.
- The policy is updated to improve expected reward.
- Model checkpoints are saved at regular intervals.

### Trained Model

The trained 10,000-step checkpoint is available at:

```text
models/ur_robot_sac_10000_steps.zip
```

This checkpoint contains the learned SAC model and can be loaded for subsequent inference.

## 6. Policy Testing and Evaluation

After training, the learned policy can be evaluated by loading the saved SAC checkpoint.

```python
model = SAC.load(
    "./models/ur_robot_sac_10000_steps"
)

action, _ = model.predict(
    obs,
    deterministic=True
)
```

The trained policy predicts actions without performing additional gradient updates.

The simulation can be used to observe the agent's learned target-reaching behaviour and the subsequent scripted grasping routine.

### Evaluation Metrics

The following metrics can be used to assess the trained policy:

- **Episode Reward:** Total reward accumulated during an episode.
- **Episode Length:** Number of steps taken before episode termination.
- **Grasp Success Rate:** Percentage of attempts resulting in an actual successful pickup.
- **Average Reward:** Mean episode reward across evaluation episodes.
- **Average Episode Length:** Mean number of steps across evaluation episodes.

Actual grasp success should be recorded separately from reward because the current reward implementation can award positive reward even if the subsequent gripper operation fails.

## 7. Training Performance Analysis

Training performance is monitored using Stable-Baselines3 episode monitoring logs.

The file `logs/monitor_sac.csv` records:

- Episode reward (`r`)
- Episode length (`l`)
- Elapsed time (`t`)

A reward visualization script, `plot_training.py`, plots episode rewards and their moving average to help analyse learning behaviour over the training process.

The objective is to examine whether the agent's performance improves with additional environment interactions.

## 8. Simulation Environment

The reinforcement learning agent interacts with a PyBullet-based robotic environment containing:

- UR5 robotic manipulator.
- Robotiq 2F-85 gripper.
- Target cube.
- Table and tray.
- Robot meshes and URDF descriptions.

Inverse kinematics is used to translate the agent's predicted end-effector target positions into robot joint movements.

## 9. Repository Structure

```text
rfl_project/
│
├── main_rl.py                 # SAC training and policy testing
├── ur5_env.py                 # RL environment and robot simulation
├── plot_training.py           # Training reward visualization
│
├── models/
│   └── ur_robot_sac_10000_steps.zip
│
├── logs/
│   ├── monitor_sac.csv
│   └── sac_training_curve.png
│
├── meshes/
│   ├── ur5/
│   └── robotiq_85/
│
├── urdf/
│   ├── ur5_robotiq_85.urdf
│   └── cube_blue.urdf
│
├── images/
├── README.md
└── LICENSE
```

## 10. Current Scope and Limitations

- The agent receives simulated target-position information rather than camera-based visual observations.
- The learned policy controls only horizontal end-effector positioning.
- The target Z-coordinate remains fixed.
- Gripper closing and object lifting are predefined routines.
- The current training experiment uses 10,000 timesteps.
- Reward improvement does not independently establish successful grasping.
- The implementation demonstrates simulation-based robotic manipulation and does not establish deployment on a physical UR5 robot.

## 11. Future Work

Potential extensions to this reinforcement learning framework include:

- Extending the action space to include additional robot joint and gripper controls.
- Learning grasp execution and object lifting through reinforcement learning.
- Introducing vision-based observations for autonomous object localization.
- Experimenting with longer training schedules and SAC hyperparameter optimization.
- Comparing SAC, PPO, and A2C under consistent experimental conditions.
- Evaluating policy generalization across different object positions.
- Investigating sim-to-real transfer for physical robotic manipulation.

## 12. Acknowledgement
- Robot model reference: [ElectronicElephant/pybullet_ur5_robotiq](https://github.com/ElectronicElephant/pybullet_ur5_robotiq)

Credit is given to the original contributors for the robot model and simulation resources.

---

**Course Project**

**UCS760 – EDGE AI AND ROBOTICS: REINFORCEMENT LEARNING & CONVERSATIONAL AI**
