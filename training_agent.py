import numpy as np
#import sys
#sys.path.append("D:\RLBot\\rlgym-tools")
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from custom_obs import CustomObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, VelocityReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym.utils.reward_functions import CombinedReward
from CustomRewards import VelocityPlayerReward, EnemyTouchBallReward, BoostUseReward, AlignBallGoal, RewardIfBehindBall, VelocityPlayerToBallReward, DribbleReward, JumpTouchReward, ClosestToBallReward, DriveForwardReward, FirstTouchReward, AirTimeReward
from customState import CustomState
from customTimeout import DropBallCondition
import torch

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 2
    team_size = 1
    num_instances = 12
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
    batch_size = target_steps//10 #getting the batch size down to something more manageable - 100k in this case
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000
    #print(torch.cuda.is_available())
    def exit_save(model):
        model.save("models/exit_save")

    timeoutCondition=TimeoutCondition(fps*300)

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityPlayerReward(),
                DriveForwardReward(),
                BoostUseReward(),
                RewardIfBehindBall(),
                VelocityBallToGoalReward(),
                FaceBallReward(),
                AlignBallGoal(),
                KickoffReward(),
                FirstTouchReward(timeoutCondition),
                AirTimeReward(),
                EnemyTouchBallReward(),
                ClosestToBallReward(),
                #DribbleReward(),
                JumpTouchReward(),
                EventReward(
                    goal=500.0,
                    team_goal=300.0,
                    concede=-600.0,
                    shot=50.0,
                    save=50.0,
                    demo=10.0,
                    touch=10.0,
                    boost_pickup=10.0
                ),
            ),
            (1.0,   #VelocityPlayerToballReward
             1.0,   #VelocityPlayerReward
             1.0,   #DriveForwardReward
             0.1,   #BoostUseReward
             1.0,   #RewardIfBehindBall
             20.0,  #VelocityBallToGoalReward
             0.2,   #FaceBallReward
             0.2,   #AlignBallGoal
             100.0, #KickoffReward
             1.0,   #FirstTouchReward
             1.0,   #AirTimeReward
             1.0,   #EnemyTouchBallReward
             1.0,   #ClosestToBallReward
             #1.0,   #DribbleReward
             1.0,   #JumpTouchReward
             1.0    #EventReward
             )),
            game_speed=100,
            boost_consumption=0,
            spawn_opponents=True,
            terminal_conditions=[timeoutCondition, NoTouchTimeoutCondition(fps * 45), GoalScoredCondition(), DropBallCondition()],
            obs_builder=CustomObs(),
            state_setter=CustomState(),
            action_parser=DiscreteAction()
        )


    env = SB3MultipleInstanceEnv(get_match, num_instances)            # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:        
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device=torch.device('cuda'),
            custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
            # If you need to adjust parameters mid training, you can use the below example as a guide
            #custom_objects={"n_envs": env.num_envs, "n_steps": steps, "batch_size": batch_size, "n_epochs": 10, "learning_rate": 5e-5}
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")
    device_used = model.device
    print("Device being used:", device_used)
    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback, reset_num_timesteps=False) #can ignore callback if training_interval < callback target
            model.save("models/exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")
        print("Saving model")
        exit_save(model)
        print("Save complete")

    print("Saving model")
    exit_save(model)
    print("Save complete")