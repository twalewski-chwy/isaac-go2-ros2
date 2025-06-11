import os
import hydra
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train Go2 to walk in a straight line")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import torch.nn as nn
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import subtract_frame_transforms

from go2.go2_env import Go2RSLEnvCfg
import go2.go2_ctrl as go2_ctrl
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner

# Custom reward functions for straight line walking
def linear_velocity_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Reward for maintaining forward velocity"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]  # Forward (x) velocity in body frame

def angular_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for angular velocity (to keep moving straight)"""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])  # Yaw angular velocity

def lateral_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for lateral movement (to keep moving straight)"""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])  # Lateral (y) velocity

def joint_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for excessive joint velocities"""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def action_smoothness_penalty(env) -> torch.Tensor:
    """Penalty for jerky movements"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def base_height_penalty(env, target_height: float = 0.34, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for deviating from target base height"""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_state_w[:, 2]
    return torch.square(base_height - target_height)

def robot_fell_down(env, minimum_height: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Check if robot has fallen down (base height too low) - returns boolean"""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_state_w[:, 2]
    return base_height < minimum_height

# Training Configuration Classes
@configclass
class RewardsCfg:
    """Reward terms for straight line walking."""
    
    # Primary rewards
    forward_velocity = RewTerm(func=linear_velocity_reward, weight=1.0)
    
    # Penalties to keep straight
    angular_velocity = RewTerm(func=angular_velocity_penalty, weight=-0.05)
    lateral_velocity = RewTerm(func=lateral_velocity_penalty, weight=-0.1)
    
    # Stability rewards
    joint_vel = RewTerm(func=joint_velocity_penalty, weight=-0.0001)
    action_smoothness = RewTerm(func=action_smoothness_penalty, weight=-0.01)
    base_height = RewTerm(func=base_height_penalty, weight=-0.3)
    
    # Alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

@configclass
class TerminationsCfg:
    """Termination terms for the environment."""
    
    # Terminate if robot falls
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=robot_fell_down,
        params={"minimum_height": 0.25, "asset_cfg": SceneEntityCfg("unitree_go2")},
    )

@configclass
class EventCfg:
    """Configuration for events."""
    
    # Reset robot pose
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("unitree_go2"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class Go2TrainingEnvCfg(Go2RSLEnvCfg):
    """Training configuration for Go2 straight line walking."""
    
    # Override with training-specific settings
    def __post_init__(self):
        super().__post_init__()
        
        # Training specific settings
        self.episode_length_s = 20.0
        self.decimation = 2  # Higher frequency for training
        self.num_envs = 4096 if not args_cli.cpu else 64
        
        # Enable all observations for training
        self.observations.policy.enable_corruption = True
        
        # Set action scale for training
        self.actions.joint_pos.scale = 0.25

    # Add training components
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

# Training configuration
train_cfg = {
    "seed": 42,
    "device": "cuda:0" if torch.cuda.is_available() and not args_cli.cpu else "cpu",
    "num_steps_per_env": 24,
    "max_iterations": 3000,
    "empirical_normalization": False,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01, 
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "save_interval": 100,
    "experiment_name": "go2_straight_line",
    "run_name": "",
    "logger": "tensorboard",
    "resume": False,
}

def main():
    """Main training function."""
    
    # Create training environment
    env_cfg = Go2TrainingEnvCfg()
    env_cfg.scene.num_envs = train_cfg["num_steps_per_env"] * 32  # Adjust based on your GPU
    env_cfg.scene.env_spacing = 2.0
    
    # Initialize velocity command system (required for observations)
    go2_ctrl.init_base_vel_cmd(env_cfg.scene.num_envs)
    print(f"✅ Initialized velocity command system for {env_cfg.scene.num_envs} environments")
    
    print(f"Creating environment with {env_cfg.scene.num_envs} robots...")
    
    # Create environment directly using ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # Create PPO runner
    log_dir = os.path.join("outputs", train_cfg["experiment_name"])
    os.makedirs(log_dir, exist_ok=True)
    
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=train_cfg["device"])
    
    # Start training
    print(f"Starting training for {train_cfg['max_iterations']} iterations...")
    print(f"Environment: {env_cfg.scene.num_envs} environments")
    print(f"Device: {train_cfg['device']}")
    print(f"Log directory: {log_dir}")
    
    ppo_runner.learn(
        num_learning_iterations=train_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )
    
    print("\nTraining completed!")
    print(f"Final model saved in: {log_dir}")

if __name__ == "__main__":
    main()
    simulation_app.close() 