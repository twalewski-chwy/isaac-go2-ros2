import os
import hydra
import torch
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
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

from go2.go2_env import Go2RSLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import env.sim_env as sim_env
import go2.go2_ctrl as go2_ctrl

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
    
    def __post_init__(self):
        super().__post_init__()
        
        # Training specific settings
        self.episode_length_s = 20.0
        self.decimation = 2
        
        # Enable observations for training
        self.observations.policy.enable_corruption = True
        
        # Set action scale for training
        self.actions.joint_pos.scale = 0.25

    # Add training components
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def train_go2(cfg):
    """Main training function."""
    
    # Create training environment configuration
    env_cfg = Go2TrainingEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.scene.env_spacing = cfg.env_spacing
    env_cfg.episode_length_s = cfg.episode_length_s
    env_cfg.decimation = cfg.decimation
    
    # Initialize velocity command system (required for observations)
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    print(f"✅ Initialized velocity command system for {cfg.num_envs} environments")
    
    print(f"Creating environment with {cfg.num_envs} robots...")
    
    # Create environment directly using ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Add environment scenery (warehouse, obstacles, etc.)
    env_name = getattr(cfg, 'env_name', 'none')
    if env_name == "obstacle-dense":
        sim_env.create_obstacle_dense_env()
        print("✅ Created obstacle-dense environment")
    elif env_name == "obstacle-medium":
        sim_env.create_obstacle_medium_env()
        print("✅ Created obstacle-medium environment")
    elif env_name == "obstacle-sparse":
        sim_env.create_obstacle_sparse_env()
        print("✅ Created obstacle-sparse environment")
    elif env_name == "warehouse":
        sim_env.create_warehouse_env()
        print("✅ Created warehouse environment")
    elif env_name == "warehouse-forklifts":
        sim_env.create_warehouse_forklifts_env()
        print("✅ Created warehouse-forklifts environment")
    elif env_name == "warehouse-shelves":
        sim_env.create_warehouse_shelves_env()
        print("✅ Created warehouse-shelves environment")
    elif env_name == "full-warehouse":
        sim_env.create_full_warehouse_env()
        print("✅ Created full-warehouse environment")
    else:
        print("✅ Using flat terrain (no additional environment)")
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Training configuration
    train_cfg = {
        "seed": cfg.seed,
        "device": cfg.device,
        "num_steps_per_env": cfg.ppo.num_steps_per_env,
        "max_iterations": cfg.max_iterations,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": cfg.policy.init_noise_std,
            "actor_hidden_dims": cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": cfg.policy.critic_hidden_dims,
            "activation": cfg.policy.activation,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": cfg.ppo.clip_param,
            "entropy_coef": cfg.ppo.entropy_coef,
            "num_learning_epochs": cfg.ppo.num_learning_epochs,
            "num_mini_batches": cfg.ppo.num_mini_batches,
            "learning_rate": cfg.ppo.learning_rate,
            "schedule": "adaptive",
            "gamma": cfg.ppo.gamma,
            "lam": cfg.ppo.lam,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "save_interval": cfg.save_interval,
        "experiment_name": "go2_straight_line",
        "run_name": "",
        "logger": "tensorboard",
        "resume": False,
    }
    
    # Create output directory
    log_dir = os.path.join("outputs", "go2_straight_line")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting training for {cfg.max_iterations} iterations...")
    print(f"Environment: {cfg.num_envs} environments")
    print(f"Device: {cfg.device}")
    print(f"Log directory: {log_dir}")
    print("Environment created successfully! Training will begin...")
    
    # Create PPO runner
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=cfg.device)
    
    # Start training
    ppo_runner.learn(
        num_learning_iterations=cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    
    print("\nTraining completed!")
    print(f"Final model saved in: {log_dir}")

if __name__ == "__main__":
    train_go2()
    simulation_app.close() 