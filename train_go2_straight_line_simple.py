import os
import hydra
import torch
import argparse
from isaaclab.app import AppLauncher
import datetime

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

def linear_velocity_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Reward for maintaining target velocity in cardinal direction"""
    asset = env.scene[asset_cfg.name]

    # Get current linear velocity in body frame (forward x, lateral y)
    current_lin_vel = asset.data.root_lin_vel_b[:, :2]
    
    # Get target velocity (only x and y)
    target_vel = go2_ctrl.base_vel_cmd(env)[:, :2]

    # Boolean mask where movement is desired (any target velocity component > 1e-6)
    movement_desired = torch.any(torch.abs(target_vel) > .05, dim=1)

    # Initialize reward tensor
    vel_reward = torch.zeros(current_lin_vel.shape[0], device=current_lin_vel.device)

    # For movement desired:
    if movement_desired.any():
        # Normalize target velocity to get direction
        target_norm = torch.norm(target_vel[movement_desired], dim=1, keepdim=True)
        target_dir = target_vel[movement_desired] / (target_norm + 1e-6)
        
        # Project current velocity onto target direction
        proj_vel = torch.sum(current_lin_vel[movement_desired] * target_dir, dim=1, keepdim=True)
        aligned_vel = proj_vel * target_dir
        
        # Calculate perpendicular component (deviation from desired direction)
        perp_vel = current_lin_vel[movement_desired] - aligned_vel
        
        # Reward aligned movement, penalize perpendicular movement
        vel_reward[movement_desired] = proj_vel.squeeze() - 2.0 * torch.norm(perp_vel, dim=1)

    # For no movement desired: penalize any movement
    no_movement = ~movement_desired
    vel_reward[no_movement] = -2.0 * torch.sum(current_lin_vel[no_movement] ** 2, dim=1)

    return vel_reward


def angular_velocity_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for angular velocity with different weights based on whether rotation is desired"""
    asset = env.scene[asset_cfg.name]
    current_ang_vel = asset.data.root_ang_vel_b[:, 2]  # Yaw angular velocity
    
    # Get target angular velocity from command
    target_ang_vel = go2_ctrl.base_vel_cmd(env)[:, 2]  # Take only the angular velocity component
    
    # Create masks for different cases
    is_rotation_desired = torch.abs(target_ang_vel) > .05  # Small threshold to account for numerical precision
    
    # Initialize error tensor
    ang_vel_error = torch.zeros_like(current_ang_vel)
    
    # For cases where rotation is desired, use normal squared error
    ang_vel_error[is_rotation_desired] = torch.clamp( 1 / torch.abs(current_ang_vel[is_rotation_desired] - target_ang_vel[is_rotation_desired]), min=0.0, max=1.5, )
    
    # For cases where no rotation is desired, use stronger penalty (squared error with higher weight)
    ang_vel_error[~is_rotation_desired] = -2.0 * torch.square(current_ang_vel[~is_rotation_desired])
    
    return ang_vel_error

#def lateral_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
#    """Penalty for lateral movement (to keep moving straight)"""
#    asset = env.scene[asset_cfg.name]
#    return torch.square(asset.data.root_lin_vel_b[:, 1])  # Lateral (y) velocity

def joint_velocity_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for excessive joint velocities"""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def base_height_penalty(env, target_height: float = 0.40, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for deviating from target base height"""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_state_w[:, 2]
    return torch.abs(base_height - target_height)

def robot_fell_down(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("body_contact"), min_force: float = .01) -> torch.Tensor:
    """Check if robot has fallen down (base height too low) - returns boolean"""
    sensor = env.scene[sensor_cfg.name]
    data = sensor.data.net_forces_w
    return torch.sum(torch.sum(data, dim=2), dim=1) > min_force

def head_contact(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_contact"), min_force: float = .01) -> torch.Tensor:
    """Penalty for head contact"""
    sensor = env.scene[sensor_cfg.name]
    data = sensor.data.net_forces_w
    return torch.sum(torch.sum(data, dim=2), dim=1) > min_force

def action_smoothness_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for jerky movements based on joint acceleration magnitude and sign changes"""
    asset = env.scene[asset_cfg.name]
    
    # Get current joint accelerations
    current_joint_acc = asset.data.joint_acc  # Shape: [num_envs, 12]
    
    # Initialize or get previous acceleration storage
    if not hasattr(env, '_prev_joint_acc'):
        # First call - initialize with zeros and return only magnitude penalty
        env._prev_joint_acc = torch.zeros_like(current_joint_acc)
        env._acc_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=current_joint_acc.device)
    
    prev_joint_acc = env._prev_joint_acc
    

    # 2. Sign change and acceleration change penalties (only after first timestep)
    sign_change_penalty = torch.zeros(env.num_envs, device=current_joint_acc.device)
    # Only compute temporal penalties for environments that have previous data
    initialized_mask = env._acc_initialized
    
    if initialized_mask.any():
        # Sign change penalty: detect when acceleration changes direction
        sign_changes = (current_joint_acc[initialized_mask] * prev_joint_acc[initialized_mask]) < 0
        sign_change_penalty[initialized_mask] = torch.sum(sign_changes.float(), dim=1)
        
    # Update previous acceleration for next timestep
    env._prev_joint_acc = current_joint_acc.clone()
    env._acc_initialized.fill_(True)  # Mark all environments as initialized
    
    return sign_change_penalty


def foot_impact_penalty(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), max_contact_force: float = 400.0) -> torch.Tensor:
    """Penalty for hard foot impacts (foot slamming)"""

    contact_sensor = env.scene[sensor_cfg.name]
    contact_data = contact_sensor.data.net_forces_w
    force_magnitudes = torch.norm(contact_data.reshape(contact_data.shape[0], -1), dim=-1)
    excess_forces = torch.clamp(force_magnitudes - max_contact_force, min=0.0)
    return torch.square(excess_forces)

def two_feet_in_contact(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), max_contact_force: float = 400.0) -> torch.Tensor:
    """Reward for maintaining two feet in contact with the ground"""
    contact_sensor = env.scene[sensor_cfg.name]
    contact_data = contact_sensor.data.net_forces_w
    # Sum forces along each axis and check if any force is non-zero (contact)
    feet_in_contact = torch.any(contact_data != 0.0, dim=2)
    # Count how many feet are in contact
    num_feet_in_contact = torch.sum(feet_in_contact, dim=1)
    # Return True if at least 2 feet are in contact
    return num_feet_in_contact >= 2

def rpy_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Penalty for large roll, pitch, and yaw"""
    asset = env.scene[asset_cfg.name]
    roll = asset.data.root_state_w[:, 3]
    pitch = asset.data.root_state_w[:, 4]
    yaw = asset.data.root_state_w[:, 5]
    return torch.abs(roll) + torch.abs(pitch) + torch.abs(yaw)

def default_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Reward for maintaining zero joint positions"""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    return torch.sum(torch.abs(joint_pos), dim=1)

def base_height_death(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    """Death if base height is too low"""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_state_w[:, 2]
    return base_height < 0.20

@configclass
class RewardsCfg:
    """Reward terms for straight line walking."""
    
    # Primary rewards
    forward_velocity = RewTerm(func=linear_velocity_reward, weight=1.0)
    angular_velocity = RewTerm(func=angular_velocity_reward, weight=1.0)
    #rpy = RewTerm(func=rpy_penalty, weight=-0.01)
    #default = RewTerm(func=default_reward, weight=-0.0001)
    
    # Penalties to keep stable
    #lateral_velocity = RewTerm(func=lateral_velocity_penalty, weight=-0.1)
    
    # Stability rewards
    base_height = RewTerm(func=base_height_penalty, weight=-.03)
    action_smoothness = RewTerm(func=action_smoothness_penalty, weight=-0.02)
    
    foot_impact = RewTerm(func=foot_impact_penalty, weight=-0.004)
    two_feet_in_contact = RewTerm(func=two_feet_in_contact, weight=0.001)
    # Alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.5)

@configclass
class TerminationsCfg:
    """Termination terms for the environment."""
    
    # Terminate if robot falls
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_down = DoneTerm(
        func=robot_fell_down,
        params={"sensor_cfg": SceneEntityCfg("body_contact")},
    )

    base_height = DoneTerm(
        func=base_height_death,
        params={},
    )

    #head_constact = DoneTerm(
    #    func=head_contact,
    #    params={"sensor_cfg": SceneEntityCfg("head_contact")},
    #)

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

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("unitree_go2"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )

    randomize_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval", 
        params={
            "asset_cfg": SceneEntityCfg("unitree_go2"),
            "velocity_range": {
                "x": (0.0, 0.5),
                "y": (0.0, 0.5),
                "z": (0.0, 0.5),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
        interval_range_s=(5.0, 10.0),
       
    )


@configclass
class Go2TrainingEnvCfg(Go2RSLEnvCfg):
    """Training configuration for Go2 straight line walking."""
    
    def __post_init__(self):
        super().__post_init__()
        
     
        # Note: decimation will be set from config file
        
        # Enable observations for training
        self.observations.policy.enable_corruption = True
        
        # Set action scale for training
        self.actions.joint_pos.scale = 0.25
        
        # Enable contact processing for foot impact penalty
        # This is deprecated, so false is the default, it does not affect the contact sensors
        self.sim.disable_contact_processing = False

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
    
    # Initialize target velocities after environment creation
    env.target_velocity = torch.zeros((cfg.num_envs, 2), device=env.device)
    env.target_angular_velocity = torch.zeros(cfg.num_envs, device=env.device)
    
    # Set random target velocities for each environment
    for env_idx in range(cfg.num_envs):
        # Randomly choose between linear or angular velocity command
        if torch.rand(1).item() < 0.67:  # 2/3 chance for linear velocity
            # Random linear velocity between -1.5 and 1.5 m/s
            if torch.rand(1).item() < 0.5:
                lin_vel_x = 1.5 if torch.rand(1).item() < 0.5 else -1.5
                lin_vel_y = 0.0
            else:
                lin_vel_x = 0.0
                lin_vel_y = 1.5 if torch.rand(1).item() < 0.5 else -1.5
            ang_vel = 0.0  # No angular velocity
        elif torch.rand(1).item() < 0.1:
            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel = 0.0
        else:  # 50% chance for angular velocity
            # Random angular velocity between -1.5 and 1.5 rad/s
            lin_vel_x = 0.0  # No linear velocity
            lin_vel_y = 0.0  # No linear velocity
            ang_vel = 1.0 if torch.rand(1).item() < 0.5 else -1.0
        
        go2_ctrl.set_target_velocity(env_idx, lin_vel_x, lin_vel_y, ang_vel)
    
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
    log_dir = os.path.join("outputs", f"go2_straight_line_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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