import os
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect Go2 asset data properties")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import pprint
from isaaclab.envs import ManagerBasedRLEnv
from go2.go2_env import Go2RSLEnvCfg
import go2.go2_ctrl as go2_ctrl

def inspect_asset_data():
    """Inspect the properties available under asset.data"""
    
    # Create output file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"asset_data_inspection_{timestamp}.txt"
    
    def write_and_print(text, file_handle=None):
        """Write to both console and file"""
        print(text)
        if file_handle:
            file_handle.write(text + "\n")
    
    # Create environment configuration
    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs = 1  # Just need one environment
    
    # Initialize velocity command system (required for observations)
    go2_ctrl.init_base_vel_cmd(1)
    print("Initialized velocity command system")
    
    print("Creating environment...")
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("Environment created successfully!")
    print(f"Writing inspection results to: {output_file}")
    
    # Get the asset
    asset = env.scene["unitree_go2"]
    
    # Open file for writing
    with open(output_file, 'w') as f:
        write_and_print("="*60, f)
        write_and_print("ASSET.DATA PROPERTIES INSPECTION", f)
        write_and_print("="*60, f)
        write_and_print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        
        write_and_print(f"\n Asset type: {type(asset)}", f)
        write_and_print(f" Asset.data type: {type(asset.data)}", f)
        
        # Get all attributes of asset.data
        data_attrs = [attr for attr in dir(asset.data) if not attr.startswith('_')]
        
        write_and_print(f"\nFound {len(data_attrs)} public attributes in asset.data:", f)
        write_and_print("-" * 50, f)
        
        for i, attr in enumerate(sorted(data_attrs), 1):
            try:
                value = getattr(asset.data, attr)
                if isinstance(value, torch.Tensor):
                    write_and_print(f"{i:2d}. {attr:<25} -> Tensor {tuple(value.shape)} ({value.dtype})", f)
                elif callable(value):
                    write_and_print(f"{i:2d}. {attr:<25} -> Method/Function", f)
                else:
                    write_and_print(f"{i:2d}. {attr:<25} -> {type(value).__name__}", f)
            except Exception as e:
                write_and_print(f"{i:2d}. {attr:<25} -> Error: {str(e)[:40]}...", f)
        
        write_and_print("\n" + "="*60, f)
        write_and_print("DETAILED TENSOR INFORMATION", f)
        write_and_print("="*60, f)
        
        # Show detailed info for tensor attributes
        tensor_attrs = []
        for attr in data_attrs:
            try:
                value = getattr(asset.data, attr)
                if isinstance(value, torch.Tensor):
                    tensor_attrs.append((attr, value))
            except:
                pass
        
        if tensor_attrs:
            for attr, tensor in tensor_attrs:
                write_and_print(f"\n{attr}:", f)
                write_and_print(f"   Shape: {tuple(tensor.shape)}", f)
                write_and_print(f"   Dtype: {tensor.dtype}", f)
                write_and_print(f"   Device: {tensor.device}", f)
                try:
                    if tensor.numel() > 0 and tensor.numel() <= 50:  # Show values for small tensors
                        values_str = f"   Values: {tensor.flatten()[:10].tolist()}{'...' if tensor.numel() > 10 else ''}"
                        write_and_print(values_str, f)
                    elif tensor.numel() > 0:
                        stats_str = f"   Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}"
                        write_and_print(stats_str, f)
                except:
                    write_and_print("   (Cannot display values)", f)
        
        write_and_print("\n" + "="*60, f)
        write_and_print("COMMONLY USED PROPERTIES", f)
        write_and_print("="*60, f)
        
        # Show some commonly used properties with descriptions
        common_props = [
            ("root_state_w", "Root state in world frame [pos(3) + quat(4)]"),
            ("root_lin_vel_b", "Root linear velocity in body frame"),
            ("root_ang_vel_b", "Root angular velocity in body frame"),
            ("joint_pos", "Joint positions"),
            ("joint_vel", "Joint velocities"),
            ("joint_acc", "Joint accelerations (if available)"),
            ("body_lin_vel_w", "Body linear velocities in world frame (if available)"),
            ("body_ang_vel_w", "Body angular velocities in world frame (if available)"),
            ("projected_gravity_b", "Projected gravity in body frame (if available)"),
            ("contact_forces", "Contact forces (if available)"),
        ]
        
        for prop, desc in common_props:
            if hasattr(asset.data, prop):
                try:
                    value = getattr(asset.data, prop)
                    if isinstance(value, torch.Tensor):
                        write_and_print(f"[YES] {prop:<20} -> {tuple(value.shape)} - {desc}", f)
                    else:
                        write_and_print(f"[YES] {prop:<20} -> {type(value).__name__} - {desc}", f)
                except Exception as e:
                    write_and_print(f"[ERR] {prop:<20} -> Error: {str(e)[:40]}...", f)
            else:
                write_and_print(f"[NO]  {prop:<20} -> Not available - {desc}", f)
        
        write_and_print("\n" + "="*60, f)
        write_and_print("EXAMPLE USAGE", f)
        write_and_print("="*60, f)
        
        example_code = """
# Example usage in reward/termination functions:

def my_reward_function(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    # Get forward velocity (x-axis in body frame)
    forward_vel = asset.data.root_lin_vel_b[:, 0]
    
    # Get base height (z-coordinate in world frame)
    base_height = asset.data.root_state_w[:, 2]
    
    # Get joint velocities
    joint_vels = asset.data.joint_vel
    
    # Get base orientation (quaternion)
    base_quat = asset.data.root_state_w[:, 3:7]  # [w, x, y, z]
    
    return forward_vel  # or whatever computation you need

# Additional examples:

def check_robot_stability(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    # Check if robot is upright (quaternion w component should be close to 1 for upright)
    base_quat = asset.data.root_state_w[:, 3:7]  # [w, x, y, z]
    upright_score = torch.abs(base_quat[:, 0])  # w component
    
    return upright_score

def get_joint_power(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    # Power = torque * angular_velocity (approximate)
    joint_velocities = asset.data.joint_vel
    # Note: You'd need joint_torques from elsewhere for actual power calculation
    
    return torch.sum(torch.abs(joint_velocities), dim=1)  # Sum of absolute velocities as proxy
"""
        write_and_print(example_code, f)
        
        write_and_print("\n" + "="*60, f)
        write_and_print("TENSOR INDEXING REFERENCE", f)
        write_and_print("="*60, f)
        
        indexing_info = """
# Common tensor indexing patterns:

root_state_w shape: [num_envs, 13]
├── [:, 0:3]   -> Position (x, y, z) in world frame
├── [:, 3:7]   -> Orientation quaternion (w, x, y, z)
├── [:, 7:10]  -> Linear velocity in world frame
└── [:, 10:13] -> Angular velocity in world frame

root_lin_vel_b shape: [num_envs, 3]
├── [:, 0] -> Forward velocity (x-axis in body frame)
├── [:, 1] -> Lateral velocity (y-axis in body frame)  
└── [:, 2] -> Vertical velocity (z-axis in body frame)

root_ang_vel_b shape: [num_envs, 3]
├── [:, 0] -> Roll rate (rotation around x-axis)
├── [:, 1] -> Pitch rate (rotation around y-axis)
└── [:, 2] -> Yaw rate (rotation around z-axis)

joint_pos shape: [num_envs, num_joints]
└── Each column represents one joint's position

joint_vel shape: [num_envs, num_joints]  
└── Each column represents one joint's velocity
"""
        write_and_print(indexing_info, f)
    
    # Clean shutdown
    env.close()
    print(f"\nInspection completed! Results saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    inspect_asset_data()
    simulation_app.close() 