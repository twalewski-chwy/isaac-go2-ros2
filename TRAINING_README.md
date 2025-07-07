# Go2 Straight Line Walking Training

This document explains how to train the Go2 robot to walk in a straight line using reinforcement learning.

## Overview

The training script converts your existing inference-only setup into a full training environment that will teach the Go2 to walk in a straight line using PPO (Proximal Policy Optimization).

## Files Created

1. **`train_go2_straight_line_simple.py`** - Main training script
2. **`cfg/train.yaml`** - Training configuration
3. **`train_go2_straight_line.py`** - Alternative comprehensive training script

## Key Differences from Inference Script

### **Inference Script (`isaac_go2_ros2.py`)**
- ✅ Loads pre-trained model
- ✅ Runs in `torch.inference_mode()`
- ✅ No learning - fixed behavior
- ✅ Real-time control with keyboard input

### **Training Script (`train_go2_straight_line_simple.py`)**
- ✅ Creates new neural network from scratch
- ✅ Enables gradient computation
- ✅ Implements reward functions
- ✅ Learns from experience
- ✅ Saves checkpoints during training

## Reward Function Design

The training optimizes for straight-line walking using these rewards:

### **Primary Rewards**
- **Forward Velocity** (+1.0): Encourages moving forward
- **Alive Bonus** (+0.5): Keeps robot upright

### **Penalties**
- **Angular Velocity** (-0.05): Discourages turning
- **Lateral Velocity** (-0.1): Discourages sideways movement  
- **Joint Velocity** (-0.0001): Prevents excessive joint speeds
- **Base Height** (-0.3): Maintains proper standing height

## Usage

### **1. Basic Training**
```bash
python train_go2_straight_line_simple.py --headless
```

### **2. Training with GUI (slower)**
```bash
python train_go2_straight_line_simple.py
```

### **3. CPU Training (if no GPU)**
```bash
python train_go2_straight_line_simple.py --cpu
```

### **4. Custom Configuration**
Modify `cfg/train.yaml` to adjust:
- Number of environments
- Learning rates
- Reward weights
- Network architecture

## Training Process

1. **Initialization**: Creates multiple Go2 robots in parallel environments
2. **Exploration**: Robots try random actions initially  
3. **Learning**: PPO algorithm updates policy based on rewards
4. **Iteration**: Process repeats for 3000 iterations (default)
5. **Checkpointing**: Models saved every 100 iterations

## Expected Training Time

- **GPU (RTX 3080+)**: ~2-4 hours for 3000 iterations
- **CPU**: ~12-24 hours for 3000 iterations
- **Multi-GPU**: ~30-60 minutes for 3000 iterations

## Monitoring Progress

### **Tensorboard Logs**
```bash
tensorboard --logdir outputs/go2_straight_line
```

### **Key Metrics to Watch**
- **Mean Reward**: Should increase over time
- **Forward Velocity**: Should approach 1.0+ m/s
- **Policy Loss**: Should decrease and stabilize

## Output Files

Training generates:
- **`outputs/go2_straight_line/model_*.pt`** - Checkpoint files
- **`outputs/go2_straight_line/summaries/`** - Tensorboard logs
- **`outputs/go2_straight_line/code/`** - Code snapshots

## Using Trained Model

After training, modify your inference script to load the new model:

```python
# In go2_ctrl_cfg.py, add new config:
your_trained_model_cfg = {
    'load_run': 'go2_straight_line',  
    'load_checkpoint': 'model_3000.pt'  # Use final model
    # ... other settings
}
```

## Troubleshooting

### **GPU Memory Issues**
- Reduce `num_envs` in `cfg/train.yaml`
- Use `--cpu` flag for CPU training

### **Slow Training**
- Increase `num_envs` for more parallelization
- Use smaller networks in `cfg/train.yaml`

### **Poor Performance**
- Adjust reward weights in `cfg/train.yaml`
- Increase `max_iterations`
- Modify reward functions in the script

## Customization

### **Changing Behavior**
Modify reward functions in `train_go2_straight_line_simple.py`:
- Add obstacle avoidance rewards
- Change target velocity
- Add turning penalties

### **Network Architecture**
Modify `cfg/train.yaml`:
```yaml
policy:
  actor_hidden_dims: [256, 256, 128]  # Larger network
  critic_hidden_dims: [256, 256, 128]
```

### **Training Length**
```yaml
max_iterations: 5000  # Longer training
save_interval: 200    # Save less frequently
```

This setup provides a complete training pipeline that builds on your existing codebase while adding the reinforcement learning components needed to train new behaviors. 