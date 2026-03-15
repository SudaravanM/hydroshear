
import torch
import torch.nn as nn

import numpy as np


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        
        # Convert ListConfig to regular Python types if needed
        if hasattr(insize, '_content'):  # Check if it's an OmegaConf object
            insize = list(insize)
        elif hasattr(insize, '__iter__') and not isinstance(insize, (int, tuple)):
            # Handle other iterable types that aren't int or tuple
            insize = tuple(insize) if len(insize) > 1 else insize[0]
        
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        print("insize: ", self.insize)
        if per_channel:
            if len(self.insize) == 3:
                # channels-last: [H, W, C] -> reduce over batch, H, W
                self.axis = [0, 1, 2]
                in_size = self.insize[-1]  # C = 2
            elif len(self.insize) == 2:
                self.axis = [0, 1]
                in_size = self.insize[-1]
            elif len(self.insize) == 1:
                self.axis = [0]
                in_size = self.insize[0]
            else:
                self.axis = [0]
                in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer('running_mean', torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer('running_var', torch.ones(in_size, dtype = torch.float64))
        self.register_buffer('count', torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis, unbiased=False)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size(0)
            )

        # change shape - use expand() instead of expand_as() to avoid referencing input tensor
        if self.per_channel:
            if len(self.insize) == 3:
                # channels-last: [batch, H, W, C]
                current_mean = self.running_mean.view([1, 1, 1, self.insize[-1]]).expand(input.shape)
                current_var = self.running_var.view([1, 1, 1, self.insize[-1]]).expand(input.shape)
            elif len(self.insize) == 2:
                # channels-last: [batch, W, C]
                current_mean = self.running_mean.view([1, 1, self.insize[-1]]).expand(input.shape)
                current_var = self.running_var.view([1, 1, self.insize[-1]]).expand(input.shape)
            elif len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand(input.shape)
                current_var = self.running_var.view([1, self.insize[0]]).expand(input.shape)        
            else:
                current_mean = self.running_mean
                current_var = self.running_var
        else:
            current_mean = self.running_mean
            current_var = self.running_var
       
        # get output
        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, per_channel_obs_list=None):
        """
        Args:
            insize: dict of observation names to shapes
            epsilon: small value for numerical stability
            per_channel: bool or dict. If bool, applies to all observations. 
                        If dict, maps observation names to bool values for per-observation control.
            norm_only: bool, if True only normalizes by std without subtracting mean
            per_channel_obs_list: list of observation names that should use per_channel=True.
                                 Alternative to passing per_channel as dict.
        """
        assert(isinstance(insize, dict))
        super(RunningMeanStdObs, self).__init__()
        
        # Handle per_channel configuration
        if per_channel_obs_list is not None:
            # Convert list to dict
            per_channel_dict = {k: (k in per_channel_obs_list) for k in insize.keys()}
        elif isinstance(per_channel, dict):
            # Use provided dict
            per_channel_dict = per_channel
        elif isinstance(per_channel, bool):
            # Apply same value to all observations
            per_channel_dict = {k: per_channel for k in insize.keys()}
        else:
            per_channel_dict = {k: False for k in insize.keys()}
        
        self.running_mean_std = nn.ModuleDict({
            k : RunningMeanStd(v, epsilon, per_channel_dict.get(k, False), norm_only) 
            for k,v in insize.items()
        })
    
    def forward(self, input, denorm=False):
        res = {k : self.running_mean_std[k](v, denorm) for k,v in input.items()}
        return res


def main():   
    # RunningMeanStd
    print("RunningMeanStd:")
    rms = RunningMeanStd(insize=5)
    data = torch.randn(1, 5)  # batch_size=10, features=5
    
    rms.train()
    normalized = rms(data)
    rms.eval()
    denormalized = rms(normalized, denorm=True)
    print(f"Input: mean={data.mean():.3f}, std={data.std():.3f}")
    print(f"Output: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    print(f"Denormalized: mean={denormalized.mean():.3f}, std={denormalized.std():.3f}")
    
    # RunningMeanStdObs
    print("\nRunningMeanStdObs")
    obs_sizes = {'vector': (3,), 'image': (256, 256, 3)}


    per_channel_obs_list = ['image']
    rms_obs = RunningMeanStdObs(insize=obs_sizes, per_channel=False)
    
    obs_data = {
        'vector': torch.randn(8, 3),
        'image': torch.randn(8, 256, 256, 3)
    }
    
    rms_obs.train()
    normalized_obs = rms_obs(obs_data)
    rms_obs.eval()
    denormalized_obs = rms_obs(normalized_obs, denorm=True)
    print("Original R channel mean:, ", obs_data['image'][..., 0].mean())
    print("Original G channel mean:, ", obs_data['image'][..., 1].mean())
    print("Original B channel mean:, ", obs_data['image'][..., 2].mean())
    print("Normalized R channel mean:, ", normalized_obs['image'][..., 0].mean())
    print("Normalized G channel mean:, ", normalized_obs['image'][..., 1].mean())
    print("Normalized B channel mean:, ", normalized_obs['image'][..., 2].mean())
    print("Denormalized R channel mean:, ", denormalized_obs['image'][..., 0].mean())
    print("Denormalized G channel mean:, ", denormalized_obs['image'][..., 1].mean())
    print("Denormalized B channel mean:, ", denormalized_obs['image'][..., 2].mean())
    
    print("\n3. Save and load:")
    torch.save(rms.state_dict(), "/tmp/rms_test.pt")
    
    rms_loaded = RunningMeanStd(insize=5)
    rms_loaded.load_state_dict(torch.load("/tmp/rms_test.pt", weights_only=True))
    print("Model saved and loaded successfully")
    
    # How adding data changes normalization
    print("\n4. How adding data changes normalization:")
    rms_demo = RunningMeanStd(insize=3)
    print(f"Initial running_mean: {rms_demo.running_mean}")
    
    rms_demo.train()  # Enable statistics updates
    
    # Add batch 1
    batch1 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    norm1 = rms_demo(batch1)
    print(f"After batch 1: running_mean={rms_demo.running_mean}, output_mean={norm1.mean():.3f}")
    
    # Add batch 2  
    batch2 = torch.tensor([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]])
    norm2 = rms_demo(batch2)
    print(f"After batch 2: running_mean={rms_demo.running_mean}, output_mean={norm2.mean():.3f}")
    
    # Cleanup
    import os
    os.remove("/tmp/rms_test.pt")
    print("Done!")


if __name__ == "__main__":
    main()