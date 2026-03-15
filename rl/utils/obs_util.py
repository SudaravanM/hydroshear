import copy
import torch

def _preproc_obs(obs_batch):
    if type(obs_batch) is dict:
        obs_batch = copy.copy(obs_batch)
        for k,v in obs_batch.items():
            if v.dtype == torch.uint8:
                obs_batch[k] = v.float() / 255.0
            else:
                obs_batch[k] = v
    else:
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
    return obs_batch