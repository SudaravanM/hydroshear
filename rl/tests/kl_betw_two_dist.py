
import torch

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()

# Between two Gaussians
# D_KL(p0 || p1) = E[log(p0_sigma/p1_sigma)] + E[(p0_mu - p1_mu)^2 / (2 * p1_sigma^2)] - 1/2

p0_mu = torch.tensor([0.0, 0.0], dtype=torch.float32)
p0_sigma = torch.tensor([1.0, 1.0], dtype=torch.float32)
p1_mu = torch.tensor([0.0, 0.0], dtype=torch.float32)
p1_sigma = torch.tensor([1.1, 1.1], dtype=torch.float32)

kl = policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma)
print(kl)