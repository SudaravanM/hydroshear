import torch



a = torch.tensor([1, 1, 1], dtype=torch.float32)
b = torch.tensor([2, 2, 2], dtype=torch.float32)

# a - b = [-1, -1, -1]
print(torch.norm(a - b, p=2)) # = sqrt((-1)^2 + (-1)^2 + (-1)^2) = sqrt(3) = 1.7320508075688772
print(torch.norm(a - b, p=1)) # = |-1| + |-1| + |-1| = 3
print(torch.norm(a - b, p=torch.inf)) #= max(|1|, |1|, |1|) = 1