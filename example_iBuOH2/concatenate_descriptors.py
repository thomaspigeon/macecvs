import torch

descriptors = torch.load('trajectory_0/descriptors.pt')[:250]
for i in range(1,200):
    descriptors = torch.cat([descriptors, torch.load('trajectory_' + str(i) + '/descriptors.pt')[:250]], dim=0)

print(descriptors.shape)
torch.save(descriptors, 'descriptors_250.pt')

