import matplotlib.pyplot as plt 
from ase.io import read
import torch
import numpy as np
import json 
from macecvs.models.permutation_invariant_ae import LineInvariantAEMultipleDecoders
from macecvs.training.train_permutation_invariant_ae import TrainLineInvariantAEMultipleDecoders 

print(torch.cuda.device_count())
print(torch.cuda.device_memory_used('cuda'))

dataset = torch.load('descriptors_250.pt')
dset = {} 
dset["points"] = dataset.detach().cpu().numpy()
del(dataset)
dset["weights"] = np.ones(len(dset["points"]))
print(dset["points"].shape)
print(dset["weights"].shape)

n_atoms = 16
n_columns_seed = 32 
bottleneck_size = 3
ae = LineInvariantAEMultipleDecoders([256, 512, 128, n_columns_seed], [n_columns_seed, 128, 512, 256], [n_columns_seed, 64, 16, bottleneck_size], [bottleneck_size, 32, n_atoms * n_columns_seed], 1, n_lines=n_atoms, n_columns_seeds=n_columns_seed)
#ae = torch.load("invar_ae_full.model", weights_only=False)

ae_training = TrainLineInvariantAEMultipleDecoders(ae, dset, l_multidec=1)

ae_training.train_test_split(train_size=40 * 10**3)
ae_training.split_training_dataset_K_folds(5)
ae_training.set_train_val_data(0)
ae_training.set_optimizer('Adam', 0.001)

loss_params = {}
loss_params["mse_weight"] = 1. * 10**(0)
loss_params["contractive_weight"] = 0. * 10**(0)
#loss_params["pen_points_weight"] = 0. * 10**(4)
#loss_params["pen_points_recons"] = 0. * 10**(0)
loss_params["n_wait"] = 10
ae_training.set_loss_weight(loss_params)

batch_size = 1*10**3
max_epochs = 10000

loss_dict = ae_training.train(batch_size, max_epochs)

#ae_training.ae.to('cuda')

train_trajs = torch.tensor(dset["points"][::5].astype('float32')).to('cuda')
train_trajs_xi = ae_training.ae.encoded(train_trajs).detach().cpu().numpy()

TS1_to_iBuOH2_c1 = torch.load('descriptor_TS1_to_iBuOH2_c1.pt').to('cuda')
TS2_to_iBuOH2_c2 = torch.load('descriptor_TS2_to_iBuOH2_c2.pt').to('cuda')
TS1_to_2BuOH2 = torch.load('descriptor_TS1_to_2BuOH2.pt').to('cuda')
TS1_to_secondaire = torch.load('descriptor_TS1_to_secondaire.pt').to('cuda')
TS2_to_tertiaire = torch.load('descriptor_TS2_to_tertiaire.pt').to('cuda')

TS1_to_iBuOH2_c1_xi = ae_training.ae.encoded(TS1_to_iBuOH2_c1).detach().cpu().numpy()
TS2_to_iBuOH2_c2_xi = ae_training.ae.encoded(TS2_to_iBuOH2_c2).detach().cpu().numpy()
TS1_to_2BuOH2_xi = ae_training.ae.encoded(TS1_to_2BuOH2).detach().cpu().numpy()
TS1_to_secondaire_xi = ae_training.ae.encoded(TS1_to_secondaire).detach().cpu().numpy()
TS2_to_tertiaire_xi = ae_training.ae.encoded(TS2_to_tertiaire).detach().cpu().numpy()

plt.figure()
plt.scatter(train_trajs_xi[:,0], train_trajs_xi[:,1], label='train_trajs', marker='.', alpha=0.2)
plt.scatter(TS1_to_iBuOH2_c1_xi[:,0], TS1_to_iBuOH2_c1_xi[:,1], label='TS1_to_iBuOH2_c1', marker='+', alpha=0.2)
plt.scatter(TS2_to_iBuOH2_c2_xi[:,0], TS2_to_iBuOH2_c2_xi[:,1], label='TS2_to_iBuOH2_c2', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:,0], TS1_to_2BuOH2_xi[:,1], label='TS1_to_2BuOH2', marker='+', alpha=0.2)
plt.scatter(TS1_to_secondaire_xi[:,0], TS1_to_secondaire_xi[:,1], label='TS1_to_secondaire', marker='+', alpha=0.2)
plt.scatter(TS2_to_tertiaire_xi[:,0], TS2_to_tertiaire_xi[:,1], label='TS2_to_tertiaire', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:1,0], TS1_to_2BuOH2_xi[:1,1], label='TS1', marker='o')
plt.scatter(TS2_to_tertiaire_xi[:1,0], TS2_to_tertiaire_xi[:1,1], label='TS2', marker='o')
plt.legend()
plt.savefig('scatter_train_trajs_0_1.png', dpi=160)
plt.close()

plt.figure()
plt.scatter(train_trajs_xi[:,0], train_trajs_xi[:,2], label='train_trajs', marker='.', alpha=0.2)
plt.scatter(TS1_to_iBuOH2_c1_xi[:,0], TS1_to_iBuOH2_c1_xi[:,2], label='TS1_to_iBuOH2_c1', marker='+', alpha=0.2)
plt.scatter(TS2_to_iBuOH2_c2_xi[:,0], TS2_to_iBuOH2_c2_xi[:,2], label='TS2_to_iBuOH2_c2', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:,0], TS1_to_2BuOH2_xi[:,2], label='TS1_to_2BuOH2', marker='+', alpha=0.2)
plt.scatter(TS1_to_secondaire_xi[:,0], TS1_to_secondaire_xi[:,2], label='TS1_to_secondaire', marker='+', alpha=0.2)
plt.scatter(TS2_to_tertiaire_xi[:,0], TS2_to_tertiaire_xi[:,2], label='TS2_to_tertiaire', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:1,0], TS1_to_2BuOH2_xi[:1,2], label='TS1', marker='o')
plt.scatter(TS2_to_tertiaire_xi[:1,0], TS2_to_tertiaire_xi[:1,2], label='TS2', marker='o')
plt.legend()
plt.savefig('scatter_train_trajs_0_2.png', dpi=160)
plt.close()

plt.figure()
plt.scatter(train_trajs_xi[:,1], train_trajs_xi[:,2], label='train_trajs', marker='.', alpha=0.2)
plt.scatter(TS1_to_iBuOH2_c1_xi[:,1], TS1_to_iBuOH2_c1_xi[:,2], label='TS1_to_iBuOH2_c1', marker='+', alpha=0.2)
plt.scatter(TS2_to_iBuOH2_c2_xi[:,1], TS2_to_iBuOH2_c2_xi[:,2], label='TS2_to_iBuOH2_c2', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:,1], TS1_to_2BuOH2_xi[:,2], label='TS1_to_2BuOH2', marker='+', alpha=0.2)
plt.scatter(TS1_to_secondaire_xi[:,1], TS1_to_secondaire_xi[:,2], label='TS1_to_secondaire', marker='+', alpha=0.2)
plt.scatter(TS2_to_tertiaire_xi[:,1], TS2_to_tertiaire_xi[:,2], label='TS2_to_tertiaire', marker='+', alpha=0.2)
plt.scatter(TS1_to_2BuOH2_xi[:1,1], TS1_to_2BuOH2_xi[:1,2], label='TS1', marker='o')
plt.scatter(TS2_to_tertiaire_xi[:1,1], TS2_to_tertiaire_xi[:1,2], label='TS2', marker='o')
plt.legend()
plt.savefig('scatter_train_trajs_1_2.png', dpi=160)
plt.close()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(train_trajs_xi[:,0], train_trajs_xi[:,1], train_trajs_xi[:,2], label='train_trajs', marker='.', alpha=0.2)
ax.scatter(TS1_to_iBuOH2_c1_xi[:,0], TS1_to_iBuOH2_c1_xi[:,1], TS1_to_iBuOH2_c1_xi[:,2], label='TS1_to_iBuOH2_c1', marker='+', alpha=0.2)
ax.scatter(TS2_to_iBuOH2_c2_xi[:,0], TS2_to_iBuOH2_c2_xi[:,1], TS2_to_iBuOH2_c2_xi[:,2], label='TS2_to_iBuOH2_c2', marker='+', alpha=0.2)
ax.scatter(TS1_to_2BuOH2_xi[:,0], TS1_to_2BuOH2_xi[:,1], TS1_to_2BuOH2_xi[:,2], label='TS1_to_2BuOH2', marker='+', alpha=0.2)
ax.scatter(TS1_to_secondaire_xi[:,0], TS1_to_secondaire_xi[:,1], TS1_to_secondaire_xi[:,2], label='TS1_to_secondaire', marker='+', alpha=0.2)
ax.scatter(TS2_to_tertiaire_xi[:,0], TS2_to_tertiaire_xi[:,1], TS2_to_tertiaire_xi[:,2], label='TS2_to_tertiaire', marker='+', alpha=0.2)
ax.scatter(TS1_to_2BuOH2_xi[:1,0], TS1_to_2BuOH2_xi[:1,1], TS1_to_2BuOH2_xi[:1,2], label='TS1', marker='o')
ax.scatter(TS2_to_tertiaire_xi[:1,0], TS2_to_tertiaire_xi[:1,1], TS2_to_tertiaire_xi[:1,2], label='TS2', marker='o')
ax.legend()
plt.savefig('scatter_3d.png', dpi=160)
plt.close()

ae_training.ae.to('cpu')
torch.save(ae_training.ae, "invar_ae_full.model")

