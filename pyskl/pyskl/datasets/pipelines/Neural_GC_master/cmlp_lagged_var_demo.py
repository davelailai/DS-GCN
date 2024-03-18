import torch
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from synthetic import simulate_var
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized, train_model_adam
device = torch.device('cuda')

X_np, beta, GC = simulate_var(p=10, T=1000, lag=3)
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
fig, axarr = plt.subplots(1, 2, figsize=(16, 5))

axarr[0].plot(X_np)
axarr[0].set_xlabel('T')
axarr[0].set_title('Entire time series')
axarr[1].plot(X_np[:50])
axarr[1].set_xlabel('T')
axarr[1].set_title('First 50 time points')
plt.tight_layout()
plt.show()

cmlp = cMLP(X.shape[-1], lag=5, hidden=[100]).cuda(device=device)

train_loss_list = train_model_ista(
    cmlp, X, lam=0.002, lam_ridge=1e-2, lr=5e-2, penalty='H', max_iter=50000,
    check_every=100)

# train_loss_list = train_model_adam(cmlp, X, lr=0.002, max_iter=5000, lam=0, lam_ridge=1e-2, penalty='H',
#                      lookback=5, check_every=100, verbose=1)

# plt.figure(figsize=(8, 5))
# train_loss_list = train_loss_list.cpu()
# plt.plot(50 * np.arange(len(train_loss_list)), train_loss_list)
# plt.title('cMLP training')
# plt.ylabel('Loss')
# plt.xlabel('Training steps')
# plt.tight_layout()
# plt.show()

GC_est = cmlp.GC().cpu().data.numpy()

print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))

# Make figures
fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
axarr[0].imshow(GC, cmap='Blues')
axarr[0].set_title('GC actual')
axarr[0].set_ylabel('Affected series')
axarr[0].set_xlabel('Causal series')
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
axarr[1].set_title('GC estimated')
axarr[1].set_ylabel('Affected series')
axarr[1].set_xlabel('Causal series')
axarr[1].set_xticks([])
axarr[1].set_yticks([])

# Mark disagreements
for i in range(len(GC_est)):
    for j in range(len(GC_est)):
        if GC[i, j] != GC_est[i, j]:
            rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
            axarr[1].add_patch(rect)

plt.show()