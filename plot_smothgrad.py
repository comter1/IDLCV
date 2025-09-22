import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -------- helpers --------
def _to_numpy(img_t):
    # img_t: [C,H,W] in model space (already normalized)
    x = img_t.detach().cpu()
    # scale to [0,1] for display without de-normalizing (purely for visualization of saliency heatmap)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x.permute(1,2,0).numpy()

def _saliency_to_heatmap(grad):
    # grad: [C,H,W] -> aggregate channels, abs and normalize to [0,1]
    g = grad.abs().mean(dim=0)
    g = g / (g.max() + 1e-8)
    return g.detach().cpu().numpy()  # [H,W] in [0,1]

@torch.no_grad()
def predict_class(model, x):
    # x: [B,C,H,W]
    logits = model(x)
    return logits.argmax(dim=1)  # [B]

def vanilla_saliency(model, x, target_idx):
    """
    x: [B,C,H,W] requires_grad=False (we'll clone+set it)
    target_idx: [B] long
    returns saliency: [B, H, W] in [0,1]
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    # gather scores of target class
    scores = logits.gather(1, target_idx.view(-1,1)).squeeze(1)
    grads = torch.autograd.grad(outputs=scores, inputs=x, grad_outputs=torch.ones_like(scores), create_graph=False, retain_graph=False)[0]
    # convert to heatmap per sample
    sal = torch.stack([grads[i].abs().mean(dim=0) for i in range(grads.size(0))], dim=0)  # [B,H,W]
    # normalize each sample to [0,1]
    sal = sal / (sal.view(sal.size(0), -1).max(dim=1)[0].view(-1,1,1) + 1e-8)
    return sal

def smoothgrad_saliency(model, x, target_idx, n_samples=25, noise_std=0.15, clip=True):
    """
    x: [B,C,H,W] (normalized inputs)
    target_idx: [B]
    n_samples: number of noisy samples to average
    noise_std: std of Gaussian noise added in model's input space
    """
    model.eval()
    B, C, H, W = x.shape
    smooth = torch.zeros(B, H, W, device=x.device)
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        if clip:
            # if model expects normalized range, clipping is optional; leave if unsure
            pass
        x_noisy = x_noisy.clone().detach().requires_grad_(True)
        logits = model(x_noisy)
        scores = logits.gather(1, target_idx.view(-1,1)).squeeze(1)
        grads = torch.autograd.grad(outputs=scores, inputs=x_noisy, grad_outputs=torch.ones_like(scores), create_graph=False, retain_graph=False)[0]
        # accumulate |grad| mean over channels
        g = grads.abs().mean(dim=1)  # [B,H,W]
        smooth += g
    smooth = smooth / n_samples
    # normalize each sample to [0,1]
    smooth = smooth / (smooth.view(B, -1).max(dim=1)[0].view(-1,1,1) + 1e-8)
    return smooth  # [B,H,W] in [0,1]

def plot_saliency_triplet(x, sal, sgrad, titles=("Original","Vanilla Grad","SmoothGrad"), max_items=3):
    """
    x: [B,C,H,W] (normalized input)
    sal: [B,H,W] vanilla
    sgrad: [B,H,W] smoothgrad
    """
    B = min(x.size(0), max_items)
    fig, axes = plt.subplots(B, 3, figsize=(9, 3*B))
    if B == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(B):
        # original (scaled for display)
        axes[i,0].imshow(_to_numpy(x[i]))
        axes[i,0].set_title(titles[0]); axes[i,0].axis('off')
        # vanilla
        axes[i,1].imshow(sal[i].detach().cpu().numpy(), cmap='inferno')
        axes[i,1].set_title(titles[1]); axes[i,1].axis('off')
        # smoothgrad
        axes[i,2].imshow(sgrad[i].detach().cpu().numpy(), cmap='inferno')
        axes[i,2].set_title(titles[2]); axes[i,2].axis('off')
    plt.tight_layout()
    plt.show()

# -------- usage example (assuming you have a model and a val_loader) --------
def plot_smoothgrad_saliency_map(model, val_loader, device, n_samples=25, noise_std=0.15, k=3):
    model.to(device).eval()
    # get one batch
    x, y = next(iter(val_loader))  # transforms already applied (including Normalize)
    x, y = x.to(device), y.to(device)

    # choose targets: predicted class (common choice) or ground-truth
    y_pred = predict_class(model, x)
    targets = y_pred  # or y

    # compute saliency maps
    sal = vanilla_saliency(model, x, targets)                  # [B,H,W]
    sgrad = smoothgrad_saliency(model, x, targets, n_samples, noise_std)  # [B,H,W]

    # plot a few examples
    plot_saliency_triplet(x, sal, sgrad, max_items=k)
