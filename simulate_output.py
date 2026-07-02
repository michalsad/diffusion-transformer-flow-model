import os
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from typing import Optional, List

from simulator import CFGVectorFieldODE, EulerSimulator


@torch.no_grad()
def visualize_output(model, device, samples_per_class: int = 10, 
                     num_timesteps: int = 100, 
                     guidance_scales: List[float] = [3.0], 
                     save_path: Optional[str] = None, use_tqdm: bool = True):
    # Graph
    fig, axes = plt.subplots(1, len(guidance_scales), 
                             figsize=(10 * len(guidance_scales), 10), 
                             squeeze=False)
    axes = axes[0]

    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(model, guidance_scale=w, null_label=10)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        y = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        num_samples = y.shape[0]
        x0 = torch.randn(num_samples, 1, 32, 32).to(device) # (num_samples, 1, 32, 32)

        # Simulate
        ts = torch.linspace(0,0.999,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y, use_tqdm=use_tqdm)

        # Plot
        v_min, v_max = x1.min(), x1.max()
        x1 = (x1 - v_min) / (v_max - v_min)
        grid = make_grid(x1, nrow=samples_per_class, normalize=True, 
                         value_range=(0,1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)

    # Save
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def checkpoint(model, optimizer, step, output_dir, device):
    # Save model
    torch.save(model.state_dict(),
               os.path.join(output_dir, f'step_{step:06d}_model.pt'))
    torch.save(optimizer.state_dict(),
               os.path.join(output_dir, f'step_{step:06d}_opt.pt'))

    # Save output visualization
    visualize_output(
        model,
        device=device,
        save_path=os.path.join(output_dir, f'step_{step:06d}_output.png'),
        use_tqdm=False
    )