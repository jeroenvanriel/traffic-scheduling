from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class TqdmCallback(BaseCallback):
    """
    Progress bar for SB3 training with tqdm.
    """
    def __init__(self, total_timesteps, update_interval=1000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_interval = update_interval
        self.pbar = None

    def _on_training_start(self):
        # Initialize tqdm
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self):
        # Update every update_interval timesteps
        if self.n_calls % self.update_interval == 0:
            self.pbar.update(self.update_interval)
        return True

    def _on_training_end(self):
        # Close tqdm at the end
        self.pbar.update(self.total_timesteps - self.pbar.n)
        self.pbar.close()


import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 14,
    'axes.labelsize': 18,
    # 'axes.titlesize': 18,
    'legend.fontsize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

textwidth_pt = 398.3386
inches_per_pt = 1.0 / 72.27
textwidth_inches = textwidth_pt * inches_per_pt 

def get_figsize(height_ratio=0.75):
    """
    Convert LaTeX textwidth to matplotlib figsize
    height_ratio: height as fraction of width (0.75 is good for subplots)
    """
    inches_per_pt = 1.0 / 72.27
    width_inches = textwidth_pt * inches_per_pt
    height_inches = width_inches * height_ratio
    return (width_inches, height_inches)

def format_duration(seconds: float) -> str:
    """Format time as (18s) or (2m) or (6h), depending on size (with rounding up/down to closest integer)."""
    seconds = round(seconds)
    if seconds < 60:
        return f"({seconds}s)"
    elif seconds < 3600:
        minutes = round(seconds / 60)
        return f"({minutes}m)"
    else:
        hours = round(seconds / 3600)
        return f"({hours}h)"
