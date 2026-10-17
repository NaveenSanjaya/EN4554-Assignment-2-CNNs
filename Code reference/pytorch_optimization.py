
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from mpl_toolkits import mplot3d

# ------------------------
# Cost function (Beale)
# ------------------------
class Beale:
    def __init__(self, xmin=-1.5, xmax=5., ymin=-3., ymax=1.5):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.x_optimum, self.y_optimum, self.z_optimum = 3.0, 0.5, 0.0  # Global optimum

    def __call__(self, x, y):
        return torch.log(
            1 + (1.5 - x + x * y) ** 2
              + (2.25 - x + x * (y ** 2)) ** 2
              + (2.625 - x + x * (y ** 3)) ** 2
        ) / 10.0

beale = Beale()
trajectories = {}

def _append_traj(xtraj, ztraj, x, z):
    xtraj = torch.cat([xtraj, x.detach().clone().unsqueeze(1)], dim=1)
    ztraj = torch.cat([ztraj, z.detach().reshape(1)], dim=0)
    return xtraj, ztraj

# ------------------------
# Optimizers
# ------------------------
def gd(x, lr=0.01, steps=1500, verbose=False):
    x = x.clone().detach().requires_grad_(True)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        with torch.no_grad():
            x -= lr * x.grad
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["GD"] = (xtraj.numpy(), ztraj.numpy(), 'lime')

def gd_noisy(x, lr=0.01, steps=1500, verbose=False):
    x = x.clone().detach().requires_grad_(True)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        gradients = x.grad.detach().clone() + torch.normal(0, 2.0, size=x.shape, dtype=x.dtype)
        with torch.no_grad():
            x -= lr * gradients
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["GD with noise"] = (xtraj.numpy(), ztraj.numpy(), 'blue')

def gdm(x, lr=0.01, beta=0.9, steps=1500, verbose=False):
    x = x.clone().detach().requires_grad_(True)
    v = torch.zeros_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        with torch.no_grad():
            v = beta * v + lr * x.grad
            x -= v
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["GD with momentum"] = (xtraj.numpy(), ztraj.numpy(), 'orange')

def gd_nesterov(x, lr=0.01, momentum=0.9, steps=1500, verbose=False):
    v = torch.zeros_like(x)
    x = x.clone().detach()
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        xla = (x - momentum * v).detach().requires_grad_(True)
        z = beale(xla[0, 0], xla[1, 0])
        z.backward()
        gradients = xla.grad.detach().clone()
        with torch.no_grad():
            v = momentum * v + lr * gradients
            x -= v
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
    trajectories["GD Nesterov"] = (xtraj.numpy(), ztraj.numpy(), 'teal')

def adagrad(x, lr=0.01, steps=1500, verbose=False):
    eps = 1e-7
    x = x.clone().detach().requires_grad_(True)
    grad_2nd_mom = torch.zeros_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        gradients = x.grad.detach().clone()
        grad_2nd_mom += gradients.pow(2)
        with torch.no_grad():
            x -= lr * gradients / (grad_2nd_mom.sqrt() + eps)
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["AdaGrad"] = (xtraj.numpy(), ztraj.numpy(), 'sienna')

def rmsprop(x, lr=0.01, decay_rate=0.9, steps=1500, verbose=False):
    eps = 1e-7
    x = x.clone().detach().requires_grad_(True)
    grad_2nd_mom = torch.zeros_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        gradients = x.grad.detach().clone()
        grad_2nd_mom = decay_rate * grad_2nd_mom + (1 - decay_rate) * gradients.pow(2)
        with torch.no_grad():
            x -= lr * gradients / (grad_2nd_mom.sqrt() + eps)
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["RMSProp"] = (xtraj.numpy(), ztraj.numpy(), 'yellow')

def adadelta(x, lr=0.01, decay_rate=0.9, steps=1500, verbose=False):
    eps = 1e-7
    x = x.clone().detach().requires_grad_(True)
    grad_2nd_mom = torch.zeros_like(x)
    update_2nd_ord = torch.ones_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        gradients = x.grad.detach().clone()
        grad_2nd_mom = decay_rate * grad_2nd_mom + (1 - decay_rate) * gradients.pow(2)
        update = gradients * (update_2nd_ord + eps).sqrt() / (grad_2nd_mom + eps).sqrt()
        update_2nd_ord = decay_rate * update_2nd_ord + (1 - decay_rate) * update.pow(2)
        with torch.no_grad():
            x -= lr * update
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["AdaDelta"] = (xtraj.numpy(), ztraj.numpy(), 'magenta')

def adam(x, lr=0.01, beta1=0.9, beta2=0.999, steps=1500, verbose=False):
    eps = 1e-7
    x = x.clone().detach().requires_grad_(True)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        g = x.grad.detach().clone()
        t = i + 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.pow(2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        with torch.no_grad():
            x -= lr * m_hat / (v_hat.sqrt() + eps)
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["Adam"] = (xtraj.numpy(), ztraj.numpy(), 'red')

def adamax(x, lr=0.01, beta1=0.9, beta2=0.999, steps=1500, verbose=False):
    eps = 1e-7
    x = x.clone().detach().requires_grad_(True)
    m = torch.zeros_like(x)
    u = torch.zeros_like(x)
    xtraj = x.detach().clone().unsqueeze(1)
    ztraj = torch.tensor([], dtype=x.dtype)
    for i in range(steps):
        z = beale(x[0, 0], x[1, 0])
        z.backward()
        g = x.grad.detach().clone()
        t = i + 1
        m = beta1 * m + (1 - beta1) * g
        u = torch.maximum(beta2 * u, g.abs())
        m_hat = m / (1 - beta1 ** t)
        with torch.no_grad():
            x -= lr * m_hat / (u + eps)
        xtraj, ztraj = _append_traj(xtraj, ztraj, x, z)
        x.grad.zero_()
    trajectories["AdaMax"] = (xtraj.numpy(), ztraj.numpy(), 'navy')

# ------------------------
# Plots & Animations
# ------------------------
def plot_trajectories():
    f2, ax2 = plt.subplots(figsize=(8, 8))
    x = np.arange(beale.xmin, beale.xmax, 0.1)
    y = np.arange(beale.ymin, beale.ymax, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([beale(torch.tensor(xi, dtype=torch.float32),
                         torch.tensor(yi, dtype=torch.float32)).item()
                   for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax2.contourf(X, Y, Z, cmap=plt.get_cmap("jet"), levels=np.linspace(zs.min(), zs.max(), 50))
    ax2.text(beale.x_optimum, beale.y_optimum, "+", color="w", size=20)
    for name, (xtraj, ztraj, color) in trajectories.items():
        ax2.plot(xtraj[0, :], xtraj[1, :], color=color, linewidth=2, label=name)
        ax2.scatter(xtraj[0, [0, -1]], xtraj[1, [0, -1]], color=color, s=50)
    ax2.legend()
    return [f2, ax2]

def plot_cost_function_3d(cost_f, grain=0.02, figsize=(10, 6)):
    x_grid = np.arange(cost_f.xmin, cost_f.xmax, grain)
    y_grid = np.arange(cost_f.ymin, cost_f.ymax, grain)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([cost_f(torch.tensor(xi, dtype=torch.float32),
                         torch.tensor(yi, dtype=torch.float32)).item()
                  for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, shade=False,
                    rstride=1, cstride=1, linewidth=0, alpha=0.7, zorder=1)
    ax.patch.set_facecolor('white')
    ax.view_init(elev=30., azim=70)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 15
    for name, (xtraj, ztraj, color) in trajectories.items():
        ax.plot(xtraj[0, :1000], xtraj[1, :1000], ztraj[:1000] + 2.0, color=color, linewidth=3, label=name)
        ax.scatter(xtraj[0, -1], xtraj[1, -1], ztraj[-1], color=color, s=50)
    ax.legend()
    return fig, ax

def generate_animation(frames=100, outfile="animation.gif", writer="pillow"):
    f, ax = plt.subplots(figsize=(8, 8))
    x = np.arange(beale.xmin, beale.xmax, 0.1)
    y = np.arange(beale.ymin, beale.ymax, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([beale(torch.tensor(xi, dtype=torch.float32),
                         torch.tensor(yi, dtype=torch.float32)).item()
                   for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.contourf(X, Y, Z, cmap=plt.get_cmap("jet"), levels=np.linspace(zs.min(), zs.max(), 50))
    ax.text(beale.x_optimum, beale.y_optimum, "+", color="w", size=20)

    dots = []
    max_frames = min(frames, min(xtraj.shape[1] for xtraj, _, _ in trajectories.values()))

    def update(frame_number):
        nonlocal dots
        for sc in dots:
            sc.remove()
        dots = []
        for name, (xtraj, ztraj, color) in trajectories.items():
            ax.plot(xtraj[0, :frame_number], xtraj[1, :frame_number], color=color, linewidth=2, label=name)
            k = ax.scatter(xtraj[0, frame_number-1], xtraj[1, frame_number-1], color=color, s=50)
            dots.append(k)
        ax.legend(trajectories.keys(), loc='upper left')

    animation = FuncAnimation(f, update, interval=1, frames=max_frames)
    if writer == "pillow":
        animation.save(outfile, writer=PillowWriter(fps=30))
    elif writer == "ffmpeg":
        animation.save(outfile, writer="ffmpeg", dpi=80)

def generate_animation_3d(frames=100, outfile="animation3d.gif", writer="pillow"):
    f, ax = plot_cost_function_3d(beale)
    dots = []
    max_frames = min(frames, min(xtraj.shape[1] for xtraj, _, _ in trajectories.values()))

    def update(frame_number):
        nonlocal dots
        for sc in dots:
            sc.remove()
        dots = []
        for name, (xtraj, ztraj, color) in trajectories.items():
            ax.plot(xtraj[0, :frame_number], xtraj[1, :frame_number], ztraj[:frame_number],
                    color=color, linewidth=2, label=name)
            k = ax.scatter(xtraj[0, frame_number-1], xtraj[1, frame_number-1], ztraj[frame_number-1], color=color, s=50)
            dots.append(k)
        ax.legend(trajectories.keys(), loc='upper left')

    animation = FuncAnimation(f, update, interval=1, frames=max_frames)
    if writer == "pillow":
        animation.save(outfile, writer=PillowWriter(fps=30))
    elif writer == "ffmpeg":
        animation.save(outfile, writer="ffmpeg", dpi=80)

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    lr = 0.01
    steps = 200
    x0 = torch.tensor([[2.4], [-2.8]], dtype=torch.float32)
    gd(x0.clone(), lr=lr, steps=steps)
    gdm(x0.clone(), lr=lr, steps=steps)
    adagrad(x0.clone(), lr=lr, steps=steps)
    rmsprop(x0.clone(), lr=lr, decay_rate=0.99, steps=steps)
    plot_trajectories()
    plt.show()
    generate_animation(frames=300, outfile="anim.gif")
    # generate_animation_3d(frames=200, outfile="anim3d.gif")
