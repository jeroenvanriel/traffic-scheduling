import matplotlib.animation as animation
from network.util import draw_vehicles, get_pos


def animate(instance, trajectories, dt, vehicle_l=2, vehicle_w=1):

    rects = draw_vehicles(G, route, positions)

    def update(frame):
        for l in range(N):
            for k in range(n[l]):
                t0, trajectory = trajectories[l][k]
                i = frame - int(t0 / dt)
                if i < 0: # not yet started
                    continue
                if i >= len(trajectory): # done
                    rects[l][k].set_visible(False)
                    continue

                pos = trajectory[min(i, len(trajectory) - 1)]
                x, y, angle = get_pos(l, k, pos, vehicle_l, vehicle_w)
                rects[l][k].set_x(x)
                rects[l][k].set_y(y)
                rects[l][k].set_angle(angle)

    # compute amount of frames required
    frames  = -1
    for l in range(N):
        for k in range(n[l]):
            t0, trajectory = trajectories[l][k]
            frames = max(frames, int(t0 / dt) + len(trajectory))

    ax.axis('off')
    ax.set_aspect('equal')
    fig.tight_layout(pad=0.05)

    return animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
