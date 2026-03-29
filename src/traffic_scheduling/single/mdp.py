import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from colors import pastel_cmap 


def current_route(base_env):
    eta = base_env.route_order
    if all(base_env.done):
        return eta[-1] # does not matter

    if len(eta) == 0 or base_env.done[eta[-1]]:
        # empty schedule or last scheduled route done?
        # take pending route with earliest arrival
        LB, k = base_env.LB, base_env.k
        lbs = []
        for r in range(base_env.R):
            if not base_env.done[r]:
                lbs.append((r, LB[r, k[r]]))
        return sorted(lbs, key=lambda i: i[1])[0][0]
    else:
        # take last scheduled route
        return eta[-1]
 

class SingleScheduleEnv(gym.Env):
    """Constructive scheduling MDP for crossing time scheduling in single intersection.
    Each state encodes a (partial) disjunctive graph augmented with crossing time lower bounds.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, instance=None, instance_generator=None,
                 dense_rewards=True, options=None):
        """Either specify a single instance, or an instance_generator"""
        super().__init__()
        self.dense_rewards = dense_rewards
        self.options = options or {}
        self.render_mode = "human"

        if instance is not None:
            # single instance specified
            self.instance = instance
        else:
            assert instance_generator is not None, "Either specify instance, or instance_generator."
            self.instance_generator = instance_generator
            # get a sample for inspecting the (fixed) number of routes
            instance = instance_generator()

        # the number of routes must be fixed!
        self.R = instance.R

        # routes are actions
        self.action_space = spaces.Discrete(self.R)
        # observation space will be defined in wrapper
        self.observation_space = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            self.options = options

        if hasattr(self, 'instance_generator'):
            self.instance = self.instance_generator()

        # switch-over time and processing time
        self.switch = self.instance.switch
        self.rho = self.instance.rho

        # number of: routes (R); vehicles per route (n); number of scheduled vehicles per route (k)
        # note that (r, self.k[r]) is the first unscheduled vehicle on route r
        self.R = self.instance.R
        self.n = self.instance.n
        self.N = self.instance.N
        self.k = [0 for _ in range(self.R)]

        self.t = 0 # count number of MDP steps
        self.done = np.array([False for _ in range(self.R)], dtype=bool)
        self.vehicle_order = []
        self.route_order = []

        # initial lower bounds are earliest crossing times, because we assume
        # that earliest crossing times satisfy a_i + \rho <= a_j for all
        # conjunctive pairs (i, j) \in \mathcal{C}
        max_k = max(len(arrivals) for arrivals in self.instance.arrivals)
        self.LB = np.zeros((self.R, max_k))
        for r in range(self.R):
            self.LB[r, :self.n[r]] = self.instance.arrivals[r]

        # keep track of which LBs needed update, for rendering
        self.LB_updated = []

        # initial total crossing time sum
        self.LB0 = self.LB.sum()

        return self._get_obs(), {}

    def step(self, action):
        if self.done[action]:
            raise gym.error.InvalidAction(f"Invalid action: all vehicle from route {action} are already scheduled.")
        action = int(action)
        vehicle = (action, self.k[action])
        self.route_order.append(action)
        self.vehicle_order.append(vehicle)

        # update lower bounds
        # also keep track of increase, to compute dense reward
        # and keep track of which LBs where updated, for rendering
        root_lb = self.LB[*vehicle] + self.rho + self.switch
        LB_inc = 0
        self.LB_updated = []
        # for all other routes
        for r in set(range(self.R)) - {action}:
            # we change the 'lb' to do conjunctive chain propagation
            lb = root_lb
            # for all unscheduled vehicles
            for k in range(self.k[r], self.n[r]):
                if lb > self.LB[r, k]:
                    # actual update necessary
                    LB_inc += lb - self.LB[r, k]
                    self.LB[r, k] = lb
                    self.LB_updated.append((r, k))
                    lb = self.LB[r, k] + self.rho
                else:
                    # stop propagating updates
                    break

        # update counts and flags
        self.t += 1
        self.k[action] += 1
        if self.k[action] == self.n[action]:
            self.done[action] = True
        done = np.all(self.done)

        reward = 0
        if self.dense_rewards:
            # dense reward is change in total sum of crossing time lower bounds
            reward = -LB_inc
        elif done:
            # sparse rewards means only final reward, corresponding to
            # increase of total sum of crossing time lower bounds
            reward = -(self.LB.sum() - self.LB0)

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return {
            'R': self.R,
            'n': self.n,
            'k': self.k,
            'vehicle_order': self.vehicle_order.copy(),
            'route_order': self.route_order.copy(),
            'done': self.done.copy(),
            'lb': self.LB.copy(),
            'partial_makespan': self.makespan(partial=True),
        }
        
    def makespan(self, partial=False):
        """Get the latest "LB+rho" of every vehicle (partial=False),
        or every *scheduled* vehicle (partial=True)."""
        m = 0
        for r in range(self.R):
            if partial:
                if self.k[r] == 0: # this route has no scheduled vehicles
                    continue
                k = self.k[r] - 1
            else:
                k = self.n[r] - 1 # last vehicle on route
            t = self.LB[r][k] + self.rho
            if t > m:
                m = t
        return m

    def render(self):
        empty    = len(self.route_order) == 0
        complete = len(self.route_order) == self.N
        partial = not empty and not complete

        # some configurable settings:
        rolled_view = self.options.get('render_roll', False)
        collapse_current = self.options.get('collapse_current', True)
        no_axis = self.options.get('no_axis', False)
        dimmed_partial = self.options.get('dimmed_partial', True)
        block_text = self.options.get('block_text', True)
        arrow_text = self.options.get('arrow_text', True)
        one_based_indices = self.options.get('one_based_indices', False)
        extra_time_lines = self.options.get('extra_time_lines', [])
        if self.options.get('tex', False): # use TeX for text
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
        plt.rcParams['axes.linewidth'] = .5
        plt.rcParams['lines.linewidth'] = .5
        plt.rcParams['patch.linewidth'] = .5

        # some fixed settings
        partial_alpha = 0.4  # for "dimmed" partial schedule
        lw = 0.5             # overall linewidth
        lw_rect = 0.5        # linewidth for vehicle boxes
        block_height = 0.6   # fixed height of vehicle boxes (relative to row)
        figure_w_scale = 0.8 # overall width scaling factor
        figure_h_scale = 0.4 # overall height scaling factor
        bottom_margin = 0.5  # margin between bottom row and x-axis

        # 0a. permute the route indices ("roll") such that the "current route" is on top
        # "current route" determines amout to "roll", and is equal to
        # - last scheduled route
        # - route with earliest arrival (for empty schedules)
        routes = list(range(self.R))
        if rolled_view:
            roll = current_route(self.unwrapped)
            routes = routes[roll:] + routes[:roll]
        
        # 0b. determine dimensions of figure
        # whether to actually collapse
        collapse_current = collapse_current and partial and not self.done[routes[0]]
        # x: determine number of required rows
        rows = (~self.done).astype(int).sum() + int(not empty) - int(collapse_current)
        # y: determine the end time ("makespan") of the schedule
        end = self.options.get('fixed_end', self.makespan() + 0.1)

        # 0c. new figure with single axis
        w, h = end, rows
        facecolor = (1,1,1) # white
        fig = plt.figure(figsize=(w * figure_w_scale, h * figure_h_scale), facecolor=facecolor)
        # add main ax
        ax = fig.add_axes([0, 0, 1, 1], facecolor=facecolor) # ax takes up full figure
        ax.set_xlim([-0.01, w])
        ax.set_ylim([- rows + 1 - bottom_margin, block_height + 0.01])

        # for drawing "switch"-arrow
        arrows, arrows_dim = [], []
        def arrow(start, end, dim=False):
            if dim: # deemphasized or "dimmed"
                arrows_dim.append(start)
            else:
                arrows.append(start)

            # add text above arrow
            if arrow_text:
                midpoint = ((start[0]+end[0])/2, (start[1]+end[1])/2)
                ax.text(midpoint[0], midpoint[1]+0.10, r"$\delta$", ha='center', fontsize=8, color='black', alpha=partial_alpha if dim else 1)

        # helper for single row of vehicles
        def draw_row(veh_indices, y=0, color=None, edgecolor='k',
                     dim=False, arrivals=False, final_switch=False):
            if color is not None: cmap = lambda r: color
            else: cmap = lambda r: list(pastel_cmap(r))

            for (r, k) in veh_indices:
                width = self.rho
                start = self.LB[r, k]

                color = cmap(r)
                linestyle = '-'
                if dim: # lighten colors, to deemphasize
                    color[-1] = partial_alpha
                    # edgecolor = (0,0,0,partial_alpha)
                    linestyle = '--'

                rect = Rectangle((start, y), width=width, height=block_height,
                                linewidth=lw_rect, facecolor=color, edgecolor=edgecolor, linestyle=linestyle)
                ax.add_patch(rect)

                # lower bound number (or arrivals)
                if block_text:
                    ri = r; ki = k
                    if one_based_indices:
                        ri += 1
                        ki += 1
                    text_i = f"{ri}{'' if ri <= 9 and k <= 9 else ','}{ki}"
                    if all(self.done):
                        text = f"$y_{{{text_i}}} = {start:.2f}$"
                    elif arrivals:
                        text = f"$a_{{{text_i}}} = {start:.2f}$"
                    else:
                        text = f"$\\beta_{{{text_i}}}(s_{{{self.t}}}) = {start:.2f}$"
                    ax.text(start + 0.08, y + 0.18, text, fontsize=9)

                # draw "switch"-arrow between scheduled vehicles from different routes
                veh = self.vehicle_order
                if (r, k) not in veh:
                    continue
                i = veh.index((r, k))
                eta = self.route_order
                if i + 1 >= len(veh) or eta[i] == eta[i+1]:
                    continue
                y_arrow = block_height / 2
                arrow((start + width, y_arrow), (start + width + self.switch, y_arrow), dim=dim)

        # 1. plot current partial schedule
        draw_row(self.vehicle_order, dim=False, final_switch=True)

        # 2. draw vertical dashed line at "current time"
        # this only makes sense if the schedule is strictly "partial" (so neither empty or complete)
        # otherwise the schedule is "empty"
        if not empty and not complete: 
            tc = self.makespan(partial=True)
            tc_margin = 0.4

            # when we "roll" the routes, let the "current time" line start at the second row
            if rolled_view: first_route = -1
            else: first_route = 0

            # calculate the vertical top and bot(tom) of the line
            ytop = first_route + block_height / 2 + tc_margin
            ybot = -rows + 1 + block_height / 2 - tc_margin
            line = Line2D((tc, tc), (ytop, ybot), color='black', linewidth=0.5, linestyle='--')
            line.set_dashes([10, 5])
            ax.add_line(line)

            for t in extra_time_lines:
                line = Line2D((t, t), (ytop, ybot), color='black', linewidth=0.5, linestyle='--')
                line.set_dashes([10, 5])
                ax.add_line(line)

        # 3. plot all unscheduled vehicles per route, each on a separate row
        row = - int(not empty) + int(collapse_current)
        for r in routes:
            indices = [(r, k) for k in range(self.k[r], self.n[r])]
            if not indices:
                continue # skip routes that are done
            draw_row(indices, y=row, dim=dimmed_partial, arrivals=empty)
            
            # draw "switch"-arrows (only for non-empty schedule and in "rolled view")
            if rolled_view and not empty and not complete:
                if r != routes[0]:
                    y = row + block_height / 2
                    arrow((tc, y), (tc + self.switch, y))

            # move to next row (so down)
            row -= 1

        # 4. draw all "switch"-arrows as matplotlib "quivers"
        # (These look nicer than the "arrow" patches, because these are actual polygons,
        #  whereas the "arrow" patches are composed of lines. The only downside is that we need
        #  to recompute the width of the quiver, because this uses a different unit system.)
        ax_width_inch = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        q_kwargs = { 'width': lw / 72 / ax_width_inch, 'headwidth': 6, 'headlength':10, 'headaxislength': 9, 'angles': 'xy', 'scale_units': 'x', 'scale': 1 }
        eps = 0.005 # slightly shift arrows right
        if arrows:
            arrows = np.array(arrows).reshape(-1, 2).T
            ax.quiver(arrows[0] + eps, arrows[1], self.switch - eps, 0, **q_kwargs)
        if arrows_dim:
            arrows_dim = np.array(arrows_dim).reshape(-1, 2).T
            ax.quiver(arrows_dim[0] + eps, arrows_dim[1], self.switch - eps, 0, alpha=partial_alpha, **q_kwargs)

        # 5. clean-up figure and fine-tuning
        # x-axis: integer ticks only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # y-axis: hide
        ax.yaxis.set_visible(False)
        # hide all spines except the bottom (x-axis)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        if no_axis:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
        # change x-axis line and ticks color
        # ax.spines['bottom'].set_color('gray')
        # ax.tick_params(axis='x', colors='gray')

        # ouput to pdf (optionally)
        if 'out' in self.options:
            plt.savefig(self.options['out'].format(t=str(self.t)), format="pdf", bbox_inches="tight")


class HorizonObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, max_k=None):
        """Produces the "horizon" observation, which is a "ragged array", encoded as a dict with
        - `'h_lengths'` with shape `(R,)`, listing the number of entries in each route horizon
        - `'horizon'` with shape `(R, max_k)`, each row is a route horizon

        If given, parameter `max_k` determines the maximum size of each single
        route horizon. If it is not given, we try to deduce it from the instance (generator) of
        the base environment."""
        super().__init__(env)
        self.R = self.unwrapped.R # fixed

        if max_k is None:
            # try to determine second dimension max size otherwise
            if hasattr(self.unwrapped, 'instance'):
                s = self.unwrapped.instance
            else:
                s = self.unwrapped.instance_generator()
            self.max_k = max(len(a) for a in s.arrivals)
        else:
            self.max_k = max_k
 
        self.observation_space = gym.spaces.Dict({
            "horizon": gym.spaces.Box(low=0, high=np.inf, shape=(self.R, self.max_k), dtype=np.float32),
            "h_lengths": gym.spaces.Box(low=0, high=self.max_k, shape=(self.R,), dtype=np.int16),
        })

    def observation(self, obs):
        n, k, lb = obs['n'], obs['k'], obs['lb']

        # compute "current time"
        if all(obs['done']):
            current_time = obs['partial_makespan']
        else:
            first = []
            for r in range(self.R):
                horizon = lb[r, k[r]:]
                if len(horizon) > 0:
                    first.append(horizon[0])
            
            if len(first) >= 2:
                first = sorted(first)
                current_time = min(first[0], first[1] - self.unwrapped.switch)
            else:
                current_time = first[0]

        # new observation is "ragged array"
        # - horizon has shape (R, max_k)
        # - h_lengths contains the number of entries in each route horizon
        horizon = np.zeros((self.R, self.max_k), dtype=np.float32)
        h_lengths = np.zeros(self.R, dtype=np.int16)
        for r in range(self.R):
            n_unscheduled = min(n[r] - k[r], self.max_k)
            h_lengths[r] = n_unscheduled
            horizon[r, :n_unscheduled] = lb[r, k[r]:k[r] + n_unscheduled] - current_time

        horizon_obs = {
            'horizon': horizon,
            'h_lengths': h_lengths
        }
        self.last_obs = horizon_obs
        return horizon_obs

    def render(self):
        horizon, h_lengths = self.last_obs['horizon'], self.last_obs['h_lengths']
        rho = self.unwrapped.rho

        end = horizon.max() + rho # maximum y value in the plot
        _, ax = plt.subplots(figsize=(end, len(horizon)))

        # some options:
        lw = 0.8 # linewidth
        height = 0.5 # fixed height of vehicle rectangles

        # plot all unscheduled vehicles, on a separate row for each route
        row = 0
        for r, h_length in enumerate(h_lengths):
            h = horizon[r]
            # plot unscheduled vehicles at their translated "horizon" location
            for k in range(int(h_length)):
                width = rho
                start = h[k]
                rect = Rectangle((start, row), width=width, height=height, linewidth=lw, facecolor='lightgray', edgecolor='k')
                ax.add_patch(rect)

                # lower bound number
                ax.text(start + 0.20, row + 0.18, f"{start:.2f}", fontsize=14)

            # move to next row (so down)
            row -= 1

        ax.set_xlim([-0.05, end + 0.05])
        ax.margins(y=0.15)

        plt.tight_layout()
        # ax.axis('off')

        # hide all spines except the bottom (x-axis)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        # hide y-axis ticks and labels
        ax.yaxis.set_visible(False)

        ax.set_aspect('equal', adjustable='box')


class HorizonRollingWrapper(gym.Wrapper):
    """Permute the routes ("roll") such that the "current route" is the first in the
    list of route horizons. The "current route" determines amout to "roll", and is equal to
    - last scheduled route if it is not done, or
    - route with earliest arrival (when last route is done or for empty schedules).
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._observation(obs)  # transform initial observation!
        return obs, info

    def step(self, action):
        action = self._action(action)
        obs, *rest = self.env.step(action)
        return self._observation(obs), *rest
    
    def _observation(self, obs):
        if all(self.unwrapped.done):
            return obs # does not matter

        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]

        # permute the routes according to "current route"
        c = current_route(self.unwrapped)
        roll = -active_routes.index(c)

        # first list the pending routes
        pending_horizon = obs['horizon'][active_routes]
        pending_h_lengths = obs['h_lengths'][active_routes]

        # fill the rest with zeros
        n_done = sum(self.unwrapped.done)
        zeros = np.zeros((n_done, obs['horizon'].shape[1]))

        return {
            'horizon':   np.vstack([np.roll(pending_horizon, roll, axis=0), zeros]),
            'h_lengths': np.concatenate([np.roll(pending_h_lengths, roll, axis=0), np.zeros(n_done)]),
        }

    def _action(self, action):
        c = current_route(self.unwrapped)
        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]

        # action = number of routes to advance from current route (while "wrapping" around the end)
        current_ix = active_routes.index(c)
        action = min(action, len(active_routes) - 1) # prevent unnecessary "wrapping around"
        return active_routes[(current_ix + action) % len(active_routes)]
    
    def _inverse_action(self, route):
        c = current_route(self.unwrapped)
        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]

        # inverse action = number of advances to take from current route to arrive at "route"
        current_ix = active_routes.index(c)
        next_ix = active_routes.index(route)
        action = next_ix - current_ix
        return action if action >= 0 else action + len(active_routes)


class NewHorizonObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, gaps=False):
        super().__init__(env)
        self.gaps = gaps
        self.R = self.unwrapped.R # fixed number of routes
    
    def observation(self, obs):
        n, k, lb = obs['n'], obs['k'], obs['lb']

        # compute "current time"
        if all(obs['done']):
            current_time = obs['partial_makespan']
        else:
            first = []
            for r in range(self.R):
                horizon = lb[r, k[r]:]
                if len(horizon) > 0:
                    first.append(horizon[0])
            
            if len(first) >= 2:
                first = sorted(first)
                current_time = min(first[0], first[1] - self.unwrapped.switch)
            else:
                current_time = first[0]

        horizons = []
        outputs = []
        for r in range(self.R):
            n_unscheduled = n[r] - k[r]
            horizon = lb[r, k[r]:k[r] + n_unscheduled] - current_time
            horizons.append(horizon)
            output = horizon
            if self.gaps:
                output = np.diff(horizon, prepend=-self.unwrapped.rho) - self.unwrapped.rho
            outputs.append(output)

        self.last_horizons = horizons
        return outputs

    def render(self):
        horizons = self.last_horizons
        rho = self.unwrapped.rho

        # compute maximum y value in the plot
        end = max(np.max(h, initial=0) for h in horizons) + rho
 
        _, ax = plt.subplots(figsize=(end, len(horizons)))

        # some options:
        lw = 0.8 # linewidth
        height = 0.5 # fixed height of vehicle rectangles

        # plot all unscheduled vehicles, on a separate row for each route
        row = 0
        for horizon in horizons:
            # plot unscheduled vehicles at their translated "horizon" location
            for start in horizon:
                rect = Rectangle((start, row), width=rho, height=height, linewidth=lw, facecolor='lightgray', edgecolor='k')
                ax.add_patch(rect)
                ax.text(start + 0.20, row + 0.18, f"{start:.2f}", fontsize=14)
            # move to next row (so down)
            row -= 1

        ax.set_xlim([-0.05, end + 0.05])
        ax.margins(y=0.15)

        plt.tight_layout()
        # ax.axis('off')

        # hide all spines except the bottom (x-axis)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        # hide y-axis ticks and labels
        ax.yaxis.set_visible(False)

        ax.set_aspect('equal', adjustable='box')


class NewHorizonRollingWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.R = self.unwrapped.R # fixed number of routes

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._observation(obs)  # transform initial observation!
        return obs, info

    def step(self, action):
        action = self._action(action)
        obs, *rest = self.env.step(action)
        return self._observation(obs), *rest
    
    def _observation(self, obs):
        if all(self.unwrapped.done):
            return obs # does not matter

        # permute the routes according to "current route"
        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]
        c = current_route(self.unwrapped)
        roll = -active_routes.index(c)
        rolled_routes = np.roll(active_routes, roll)
        horizons = [obs[r] for r in rolled_routes]
        return horizons

    def _action(self, action):
        c = current_route(self.unwrapped)
        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]

        roll = -active_routes.index(c)
        rolled_routes = np.roll(active_routes, roll)

        return rolled_routes[action % len(rolled_routes)]

        # action = number of routes to advance from current route (while "wrapping" around the end)
        current_ix = active_routes.index(c)
        # action = min(action, len(active_routes) - 1) # prevent unnecessary "wrapping around"
        return active_routes[(current_ix + action) % len(active_routes)]
    
    def _inverse_action(self, route):
        c = current_route(self.unwrapped)
        active_routes = [r for r in range(self.unwrapped.R) if not self.unwrapped.done[r]]

        # inverse action = number of advances to take from current route to arrive at "route"
        current_ix = active_routes.index(c)
        next_ix = active_routes.index(route)
        action = next_ix - current_ix
        return action if action >= 0 else action + len(active_routes)


class PaddedHorizonWrapper(gym.ObservationWrapper):

    def __init__(self, env, max_k=None, reversed=False):
        super().__init__(env)
        self.env = env
        self.R = self.unwrapped.R # fixed
        self.reversed = reversed

        if max_k is None:
            # try to determine second dimension max size otherwise
            if hasattr(self.unwrapped, 'instance'):
                s = self.unwrapped.instance
            else:
                s = self.unwrapped.instance_generator()
            self.max_k = max(len(a) for a in s.arrivals)
        else:
            self.max_k = max_k
        self.observation_space = gym.spaces.Dict({
            "horizon": gym.spaces.Box(low=0, high=np.inf, shape=(self.R, self.max_k), dtype=np.float32),
            "h_lengths": gym.spaces.Box(low=0, high=np.inf, shape=(self.R,), dtype=np.int16),
            "k_lengths": gym.spaces.Box(low=0, high=self.max_k, shape=(self.R,), dtype=np.int16),
        })

    def observation(self, obs):
        horizons = np.zeros((self.R, self.max_k))
        h_lengths = np.zeros(self.R)
        k_lengths = np.zeros(self.R)
        for r, horizon in enumerate(obs):
            k = min(len(horizon), self.max_k)

            h_lengths[r] = len(horizon)
            k_lengths[r] = k
            k = min(len(horizon), self.max_k)
            if self.reversed:
                horizons[r,:k] = list(reversed(horizon[:k]))
            else:
                horizons[r,:k] = horizon[:k]

        return {
            'horizon':   horizons,
            'h_lengths': h_lengths,
            'k_lengths': k_lengths,
        }
