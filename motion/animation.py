import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10,10))

vehicle_w = 1
vehicle_l = 2

in_lane_length = 10
out_lane_length = 2
margin = 1

ax.set_xlim(-in_lane_length - margin, vehicle_w + out_lane_length + margin)
ax.set_ylim(-in_lane_length - margin, vehicle_w + out_lane_length + margin)

lineargs = { 'color': 'k', 'linewidth': 0.8 }
vehicleargs = { 'facecolor': 'darkgrey', 'edgecolor': 'k', 'linewidth': 0.8 }

# horizontal lane (1)
ax.axline((in_lane_length, 0), (-vehicle_w - out_lane_length, 0), **lineargs)
ax.axline((in_lane_length, vehicle_w), (-vehicle_w - out_lane_length, vehicle_w), **lineargs)
# vertical lane (2)
ax.axline((0, in_lane_length), (0, -vehicle_w - out_lane_length), **lineargs)
ax.axline((vehicle_w, in_lane_length), (vehicle_w, -vehicle_w - out_lane_length), **lineargs)


# draw a vehicle on horizontal lane (1)
p = -in_lane_length
rect1 = Rectangle((p - vehicle_l, 0), width=vehicle_l, height=vehicle_w, **vehicleargs)
ax.add_patch(rect1)

# draw a vehicle on vertical lane (2)
p = -in_lane_length - vehicle_l - vehicle_w
rect2 = Rectangle((0, p - vehicle_l), width=vehicle_w, height=vehicle_l, **vehicleargs)
ax.add_patch(rect2)

def update(frame):
    rect1.set_x(rect1.get_x() + 0.05)
    rect2.set_y(rect2.get_y() + 0.05)


ani = animation.FuncAnimation(fig=fig, func=update, frames=400, interval=10)
ax.axis('off')
plt.show()
#ani.save(filename="animation.gif", writer="pillow")
