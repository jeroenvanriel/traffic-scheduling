import numpy as np
from exact import solve
from expert_demonstration import expert_demonstration
import single_intersection_gym

from plotting import plot_instance, plot_schedule

# t = 45.08385121273484
# start = 44.08385121273484


instance = {
    'K': 2,
    's': 2,
    'arrival0': np.array([3.67969241e-02, 4.20525098e+00, 6.51133950e+00, 1.20129413e+01,
                       1.64460666e+01, 2.70842060e+01, 3.15670811e+01, 3.48634853e+01,
                       4.10980082e+01, 4.40838512e+01]),
    'length0': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    'arrival1': np.array([2.14758715, 6.70408841, 9.61579576, 11.56997665, 12.96480807,
                       17.62657081, 19.33690183, 24.8126108 , 25.96657261, 31.88123968]),
    'length1': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
}
plot_instance(instance, out='instance.pdf')

solution = solve(instance)
plot_schedule(solution, out='exact.pdf')

sa = expert_demonstration(instance, solution)
