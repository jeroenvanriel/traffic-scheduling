## 🚘 Autonomous Intersection Coordination

> How to optimally guide autonomous vehicles through a busy network of intersections?

Suppose you can control every autonomous vehicle in a certain neighborhood of tightly interconnected intersections, maybe something like downtown Manhattan. Assume that every driver (or rather, *passenger* in this scenario) has communicated their next destination, then how would we have to guide all vehicles in order to optimize measures such as overall travel time and energy efficiency?

Autonomous vehicle technology is advancing rapidly, making large-scale coordinated traffic control an increasingly relevant problem. There are many modeling facets, but this project focuses on the following two main aspects:
1. In which order are vehicles going to cross intersections?
2. Given fixed routes, how to control the speed of each vehicle?

![Animation of vehicle movement in a grid-like network](grid.gif)

## 📚 Thesis

[Efficient and Provably Safe Autonomous Intersection Coordination](report/thesis.pdf)

**Abstract**: The growing adoption of autonomous vehicles motivates the need for systems that coordinate joint motion across traffic networks, aiming to reduce travel time, fuel consumption, and improve comfort. We study this coordination problem under ideal conditions, assuming perfect communication and a centralized controller that prescribes precise vehicle trajectories. Focusing on intersection management, we identify the determination of optimal crossing orders as a key combinatorial challenge. Formulating delay minimization as a scheduling problem, we develop an integer programming model and introduce two types of cutting planes that significantly accelerate solution time for single-intersection cases. As exact optimization scales poorly, we investigate step-by-step scheduling as a basis for fast heuristics. For a single intersection, a simple one-parameter threshold policy achieves less than 2% and 10% optimality gaps for two and three crossing routes, respectively, with up to 60 vehicles per route. Neural network policies trained via imitation and reinforcement learning offer limited additional benefit. Finally, we outline challenges in extending the framework to networks of intersections, particularly in modeling finite lane capacities, and present preliminary insights into how finite lane capacity defines the space of feasible crossing time schedules. 

## 📂 Folder Organisation

- *single* - Main experiments for a single intersection
- *network* - Extensions of experiments to a network of intersections
- *motion* - Various ways of solving the optimal control problem to determine speed profiles
- *report* - Thesis and miscellaneous personal notes
