# MO-RS-POMDP
## Description
This package of codes is the simulation part of our paper:<br>
["Risk-Sensitive Partially Observable Markov Decision Processes as Fully Observable Multivariate Utility Optimization problems."][paper]
<br>
<br>
In this project, we have introduced a new method to solve Partially Observable Markov Decission Processes (POMDPs) under Risk-sensitivity while the accumulated reward is hidden as 
well as state of environment. In our method, we use a class of utility functions made by weighted sum of exponential terms. We have proven that:
1. We can formulate the RS-POMDP problem and design the solver agent in a way that can work with multi-variate exponential utility function 
2. Using multi-variate exponential utility function, makes the problem's state space much smaller and more feasible to solve (*)
3. By using this class of utility functions, we can approximate any increasing utility function in bounded intervals.

(*) [Cavazos-Cadena and Hernández-Hernández(2005)][Cavanoz] had previously shown this for one exponential term.<br>

Therefore, our method has generaliy, altough it esmiates the values. The precision of our models estimations depends on the number of exponential terms in its utility function.
So, by using our model we can manage the tradeoff between speed+memory complexity and precision, which , as much as we know, is a unique feature of our method in comparison
with other methods.
<br>
### Simulation and numerical analysis

In numerical part we have first shown the competency of our model to approximate the famous utility functions like Risk-seeker, Risk-averse, Risk-neutral and Sigmoid utility functions.
Then, we have compared our method with the well-known most general method in the field ([Bäuerle and Rieder 2017][Bauerle]).
 
We have implemented their method as well as ours in two different settings: <b>continious_optimized</b> and <b>discrete versions</b>:<br>
* In both methods, in the continious_optimized version the agent expands the decision tree until the maximium length of planning in a forward search, then applies Value iteration only on the reachable 
states. Altogh the forward search is a reasonable step for our simulation paradigm, but it is not applicable in the general cases. <br>
* In descrite settings, both methods make their continious internal states partitioned to have descrite points in order to apply Value iteration on them.

For simulations we used an extension of the famous "Tiger problem" (Kaelbling, Littman, & Cassandra, 1998) that is explained in our paper.

## Code Desciption
In the final version of code (until now), there are 7 files:<br>
* <b>Compact_Beurele.py:</b> Implementation of the of Bauerle and Rieder's model<br>
* <b>Compact_MO.py:</b> Implementation of our model (Multi-variate method)<br>
* <b>env.py:</b> Implementation of class of environment which is the extended Tiger problem<br>
* <b>Tiger2.json:</b> Contains the config of enviroment<br>
* <b>Both methods run.ipynb:</b> Jypyter-notebook of examples for creating environment and creating and running both methods in both continious and discrete modes and their interactions with the environment<br>
* <b>Sigmoid.ipynb:</b> A Jypyter-notebook that contains implemenation of approximating the Sigmoid function with weighted sum of exponential functions by gradient descent<br>
* <b>Plot.ipynb:</b> Jypyter-notebook that is using for plotting some paper's figures<br>



[paper]:https://www.researchgate.net/publication/362068391_Risk-Sensitive_Partially_Observable_Markov_Decision_Processes_as_Fully_Observable_Multivariate_Utility_Optimization_problems
[Cavanoz]:https://www.tandfonline.com/doi/abs/10.1080/17442500500428833
[Bauerle]:https://pubsonline.informs.org/doi/abs/10.1287/moor.2016.0844?casa_token=NQ6QYrIyjbwAAAAA%3ANntGN3SyYj8ONO5SI8eyv2fVEJMrormclALg_C1hhIXXV-wfu3Sf3mhBnt4ZEdGLMvY4jgWzxu_y&journalCode=moor
