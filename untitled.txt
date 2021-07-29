import random
import agent as agent
from env import *
import scipy.special
import time

maximum_depth=4
num_of_Mu_chunks=3

e=tiger_POMDP_env(read_config=True,config_address='./tiger.json',parameters=None)
e.discount_factor = 1
for i in range(1,len(e.actions)):
    e.observation_matrix[i][0][0]=0.5
    e.observation_matrix[i][0][1]=0.5
    e.observation_matrix[i][1][0]=0.5
    e.observation_matrix[i][1][1]=0.5

# listening cost
e.rewards[:,0]=-1.
# correct and low
e.rewards[0,1]=e.rewards[1,2]=10.
# incorrect and low
e.rewards[0,2]=e.rewards[1,1]=-100.
# correct and high
e.rewards[0,3]=e.rewards[1,4]=20.
# incorrect and high
e.rewards[0,4]=e.rewards[1,3]=-200.

ag=agent.Bauerle_Rieder_agent(environment=e, num_of_Mu_chunks=num_of_Mu_chunks,max_iterations=maximum_depth)

ag.pre_planning_paration(make_and_save_Mu=True,save_Mu2index_chunks=True)

xa = ag.value_iteration(utility_function='risk-neutral')

get_possible_s = lambda step : ag.generate_possible_wealths(np.unique(ag.env.rewards),ag.initial_wealth,ag.env.discount_factor,step)

num_to_act=dict(zip(list(e.actions.values()),list(e.actions.keys())))

# reset the agent with random starting x and probability of P(y=0 and s=0)=1/3 and P(y=1 and s=0)=2/3
init_observation = random.randint(0,1)
print("Initial Observation:",init_observation)
ag.reset(initiative_observation=init_observation)


max_s = list(get_possible_s(maximum_depth))*2
acc_reward = 0

for t in range(maximum_depth):
    action,value,belief=ag.do_action()   
    print("Step: {} Action: {} Value: {} Belief: {} State: {} ".format(ag.current_internal_timeStep,action,value,np.argmax(belief),e.current_state),end="")
    t1,t2,new_state,last_reward,new_observation=e.step(num_to_act[action])
    print("Reward {}".format(last_reward))
    
    acc_reward += last_reward
    
    ag.update_agent(new_observation)

    print('Step: {} New State: {} New Observation: {} New Belief {}'.format(ag.current_internal_timeStep,e.current_state,new_observation,np.argmax(ag.current_internal_belief)))
    print(ag.current_internal_belief)

    
print("Actual Reward {}, Beliefed Reward: {}".format(acc_reward,max_s[np.argmax(ag.current_internal_belief)]))


    