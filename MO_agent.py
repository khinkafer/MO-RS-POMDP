import numpy as np
from datetime import datetime
import os
import itertools as it
import pandas as pd

class Multi_Variate_agent(object):
    
    def __init__(self,environment, planning_depth, partitioning_chunk_number=100,agent_mode='naive'):
        
        
        
        # parameters       
        self.agent_mode=agent_mode                                        
        self.partitioning_chunk_number=partitioning_chunk_number        
        self.planning_depth=planning_depth  
        self.initial_state=[]
        self.initial_observation=0
        # both env and env_dynamics point to the same thing, so they are euqaal
        self.env_dynamics=environment
        self.env=environment
        
        
        # results of planning
        
        self.q_func={}
        self.action_func={}
        self.value_func={}  
        
        
        #aux variables
        self.value_function={}
        self.reachable_states={}
        self.all_theta=[]           
        self.M=[]                   # M matrix
        self.exp_vorfaktoren=[]     # powers of exponential parts of the utility function
        self.exp_weights=[]         # weights of exp. parts of the ytility func.
        self.x_map={}
        self.F=[]
        self.exp_vorfaktoren=[]
        #simulation parameters
        self.current_internal_state=[]
        self.time_step=0
        self.last_action=-1
        self.rounding_prec_coeff=100000
        
        
        return


    def pre_planning(self,exp_vorfaktoren,exp_weights, initial_theta=[0.5,0.5],initial_observation=0,initial_wealth=0 ):
        
        # parameters
        env_dynamics=self.env_dynamics
        planning_depth=self.planning_depth
        partitioning_chunk_number=self.partitioning_chunk_number
        agent_mode=self.agent_mode
        self.exp_vorfaktoren=exp_vorfaktoren
        self.exp_weights=exp_weights
        exp_num=len(self.exp_vorfaktoren)
        self.initial_observation=initial_observation
        self.initial_wealth=initial_wealth
        
        # predefine all theta points(represented by integer values) and calculate their mappings by F and values by G
        int_theta,F,G,M=self.theta_transition_function_md(exp_vorfaktoren=exp_vorfaktoren,env_dynamics=env_dynamics,partitioning_chunk_number=partitioning_chunk_number)
        
        # universal theta (represented by float)
        all_theta=np.array(int_theta)/partitioning_chunk_number

        # initialize extended internal space ( theta_i1, theta_i2, .. , y)
        x0=self.initialize_internal_state(theta_0=initial_theta, observation_0=initial_observation, exp_num=exp_num,initial_wealth=initial_wealth)
        self.initial_state=x0
        self.current_internal_state=x0
        
        ## Save calculated variables
        self.M=M
        self.F=F
        self.all_theta=all_theta
        
        # calculate possible states for each depth of planning
        # in the 'optimized' mode: it finds all reachable successive internal states and set their values in the value function to Zero.
        # in the 'naive' mode: it finds  all internal states (regardless of the planning procedure) and set their related value in value function to Zero.
        if agent_mode=='discrete_optimized':
            
            # find reachable internal states in each time step 
            reachable_states=self.find_reachable_states(intial_x_state=x0,env_dynamics=env_dynamics,F_map_md=F,max_depth=planning_depth,partitioning_chunk_number=partitioning_chunk_number,exp_num=exp_num,all_theta=all_theta)
            
            #initialize value function ( dictionary from each possible state at each depth to their value (which are set to zero))
            # Also record a mapping from each state (regarding the time step) to its index in the reachable_states variable 
            value_function,x_map=self.intial_value_function(reachable_states)
            X=reachable_states
            
        elif agent_mode=='cheating':
            # find continoius (without partitioning)reachable internal states for each time step
            reachable_states,x_map=self.continious_optimized_reachable_states(initial_aug_state=x0,initial_wealth=initial_wealth,env_dynamics=env_dynamics,max_depth=planning_depth,rounding_prec_coeff=100000)
            
            #initialize value function ( dictionary from each possible state at each depth to their value (which are set to zero))
            # Here we don't consider the mapping of internal states that 'initial_value_function()' gives us. Instead, we use the second output of the 
            # 'continious_optimized_reachable_states()' which is an exact mapping of (internal_state,action, next_observation) of each timeStep to
            # the succesor internal state(th1',th2',...,y').
            value_function,_=self.intial_value_function(reachable_states)
            X=reachable_states
            
            
        elif agent_mode=='naive':
            
            # make all possible internal states regardless of time step
            #combinations of: theta points of all exponential axis and observations (all possible (theta0,theta1,..observation))
            possible_states=self.naive_reachable_states(env_dynamics=env_dynamics,exp_num=exp_num,all_theta=all_theta)
            
            # initialize value function
            value_function,x_map=self.naive_initial_value_function(possible_states=possible_states)
            X=possible_states
        else:
            print('bad agent_mode input')
            return -1
        
        ## Save calculated variables
        
        self.reachable_states=X
        self.x_map=x_map    
        self.value_function=value_function
        
        
        
               
        
        if agent_mode=='discrete_optimized':
            return x_map,M,F,G,X,value_function,all_theta
        elif agent_mode=='cheating':
            return x_map,M,F,G,X,value_function,all_theta
        elif agent_mode=='naive':
            return x_map,M,F,G,X,value_function,all_theta
        
        
        


    def make_M_md(self,exp_vorfaktoren,env_dynamics):
        
        # setting parameters
        number_of_exp=len(exp_vorfaktoren)
        vorfaktoren=exp_vorfaktoren
        #initializing M as a 5-dim matrix: M(i,a,y,s,s')
        M= np.zeros((number_of_exp, len(env_dynamics.actions), len(env_dynamics.observations), len(env_dynamics.states), len(env_dynamics.states)))

        #calculation of M as a 5-dim matrix, i.e. the multi-variation/i is seen as an extra variable - 
        #M[i][a][y][:][:] then yields the 2x2 that is used for forward propagation

        for i in range(len(vorfaktoren)):
            for a in range(len(env_dynamics.actions)):
                for y in env_dynamics.observations:
                    for s in env_dynamics.states:
                        for ss in env_dynamics.states:
                            M[i][a][y][s][ss]= np.exp(vorfaktoren[i]*env_dynamics.rewards[s][a])*env_dynamics.transition_matrix[s][a][ss]*env_dynamics.observation_matrix[a][ss][y]
                            
                    # for each M(i,a,y)[s,s'] the M Matrix should become Transpose. (see the paper)       
                    M[i][a][y]=M[i][a][y].T
        return M


    def make_possible_theta_universal(self,states, observations, partitioning_chunk_number, save=False ):
        
        # make discrete theta for one sample exponential as prototype. Theta has two elements becasue in the design of our experiment we have 
        # only two real states (tiger in right/left)
        actual_partition_num=partitioning_chunk_number+1
        univariate_x=[[s1,partitioning_chunk_number-s1] for s1 in range(actual_partition_num) ]
        # contents are dictionaries with keys: different possible Theta values, and values: index/indicator
        indicator=np.arange(len(univariate_x))
        x_to_index=dict(zip(list(map(tuple,univariate_x)),indicator)) 
        return univariate_x,x_to_index


    def find_next_theta_index_universal(self,theta_values,partitioning_chunk_number,mode='multi'):
        
        # make valid discrete theta points 
        in_base_partitioning=np.array(theta_values)*partitioning_chunk_number
        valid_points=np.round(in_base_partitioning)
        
        # if mode is 'single' there is just one point in the input
        if mode=='single':
            return valid_points[0]
        # find their index. Because we have only 2 states, the index of theta point is equal to the value of first element in integer representation.
        return valid_points[:,0]


    def theta_transition_function_md(self,exp_vorfaktoren,env_dynamics,partitioning_chunk_number):
        
        # make possible theta values (for one sample exponential axis)
        int_theta,_=self.make_possible_theta_universal(states=env_dynamics.states, observations=env_dynamics.observations, 
                                                  partitioning_chunk_number=partitioning_chunk_number, save=False)
        int_theta=np.array(int_theta)

        ### Allocate theta mappings variables

        # for F-function
        # (i x len(theta) x actions x observation) -> index_of_next_theta_point
        theta_map_md=np.empty((len(exp_vorfaktoren),len(int_theta),len(env_dynamics.actions),len(env_dynamics.observations)),dtype=int)

        # for G -function : (i x len(theta) x actions x observations) -> g-value
        G_map_md=np.empty((len(exp_vorfaktoren),len(int_theta),len(env_dynamics.actions),len(env_dynamics.observations)),dtype=int)


        # actual theta (floating point representation)
        theta=np.array(int_theta)/partitioning_chunk_number
        
        ## Calculate values

        # M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        M=self.make_M_md(exp_vorfaktoren,env_dynamics)

        # theta mapping function
        for i,vorfaktor in enumerate(exp_vorfaktoren):
            for a,_ in enumerate(env_dynamics.actions):
                for y_prim,_ in enumerate(env_dynamics.observations):
                    
                    # M x theta_i
                    z=np.matmul(M[i,a,y_prim,:,:],theta.T)
                    z=z.T
                    # integral of M x Theta_i
                    integral=np.sum(z,axis=1).reshape(len(z),1) 
                    
                    # F- function : the index of the successive state. Note: its input is normalized z (means z / integral of z)
                    theta_map_md[i,:,a,y_prim]=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number)
                    
                    # G- function
                    G_map_md[i,:,a,y_prim]=(np.log(integral)/vorfaktor).reshape(len(z),)
                    
        F_map_md=theta_map_md      
        return int_theta,F_map_md,G_map_md,M

    def initialize_internal_state(self,theta_0, observation_0, exp_num,initial_wealth):
        # internal state
        x=[]
        r=[]
        aug=[]
        # fill it by initial values of theta (2 elemnts) for each variate dimention (number of exponentials)
        for t in range(exp_num):
            x.append(theta_0[0])
            x.append(theta_0[1])
            r.append(initial_wealth)
        # append the value of the initial observation (y) to make internal X-state
        x.extend([observation_0])
        
        # to make augmented states
        aug=[tuple(x), tuple(r),0]
        
        return tuple(aug)    

    def find_reachable_states(self,intial_x_state,env_dynamics,F_map_md,max_depth,partitioning_chunk_number,exp_num,all_theta):
        
        # reachables is a dictionary with keys: time steps & values: list of all possible internal states(Xs). 
        # Each internal state has represented by a tuple contains (theta of the first exponential, theta of the second, ..., last (current) observation).
        reachables={}
        
        # set the first step
        reachables[0]=[tuple(intial_x_state)]

        # set some parameters
        num_actions=F_map_md.shape[2]
        num_observations=F_map_md.shape[3]       
        observations=list(env_dynamics.observations.keys())
        
        
        # until one step before end
        for depth in range(1,max_depth):
            reachables[depth]=[]
            for to_extend in reachables[depth-1]:      
                
                this_extention=[]
                for a in range(num_actions):  

                    for y_prim in range(num_observations):
                        next_x=[]
                        for i in range(exp_num):

                            # find current index of theta point
                            tt=int(self.find_next_theta_index_universal([to_extend[i*2],to_extend[(i*2)+1]],partitioning_chunk_number,mode='single'))

                            # find next theta based on: action,next observation and i
                            next_theta_index=F_map_md[i][tt][a][y_prim]
                                                        
                            next_x.extend(all_theta[next_theta_index])
                            
                            
                        next_x.append(y_prim)
                        
                        new_entry=tuple(next_x)

                        if not (new_entry in reachables[depth]):
                            reachables[depth].append(new_entry)

        return reachables

    def naive_reachable_states(self,env_dynamics,exp_num,all_theta):
        
        # setting parameters
        observations=list(env_dynamics.observations.keys())
        
        # make a copy from all_possible thetas. Theta_part contains all possible combinations of theta values for all exponential axes.
        # Theta_part is now contains all possible theta points for the first exponential 
        theta_part=all_theta.copy()
        
        # make combination of theta for other exponentials one by one.
        for i in range(1,exp_num):
            theta_part=it.product(theta_part,all_theta)
        
        # change the theta_part variable to the type and the shape that we want
        theta_part= list(map(list,theta_part))
        theta_part=np.array(theta_part)
        theta_part=theta_part.reshape(len(theta_part),exp_num*2)
        
        # make a list of observations (each observation has repeated for times equal to possible theta combinations) 
        tmp1=[]
        for y in observations:
            tmp1.extend([y]*len(theta_part))

        # repeating possible thetas for number_of_observations times
        theta_part=np.tile(theta_part,(len(observations),1))
        
        # concatenate two arrays to make all possible combinations of thetas and observations
        tmp1=np.array(tmp1).reshape(len(tmp1),1)
        all_possible_internal_x=np.hstack((theta_part,tmp1))

        return all_possible_internal_x

    def intial_value_function(self,reachable_states):
        
        # a mapping form each internal state to its value, which is set to 0 for initialization.
        value_function={}
        # a mapping from each internal state to its index in the reachable_states[step] variable.
        x_map={}
        
        # for each time step:
        for step in range(len(reachable_states.keys())):
            # setting value of that time step's states to zero (make a dict. from states to values(0s)).
            value_function[step]= dict((key,0) for key in reachable_states[step])
            # setting a mapping from internal states to their indexes (in this time step)
            indexes=np.arange(len(reachable_states[step]))
            x_map[step]=dict(zip(reachable_states[step],indexes))
            
        return value_function,x_map

    def naive_initial_value_function(self,possible_states):
        
        # make a dictionary from keys: all possible internal states (each of them is a tuple) to values:initial values (which are 0 )
        init_values=np.zeros(len(possible_states))
        possible_states=list(map(tuple,possible_states))
        value_function=dict(zip(possible_states,init_values))
        
        # make a dictionary from keys: all possible internal states to values: their indexes in the 'possible_states' variable
        indexes=np.arange(len(possible_states))
        x_map=dict(zip(possible_states,indexes))
        
        return value_function,x_map






    def value_iteration(self):
        
        #env_dynamics=self.env_dynamics
        #planning_depth=self.planning_depth
        #reachable_states=self.reachable_states
        #value_function=self.value_function
        #agent_mode=self.agent_mode
        #exp_vorfaktoren=self.exp_vorfaktoren
        
        if self.agent_mode=='discrete_optimized':
            print('this function has not been finilized!')
            v,a,q,vf =self.value_iteration_optimized(env_dynamics=self.env_dynamics,M_matrix=self.M,exp_vorfaktoren=self.exp_vorfaktoren,exp_weights=self.exp_weights,planning_depth=self.planning_depth,reachable_states=self.reachable_states,value_function=self.value_function,all_theta=self.all_theta,partitioning_chunk_number=self.partitioning_chunk_number) 
        elif self.agent_mode=='naive':
            v,a,q,vf =self.value_iteration_naive(env_dynamics=self.env_dynamics,M_matrix=self.M,exp_vorfaktoren=self.exp_vorfaktoren,exp_weights=self.exp_weights,planning_depth=self.planning_depth,reachable_states=self.reachable_states,value_function=self.value_function,all_theta=self.all_theta,partitioning_chunk_number=self.partitioning_chunk_number)
        elif self.agent_mode=='cheating':
            v,a,q =self.MO_cheating(self,planning_depth=self.planning_depth,exp_vorfaktoren=self.exp_vorfaktoren,exp_weights=self.exp_weights,rounding_prec_coeff=100000)
            
        else:
            print('bad agent_mode input!')
            return -1

        if self.agent_mode == 'cheating':
            self.value_function=0
        else:
            self.value_function=vf
        self.value_func=v
        self.action_func=a
        self.q_func=q
        
        return self.value_func,self.action_func,self.q_func,self.value_function
     

    def value_iteration_optimized(self,env_dynamics,M_matrix,exp_vorfaktoren,exp_weights,planning_depth,reachable_states,value_function,all_theta,partitioning_chunk_number):

        M=M_matrix
        exp_num=len(exp_vorfaktoren)
        max_depth=planning_depth

        discount=env_dynamics.discount_factor

        num_action=len(env_dynamics.actions.keys())
        num_observations=len(env_dynamics.observations.keys())


        # backwardly
        q_func={}
        action_func={}
        value_func={}
        #for step in range(max_depth-1,-1,-1):
        for step in range(max_depth-1,-1,-1):
            this_step=list(map(list,reachable_states[step])).copy()

            q_tmp=np.empty((len(this_step),num_action))
            for a in range(num_action):
                this_action_q=np.zeros((len(this_step),1))

                for y_prim in range(num_observations):
                    this_y_prim_nextState=np.empty((len(this_step),2))
                    this_y_prim_cost=np.zeros((len(this_step),1))
                    for i in range(exp_num):

                        this_step=np.array(this_step)

                        theta_i=this_step[:,i*2:2*i+2].copy().T
                        
                        # M x Theta_i
                        z=np.matmul(M[i,a,y_prim,:,:],theta_i)
                        z=z.T
                        
                        # Integral of M x theta_i
                        integral=np.sum(z,axis=1).reshape(len(z),1)          
                        # F- function: next theta_i
                        next_theta_i=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number).reshape(len(this_step),1)

                        # Cost function
                        g_i=(np.log(integral)/exp_vorfaktoren[i]).reshape(len(z),)
                        c_i=(g_i+np.log(np.power(num_observations,1/exp_vorfaktoren[i]))).reshape(len(this_step),1)                  

                        # utility of cost
                        this_y_prim_cost=this_y_prim_cost+exp_weights[i]*np.exp(exp_vorfaktoren[i]*c_i)


                        if i==0:

                            this_y_prim_nextState=np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)
                        else:
                            this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)),axis=1)
                    this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.array([y_prim]*len(this_step)).reshape(len(this_step),1)),axis=1)

                    # if max step
                    if step==max_depth-1:

                        this_action_q+=this_y_prim_cost * (1./num_observations)
                        
                    else:
                        # else
                        next_v=[]
                        for r,thet in enumerate(this_y_prim_nextState):
                            thet=tuple(thet)

                            next_v.append(value_function[step+1][thet])
                        next_v=np.array(next_v).reshape(len(this_step),1)
                        this_action_q+= (this_y_prim_cost + discount* next_v) * (1./num_observations)


                q_tmp[:,a]= this_action_q.reshape(len(this_step),)
                # choose best action, record values, record actions
                # check to see if it has equal values for action=1,2 and 3,4 again

            best_actions=q_tmp.argmax(axis=1)
            best_values=q_tmp.max(axis=1)

            q_func[step]=q_tmp.copy()
            action_func[step]=best_actions.copy()
            value_func[step]=best_values.copy()

            # update the values

            value_function[step]= dict((tuple(this_step[k]),best_values[k]) for k in range(len(best_values)))

        return value_func,action_func,q_func ,value_function

    def value_iteration_naive(self,env_dynamics,M_matrix,exp_vorfaktoren,exp_weights,planning_depth,reachable_states,value_function,all_theta,partitioning_chunk_number):

        M=M_matrix
        exp_num=len(exp_vorfaktoren)
        max_depth=planning_depth

        discount=env_dynamics.discount_factor

        num_action=len(env_dynamics.actions.keys())
        num_observations=len(env_dynamics.observations.keys())

        # cast dictionary to pandas series to use its vectorize ability
        value_function=pd.Series(value_function)

        # backwardly
        q_func={}
        action_func={}
        value_func={}

        for step in range(max_depth-1,-1,-1):


            q_tmp=np.empty((len(value_function),num_action))
            for a in range(num_action):
                this_action_q=np.zeros((len(value_function),1))

                for y_prim in range(num_observations):
                    this_y_prim_nextState=np.empty((len(value_function),2))
                    this_y_prim_cost=np.zeros((len(value_function),1))


                    ### calculations #############################

                    # for each exponential part
                    for i in range(exp_num):
                        # fetch data of related theta
                        theta_i=reachable_states[:,i*2:i*2+2].copy()

                        # M x theta_i
                        z=np.matmul(M[i,a,y_prim,:,:],theta_i.T)
                        z=z.T
                        #Integral of M x theta_i
                        integral=np.sum(z,axis=1).reshape(len(z),1) 
                        # F- function: next theta_is
                        next_theta_i=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number).reshape(len(value_function),1)

                        # Calculate G-function
                        g_i=(np.log(integral)/exp_vorfaktoren[i]).reshape(len(z),)

                        # Calculate cost-function
                        c_i=(g_i+np.log(np.power(num_observations,1/exp_vorfaktoren[i]))).reshape(len(value_function),1)  
                        
                        # Aggregate sum of all exponential results 
                        # utility of cost
                        this_y_prim_cost=this_y_prim_cost+exp_weights[i]*np.exp(exp_vorfaktoren[i]*c_i)

                        # put next theta of all exponentials together
                        if i==0:

                            this_y_prim_nextState=np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)
                        else:
                            this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)),axis=1)

                    # put next observation (y_prim) at the end to shape the next internal state (x)
                    this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.array([y_prim]*len(value_function)).reshape(len(value_function),1)),axis=1)

                    # if max step
                    if step==max_depth-1:

                        # value is equal to immediate rewards
                        this_action_q+=this_y_prim_cost * (1./num_observations)
                        
                    else:

                        #find value of the next related states
                        this_y_prim_nextState=list(map(tuple,this_y_prim_nextState))
                        next_v=value_function[this_y_prim_nextState]                    
                        next_v=np.array(next_v).reshape(len(next_v),1)

                        # add next states' values to immediate rewards                        
                        this_action_q+= (this_y_prim_cost + discount* next_v) * (1./num_observations)  
                # fill Q-function fiekds
                q_tmp[:,a]= this_action_q.reshape(len(value_function),)


            # Choose best actions and its related q-values

            best_actions=q_tmp.argmax(axis=1)
            best_values=q_tmp.max(axis=1)

            q_func[step]=q_tmp.copy()
            action_func[step]=best_actions.copy()
            value_func[step]=best_values.copy()

            # update the values
            value_function_indexes=value_function.index
            value_function[value_function_indexes]=best_values
            #dict((tuple(this_step[k]),best_values[k]) for k in range(len(best_values)))

        return value_func,action_func,q_func ,value_function
    
    
    
    def continious_optimized_reachable_states(self,initial_aug_state,initial_wealth, env_dynamics,max_depth,rounding_prec_coeff):
              
        # setting parameters
        exp_num=len(self.exp_vorfaktoren)
        num_actions=len(env_dynamics.actions)
        num_observations=len(env_dynamics.observations)
        observations=list(env_dynamics.observations.keys())
        
        # result variables
        
        # reachables is a dictionary with keys: time steps & values: list of all possible internal states. 
        # Each internal state has represented by a tuple contains: (X:which is a tuple of (theta of the first exponential, theta of the second, ..., last (current) observation) , R: which is a tupel of (accumulated reward
        # regarding the first exp, acc_reward of exp2 ...) , and Z: which is the time step.
        # e.g for u=e1+e2 : ((th1,th2,y'),(r1,r2),timeStep)
        reachables={}   
        
        # mapping from each internal state and action and next observation to the successive internal state.
        # (((th1,th2,..y),(r1,r2,...),time),a,y')--> ((th1',th2',...),(r1,r2,...)),y')
        universal_map={}    
        
        # set the first step
        r_state_vector=tuple([initial_wealth]*exp_num)
        reachables[0]=[initial_aug_state]
        
        # M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        M=self.make_M_md(self.exp_vorfaktoren,env_dynamics)
        
        # until the last step
        for depth in range(1,max_depth+1):
            
            reachables[depth]=[]
            
            for to_extend in reachables[depth-1]: 
                
                # State which we want to find its successors
                to_extend_r=np.array(to_extend[1])
                to_extend_theta=np.array(to_extend[0])
                
                # this expansion of the decision three (by a specific action and a specific observation)
                this_extention=[]
                
                for a in range(num_actions):  
                    for y_prim in range(num_observations):
                        
                        next_x_theta=[]
                        transition_cost_vector=[]
                        next_x_r=[]
                        
                        for i in range(exp_num):
                            
                            # M x theta_i
                            z=np.matmul(M[i,a,y_prim,:,:],to_extend_theta[i*2:(i+1)*2].reshape(2,1))
                            
                            # integral of M x Theta_i
                            integral=np.sum(z)
                            
                            # F- function : next theta part of the next state
                            next_theta=z/integral
                            
                            next_theta=next_theta.reshape(-1)
                            
                            next_theta=self.myRound(next_theta)
                            
                            next_theta=next_theta.tolist()
                            next_x_theta.extend(next_theta)
                            
                            # Cost part
                            # G- function: next imediate reward
                            g_i=np.log(integral)/self.exp_vorfaktoren[i]
                            # C= G + log(|y|^(1/lambda_i))
                            c_i=g_i +np.log(np.power(num_observations,1/self.exp_vorfaktoren[i]))
                            
                            
                            c_i=self.myRound(c_i)
                            c_i=c_i.tolist()
                            # make vector of costs of each exponential dimension
                            transition_cost_vector.append(c_i)
                            
                            
                            
                            
                        # Add y' to the next theta to make next state: (th1', th2',... , y')
                        next_x_theta.append(y_prim)
                        
                        # update R part of the state
                        z=np.power(self.env.discount_factor,(depth-1))                       
                        z=self.myRound(z)
                        
                        next_x_r=to_extend_r+ np.array(transition_cost_vector)*z
                        next_x_r=self.myRound(next_x_r)
                        
                        # make new internal state
                        new_entry=(tuple(next_x_theta),tuple(next_x_r),depth)
                        
                        # fetch different parts (maybe it is redundant)
                        current_theta=np.array(new_entry[0])
                        current_acc_reward=np.array(new_entry[1])
                        
                       
                        
                        
                        # Add the mapping of (internal state,action, next_observation) --> next internal state
                        universal_map[(tuple(to_extend),a,y_prim)]= new_entry
                        
                        # Check if the successive internal state is unique in the reachable states of the time step
                        # here we use rounding_prec_coeff to cut the decimal digits in a certain number. We also use this number to compare
                        # floating points, however we use 10x of that number in floating points for better precision.
                        
                        
                        # let's take the next state unique
                        redundants=0
                        
                        redundant_theta=False
                        redundant_r=False
                        
                        
                        
                        # for each next state which has been found until now
                        for rec in reachables[depth]:
                            
                            theta=np.array(rec[0])
                            acc_reward=np.array(rec[1])
                            
                            # if next theta is redundant:
                            if (np.max(np.abs(theta-current_theta))<1./(10*rounding_prec_coeff)):
                                redundant_theta=True
                                
                            else:
                                redundant_theta=False
                                
                            # if next accumulated reward is redundant
                            if (np.max(np.abs(acc_reward-current_acc_reward))<1./(10*rounding_prec_coeff)):
                                redundant_r=True
                                
                            else:
                                redundant_r=False
                                
                            # if both theta and accumulated reward are redundant    
                            if (redundant_theta and redundant_r):
                                redundants+=1
                                
                        # if it is a unique (theta,acc_reward) pair        
                        if redundants==0:
                             
                            # add next step to reachable states of the time step
                            reachables[depth].append((tuple(current_theta),tuple(current_acc_reward),depth))
                            
        return reachables,universal_map

                        
    def myRound(self,f,rounding_prec_coeff=100000,decimals=10):
        #f=np.multiply(f,rounding_prec_coeff)
        #f=f.astype(np.int64)
        #f=f.astype(np.float64)/rounding_prec_coeff
        f=np.round(f,decimals=decimals)
        return f
        
    def MO_cheating(self,env_dynamics,planning_depth,exp_vorfaktoren,exp_weights,rounding_prec_coeff=100000):
            
        # 
        env_dynamics=self.env_dynamics
        discount=env_dynamics.discount_factor
        reachable_states=self.reachable_states
        exp_num=len(self.exp_vorfaktoren)
        num_actions=len(env_dynamics.actions)
        num_observations=len(env_dynamics.observations)
        observations=list(env_dynamics.observations.keys())
        M=self.M
        max_depth=planning_depth
    
        ## results
        
        # q value of each state and action. q_func is a nested dictionary form time_steps to an inner dict. 
        # the inner dict. is a mapping from keys: internal states to values: a list of values of each action at that state
        q_func={}
        # A list of dictionaries. For each time step there is a dict. Dict.s are mapping from keys: internal states in that timeStep
        # to values: index of the best action at that state
        action_func={}
        # A list of dictionaries. For each time step there is a dict. Dict.s are mapping from keys: internal states in that timeStep
        # to values: value of that state at that time_step
        value_function={}
        
        ## backwardly
        
        # from max_depth-1: becasue in our modeling the initial state is 0 and for n step of planning, 
        # the last step of decision-making will happends at the time step= n-1
        for step in range(max_depth,-1,-1):
            
            # fetch list of all possible states in this time Step
            this_step=list(map(list,reachable_states[step])).copy()
            
            this_step_theta=list(map(list,(np.array(this_step,dtype=object)[:,0])))
            this_step_r=list(map(list,(np.array(this_step,dtype=object)[:,1])))
            
            # make an empty Q(internal_state, actions)
            
            q_func[step]={}
            action_func[step]={}
            value_function[step]={}
            
            if step==max_depth:
                val=np.zeros((len(this_step),1))
                this_step_r=np.array(this_step_r)
                for i in range(exp_num):
                    
                    val[:,0]+=np.exp(this_step_r[:,i]*exp_vorfaktoren[i])*exp_weights[i]
                for p,point in enumerate(reachable_states[step]):
                    
                    q_func[step][point]=[val[p,0].copy()]*num_actions
                   
                    action_func[step][point]=-1
                    value_function[step][point]=val[p,0].copy()
            else:
                
                
                for p,point in enumerate(reachable_states[step]):
                    
                    this_state_q_values=[0]*num_actions
                    for action in range(num_actions):
                        
                        for y_prim in range(num_observations):
                            
                            val=value_function[step+1][self.x_map[(point,action,y_prim)]]
                            pv=(1./num_observations)*val
                            
                            pv=self.myRound(pv)
                            
                            this_state_q_values[action]+=pv
                            
                    q_func[step][point]=this_state_q_values.copy()
                    action_func[step][point]=np.argmax(this_state_q_values)
                    value_function[step][point]=np.max(this_state_q_values)
                                
                
                
            #     for a in range(num_actions):
                
            #         # value of all theta points for just this action
            #         this_action_q=np.zeros((len(this_step),1))
        
            #         for y_prim in range(num_observations):
                    
                        
            #             # next state and cost of this (state,action, y_prim)
            #             this_y_prim_nextState=np.empty((len(this_step),2))
            #             this_y_prim_cost=np.zeros((len(this_step),1))
                        
            #             for i in range(exp_num):
        
            #                 this_step_theta=np.array(this_step_theta)
            #                 theta_i=this_step_theta[:,i*2:2*i+2].copy().T
                            
            #                 # M x Theta_i
            #                 z=np.matmul(M[i,a,y_prim,:,:],theta_i)
            #                 z=z.T
                            
            #                 # Integral of M x theta_t
            #                 integral=np.sum(z,axis=1).reshape(len(z),1)    
                            
            #                 # F- function: next theta_i
            #                 next_theta_i=(z/integral)
        
            #                 # Cut the decimal digits with certain number 
            #                 next_theta_i*=rounding_prec_coeff
            #                 next_theta_i=next_theta_i.astype(np.int64)
            #                 next_theta_i=next_theta_i.astype(np.float64)/rounding_prec_coeff
        
            #                 # Cost function
            #                 # G- function
            #                 g_i=np.log(integral)/exp_vorfaktoren[i]
            #                 # C= G + log(|y|^(1/lambda_i))
            #                 c_i=g_i +np.log(np.power(num_observations,1/exp_vorfaktoren[i])) 
                                  
        
            #                 # utility of cost
            #                 this_y_prim_cost=this_y_prim_cost+exp_weights[i]*np.exp(exp_vorfaktoren[i]*c_i)
                            
        
            #                 # concatenate next theta of all exponentional axes
            #                 if i==0:
        
            #                     this_y_prim_nextState=next_theta_i
            #                 else:
            #                     this_y_prim_nextState=np.hstack((this_y_prim_nextState,next_theta_i))
    
            #         # concatenate theta_prims with y_prim to make the next internal states
            #         this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.array([y_prim]*len(this_step)).reshape(len(this_step),1)),axis=1)
                    
            #         # Accumulate cost of this action. This action can leads us to num_of_observations different successive internal states. 
            #         # Each of them is related to one of the next y_prims. So, we accumulate expected value of them to make the value of this action.
            #         # Note: to calculate expected value of actions we have P(x'|x,a)=1/|y| where x'=(th1',th2',...,y')               
            #         # if last decision-making step
            #         if step==max_depth-1:
    
            #             this_action_q+=this_y_prim_cost * (1./num_observations)
                        
            #         else:
            #             # to contain values of successive states 
            #             next_v=[]
                        
            #             # for each element in the next states (not vectorized)
            #             for r,thet in enumerate(this_y_prim_nextState):
            #                 thet=tuple(thet)
            #                 # a bug
            #                 if (exp_vorfaktoren[0]==-1.5 and exp_vorfaktoren[1]==-1.5 and exp_weights[0]==-1 and exp_weights[1]==-1 and self.initial_state[0]==0.75):
            #                     pass
            #                 next_v.append(value_function[step+1][thet])  
            #             next_v=np.array(next_v).reshape(len(this_step),1)
                        
            #             # accumulate the weighted values of each y_prim case to make the expected value of this_action's value
            #             this_action_q+= (this_y_prim_cost + discount* next_v) * (1./num_observations)   
    
            #     # Q(current internal states, this action)
            #     q_tmp[:,a]= this_action_q.reshape(len(this_step),)
                
            # # choose best action, record values, record action values
            # best_actions=q_tmp.argmax(axis=1)
            # best_values=q_tmp.max(axis=1)
    
            # # make timeStep-related element of result variables 
            # q_func[step]={}
            # action_func[step]={}
            # value_function[step]={}
            
            # # fill the result variables in sequential manner
            # for p,point in enumerate(this_step):
            #     point=tuple(point)
    
            #     q_func[step][point]=q_tmp[p].copy()
            #     action_func[step][point]=best_actions[p].copy()
            #     value_function[step][point]=best_values[p].copy()
    
            
        self.q_func=q_func
        self.action_func=action_func
        self.value_func=value_function
        
        return value_function,action_func,q_func


 
    
    def reset(self):
        
        #exp_num=len(self.exp_vorfaktoren)
        self.time_step=0
        self.current_internal_state=self.initial_state
        
        return
    
    def do_action(self):
        
        if self.agent_mode=='discrete_optimized':
            action=np.argmax(self.q_func[self.time_step][self.x_map[self.time_step][tuple(self.current_internal_state)]])
            value_of_action=np.max(self.q_func[self.time_step][self.x_map[self.time_step][tuple(self.current_internal_state)]])
        elif self.agent_mode=='naive':
            action=np.argmax(self.q_func[self.time_step][self.x_map[tuple(self.current_internal_state)]])
            value_of_action=np.max(self.q_func[self.time_step][self.x_map[tuple(self.current_internal_state)]])
        elif self.agent_mode=='cheating':
            action=self.action_func[self.time_step][self.current_internal_state]
            value_of_action=self.value_func[self.time_step][self.current_internal_state]
        
        self.last_action=action
        
        return action,value_of_action
    
    def update_agent(self,new_observation):
        num_observations=len(self.env.observations)
        # update internal state
        next_internal_x=[]
        transition_cost_vector=[]
        if self.agent_mode != 'cheating':
            for i in range(len(self.exp_vorfaktoren)):
                theta_i=self.current_internal_state[i*2:i*2+2]
                theta_index=int(self.find_next_theta_index_universal(theta_i,partitioning_chunk_number=self.partitioning_chunk_number,mode='single'))
                theta_i_next=self.all_theta[self.F[i][theta_index][self.last_action][new_observation]]
                
                if i==0:
                    next_internal_x=list(theta_i_next)
                else:               
                    next_internal_x.extend(theta_i_next)
            
            next_internal_x.extend([new_observation])

        else:
            
            for i in range(len(self.exp_vorfaktoren)):
                theta_i=self.current_internal_state[0][i*2:i*2+2]
                z=np.matmul(self.M[i,self.last_action,new_observation,:,:],np.array(theta_i).reshape(2,1))
                integral=np.sum(z)
                next_theta_i=z/integral
                
                # Cut the decimal digits with certain number 
                next_theta_i=self.myRound(next_theta_i)
                
                next_theta_i=next_theta_i.reshape(-1)
                next_theta_i=next_theta_i.tolist()
                next_internal_x.extend(next_theta_i)
                
                
                # Cost part
                # G- function: next imediate reward
                g_i=np.log(integral)/self.exp_vorfaktoren[i]
                # C= G + log(|y|^(1/lambda_i))
                c_i=g_i +np.log(np.power(num_observations,1/self.exp_vorfaktoren[i]))
                
                
                c_i=self.myRound(c_i)
                c_i=c_i.tolist()
                # make vector of costs of each exponential dimension
                transition_cost_vector.append(c_i)
                
                
                
                
            # Add y' to the next theta to make next state: (th1', th2',... , y')
            next_internal_x.append(new_observation)
            
            z=np.power(self.env.discount_factor,(self.time_step))
                        
            z=self.myRound(z)
            
            next_x_r=np.array(list(self.current_internal_state[1]))+ np.array(transition_cost_vector)*z
            
            next_x_r=self.myRound(next_x_r)
            
            next_aug_state=(tuple(next_internal_x),tuple(next_x_r),self.time_step+1)
       
        self.current_internal_state=next_aug_state
        
        # update time
        self.time_step+=1
        
        return self.current_internal_state
        
        