import numpy as np
from datetime import datetime
import os
import itertools as it
import pandas as pd

class Multi_Variate_agent(object):
    
    def __init__(self,environment, planning_depth, partitioning_chunk_number=100,agent_mode='naive'):
        
        self.env=environment
        
        # parameters       
        self.agent_mode=agent_mode
        self.partitioning_chunk_number=partitioning_chunk_number        
        self.planning_depth=planning_depth   
        self.env_dynamics=environment
        # results
        
        self.q_func={}
        self.action_func={}
        self.value_func={}  
        
        
        #aux variables
        self.value_function={}
        self.reachable_states={}
        self.all_theta=[]
        self.M=[]
        self.exp_vorfaktoren=[]
        self.x_map={}
        self.F=[]
        self.exp_vorfaktoren=[]
        #simulation parameters
        self.current_internal_state=[]
        self.time_step=0
        self.last_action=-1
        
        return


    def pre_planning(self,exp_vorfaktoren, initial_theta=[0.5,0.5],initial_observation=0 ):
        
        env_dynamics=self.env_dynamics
        planning_depth=self.planning_depth
        partitioning_chunk_number=self.partitioning_chunk_number
        agent_mode=self.agent_mode
        self.exp_vorfaktoren=exp_vorfaktoren
        
        exp_num=len(exp_vorfaktoren)
        # predine theta and its mappings by F and values by G
        int_theta,F,G,M=self.theta_transition_function_md(exp_vorfaktoren=exp_vorfaktoren,env_dynamics=env_dynamics,partitioning_chunk_number=partitioning_chunk_number)
        # universal theta (floadt)
        all_theta=np.array(int_theta)/partitioning_chunk_number
        # initialize extended internal space ( theta_i1, theta_i2, .. , y)
        x0=self.initialize_internal_state(theta_0=initial_theta, observation_0=initial_observation, exp_num=exp_num)
        # calculate possible states for each depth of planning
        if agent_mode=='optimized':
            reachable_states=self.find_reachable_states(intial_x_state=x0,env_dynamics=env_dynamics,F_map_md=F,max_depth=planning_depth,partitioning_chunk_number=partitioning_chunk_number,exp_num=exp_num,all_theta=all_theta)
            #initialize value function ( dictionary from each possible state at each depth to their value (which are set to zero))
            value_function,x_map=self.intial_value_function(reachable_states)
            X=reachable_states
            self.reachable_states=reachable_states
            

        else:
            possible_states=self.naive_reachable_states(env_dynamics=env_dynamics,exp_num=exp_num,all_theta=all_theta)
            # initialize value function
            value_function,x_map=self.naive_initial_value_function(possible_states=possible_states)
            X=possible_states
            
            self.reachable_states=possible_states
        
        self.x_map=x_map    
        self.value_function=value_function
        self.M=M
        self.all_theta=all_theta
        self.exp_vorfaktoren=exp_vorfaktoren
        self.F=F
        self.current_internal_state=x0       
        
        if agent_mode=='optimized':
            return -1,M,F,G,X,value_function,all_theta
        else:
            return x_map,M,F,G,X,value_function,all_theta
        
        
        
    def value_iteration(self):
        
        env_dynamics=self.env_dynamics
        planning_depth=self.planning_depth
        reachable_states=self.reachable_states
        value_function=self.value_function
        agent_mode=self.agent_mode
        
        exp_vorfaktoren=self.exp_vorfaktoren
        
        if agent_mode=='optimized':
            v,a,q,vf=self.value_iteration_optimized(env_dynamics=self.env_dynamics,M_matrix=self.M,exp_vorfaktoren=self.exp_vorfaktoren,planning_depth=self.planning_depth,reachable_states=self.reachable_states,value_function=self.value_function,all_theta=self.all_theta,partitioning_chunk_number=self.partitioning_chunk_number) 
        else:
            v,a,q,vf=self.value_iteration_naive(env_dynamics=self.env_dynamics,M_matrix=self.M,exp_vorfaktoren=self.exp_vorfaktoren,planning_depth=self.planning_depth,reachable_states=self.reachable_states,value_function=self.value_function,all_theta=self.all_theta,partitioning_chunk_number=self.partitioning_chunk_number)

        self.value_function=vf
        self.value_func=v
        self.action_func=a
        self.q_func=q
        
        return self.value_func,self.action_func,self.q_func,self.value_function
     



    def make_M_md(self,exp_vorfaktoren,env_dynamics):
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
        return M


    def make_possible_theta_universal(self,states, observations, partitioning_chunk_number, save=False ):

        actual_partition_num=partitioning_chunk_number+1
        univariate_x=[[s1,partitioning_chunk_number-s1] for s1 in range(actual_partition_num) ]

        indicator=np.arange(len(univariate_x))

        # content of files are dictionaries with keys: different possible Theta values, and values: index/indicator
        x_to_index=dict(zip(list(map(tuple,univariate_x)),indicator)) 

        # save data.
        # not needed for now


        return univariate_x,x_to_index


    def find_next_theta_index_universal(self,theta_values,partitioning_chunk_number,mode='multi'):
        # make valid theta points
       
        in_base_partitioning=np.array(theta_values)*partitioning_chunk_number
        valid_points=np.round(in_base_partitioning)
        if mode=='single':
            return valid_points[0]
        # find their index. Because we have only 2 states, the index of theta point is equal to the value of first element in integer representation.
        return valid_points[:,0]


    def theta_transition_function_md(self,exp_vorfaktoren,env_dynamics,partitioning_chunk_number):
        # make possible theta values
        int_theta,_=self.make_possible_theta_universal(states=env_dynamics.states, observations=env_dynamics.observations, 
                                                  partitioning_chunk_number=partitioning_chunk_number, save=False)
        int_theta=np.array(int_theta)

        ### Allocate theta mappings variables

        # for F-function
        # (i x len(theta) x actions x observation) -> index_of_next_theta_point
        theta_map_md=np.empty((len(exp_vorfaktoren),len(int_theta),len(env_dynamics.actions),len(env_dynamics.observations)),dtype=int)

        # for G -function : (i x len(theta) x actions x observations) -> g-value
        G_map_md=np.empty((len(exp_vorfaktoren),len(int_theta),len(env_dynamics.actions),len(env_dynamics.observations)),dtype=int)


        # actual theta 
        theta=np.array(int_theta)/partitioning_chunk_number

        ## Calculate values

        # M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        M=self.make_M_md(exp_vorfaktoren,env_dynamics)

        # theta mapping function
        for i,vorfaktor in enumerate(exp_vorfaktoren):
            for a,_ in enumerate(env_dynamics.actions):
                for y_prim,_ in enumerate(env_dynamics.observations):

                    z=np.matmul(theta,M[i,a,y_prim,:,:])

                    integral=np.sum(z,axis=1).reshape(len(z),1)          
                    # F- function
                    theta_map_md[i,:,a,y_prim]=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number)
                    # G- function
                    G_map_md[i,:,a,y_prim]=(np.log(integral)/vorfaktor).reshape(len(z),)
        F_map_md=theta_map_md            
        return int_theta,F_map_md,G_map_md,M


    def initialize_internal_state(self,theta_0, observation_0, exp_num):
        # internal state
        x=[]
        # fill it by initial values of theta (2 elemnts) for each variate dimention (number of exponentials)
        for t in range(exp_num):
            x.append(theta_0[0])
            x.append(theta_0[1])
        # append the value of the initial observation (y) to make internal state
        x.extend([observation_0])
        return x    

    def find_reachable_states(self,intial_x_state,env_dynamics,F_map_md,max_depth,partitioning_chunk_number,exp_num,all_theta):
        reachables={}
        reachables[0]=[tuple(intial_x_state)]

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

                            #next_theta_index=F_map_md[i][int(to_extend[i*2]*partitioning_chunk_number)][a][y_prim]
                            next_theta_index=F_map_md[i][tt][a][y_prim]
                            for y in range(len(observations)):
                                next_x.append(all_theta[next_theta_index,observations[y]])
                                #next_x.append(all_theta[next_theta_index,1])
                        next_x.append(y_prim)

                        new_entry=tuple(next_x)

                        if not (new_entry in reachables[depth]):
                            reachables[depth].append(new_entry)


        return reachables

    def naive_reachable_states(self,env_dynamics,exp_num,all_theta):
        observations=list(env_dynamics.observations.keys())
        theta_part=all_theta.copy()
        for i in range(1,exp_num):
            theta_part=it.product(theta_part,all_theta)
        theta_part= list(map(list,theta_part))
        theta_part=np.array(theta_part)
        theta_part=theta_part.reshape(len(theta_part),exp_num*2)

        tmp1=[]
        for y in observations:
            tmp1.extend([y]*len(theta_part))


        theta_part=np.tile(theta_part,(len(observations),1))
        tmp1=np.array(tmp1).reshape(len(tmp1),1)

        all_possible_internal_x=np.hstack((theta_part,tmp1))

        return all_possible_internal_x

    def intial_value_function(self,reachable_states):
        value_function={}
        x_map={}
        for step in range(len(reachable_states.keys())):
            value_function= dict((key,0) for key in reachable_states[step])
            indexes=np.arange(len(reachable_states[step]))
            x_map[step]=dict(zip(reachable_states[step],indexes))
        return value_function,x_map

    def naive_initial_value_function(self,possible_states):
        init_values=np.zeros(len(possible_states))
        possible_states=list(map(tuple,possible_states))
        value_function=dict(zip(possible_states,init_values))
        indexes=np.arange(len(possible_states))
        x_map=dict(zip(possible_states,indexes))
        return value_function,x_map


    def value_iteration_optimized(self,env_dynamics,M_matrix,exp_vorfaktoren,planning_depth,reachable_states,value_function,all_theta,partitioning_chunk_number):

        M=M_matrix
        exp_num=len(exp_vorfaktoren)
        max_depth=planning_depth

        discount=env_dynamics.discount_factor

        num_action=len(env_dynamics.actions.keys())
        num_observation=len(env_dynamics.observations.keys())


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

                for y_prim in range(num_observation):
                    this_y_prim_nextState=np.empty((len(this_step),2))
                    this_y_prim_cost=np.zeros((len(this_step),1))
                    for i in range(exp_num):

                        this_step=np.array(this_step)

                        theta_i=this_step[:,i*2:2*i+2].copy()

                        z=np.matmul(theta_i,M[i,a,y_prim,:,:])


                        integral=np.sum(z,axis=1).reshape(len(z),1)          
                        # F- function
                        next_theta_i=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number).reshape(len(this_step),1)

                        # cost function
                        g_i=(np.log(integral)/exp_vorfaktoren[i]).reshape(len(z),)
                        c_i=(g_i+np.log(num_observation)).reshape(len(this_step),1)                  

                        # utility of cost
                        this_y_prim_cost+=this_y_prim_cost+np.exp(c_i)


                        if i==0:

                            this_y_prim_nextState=np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)
                        else:
                            this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.squeeze(all_theta[next_theta_i.astype(int)],axis=1)),axis=1)
                    this_y_prim_nextState=np.concatenate((this_y_prim_nextState,np.array([y_prim]*len(this_step)).reshape(len(this_step),1)),axis=1)

                    # if max step
                    if step==max_depth-1:

                        this_action_q+=this_y_prim_cost * (1./num_observation)
                    else:
                        # else
                        next_v=[]
                        for r,thet in enumerate(this_y_prim_nextState):
                            thet=tuple(thet)

                            next_v.append(value_function[step+1][thet])
                        next_v=np.array(next_v).reshape(len(this_step),1)
                        this_action_q+= (1./num_observation)*(this_y_prim_cost + discount* next_v )  


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

        return value_func,action_func,q_func,value_function

    def value_iteration_naive(self,env_dynamics,M_matrix,exp_vorfaktoren,planning_depth,reachable_states,value_function,all_theta,partitioning_chunk_number):

        M=M_matrix
        exp_num=len(exp_vorfaktoren)
        max_depth=planning_depth

        discount=env_dynamics.discount_factor

        num_action=len(env_dynamics.actions.keys())
        num_observation=len(env_dynamics.observations.keys())

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

                for y_prim in range(num_observation):
                    this_y_prim_nextState=np.empty((len(value_function),2))
                    this_y_prim_cost=np.zeros((len(value_function),1))


                    ### calculations #############################

                    # for each exponential part
                    for i in range(exp_num):
                        # fetch data of related theta
                        theta_i=reachable_states[:,i*2:i*2+2].copy()

                        # Calculate F- function
                        z=np.matmul(theta_i,M[i,a,y_prim,:,:])
                        integral=np.sum(z,axis=1).reshape(len(z),1) 
                        next_theta_i=self.find_next_theta_index_universal(z/integral,partitioning_chunk_number).reshape(len(value_function),1)

                        # Calculate G-function
                        g_i=(np.log(integral)/exp_vorfaktoren[i]).reshape(len(z),)

                        # Calculate cost-function
                        c_i=(g_i+np.log(num_observation)).reshape(len(value_function),1)  


                        # Aggregate sum of all exponential results 
                        # utility of cost
                        this_y_prim_cost+=this_y_prim_cost+np.exp(c_i)

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
                        this_action_q+=this_y_prim_cost * (1./num_observation)
                    else:

                        #find value of the next related states
                        this_y_prim_nextState=list(map(tuple,this_y_prim_nextState))
                        next_v=value_function[this_y_prim_nextState]                    
                        next_v=np.array(next_v).reshape(len(next_v),1)

                        # add next states' values to immediate rewards
                        this_action_q+= (1./num_observation)*(this_y_prim_cost + discount* next_v )

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

        return value_func,action_func,q_func,value_function
    
    def reset(self,initial_theta=[0.5,0.5],initial_observation=0):
        
        exp_num=len(self.exp_vorfaktoren)
        self.time_step=0
        self.current_internal_state=self.initialize_internal_state(theta_0=initial_theta, observation_0=initial_observation, exp_num=exp_num)
        
        return
    
    def do_action(self):
        
        if self.agent_mode=='optimized':
            action=np.argmax(self.q_func[self.time_step][self.x_map[self.time_step][tuple(self.current_internal_state)]])
            value_of_action=np.max(self.q_func[self.time_step][self.x_map[self.time_step][tuple(self.current_internal_state)]])
        else:
            action=np.argmax(self.q_func[self.time_step][self.x_map[tuple(self.current_internal_state)]])
            value_of_action=np.max(self.q_func[self.time_step][self.x_map[tuple(self.current_internal_state)]])
        
        self.last_action=action
        
        return action,value_of_action
    
    def update_agent(self,new_observation):
        # update internal state
        next_internal_x=[]
        for i in range(len(self.exp_vorfaktoren)):
            theta_i=self.current_internal_state[i*2:i*2+2]
            theta_index=int(self.find_next_theta_index_universal(theta_i,partitioning_chunk_number=self.partitioning_chunk_number,mode='single'))
            theta_i_next=self.all_theta[self.F[i][theta_index][self.last_action][new_observation]]
            
            if i==0:
                next_internal_x=list(theta_i_next)
            else:               
                next_internal_x.extend(theta_i_next)
        
        next_internal_x.extend([new_observation])
        
        self.current_internal_state=next_internal_x
        
        # update time
        self.time_step+=1
        
        return self.current_internal_state
        
        