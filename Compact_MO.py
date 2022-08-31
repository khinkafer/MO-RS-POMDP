import itertools as it
import pandas as pd
import numpy as np
import os as os
import gc

class Multi_Variate_agent(object):
    
    def __init__(self,environment, max_time_step):
        
        '''
        initialize the agent.
        
        Parameters:
        -----------
        environment : env class 
            environment
        max_time_step: int
            the decision depth
        
        Returns:
        --------
        
        Nothing. But create the value_func, action_func and q_func dictionaries.
            
        '''
        
        # environment
        self.env=environment
                        
        # time_step=0 means initial point, therefore time_step=n means decision-tree with n actions depth. 
        self.max_time_step=max_time_step
        

        # results
        
        # Continious optimized method results
        # value function of each internal state in each time step. value function is a 2-level nested dictionaries: The outer dict.
        # is a mapping from time_steps to inner dicts. Inner dicts are mappings from interal states to their values. Internal states 
        # are represented as tuples of (X:which is a tuple of (theta of the first exponential, 
        # theta of the second, ..., last (current) observation) , R: which is a tupel of (accumulated reward
        # regarding the first exp, acc_reward of exp2 ...) , and Z: which is the time step.
        # e.g for u=e1+e2 : ((th1,th2,y'),(r1,r2),timeStep)
        # In our internal state representation, storing time_step is redundant, however
        # to be similar with original paper's modeling, we used internal_states excactly like that.
        self.value_func={}
        # Like the above, but final values are the index of the best actions (the actions which had those maximum values)
        self.action_func={}
        # Like the aboves, but the values are a list of values of all actions in that internal state.
        self.q_func={}
        
        
        
        # aux-results
       
        # reachables is a dictionary with keys: time steps & values: list of all possible internal states. 
        self.reachable_states={}
               
        # M matrix
        self.M=[]  
        
        # Transition Kernel
        # a mapping from each internal state and action and next observation to the successive internal state.
        # (((th1,th2,..y),(r1,r2,...),time),a,y')--> ((th1',th2',...),(r1,r2,...)),y')
        self.transition_kernel={}    
        
     
        # aux-parameters
        
        self.initial_wealth=self.env.initial_wealth
        self.discount_factor=self.env.discount_factor
        self.num_of_actions=len(self.env.actions)
        self.num_of_observable_states=len(self.env.observations)
        self.num_of_unbservable_states=len(self.env.states)
        self.num_of_observations=self.num_of_observable_states
        self.num_of_states=self.num_of_unbservable_states 
        
        #set x_round to 5 as a default
        self.x_round=5
        return
    

    def make_M_md(self,env_dynamics,exp_vorfaktoren):
        '''
        Calculating the M matrix

        Parameters
        ----------
        env_dynamics : env class
            Dynamics of environment.
        exp_vorfaktoren : list of floats
            Power of exponential terms in the Utility function.

        Returns
        -------
        M : a 5-dimensional Matrix of floats. 
            Calculated M matrix (see the paper). The dimensions of M are: [each exponential terms] x [actions] x [observations] x [state] x [next state]

        '''
        
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
    
    
    def continious_optimized_reachable_states(self, initial_aug_state,max_time_step,rounding_prec_coeff,x_round,r_round):
        '''
        
        This function takes the initial internal state and finds all successive internal states. It also returns 
        the transition kernel between internal states, and make and initialize a value function with zero values.
        
        Parameters
        ----------
        initial_aug_state : tuple 
            the initial state. It is a tuple contains 3 parts: X(tuple of theta1 dist., theta2, ... ,and last_observation ), R(tuple of accumulated costs for each variant: r1,r2,...) , and time step. Time step is
            redundant here (because we know it is in the time_step=0 ), but in sake of similarity with the paper we record time_step here as well.
        max_time_step : int
            maximum depth for planning.
        rounding_prec_coeff : big int
            Used for float variables comparisons (not used now).
        x_round: int
            number of integer points for roundings regarding theta points
        r_round: int
            number of integer points for roundings regarding accumulated reward points

        Returns
        -------
        reachables : dict. (int->list)
            From time_steps(int) to a list of tuples (possible successive internal states).
        universal_map : dict [int -> dict. (tuple -> tuple)]
            Transition kernel between internal states. A nested dictionary whose outer layer is mapping from time_step to inner dict. inner dict's keys are tuples contain(internal_state, action, next_observation) and values are tuples that contain next_internal_state.
        value_func : dict [int -> dict ( tuples -> float)]
            Initialized Value function (with zero). A nested dict like the above. Inner dict is a mapping from internal states to their assigned value (0).

        '''
        
        # setting parameters
        exp_num=self.exp_num
        num_of_actions=self.num_of_actions
        num_of_observations=self.num_of_observations
        
        
        # result variables
        
        # reachables is a dictionary with keys: time steps & values: list of all possible internal states. 
        # Each internal state has represented by a tuple contains: (X:which is a tuple of (theta of the first exponential, theta of the second, ..., last (current) observation) , R: which is a tupel of (accumulated reward
        # regarding the first exp, acc_reward of exp2 ...) , and Z: which is the time step.
        # e.g for u=e1+e2 : ((th1,th2,y'),(r1,r2),timeStep)
        reachables={}   
        
        # mapping from each internal state and action and next observation to the successive internal state.
        # (((th1,th2,..y),(r1,r2,...),time),a,y')--> ((th1',th2',...),(r1,r2,...)),y')
        universal_map={}   
        
        # a mapping form each internal state to its value, which is set to 0 for initialization.
        value_func={}
        
        
        # set the first step
        reachables[0]=[initial_aug_state]
        
        # M matrix
        M=self.M
        
        # until the last step
        for depth in range(1,max_time_step+1):
            
            reachables[depth]=[]
            
            for to_extend in reachables[depth-1]: 
                
                # State which we want to find its successors
                to_extend_r=np.array(to_extend[1])
                to_extend_theta=np.array(to_extend[0])
                
                # this expansion of the decision three (by a specific action and a specific observation)
                #this_extention=[]
                
                for a in range(num_of_actions):  
                    for y_prim in range(num_of_observations):
                        
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
                            
                            next_theta=self.myRound(next_theta,decimals=x_round)
                            
                            next_theta=next_theta.tolist()
                            next_x_theta.extend(next_theta)
                            
                            # Cost part
                            # G- function: next imediate reward
                            g_i=np.log(integral)/self.exp_vorfaktoren[i]
                            # C= G + log(|y|^(1/lambda_i))
                            c_i=g_i +np.log(np.power(num_of_observations,1/self.exp_vorfaktoren[i]))
                            
                            
                            c_i=self.myRound(c_i,decimals=x_round)
                            c_i=c_i.tolist()
                            # make vector of costs of each exponential dimension
                            transition_cost_vector.append(c_i)
                            
                            
                            
                            
                        # Add y' to the next theta to make next state: (th1', th2',... , y')
                        next_x_theta.append(y_prim)
                        
                        # update R part of the state
                        z=np.power(self.env.discount_factor,(depth-1))                       
                        z=self.myRound(z,decimals=r_round)
                        
                        next_x_r=to_extend_r+ np.array(transition_cost_vector)*z
                        next_x_r=self.myRound(next_x_r,decimals=r_round)
                        
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
         
        # Initialize the value_func 
        # for each time step
        
        for step in range(len(reachables)):
            # setting value of that time step's states to zero (make a dict. from states to values(0s)).
            value_func[step]= dict((key,0) for key in reachables[step])
             
                           
        return reachables,universal_map,value_func
      
        
    def continious_optimized_planning(self, initial_theta,initial_observation,initial_wealth,exp_weights,exp_vorfaktoren,
                                      rounding_prec_coeff=100000,x_round=5,r_round=5,utility_approx=False, apx_terms_num=2):
        
        '''
        This function first reset the agent to its initial internal state, then based on that, it performs forward and backward 
        steps srerialy: First: it finds all possible internal states (in each time step) and make a transition kernel that 
        determines all possible internal state transitions during the planning (via continious_optimized_reachable_states()).
        Second, it  performs value iteration backwardly.
        
        Parameters:
        -----------
        initial_theta : list of two floats
            the probability of being in each hidden state in step 0, when the wealth is equal to initial_wealth (theta0)
        initiative_observation : int
            The first x (observable state).
        initial_wealth: float
        
        exp_weights: list of floats
            weights of exponential elements
        exp_vorfaktoren: list of floats
            power value of exponential elements
        rounding_prec_coeff: big int
            1 divide by this value will be the threshhold which we use to check floating point equalities
        x_round: int
            number of integer points for roundings regarding theta points
        r_round: int
            number of integer points for roundings regarding accumulated reward points
        Returns:
        --------
        0: int
            A dummy return. It is here just to the shape of output be similiar to this function in Beurele method's implementation
        reachable_states: dict of list of tuples 
            time_steps -> list of possibe internal states in that time step.
        Transition_kernel: dict of dicts( tuples -> tuples)
            outer dict is time_steps to inner dicts. inner dict is a mapping from 'all combinations of possible internal_states, actions 
            and succesive observations' to successive next internal_states 
        q_func: dict of dicts (tuples -> list)
            outter dict is a mapping of time_steps to inner dicts. Inner dicts are mappings from tuple of internal_states to list of 
            their Q-values (values for doing each action in them)
        value_func: dict of dicts(tuples -> floats)
            like the above, but the inner dicts are from internal_states to their values (V-values or max(Q-values))
        action_func:  dict of dicts(tuples -> int)
            like the above but the inner dicts are from internal_states to their best actions (argmax(Q-values)).
        '''
        
        # set agent's mode to 'optimized_continious' or 'naive_discrete'
        self.agent_mode='optimized_continious'
        
        # setting parameters
        observations=list(self.env.observations.keys())
        
        terms_weights=[1]*apx_terms_num
        if utility_approx==True:
            self.exp_vorfaktoren=[1]*apx_terms_num
            self.exp_weights=terms_weights
        else:
            self.exp_vorfaktoren=exp_vorfaktoren
            self.exp_weights=exp_weights
            
        self.exp_num=len(self.exp_vorfaktoren)
        self.intial_theta=initial_theta
        self.initial_observation=initial_observation
        
        # initial states         
        self.reset(initial_theta,initial_observation,initial_wealth)
        
        # make M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        if utility_approx:
            self.M=self.make_apx_M(self.env,apx_terms_num)
        else:
            self.M=self.make_M_md(self.env,self.exp_vorfaktoren)
        

        
        #######################################################################
        #                              Forward                                #
        #######################################################################
        
        # find continoius (without partitioning)reachable internal states for each time step
        # also find their transition kernel
        
        self.reachable_states,self.transition_kernel,self.value_func=self.continious_optimized_reachable_states(
            initial_aug_state=self.initial_state,max_time_step=self.max_time_step,rounding_prec_coeff=rounding_prec_coeff,x_round=x_round,r_round=r_round)
        
        
        #######################################################################
        #                              Backward                               #
        #######################################################################
        
        # backwardly
        
        # from self.max_time_step-1: becasue in our modeling the initial state is 0 and for n step of planning, 
        # the last step of decision-making will happends at the time step= n-1
        for step in range(self.max_time_step,-1,-1):
            
            # fetch list of all possible states in this time Step
            this_step=list(map(list,self.reachable_states[step])).copy()
            
            this_step_theta=list(map(list,(np.array(this_step,dtype=object)[:,0])))
            this_step_r=list(map(list,(np.array(this_step,dtype=object)[:,1])))
            
            # make an empty Q(internal_state, actions)
            
            self.q_func[step]={}
            self.action_func[step]={}
            self.value_func[step]={}
            
            if step==self.max_time_step:
                val=np.zeros((len(this_step),1))
                this_step_r=np.array(this_step_r)
                for i in range(self.exp_num):
                    
                    val[:,0]+=np.exp(this_step_r[:,i]*self.exp_vorfaktoren[i])*self.exp_weights[i]
                    
                for p,point in enumerate(self.reachable_states[step]):
                    
                    self.q_func[step][point]=[val[p,0].copy()]*self.num_of_actions
                   
                    self.action_func[step][point]=-1
                    self.value_func[step][point]=val[p,0].copy()
            else:
                
                
                for p,point in enumerate(self.reachable_states[step]):
                    
                    this_state_q_values=[0]*self.num_of_actions
                    for action in range(self.num_of_actions):
                        
                        for y_prim in range(self.num_of_observations):
                            
                            val=self.value_func[step+1][self.transition_kernel[(point,action,y_prim)]]
                            pv=(1./self.num_of_observations)*val
                            
                            pv=self.myRound(pv,decimals=r_round)
                            
                            this_state_q_values[action]+=pv
                            this_state_q_values[action]=self.myRound(this_state_q_values[action],decimals=r_round)
                            
                    self.q_func[step][point]=this_state_q_values.copy()
                    self.action_func[step][point]=np.argmax(this_state_q_values)
                    self.value_func[step][point]=np.max(this_state_q_values)
                                
        
        return 0,self.reachable_states,self.transition_kernel,self.q_func,self.value_func,self.action_func

    def myRound(self,f,rounding_prec_coeff=100000,decimals=10):
        '''
        This function rounds the float decimals. In the latest version it rounds to 10 decimal points but it can do smthg else.
        
        Parameters:
        ----------
        f : floating point
           the number that we want to round
        rounding_prec_coeff : big int
            coefficient for some rounding methods
        decimals: int
            number of integer points we want to keep
        Returns:
        -------
        f : float
            the rounded value of the input
        '''
        #f=np.multiply(f,rounding_prec_coeff)
        #f=f.astype(np.int64)
        #f=f.astype(np.float64)/rounding_prec_coeff
        f=np.round(f,decimals=decimals)
        return f

    def reset(self,theta_0,initial_observation,initial_wealth):
        '''
        This function resets the agent to its belief about the initial wealth and set time-step to zero (begining of the simulation).
        If the agent_mode is 'naive_discrete', then it starts from the initial state and extract all possible paths from that state during the planning and make 
        reachable_states, value_func,action_func, q_func and transition_kernel only for the states in these paths. the represaentation of states here is like what we have in
        the paper and the 'optimized_continious' agent_mode. 

        Parameters
        ----------
        theta_0 : list of two floats
            the probability of being in each hidden state at step time =0 , when the wealth is equal to initial_wealth
        initiative_observation : int
            The first x (observable state).
        initial_wealth: float
        Returns
        -------
        Noting. makes the initial internal_state and makes the agent ready for a new simulation.

        '''
        
        # internal state
        x=[]
        r=[]
        aug=[]
        # fill it by initial values of theta (2 elemnts) for each variate dimension (number of exponentials)
        for t in range(self.exp_num):
            x.append(theta_0[0])
            x.append(theta_0[1])
            r.append(initial_wealth)
            
        # append the value of the initial observation (y) to make internal X-state
        x.extend([initial_observation])
        
        # to make augmented states
        aug=tuple([tuple(x), tuple(r),0])
        
        ############ setting internal variables
        # set the time step to 0
        self.current_internal_timeStep=0
        
        # set the initial state
        self.initial_state=aug
        
        # set the current state to initial state
        self.current_internal_state=self.initial_state
        
        # defien last action
        self.last_action=None
        
        # to load all possible actions in the 'naive_discrete' mode
        if self.agent_mode=='naive_discrete':
            self.summarized_reachables={}
            self.summarized_reachables_indexes={}
            
            self.summarized_transition_kernel={}
            self.summarized_value_func={}
            self.summarized_action_func={}
            self.summarized_q_func={}
            
            
            for step in range(self.max_time_step+1):
                
                #################### set variables
                
                self.summarized_reachables_indexes[step]=[]  
                self.summarized_reachables[step]=[]
                
                self.summarized_transition_kernel[step]={}
                
                self.summarized_value_func[step]={}
                self.summarized_action_func[step]={}
                self.summarized_q_func[step]={}
                ##################### Fetch data
                
                # fetch reachable states and transition kernel of that step
                reachables_path=os.path.join(os.getcwd(),'MO_state_space_discrete')
                reachables=np.load(os.path.join(reachables_path,'depth'+str(step)+'.npy'))
                
                # don't read any file for transition kernal from the last depth
                if step<self.max_time_step:
                    if self.saved_transition_kernel:
                        tk_path=os.path.join(os.getcwd(),'MO_transition_kernel_discrete')
                        tk=pd.read_pickle(os.path.join(tk_path,'depth'+str(step)+'_'+str(step+1)+'.pkl'))
                    else:
                        tk=pd.DataFrame(self.transition_kernel[step])
                
                #################### state complete representation
                # for initial state
                if step==0:
                    
                    # where it is equal to initial state
                    # to fetch just first elements of thetas
                    first_th=[]
                    # find the value and index of compelete representaition of initial state                   
                    for i in range(self.exp_num):
                        first_th.append(self.current_internal_state[0][i*2])
                    first_th=np.round(np.array(first_th)*self.partitioning_points_num).astype(np.int8)    
                    first_th=tuple(first_th)
                    reachables=dict(zip(list(map(tuple,reachables[:,:self.exp_num])),np.arange(len(reachables))))
                    
                    self.summarized_reachables_indexes[step]=[reachables[first_th] ]  
                    self.summarized_reachables[step]=[self.current_internal_state]
                else:
                    # make state by indexes
                    for rec in list(self.summarized_transition_kernel[step-1].keys()):
                        idx=self.summarized_transition_kernel[step-1][rec]
                        raw_state=reachables[idx]
                        
                        # make full representation of state
                        x=[]
                        r=[]
                        for i in range(self.exp_num):
                            x.append(raw_state[i])
                            x.append(self.partitioning_points_num-raw_state[i])
                            r.append(raw_state[-self.exp_num+i])
                        x=self.myRound(np.array(x)/self.partitioning_points_num,decimals=self.x_round)
                        x=tuple(x)
                        r=tuple(r)
                        y=rec[2]
                        state=tuple([x,r,y])
                        
                        # replace the index of next_state in the previous transition function by the complete representation of the state
                        self.summarized_transition_kernel[step-1][rec]=state
                        # add the state to reachables
                        self.summarized_reachables[step].append(state)
                        # add the index of the state
                        self.summarized_reachables_indexes[step].append(idx)
                        
                    
                
                ###################     update states data
                #print(self.summarized_reachables_indexes[step])
                for s,state_idx in enumerate(self.summarized_reachables_indexes[step]):
                    state=self.summarized_reachables[step][s]
                    self.summarized_value_func[step][state]=self.value_func[step][state_idx]
                    self.summarized_action_func[step][state]=self.action_func[step][state_idx]
                    
                    ###############     expand successive states
                    
                    
                        
                    if step <self.max_time_step:
                        # do not run it for the last step
                        if self.keep_q:
                            #print(self.q_func)
                            self.summarized_q_func[step][state]=self.q_func[step][state_idx,:]
                        else:
                            self.summarized_q_func[step]=[]
                            
                        for a in range(self.num_of_actions):
                            
                            for y_prime in range(self.num_of_observations):
                                
                                next_index=tk.loc[state_idx,str(a)+'_'+str(y_prime)]
                                
                                self.summarized_transition_kernel[step][tuple([state,a,y_prime])]= next_index
            

        return 
    
    # simulation functs
    def do_action(self):
        '''
        This function returns the best action based-on: current time-step, current internal_state
        
        Returns
        -------
        best_action: int 
            Among actions' indexes (1-5)
        value_of_action: float
            value of doing that action at that (x,Mu,z) point
        current_internal_state: tuple 
            the current internal state which contains (X:((theta1_1,th1_2, th2_1, ..., last_observation)), R:(r1,r2,...), and time_step)
        '''
        # for optimize continious mode
        if self.agent_mode=='optimized_continious':
            action=self.action_func[self.current_internal_timeStep][self.current_internal_state]
            value_of_action=self.value_func[self.current_internal_timeStep][self.current_internal_state]
        # for naive_discrete mode
        else:
            action=self.summarized_action_func[self.current_internal_timeStep][self.current_internal_state]
            value_of_action=self.summarized_value_func[self.current_internal_timeStep][self.current_internal_state]
        
        self.last_action=action
        
        return action,value_of_action,self.current_internal_state
     
    def update_agent(self,new_observation):
        
        '''
        This function takes the new observable state, and based-on the Transition kernel updates the internal state, as well as
        agent's current time_step
        
        Parameters
        ----------
        new_oservation : int ( 0 or 1)
            the observation from environment, after the agent's action
        Returns:
        --------
        current_internal_state: tuple of (X:((theta1_1,th1_2, th2_1, ..., last_observation)), R:(r1,r2,...), and time_step)
            It is the current internal_state AFTER updating.
        '''
        
        
        if self.current_internal_timeStep>=self.max_time_step:
            print('too much iterations!')
            return
        else:      
            
            # update the current internal state
            if self.agent_mode=='optimized_continious':
                self.current_internal_state=self.transition_kernel[(self.current_internal_state,self.last_action,new_observation)]
            else:
                self.current_internal_state=self.summarized_transition_kernel[self.current_internal_timeStep][(self.current_internal_state,self.last_action,new_observation)]
            
            # increase time-step
            self.current_internal_timeStep=self.current_internal_timeStep+1
            
            # update current observable-state 
            self.current_internal_x=new_observation
            
            return self.current_internal_state    
        
        
        
    # and functions to make all possible internal states
    
    def make_all_internal_states(self,max_time_step,partitioning_points_num,initial_wealth,exp_weights,exp_vorfaktoren,rounding_prec_coeff=100000,r_round=5,save_space=True, keep_all_steps=True):
        '''
        
        This function starts from all possible discretized internal states. i.e: disceretized thetas and initiail wealth. And, generates all reachable internal states during the depth
        of planning. It can save the ruslts and can contain all calculated internal states in memory.
        Note: in this function internal states are defined as tuple of : tuple of thetas , and tuple of accumulated rewards. Time is not recorded in the state representation becasue we record it as time_step iun the keys of the dictionary so
        it was redundant. Also, the last observation is not recorded because it has no effect on successive state and cost. The last observation's impact has been applied in the calculation of current state and current accumulated cost in
        the a loop. i.e: if we reach the same theta and accumulated cost by different last observations, there is no difference in continiue of the planning so because of Markovian property we can assume they are the same.
         
        
        Parameters
        ----------
        max_time_step : int
            maximum depth of planning.
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the theta distributions.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}     
        initial_wealth : float
            initial wealth.
        exp_weights : list of floats
            weight of exponential terms in the Utility function.
        exp_vorfaktoren : list of floats
            power of exponential terms in the Utility function..
        rounding_prec_coeff : big int, optional
            Used as measure of comparison between floating points.difference less than 1 dvided by this number would treated as equal . The default is 100000.
        r_round : int, optional
            Number of decimal digits in rounding accumulated rewards. The default is 5.
        save_space : Boolean, optional
            Save rechable internal states or not. It saves time_step by time_step in distict files. The default is True.
        keep_all_steps : Boolean, optional
            Keep calculated internal states during all time_steps in memory and return them. The default is True.

        Returns
        -------
        if keep_all_steps is True: 
            universal_int_states: dict from time_step to list of all possible diceretized internal_states at that time_step
            counts: a list containing number of points in internal_state in time_steps
        else:
            None.

        '''
        # setting parameters
        observations=list(self.env.observations.keys())
        self.max_time_step=max_time_step
        self.r_round=r_round
        self.exp_vorfaktoren=exp_vorfaktoren
        self.exp_weights=exp_weights
        self.exp_num=len(self.exp_vorfaktoren)
        # actual number is one extra point. i.e: self.partitioning_points_num=3 means valid points of {0,1,2,3} or {0%,33%,66%,100%}
        self.partitioning_points_num=partitioning_points_num
        
        # make M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        self.M=self.make_M_md(self.env,self.exp_vorfaktoren)
        
        #######################################################################
        #                   First Depth's Possible Points                     #
        #######################################################################
        
        # a prototype for all thata combinations over 2 states
        t0=[[i,self.partitioning_points_num-i] for i in range(self.partitioning_points_num+1)]
        
        # theta-part
        for i in range(self.exp_num):
            # save the first theta
            if i==0:
                pre=np.array(t0).reshape((len(t0),2))
            else:
                # for each elemnt of thetas, product it by all previous combinations to make all together_combinations
                for r,rec in enumerate(t0):
                    cons=np.tile(rec,len(pre)).reshape(len(pre),2)
                    if r==0:
                        post=np.hstack([cons,pre])
                    else:
                        post=np.vstack([post,np.hstack([cons,pre])])
                # the new combination should be used as previous combinations of new theta points
                pre=post
        # Theta parts of the internal state. In the original setting, the last observation is also a part of x, but it was unneccessary for us.
        x0=pre
        
        
        # R-part
        # first accumulated reward prototype that filled by initial_wealth
        r0=np.array([initial_wealth]*np.power(len(t0),len(exp_vorfaktoren)).astype(np.int32)).reshape(-1,1)
        
        for i in range(len(exp_vorfaktoren)):
            if i==0:
                final_r0=r0.copy()
            else:
                final_r0=np.hstack([final_r0,r0])
                
        # Whole internal_state at the first step (step=0). Again, there is a time_step elemnt in the original setting which is represented here by seperating each time_steps' interal_states. 
        state0=np.hstack([x0,final_r0])
        
        #######################################################################
        #                   Other Depth's Possible Points                     #
        #######################################################################
         
        
        # make the path of saving the internal-statespace. 
        # by state_space we mean that a disceretized thetas (2d) for each exponential and accumulated rewards (1d) for each of them
        # in each time step (because possible accumulated rewards values depend on time_step)
        self.space_path=os.path.join(os.getcwd(),'MO_state_space_continious')
        
        # all reachable internal states in each time_step (without observation part) 
        universal_int_states={}
        # number of each time_step's space points
        counts=[]
        
        
        ##### other steps
        for step in range(0,self.max_time_step+1):
            print(step)
            if step==0:
                # previous step's state space
                #prev=state0.copy()
                
                # new step's state space
                reachables=state0
                
                
            else:
                #prev=reachables.copy()
                
                reachables=self.expand_nextStep_states(exp_vorfaktoren=self.exp_vorfaktoren,prev_state=reachables,M=self.M,partitioning_points_num=self.partitioning_points_num,r_round=self.r_round)
                
                
            ################## save
            if save_space==True:
                if not os.path.exists(self.space_path):
                    os.makedirs(self.space_path)
                file_path=os.path.join(self.space_path,'time_step_'+str(step)+'.npy')
                np.save(file_path,reachables)
            
            ################# keep steps in memmory
            if keep_all_steps==True:
                counts.append(len(reachables))
                universal_int_states[step]=reachables
        
        
            
        if keep_all_steps==True:        
            return universal_int_states,counts
        else:
            return 
       
    def expand_nextStep_states(self,exp_vorfaktoren,prev_state,M,partitioning_points_num,r_round):
        '''
        It takes the previous step's disceretized internal_state and produce the successive diceretized internal_state points. The next internal_states are only those points that are reachable by the values of environment dynamics and
        utility function.

        Parameters
        ----------
        exp_vorfaktoren : list of floats
            power of exponential terms in the Utility function..
        prev_state : numpy array of tuples 
            All possible internal_state points in the previous time_step. A list or numpy array which contains the tuples which are internal states in the previous time_step.
        M : 5-dimensional numpy array of floats. 
            Calculated M matrix (see the paper). The dimensions of M are: [each exponential terms] x [actions] x [observations] x [state] x [next state]
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the theta distributions.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}  
        r_round : int, optional
            Number of decimal digits in rounding accumulated rewards.

        Returns
        -------
        reachables: numpy array
            an array of tuples which are the next time_step's reachable internal states. The representation of states here is like make_all_internal_states(). i.e: Neither time_step nor last_observation has recorded. 

        '''
        
        # fetch data
        prev_state=np.array(prev_state)
        x_prev=prev_state[:,:self.exp_num*2]
        r_prev=prev_state[:,self.exp_num*2:]
        
        
        # prepare result variables
        x=np.empty((len(x_prev)*self.num_of_actions*self.num_of_observations,1))
        r=np.empty((len(r_prev)*self.num_of_actions*self.num_of_observations,self.exp_num))
        
        
        for i in range(self.exp_num):
            
            # fetch each variate data
            this_theta=x_prev[:,2*i:2*i+2]
            # make theta floating point in [0,1] 
            this_theta=this_theta/np.sum(this_theta,axis=1).reshape(len(this_theta),1)
            this_theta=self.myRound(this_theta,rounding_prec_coeff=100000,decimals=r_round)
            
            this_r=r_prev[:,i]
            # to save all successive states regarding (action , observation) combinations 
            ct=[]
            ft=[]
            for a in range(self.num_of_actions):
            
                for y_prime in range(self.num_of_observations):
    
                    z=np.matmul(M[i,a,y_prime,:,:],this_theta.T)
                    integral=np.sum(z,axis=0)
                    lg=np.log(integral)
                    G=1./exp_vorfaktoren[i] * lg
                    c=G+np.log(np.power(self.num_of_observations,1./exp_vorfaktoren[i]))
                    ct.extend(list(c))
    
                    f=(z/integral).T
                    
                    # make integer theta again
                    # first column
                    fr=list(np.round(f[:,0]*self.partitioning_points_num).astype(np.int32))
                    # the other column
                    F=[[i,self.partitioning_points_num-i] for i in fr]
                    ft.extend(F)
                    
            # put all thetas together and make x
            if i==0:
                x=np.array(ft)[:,0].reshape(len(ft),1)
            else:    
                x=np.hstack([x,np.array(ft)[:,0].reshape(len(ft),1)])
            x=np.hstack([x,np.array(ft)[:,1].reshape(len(ft),1)])        
    
            # make overal cost
            r[:,i]=np.tile(this_r,self.num_of_actions*self.num_of_observations)+self.discount_factor*np.array(ct)
    
            
        # round accumulated costs
        r=self.myRound(r,decimals=self.r_round)
        
        # make internal states
        reachables=np.hstack([x,r])
        # remove redundant states
        reachables=list((map(tuple,reachables)))
        reachables=list(dict.fromkeys(reachables,0).keys())
        
        return np.array(reachables)
    
    def load_continious_space(self):
        '''
        Load all saved Mu_space points from disk (address is same as saving function) and make a universal variable of them.
        
        Returns:
        --------
        universal_int_states: a dict from time_steps to list of all possible internal state points in that step. 
        theta part of the internal_states expressed by integers to reduce the size, like 'make_all_integer_states()
        '''
        universal_int_states={}
        self.space_path=os.path.join(os.getcwd(),'MO_state_space_continious')
        for step in range(self.max_time_step+1):
            if not os.path.exists(self.space_path):
                print('directory not found!')
                return
            file_path=os.path.join(self.space_path,'time_step_'+str(step)+'.npy')
            thisStep_stateSpace=np.load(file_path,allow_pickle=True)
            universal_int_states[step]=thisStep_stateSpace
        return universal_int_states
    

    def make_all_internal_states_CN_discrete(self,max_time_step,partitioning_points_num,initial_wealth,exp_weights,exp_vorfaktoren,rounding_prec_coeff=100000,r_round=5,save_space=True, keep_all_steps=True):
        '''
        
        This function starts from all possible discretized internal states. i.e: disceretized thetas and initiail wealth. And, generates all reachable internal states during the depth
        of planning. It can save the ruslts and can contain all calculated internal states in memory.
        Note: in this function internal states are defined as tuple of : tuple of thetas , and tuple of accumulated rewards. Time is not recorded in the state representation becasue we record it as time_step iun the keys of the dictionary so
        it was redundant. Also, the last observation is not recorded because it has no effect on successive state and cost. The last observation's impact has been applied in the calculation of current state and current accumulated cost in
        the a loop. i.e: if we reach the same theta and accumulated cost by different last observations, there is no difference in continiue of the planning so because of Markovian property we can assume they are the same.
         
        
        Parameters
        ----------
        max_time_step : int
            maximum depth of planning.
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the theta distributions.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}     
        initial_wealth : float
            initial wealth.
        exp_weights : list of floats
            weight of exponential terms in the Utility function.
        exp_vorfaktoren : list of floats
            power of exponential terms in the Utility function..
        rounding_prec_coeff : big int, optional
            Used as measure of comparison between floating points.difference less than 1 dvided by this number would treated as equal . The default is 100000.
        r_round : int, optional
            Number of decimal digits in rounding accumulated rewards. The default is 5.
        save_space : Boolean, optional
            Save rechable internal states or not. It saves time_step by time_step in distict files. The default is True.
        keep_all_steps : Boolean, optional
            Keep calculated internal states during all time_steps in memory and return them. The default is True.

        Returns
        -------
        if keep_all_steps is True: 
            universal_int_states: dict from time_step to list of all possible diceretized internal_states at that time_step
            counts: a list containing number of points in internal_state in time_steps
        else:
            None.

        '''
        # setting parameters
        observations=list(self.env.observations.keys())
        self.max_time_step=max_time_step
        self.r_round=r_round
        self.exp_vorfaktoren=exp_vorfaktoren
        self.exp_weights=exp_weights
        self.exp_num=len(self.exp_vorfaktoren)
        # actual number is one extra point. i.e: self.partitioning_points_num=3 means valid points of {0,1,2,3} or {0%,33%,66%,100%}
        self.partitioning_points_num=partitioning_points_num
        
        # make M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        self.M=self.make_M_md(self.env,self.exp_vorfaktoren)
        
        #######################################################################
        #                   First Depth's Possible Points                     #
        #######################################################################
        
        # a prototype for all thata combinations over 2 states
        t0=[[i,self.partitioning_points_num-i] for i in range(self.partitioning_points_num+1)]
        
        # theta-part
        for i in range(self.exp_num):
            # save the first theta
            if i==0:
                pre=np.array(t0).reshape((len(t0),2))
            else:
                # for each elemnt of thetas, product it by all previous combinations to make all together_combinations
                for r,rec in enumerate(t0):
                    cons=np.tile(rec,len(pre)).reshape(len(pre),2)
                    if r==0:
                        post=np.hstack([cons,pre])
                    else:
                        post=np.vstack([post,np.hstack([cons,pre])])
                # the new combination should be used as previous combinations of new theta points
                pre=post
        # Theta parts of the internal state. In the original setting, the last observation is also a part of x, but it was unneccessary for us.
        x0=pre
        
        
        # R-part
        # first accumulated reward prototype that filled by initial_wealth
        r0=np.array([initial_wealth]*np.power(len(t0),len(exp_vorfaktoren)).astype(np.int32)).reshape(-1,1)
        
        for i in range(len(exp_vorfaktoren)):
            if i==0:
                final_r0=r0.copy()
            else:
                final_r0=np.hstack([final_r0,r0])
                
        # Whole internal_state at the first step (step=0). Again, there is a time_step elemnt in the original setting which is represented here by seperating each time_steps' interal_states. 
        state0=np.hstack([x0,final_r0])
        
        #######################################################################
        #                   Other Depth's Possible Points                     #
        #######################################################################
         
        
        # make the path of saving the internal-statespace. 
        # by state_space we mean that a disceretized thetas (2d) for each exponential and accumulated rewards (1d) for each of them
        # in each time step (because possible accumulated rewards values depend on time_step)
        self.space_path=os.path.join(os.getcwd(),'MO_state_space_continious')
        
        # all reachable internal states in each time_step (without observation part) 
        universal_int_states={}
        # number of each time_step's space points
        counts=[]
        
        
        ##### other steps
        for step in range(0,self.max_time_step+1):
            print(step)
            if step==0:
                # previous step's state space
                #prev=state0.copy()
                
                # new step's state space
                reachables=state0
                
                
            else:
                #prev=reachables.copy()
                
                reachables=self.expand_nextStep_states(exp_vorfaktoren=self.exp_vorfaktoren,prev_state=reachables,M=self.M,partitioning_points_num=self.partitioning_points_num,r_round=self.r_round)
                
                
            ################## save
            if save_space==True:
                if not os.path.exists(self.space_path):
                    os.makedirs(self.space_path)
                file_path=os.path.join(self.space_path,'time_step_'+str(step)+'.npy')
                np.save(file_path,reachables)
            
            ################# keep steps in memmory
            if keep_all_steps==True:
                counts.append(len(reachables))
                universal_int_states[step]=reachables
        
        
            
        if keep_all_steps==True:        
            return universal_int_states,counts
        else:
            return 
        
         
    # Discrete
    def discrete_value_iteration(self,partitioning_points_num,initial_wealth,exp_weights,exp_vorfaktoren,rounding_prec_coeff=100000,
                                 r_round=5,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=True,
                                 memory_threshold=1000000,keep_q_func=True,utility_approx=False, apx_terms_num=2):
        '''
        This function starts from all possible discretized internal states. i.e: disceretized thetas and initiail wealth. And, generates all reachable internal states during the depth
        of planning. It can save the ruslts and can contain all calculated internal states in memory.
        Note: in this function internal states are defined as tuple of : tuple of thetas , and tuple of accumulated rewards. Time is not recorded in the state representation becasue we record it as time_step iun the keys of the dictionary so
        it was redundant. Also, the last observation is not recorded because it has no effect on successive state and cost. The last observation's impact has been applied in the calculation of current state and current accumulated cost in
        the a loop. i.e: if we reach the same theta and accumulated cost by different last observations, there is no difference in continiue of the planning so because of Markovian property we can assume they are the same.
    

        Parameters
        ----------
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the theta distributions.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%} 
        initial_wealth : float
            initial wealth.
        exp_weights : list of floats
            weight of exponential terms in the Utility function.
        exp_vorfaktoren : list of floats
            power of exponential terms in the Utility function.
        rounding_prec_coeff : big int, optional
            Used as measure of comparison between floating points.difference less than 1 dvided by this number would treated as equal . 
            The default is 100000.
        r_round : int, optional
            Number of decimal digits in rounding accumulated rewards. The default is 5.
            
        save_reachables : Boolean, optional (used only in expander2)
            Save the reachable states in each time step on disk. Each time step's data will be saved in a separate file in the directory:
            'MO_state_space_discrete'. The default is True.
            
        make_transition_kernel : Boolean, optional (used only in expander2)
            Make a mapping from indexes of this step's internal states to the index of next state's internal_states. Transition kernel
            is also a function of action and next observation. If this option is not set, the extender_function bypass the part that makes
            transition kernels. The default is True.
            
        save_transition_kernel : Boolean, optional (used in expander2, and in value_iteration for loading data)
            Save the made transition kernels of each time step in a seperate file in directory: 'MO_transition_kernel_discrete'. If set to True,
            each file contains a mapping from indexes of the current_time_step's internal states to indexes of the next_time_step's internal_states.
            Th order of numbers represents their index in current time and the index of successive states are similiar to order of the states in reachables of
            the next state. Each time step's transition kernel (file or in memory) is a dictionary with keys: action, observation pair and 
            values: index of successive state in next step's reachable states. The default is True.
            
        use_buffer : Boolean, optional (used only in expander2)
            Activate the option of temporarily writing chunks of big state spaces in memory to free memory. 
            If set to False, all calculations will happend in the memory.The default is True.
            
        memory_threshold : big int, optional (used only in expander2)
            A number that if size of internal state was bigger than that, the extender2() function will run in mode 2, 
            so in the itermediate calculations save successive reachable states of each action seperately to prevent cntaining big data in the memory.
            If the size was 10x 'memory_threshold', extender2() will run in mode 3 and for each action, observation pair make a seperate temporary file.
            It will be useless if 'use_buffer' set to False. The default is 1000000.
            
        keep_q_func : Boolean, optional
            need the q_function as a result for free its memory. If set to True there are one extra output (4). The default is True.

        Returns
        -------
        step_sizes : a list of int
            size of each time_step's possible internal states.
        value_func: a dict from time_step to a list of floats
            each value is a list of value of states (ordered similar to reachable states)
        action_func: a dict from time_step to a list of integers
            each value is a list of best actions in the current states (ordered similar to reachable states)
        q_func : a dict from time_step to a 2-D numpy array
            ONLY IF 'keep_q_func' is set to True.
            like value_func, but for each state we have alist of values of each action.

        '''
       
        # set agent's mode to 'optimized_continious' or 'naive_discrete'
        self.agent_mode='naive_discrete'
        
        # setting parameters
        observations=list(self.env.observations.keys())
        
        self.r_round=r_round
        terms_weights=[1]*apx_terms_num
        if utility_approx==True:
            self.exp_vorfaktoren=[1]*apx_terms_num
            self.exp_weights=terms_weights[0:apx_terms_num]
        else:
            self.exp_vorfaktoren=exp_vorfaktoren
            self.exp_weights=exp_weights
            
        self.initial_wealth=initial_wealth
        self.exp_num=len(self.exp_vorfaktoren)
        # actual number is one extra point. i.e: self.partitioning_points_num=3 means valid points of {0,1,2,3} or {0%,33%,66%,100%}
        self.partitioning_points_num=partitioning_points_num
    
        # make M matrix
        # (i x action x observation x state x next_state)-> scalar   which contains smth like: (e^(lambda x cost) x probability)
        if utility_approx==True:
            self.M=self.make_apx_M(self.env,apx_terms_num)
        else:
            self.M=self.make_M_md(self.env,self.exp_vorfaktoren)
        
        # make emtpy dictionaries to store results
        self.keep_q=keep_q_func
        self.q_func={}
        self.value_func={}
        self.action_func={}
        
        ##### saving variables
        
        # save reachable states in each time step (each step's data separately )
        if save_reachables:
            MO_state_path=os.path.join(os.getcwd(),'MO_state_space_discrete')    # name of reachable states' folder 
            if not os.path.exists(MO_state_path):
                os.makedirs(MO_state_path)
        
        # save transition kernel of each step in disk separately. A dict contains keys: each (action, observation) pair and values: index
        # of internal states in depth i-th and the successive internal state in depth i+1-th 
        if save_transition_kernel:
            MO_TK_path=os.path.join(os.getcwd(),'MO_transition_kernel_discrete')  # name of Transition kernels' folder
            if not os.path.exists(MO_TK_path):
                os.makedirs(MO_TK_path)
            
        #######################################################################
        #                   First Depth's Possible Points                     #
        #######################################################################
    
        # a prototype for all thata combinations over only the first state (2-nd one can be infered by 1 minus the first element)
        t0=[i for i in range(self.partitioning_points_num+1)]
                
        # theta-part
        for i in range(self.exp_num):
            
            # save the first theta
            if i==0:
                pre=np.array(t0).reshape((len(t0),1))
            else:
                # for each elemnt of thetas, product it by all previous combinations to make all together_combinations
                for r,rec in enumerate(t0):
                    cons=np.tile(rec,len(pre)).reshape(len(pre),1)
                    if r==0:
                        post=np.hstack([cons,pre])
                    else:
                        post=np.vstack([post,np.hstack([cons,pre])])
                # the new combination should be used as previous combinations of new theta points
                pre=post
                
        # Theta parts of the internal state. In the original setting, the last observation is also a part of x, but it was unneccessary for us.
        x0=pre
        del(pre)
    
        # R-part
        # first accumulated reward prototype that filled by initial_wealth
        r0=np.array([self.initial_wealth]*np.power(len(t0),self.exp_num).astype(np.int32)).reshape(-1,1)
    
        for i in range(self.exp_num):
            if i==0:
                final_r0=r0.copy()
            else:
                final_r0=np.hstack([final_r0,r0])
    
        # Whole internal_state at the first step (step=0). Again, there is a time_step elemnt in the original setting which is represented here by seperating each time_steps' interal_states. 
        state0=np.hstack([x0,final_r0])
        
        #######################################################################
        #                   Forward search and expantion                      #
        #######################################################################
        # starting from the state0 and expand the decision tree until the end.
        # make reachable states in all steps and the transition kernel between them.
        if utility_approx==True:
            self.exp_vorfaktoren=[1]*apx_terms_num
        
        if make_transition_kernel and not save_transition_kernel :
            in_memory=True
            self.step_sizes,self.transition_kernel=self.discrete_state_expander2(self.exp_vorfaktoren,self.max_time_step,state0,self.M,self.partitioning_points_num,self.r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=False,use_buffer=use_buffer,memory_threshold=memory_threshold)
        else:
            in_memory=False
            self.step_sizes=self.discrete_state_expander2(self.exp_vorfaktoren,self.max_time_step,state0,self.M,self.partitioning_points_num,self.r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=use_buffer,memory_threshold=memory_threshold)
            
        #######################################################################
        #                        Backward evaluation                          #
        #######################################################################
        del(state0)
        gc.collect()
        
        
        # backwardly
                        
        for step in range(self.max_time_step,-1,-1):
                
            ############### filling last step
            if step==self.max_time_step:
                
                # fetch list of all possible states in the last time Step
                if save_reachables:
                    filename='depth'+str(step)+'.npy'
                    filepath=os.path.join(MO_state_path,filename)
                    this_step=np.load(filepath,allow_pickle=True)
                else:
                    # not implememnted
                    pass
                # Fetch R-parts
                this_step_r=this_step[:,-self.exp_num:]
                
                # release memory
                del(this_step)
                gc.collect()
                
                val=np.zeros((len(this_step_r),1))
                
                # caclualte total utility of states
                for i in range(self.exp_num):
                    val[:,0]+=np.exp(this_step_r[:,i]*self.exp_vorfaktoren[i])*self.exp_weights[i]
                
                # release memory
                del(this_step_r)
                gc.collect()
                
                # fill the results
                # we don't feel q_func in this last step
                self.value_func[step]=val.copy()
                self.action_func[step]=np.array([-1]*len(val)).astype(np.int8)
                
                del(val)
                gc.collect()
                
            #################### filling other steps 
            else:
                if save_transition_kernel:
                    filename='depth'+str(step)+'_'+str(step+1)+'.pkl'
                    filepath=os.path.join(MO_TK_path,filename)
                    tk=pd.read_pickle(filepath)
                else:
                    # not implemented
                    tk=pd.DataFrame(self.transition_kernel[step])
                    pass
    
                self.q_func[step]=np.array([[0]*self.num_of_actions]*len(tk)).astype(np.float64)
                
                
                for action in range(self.num_of_actions):
    
                    for y_prime in range(self.num_of_observations):
                        
                        # make transition code to find the related column(index of the next states) in the saved transition kernels
                        a_y_code=str(action)+'_'+str(y_prime)
                         
                        # fetch related values of this action,y_prime transition, multiply thom with 1/|y| and accumulate them for each action
                        tmp=self.myRound((1./self.num_of_observations)*self.value_func[step+1][tk.loc[:,a_y_code].to_list()],decimals=r_round)
                        # add related value of each next observation to make the expected value of the action
                        self.q_func[step][:,action]+=tmp.reshape(-1)
                        # Final Roundig of q_func
                        self.q_func[step][:,action]=self.myRound(self.q_func[step][:,action],decimals=r_round)
    
    
                # make value func and action_func based on q_func
                self.action_func[step]=np.argmax(self.q_func[step],axis=1)
                self.value_func[step]=np.max(self.q_func[step],axis=1)
                
                # if not save q_func, release the memory by assiging an int, instead of the q_func array 
                if keep_q_func==False:
                    self.q_func[step]=[]
        # if not keep q_func, we can only return value_func and action_func            
        if keep_q_func:
            return self.step_sizes,self.value_func,self.action_func,self.q_func
        else:
            return self.step_sizes,self.value_func,self.action_func
    
    def discrete_state_expander2(self,exp_vorfaktoren,max_time_step,state0,M,partitioning_points_num,
                                 r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=True,
                                 memory_threshold=1000000):
        '''
        

        Parameters
        ----------
        exp_vorfaktoren : list of floats
            power of exponential terms in the Utility function.
        max_time_step : int
            Maximum depth of planning.
        state0 : 2-d numpy array of floats
            All possible internal_states in depth 0 when their r values are set to initial_wealth.
            one column for each exponential theta (only theta on the first state, the second can be infered by 1 minus the first ) and one 
            column for each r(accumulated wealth) related to each theta. Note: The last observation is also a part of internal state, however
            here it was redundant because it had no effect on the successive states and wealths. 
        M : (i x action x observation x state x next_state)-d numpy array of floats
            M matrix in the paper.
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the theta distributions.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}
        r_round : int, optional
            Number of decimal digits in rounding accumulated rewards. The default is 5.
            
        save_reachables : Boolean, optional 
            Save the reachable states in each time step on disk. Each time step's data will be saved in a separate file in the directory:
            'MO_state_space_discrete'. The default is True.
            
        make_transition_kernel : Boolean, optional
            Make a mapping from indexes of this step's internal states to the index of next state's internal_states. Transition kernel
            is also a function of action and next observation. If this option is not set, the extender_function bypass the part that makes
            transition kernels. The default is True.
            
        save_transition_kernel : Boolean, optional 
            Save the made transition kernels of each time step in a seperate file in directory: 'MO_transition_kernel_discrete'. If set to True,
            each file contains a mapping from indexes of the current_time_step's internal states to indexes of the next_time_step's internal_states.
            Th order of numbers represents their index in current time and the index of successive states are similiar to order of the states in reachables of
            the next state. Each time step's transition kernel (file or in memory) is a dictionary with keys: action, observation pair and 
            values: index of successive state in next step's reachable states. The default is True.
            
        use_buffer : Boolean, optional 
            Activate the option of temporarily writing chunks of big state spaces in memory to free memory. 
            If set to False, all calculations will happend in the memory.The default is True.
            
        memory_threshold : big int, optional 
            A number that if size of internal state was bigger than that, the extender2() function will run in mode 2, 
            so in the itermediate calculations save successive reachable states of each action seperately to prevent cntaining big data in the memory.
            If the size was 10x 'memory_threshold', extender2() will run in mode 3 and for each action, observation pair make a seperate temporary file.
            It will be useless if 'use_buffer' set to False. The default is 1000000.
            
        Returns
        -------
        reachable_counts: list of int
            alist contains ordered sizes of state spaces in each time_step
        all_TK
            ONLY IF save_transition_kernel is set to False: returns ordered list contains all transition kernels. 

        '''

        # wrong case of not making the kernel but saving it!
        if make_transition_kernel==False and save_transition_kernel==True:
            print('bad_input!')
            return -1
    
        ##### saving variables
        
        #saving reachable states of each time step separately
        if save_reachables:
            MO_state_path=os.path.join(os.getcwd(),'MO_state_space_discrete')
            if not os.path.exists(MO_state_path):
                os.makedirs(MO_state_path)
                
        # saving Transition kernel of each time step seperately. Transition kernel is a dict contains keys: each (action, observation) pair and values: index
        # of internal states in depth i-th and the successive internal state in depth i+1-th 
        if save_transition_kernel:
            MO_TK_path=os.path.join(os.getcwd(),'MO_transition_kernel_discrete')
            if not os.path.exists(MO_TK_path):
                os.makedirs(MO_TK_path)
                
        # use disk in case of big state spaces
        if use_buffer:
            buffer_path=os.path.join(os.getcwd(),'MO_buffer_discrete')
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
        
        # set an empty list to contain the size of state space in each time step
        reachable_counts=[]
        
        if save_transition_kernel==False:
            # save calculated data in case we don't want to write it on the disk
            all_TK=[] # if we want to have transition kernel of all steps togther
            self.saved_transition_kernel=False
        else:
            self.saved_transition_kernel=True
        
        
        
                
                
        # expand state space for each time step
        
        for step in range(self.max_time_step):
            
            if step==0:
                
                prev_state=state0
                prev_state=np.array(prev_state)
                
                # save possible initial states 
                if save_reachables:
                    filename='depth0.npy'
                    filepath=os.path.join(MO_state_path,filename)
                    np.save(filepath,state0)
                    
                # save size of possible initial states    
                reachable_counts.append(len(state0))
            
            
            
            # Do for all steps
            # fetch previous step's internal state's data
            prev_num=len(prev_state)
            x_prev=prev_state[:,:self.exp_num]
            r_prev=prev_state[:,self.exp_num:]
            
            #
            # setting mode before each depth
            if prev_num<memory_threshold:
                mode=0      # keep the whole next states in the memory
            elif prev_num<memory_threshold*5:
                mode=2     # save next states' data for each action seperately
            else:
                mode=3       # save next states' data for each action and observation separately
                
            if use_buffer==False:
                # if we want to calculate things while not using the buffers(read and write on/from disk) it is equal to make writing threshold infinity and keep the variable in the memmory. 
                memory_threshold=np.inf
                mode=0
            
            # allocate results arrays in each depth
            if (mode==1 or mode==0):
                next_states=np.empty((self.num_of_actions,self.num_of_observations,prev_num,2*self.exp_num))
            elif mode==2:
                next_states=np.empty((self.num_of_observations,prev_num,2*self.exp_num))
            else:
                next_states=np.empty((prev_num,2*self.exp_num))
            
            # reachable states in the next step
            reachables=[]
    
            for a in range(self.num_of_actions):
                for y_prime in range(self.num_of_observations):
        
                    # to save all exponential terms
                    rt=[]    # accumulated rewards
                    ft=[]    # next theta points
                    
                    for i in range(self.exp_num):
    
                        # fetch each variate data
                        # data is presenting only the first element of thetas
                        this_theta=np.array(x_prev[:,i]).reshape(len(prev_state),1)
                        # make the full theta (on s1 and s2)
                        this_theta_2nd=self.partitioning_points_num-this_theta
                        this_theta=np.hstack([this_theta,this_theta_2nd])
                        
                        # make theta floating point in [0,1] 
                        this_theta=this_theta/np.sum(this_theta,axis=1).reshape(len(this_theta),1)
                        this_theta=self.myRound(this_theta,decimals=r_round)
                        
                        # accumulated cost until now
                        this_r=r_prev[:,i]
    
                        z=np.matmul(self.M[i,a,y_prime,:,:],this_theta.T)
                        integral=np.sum(z,axis=0)
                        lg=np.log(integral)
                        G=1./exp_vorfaktoren[i] * lg
                        c=G+np.log(np.power(self.num_of_observations,1./exp_vorfaktoren[i]))
                        
                        # reward
                        r=self.myRound(this_r.reshape(-1)+self.discount_factor*np.array(c), decimals=r_round)
                        rt.append(list(r))
                        
                        # next theta
                        f=(z/integral).T
                        # make theta integer again!
                        # first column
                        f1=list(np.round(f[:,0]*self.partitioning_points_num).astype(np.int32)) # use the np.round and integer casting
                        ft.append(f1)
    
                    
                    st=np.hstack([np.array(ft).T,np.array(rt).T])   # next internal states
                    
                    # fill the results based on the mode
                    # all together ( fill the action-&-observation related positions )
                    if (mode==1 or mode==0):
                        next_states[a,y_prime]=st
                        
                    # save for each action (fill the observation related positions) 
                    elif mode==2:
                        next_states[y_prime]=st
                        
                    # save for each action and observation ( fill the the whole variable, becasue it should represent 
                    # one unspecified action_observation pair). Then save that.
                    else:
                        # mode 3
                        next_states=st                      
                        # save the full_next_states in the buffer 
                        filename='a'+str(a)+'_yprim'+str(y_prime)+'.npy' # filename example: a2_yprime1.npy
                        filepath=os.path.join(buffer_path,filename)
                        np.save(filepath, next_states)
    
    
                    ############### find next_steps's unique reachable states
                    # remove redundant elements. By making all next states tuple and make them keys of a dict, then retrieve the keys as a list.
                    new_reachables=list((map(tuple,st)))
                    new_reachables=list(dict.fromkeys(new_reachables,0).keys())
                    
                    # accumulate all unique next states
                    # And, remove the records which are redundant between different sets of appended values 
                    if a==0 and y_prime==0 and i==0:
                        reachables=new_reachables.copy()
                    else:
    
                        reachables.extend(new_reachables)
                        # remove redundancies in the overal next states' variable
                        reachables=list(dict.fromkeys(reachables,0).keys())
                        
                    ##############################################################
                    
                # mode 2 : for each action save a file
                if mode==2:
                    filename='a'+str(a)+'.npy' # filename example : a2.npy
                    filepath=os.path.join(buffer_path,filename)
                    np.save(filepath, next_states)
                    
            # mode 1: save all states togheter       
            if mode==1:
                filename='all.npy'          # filename: 'all.npy'
                filepath=os.path.join(buffer_path,filename)
                np.save(filepath, next_states)
                
            # save unique reachable states
            if save_reachables:
                filepath=os.path.join(MO_state_path,'depth'+str(step+1)+'.npy')
                np.save(filepath,reachables)
                
            # count unique next states
            reachable_counts.append(len(reachables))
            
            ##############################################################################################
            #                                making transition kernal                                    #
            ##############################################################################################
            # if making transition kernel is not neccessary do not execute the rest
            if make_transition_kernel==False:
                continue
            
            # a mapping from unique next_states to their index
            reachables=dict(zip(reachables,np.arange(len(reachables))))
            
            # Transition kernel
            TK_indexes={}
            
            # read file by file
            if mode==3:
                for y_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        # read a file
                        filename='a'+str(a)+'_yprim'+str(y_prime)+'.npy'
                        filepath=os.path.join(buffer_path,filename)            
                        read=np.load(filepath,allow_pickle=True)
                        
                        # make index list for this action,observation pair
                        indexes=[]
                        
                        s_primes=read
                        for s_prime in s_primes:
                            # adding the index of unique reachable next_states for each for this current internal state under a and y_prime
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(y_prime)]=indexes
                        
            elif mode==2:
                for a in range(self.num_of_actions):
                    # read a file
                    filename='a'+str(a)+'.npy'
                    filepath=os.path.join(buffer_path,filename)            
                    read=np.load(filepath,allow_pickle=True)
                    
                    # the files contain data of y_prime =0 and 1
                    for y_prime in range(self.num_of_observations):
                        s_primes=read[y_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(y_prime)]=indexes
            
            # if all data were in one file            
            elif mode==1:
                # read a file
                filename='all.npy'           
                filepath=os.path.join(buffer_path,filename)            
                read=np.load(filepath,allow_pickle=True)
                
                # the file contains all data so it needs loop over all action and observations to seperate them
                for y_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        s_primes=read[a,y_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(y_prime)]=indexes
            
            # if not using the buffer
            elif mode==0:
                for y_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        # use the next_state variable which we had previously
                        s_primes=next_states[a,y_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(y_prime)]=indexes
            
            # if want to save the transition kernel
            # we use Pandas pickle write function
            if save_transition_kernel:
                filepath=os.path.join(MO_TK_path,'depth'+str(step)+'_'+str(step+1)+'.pkl')
                pp=pd.DataFrame(TK_indexes)
                pp.to_pickle(filepath)
                
            # if not save the transition data on disk, then we should keep all of them in memory     
            else:
                all_TK.append(TK_indexes)
            
            
            ##### set prevous state for the next iteration
            prev_state=np.array(list(reachables.keys()))       
            
        # In current implementation, we have saved reachable states on the disk.
        # if don't want to write transition kernel on disk we should return it.
        if make_transition_kernel and not save_transition_kernel :
            return reachable_counts,all_TK
        else:
            return reachable_counts
                                                 
    #def discrete_value_iteration_2sigmoid:
    def make_apx_M(self,env_dynamics,num_of_tailor_terms):
        '''
        Calculating the mapped M matrix. In th mapped M matrix each ecponnetial dimension is calculated by a mapping of reward function to a 
        term of logarithm of tairlor expansion of the utilty func. Here: 4* sigmoid(x-4)

        Parameters
        ----------
        env_dynamics : env class
            Dynamics of environment.
        num_of_tailor_terms : int
            number of tairlor expansion terms that we want to use to approximate the original utility function.

        Returns
        -------
        M : a 5-dimensional Matrix of floats. 
            Calculated mapped_M matrix (see the paper). The dimensions of M are: [each exponential terms] x [actions] x [observations] x [state] x [next state]

        '''
        
        # setting parameters
        number_of_exp=num_of_tailor_terms
        vorfaktoren=[1]*num_of_tailor_terms
        #initializing M as a 5-dim matrix: M(i,a,y,s,s')
        M= np.zeros((number_of_exp, len(env_dynamics.actions), len(env_dynamics.observations), len(env_dynamics.states), len(env_dynamics.states)))

        #calculation of M as a 5-dim matrix, i.e. the multi-variation/i is seen as an extra variable - 
        #M[i][a][y][:][:] then yields the 2x2 that is used for forward propagation
        
        ### 4*sigmoid(x-4)
        
        for i in range(len(vorfaktoren)):
            for a in range(len(env_dynamics.actions)):
                for y in env_dynamics.observations:
                    for s in env_dynamics.states:
                        for ss in env_dynamics.states:
                            new_r=env_dynamics.rewards[s][a]+4
                            e4=np.exp(4)
                            denom=1+e4
                            if i==0:
                                term=1/denom
                                #mapped_reward=np.log(term)
                            if i==1:
                                
                                term=e4*new_r/np.power(denom,2)
                                
                            elif i==2:
                                term=e4*(e4-1)*np.power(new_r,2)/(2*np.power(denom,3))
                                                               
                            elif i==3:
                                term=(e4-4*np.power(e4,2)+np.power(e4,3))*np.power(new_r,3)/(6*np.power(denom,4))
                               
                            elif i==4:
                                term=e4*(-1+11*e4-11*np.power(e4,2)+np.power(e4,3))* np.power(new_r,4)/(24*np.power(denom,5))
                                
                            mapped_reward=np.log(4*term)
                            M[i][a][y][s][ss]= np.exp(vorfaktoren[i]*mapped_reward)*env_dynamics.transition_matrix[s][a][ss]*env_dynamics.observation_matrix[a][ss][y]
                            
                    # for each M(i,a,y)[s,s'] the M Matrix should become Transpose. (see the paper)       
                    M[i][a][y]=M[i][a][y].T
        return M