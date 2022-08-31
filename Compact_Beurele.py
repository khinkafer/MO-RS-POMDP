import numpy as np
#from datetime import datetime
#import sys
import gc
import os
#from env import *
import pandas as pd



class Bauerle_Rieder_agent(object):
    
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
        # are represented as tuples of (observable part of environment or here "observation", Mu-state,time_step). Here, mu-state is
        # represented by a tuple of probability distributions over possible wealth in possible hidden states.
        # In our internal state representation, storing time_step is redundant, and in our simulation storing last observation is also redundant, however
        # to be similar with original paper's modeling, we used internal_states excactly like that.
        self.value_func={}
        # Like the above, but final values are the index of the best actions (the actions which had those maximum values)
        self.action_func={}
        # Like the aboves, but the values are a list of values of all actions in that internal state.
        self.q_func={}
        # A mapping from tuple of (internal_state, action, next_observation) to next_internal_state in each time step(outer dictionary)
        self.transition_kernel={}
        
        # aux-parameters
        
        self.initial_wealth=self.env.initial_wealth
        
        self.num_of_actions=len(self.env.actions)
        self.num_of_observable_states=len(self.env.observations)
        self.num_of_unbservable_states=len(self.env.states)
        self.num_of_observations=self.num_of_observable_states
        self.num_of_states=self.num_of_unbservable_states
        
        
        
        return
    
    
    
    def generate_possible_wealths(self,cost_reward_values,initial_wealth,discount_factor,trials,comparison_precision=100000):
        '''
        
        It Generates all possible (unique) amounts of cost/reward combinations during the trials.S-dimension of Mu. 
         

        Parameters
        ----------
        cost_reward_values : array of numbers
            set of all possible cost and rewards of actions.
        discount_factor : float
            Gamma value (Beta in the paper).
        trials : int
            indicates the depth of decision-making and consequently number of possible wealth amounts .
        comparison_precision: float 
            its a coefficient that used to avoid floating point imprecise operations. If two numbers have difference less than comparison_precision, they assumed to be equal. 
        Returns
        -------
        a dictionary that its keys are the step-numbers of the planning depth (strats: from 0(initial value) to maximum: trials variable), and their related values are the possible amounts of wealth in that step.

        '''
        # add initial_wealth as the value of the first time-step. 
        final_values={0:[initial_wealth]}
        for t in range(trials+1):
            # if it is the first time-step, pass
            if t==0:
                pass
            else:
                values=np.array(final_values[t-1])               
                tmp_values=np.empty(0,float)

                for val in cost_reward_values:
                    # next step values can be current ones + discounted value of each action, (except for the firts trial)
                    if t==1:
                        tmp=values+val
                    else:
                        z=np.power(discount_factor,(t-1))
                        
                        z=self.myRound(z)
                        
                        tmp=values+val*z
                        
                    tmp=self.myRound(tmp)
                    
                    # check if calculated wealth levels are redundant
                    
                    for to_add_val in tmp:
                        is_redundant=False
                        if len(tmp_values)==0:
                            tmp_values=np.append(tmp_values,to_add_val)
                        else:
                            for prev_saved_val in tmp_values:
                                # Do floating point comparison: if their difference is less than a threshold (comparison_precision) we assume them equal.
                                if (np.sum(np.abs(prev_saved_val-to_add_val))<1./(10*comparison_precision)):
                                    is_redundant=True
                            if is_redundant==False:
                                tmp_values=np.append(tmp_values,to_add_val)
                
                #sort the values in ascending order
                final_values[t]=np.sort(tmp_values).tolist()
                
        return final_values
   
    def make_Q_kernel(self,transition_matrix,observation_matrix):
        '''
        This funtion, returns the transition kernel of the Paper. Q=P(x',y'| x,y,a)
        
        In the extended tiger problem, the next observable and unobservable states (Or observations and states) are only depend on the current unobservable state and the action. 
        So, we can rewrite the P(x',y'| x,y,a) as P(x',y'|y,a).
        The final output is a 4-D matrix Q[y][a][y'][x'] whose values are probability of transtion by having y and a and recieving y' and x'.
        In the paper's notation x, x': observable states (Or Observations in the tiger problem paradigm), and y,y' : unobservable states (Or States in the tiger problem paradigm)    
        

        Parameters
        ----------
        transition_matrix : 3-D float 
            with shape (2x5x2)[YxAxY]. It codes P(y,a)->y'.
        observation_matrix : 3-D float
            with shape (5x2x2)[AxYxX]. It codes P(x|y,a).

        Returns
        -------
        4-D floats which codes P(x',y'|y,a).[Y x A x Y' x X']

        '''
    
        #Q[y][a][y'][x']
        Q=np.zeros((self.num_of_states, self.num_of_actions, self.num_of_states,self.num_of_observations))
        
        # product of P(y'|y,a) and P(x'|y',a)
        Q=np.array([[transition_matrix[s,a,:].reshape(1,self.num_of_states).T * observation_matrix[a] for a in range(self.num_of_actions)] for s in range(self.num_of_states)])
        return Q

    def make_marginal_Q_kernel(self,transition_matrix,observation_matrix):
        '''
        This funtion, returns the marginal transition kernel of the Paper. Marginal_Q=P(x'| x,y,a)=  Integral(P(x',y'| x,y,a) dy')
        
        In the extended tiger problem, the next observable and unobservable states (Or observations and states) are only depend on the current unobservable state and the action. 
        So, we can rewrite the P(x',y'| x,y,a) as P(x',y'|y,a).
        The final output is a 3-D matrix Q[y][a][x'] whose values are: probability of transtion by having y and a and recieving x'.
        In the paper's notation x, x': observable states (Or Observations in the tiger problem paradigm), and y,y' : unobservable states (Or States in the tiger problem paradigm)    
        
    
        Parameters
        ----------
        transition_matrix : 3-D float 
            with shape (2x5x2)[YxAxY]. It codes P(y,a)->y'.
        observation_matrix : 3-D float
            with shape (5x2x2)[AxYxX]. It codes P(x|y,a).
    
        Returns
        -------
        3-D floats which codes P(x'|y,a).[Y x A x X']
    
        '''
        
        # MQ[y][a][y']
        MQ=np.zeros((self.num_of_states,self.num_of_actions,self.num_of_observations))
        
        # sum_on_y'_for_each_x'(dot product of P(y'|y,a) and P(x'|y',a))
        MQ=np.array([[np.dot(transition_matrix[s,a,:].reshape(1,self.num_of_states), observation_matrix[a]).reshape(-1) for a in range(self.num_of_actions)] for s in range(self.num_of_states)])
    
        return MQ


    def continious_optimized_planning(self, initial_mu_state,initial_observation,initial_wealth,exp_weights,exp_vorfaktoren,
                                      rounding_prec_coeff=100000,exp_util=True,utility_approx=False,apx_terms_num=2):
        
        '''
        This function first reset the agent to its initial internal state, then based on that, it performs forward and backward 
        steps srerialy: First: it finds all possible internal states (in each time step) and make a transition kernel that 
        determines all possible internal state transitions during the planning. Second, it  performs value iteration backwardly.
        
        Parameters:
        -----------
        initial_mu_state : list of two floats
            the probability of being in each hidden state when the wealth is equal to initial_wealth
        initiative_observation : int
            The first x (observable state).
        initial_wealth: float
        
        exp_weights: list of floats
            weights of exponential elements
        exp_vorfaktoren: list of floats
            power value of exponential elements
        rounding_prec_coeff: big int
            1 divide by this value will be the threshhold which we use to check floating point equalities
            
        Returns:
        --------
        Ss: dict of lists of floats
            each key of dictionary is related to a time_step. each value is a list of possible wealth values in that step. 
        b_reachables: dict of list of tuples 
            time_steps -> list of possibe internal states in that time step.
        TK: dict of dicts( tuples -> tuples)
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
        
        # default initial states 
        # it set the agent to have equal probability over hidden states and having the initial observation equal to 0 (which has no effect).
        
        self.reset(initial_mu_state,initial_observation,initial_wealth)
        
        
        #######################################################################
        #                              Forward                                #
        #######################################################################
        
        
        
        # reachable internal-states in each time step. It is a dictionary from time-steps to list of reachable internal states in that depth. The representation of internal states is:
        # a tuple contains: (X: observable part of state which here is equal to observation, Mu-state: a tuple that represents probability dist. over space Wealths x hiddenStates, Z: which is time step).
        b_reachables={}
        
       
        
        # Transition Kernel ( and not TK insurance company! :) ). a mapping from (internal state, action,next_observation) to the (next internal state). 
        # 'TK' is a nested dictionarty. The outer dict.'s keys are time steps and the values are the inner dicts. The inner dicts are mappings 
        # from (current internal-state, action, next observation) in timeStep t --> (next internal_state) of timeStep t+1. 
        # Note, each Mu point is also represented by a tuple.
        TK={}

        
        
        # make possible wealth values
        self.Ss=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step,comparison_precision=100000)
        
        ## precalculate needed variables
        
        # q(x_prim, y_prim | x,y,a) while in our settign it is equal to q(x_prim,y_prim|y,a)
        # make Q-kernel, the probability of reaching each y_prim x_prim pair when the real state is y and doing action a
        q=self.make_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        # q(x_prim | x,y,a) while in our settign it is equal to q(x_prim|y,a)
        # make marginal-Q-kernel, the probability of getting observation x_prim, when the real state is y and doing action a
        mq=self.make_marginal_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        # make and add all possible internal states in each timeStep
        for step in range(self.max_time_step):
            # set initial internal-state
            if step==0:
                # add two possible initial states (for two possible initial observation)
                #b_reachables[0]=[tuple([0,self.current_internal_state[1],self.current_internal_state[2]]),tuple([1,self.current_internal_state[1],self.current_internal_state[2]])]
                b_reachables[0]=[self.current_internal_state]
            # fetch just Mu points of internal-states
            b_reachable_mu=np.array(list(map(list,b_reachables[step])),dtype=object)[:,1].tolist()
            #print(b_reachable_mu)
            
            thisStep_s=self.Ss[step]
            # ls is the number of possible wealth levels in this timeStep as well as half of Mu points' lenght in this timeStep 
            ls=len(thisStep_s)
            mus=np.reshape(b_reachable_mu,(len(b_reachable_mu),len(b_reachable_mu[0])))
            
            z=np.power(self.env.discount_factor,step)
            
            z=self.myRound(z)
            # fetch all wealth values of this step and the successive ones
            all_current_s=thisStep_s
            all_next_s=self.Ss[step+1]
        
        
            # decompose Mu-beliefs about being in each state (y=0 or 1)
            mus_y0=mus[:,0:ls]
            mus_y1=mus[:,ls:2*ls]
            mus_ys=[mus_y0,mus_y1]
        
            # Marginal Mu (Mu superscript Y in the paper, or Mu(dy,R)). This variable expresses the probability of being in each state(y)
            marginal_mus_y0=mus_y0.sum(axis=1)
            marginal_mus_y1=mus_y1.sum(axis=1)
            marginal_mus_ys=[marginal_mus_y0,marginal_mus_y1]
          
            # to pick unique Mu' points
            nextStep_uniques=[]
            
            # prepare the timeStep related elements of the result variabes
            b_reachables[step+1]=[]
            
            TK[step]={}
                
                
            
            for action in range(self.num_of_actions):
                for x_prim in range(self.num_of_observations):
                    ########################################################
                    
                    ## prepare
                    
                    # C(x,y,a) while C depends only on y
                    #rewards/costs of doing action "a" when the real state is "y" 
                    y0=0
                    y1=1
                    y_prim0=0
                    y_prim1=1
                    c_y0=self.env.rewards[y0][action]
                    c_y1=self.env.rewards[y1][action]
                    c=[c_y0,c_y1]
                    
                    ### Calculate the denominator of psi function
                
                    # It is equal to probaility of recieving observation x_prim, regardless of what are the wealth(s) or the state(y). 
                    # It calculate sum of probabilities of reaching each state (regardless of wealth level) (marginal_mus_y), while reaciving x_prime observation
                    psi_denominator=marginal_mus_y0*mq[0][action][x_prim]+marginal_mus_y1*mq[1][action][x_prim]
                    psi_denominator=self.myRound(psi_denominator)
                    
                    ### calculate the nomerator of the psi function
                
                    # The dimensions of the Mu-space doesn't change with just one action and one observation.  Because in our experiment the reward function is deterministic, all of possible wealths of this step, will transfer to just one other value 
                    # based on state and action. So, the size of Mu-space remains constant in psi calculator function (for doing only one action). Also, the number of possible distributions over Mu, is not a concern for this function: It maps all current possible values to
                    # continious values
        
                    ## allocate variables for results of nomerator calculations for each current state
        
                    # these arrays are here to represent Mu distribution of the next state (naturally over its own (next state's) wealth levels) 
                    for_y0=np.zeros((len(mus),len(all_next_s)*2))
                    for_y1=np.zeros((len(mus),len(all_next_s)*2))
                    next_mus=[for_y0,for_y1]
        
                    # tmp_mus[0] for y=0 and tmp_mus[1] for y=1 calculations
                    tmp_mus=np.zeros((2,len(mus),len(mus[0])))
                    
                    ##
                     # for each current state
                    for y,y_mus in enumerate(tmp_mus):
                        
                        # mus_ys is Mu-beliefs about being in a staet (y)
                        # probability of reaching y_prim=0
                        y_mus[:,:len(all_current_s)]=mus_ys[y]*q[y][action][y_prim0][x_prim]
                        # probability of reaching y_prim=1
                        y_mus[:,len(all_current_s):]=mus_ys[y]*q[y][action][y_prim1][x_prim]
        
                        # Dirac function: d(s+zc(x,y,a))
                        # Here, we compute the possible values of s for next Mu distribution         
                        next_possible_s=np.array(all_current_s)+z*c[y]
                        next_possible_s=self.myRound(next_possible_s)
        
                        #The calculated Mu distributions on the S-axis, are defined on the current S values, however these probabilities are for s+zc(y,a).
                        # so as the Psi function inputs are (x,a,x_prim), for each y we can rotate Mu(s) to be matched with next stage's possible values

                        # indexes of the next wealth levels which are the successors ( after current S-values recieving c(y,a) )
                        # Here, we used comparison method to check the equality of next time-spte's S-values and current S-values+ z*c 
                        next_s_indexes=np.empty(0,np.int32)
                        for ind,ns in enumerate(next_possible_s):
                            
                            next_related_index=np.where(np.abs(np.array(all_next_s)-ns)<(1/(rounding_prec_coeff*10)))[0][0]
                            next_s_indexes=np.append(next_s_indexes,next_related_index)
        
        
                        #assign Mu-probability to each possible S(=previous_s + c) point
                        # for S points in y_prim=0        
                        next_mus[y][:,next_s_indexes]=y_mus[:,:len(all_current_s)]
                        # for S points in y_prim=1
                        next_mus[y][:,next_s_indexes+len(all_next_s)]=y_mus[:,len(all_current_s):]
        
        
                    # sum of probabilities of next Mu-space for both conditions : p(mu|y0), p(mu|y1)
                    psi_nomerator=next_mus[0]+next_mus[1]
                    psi_nomerator=self.myRound(psi_nomerator)
                    # calculate final psi result
                    # normalize the whole Mu-space dist. by dividing it by totoal probability of taking x_prim
                    psi_result=psi_nomerator/psi_denominator[:,None]  
                    psi_result=self.myRound(psi_result)
                    ### make extended states and transition kernel between them
                    # loop over the elements of psi_result(not vectorized):
                    for p in range(len(psi_result)):
         
                        
                        
                        ## reachable Mu states
                        redundants=[]
                        is_unique=True
                        for i in range(len(nextStep_uniques)):
                            
                            # fecth each part of the successive states
                            nst_mu=np.array(list(nextStep_uniques[i][1]))
                            nst_x=nextStep_uniques[i][0]
                            
                            # if the successive internal_state is redundant .(its X part as well as its Mu part ) 
                            if ((np.sum(np.abs(np.array(psi_result[p])-nst_mu))<1/(10*rounding_prec_coeff)) and (x_prim==nst_x)):
                                is_unique=False
                                redundants.append(i)
                              
                        if is_unique:
                            
                            # add unique values for other comparisons
                            nextStep_uniques.append((x_prim,tuple(psi_result[p]),step+1))
                            
                            # adding to extended reachable states
                            b_reachables[step+1].append((x_prim,tuple(psi_result[p]),step+1))
                            
                            # transition kernel of extended states
                            # dict value                            
                            ex_state=(x_prim,tuple(psi_result[p]),step+1)
                            # dict key
                            ex_kernel_element=(b_reachables[step][p],action,x_prim)
                            
                            TK[step][ex_kernel_element]=ex_state
                            
                           
                        else:
                            
                            # use that previously saved internal-state that was equal to current successive
                            
                            ex_state=nextStep_uniques[redundants[0]]
                            
                            ex_kernel_element=(b_reachables[step][p],action,x_prim)
        
                            TK[step][ex_kernel_element]=ex_state
                            
                        
                        
                            
        #######################################################################
        #                           Value Iteration                           #
        #######################################################################
        
 
        # Start from the last step to the first
        # Note: In this modeling all the value of decision making process are assigned to the last step
        # for example if we want to do a depth 2 decision making, we start from timeStep 0, then choose action 1, then reach timeStep 1's state
        # and the only valuable states are only those are in the timeStep 2 (after 2 decisions).
        for step in range(self.max_time_step,-1,-1):
            
            points=b_reachables[step]
            
            self.value_func[step]={}
            self.action_func[step]={}
            self.q_func[step]={}
            
                
            if step==self.max_time_step:
                # fecth just Mu parts of rechable states
                points_mu=[]
                for i in range(len(points)):
                    points_mu.append(points[i][1])
                
                
                # Calculate the utility of each possible wealth level in the last timeStep
                each_wealth_level_value=self.utility_evaluator(self.Ss[step],exp_weights,exp_vorfaktoren,
                                                               exp_util,utility_approx,apx_terms_num)
                #print(each_wealth_level_value.shape)
                # Calculate the utility of each possible last Mu state
                each_mu_value=np.dot(np.array(list(map(list,points_mu))),np.hstack([each_wealth_level_value,each_wealth_level_value]).T)
                each_mu_value=self.myRound(each_mu_value)
                
                
                # fill valu_func and q_func variables
                self.value_func[step]=dict(zip(points,each_mu_value.reshape(-1))).copy()               
                self.q_func[step]=dict(zip(points,each_mu_value.reshape(-1))).copy()
                
                #value_func[step]=np.hstack([dict(zip(points,each_mu_value.reshape(-1))).copy(),dict(zip(points,each_mu_value.reshape(-1))).copy()])               
                #q_func[step]=[dict(zip(points,each_mu_value.reshape(-1))).copy()]
                
            else:
                tmp_q=np.zeros((len(points),self.num_of_actions))
                              
                for p,point in enumerate(points):
                    
                    for a in range(self.num_of_actions):
                        this_action_value=0
                        for x_prim in observations:
                            # value of this internal state and action is equal to the value of the successive state (only its Mu part)
                            val_xprim=self.value_func[step+1][TK[step][(point,a,x_prim)]]
                            p_y0= np.sum(list(point[1])[:len(self.Ss[step])])
                            p_y1= np.sum(list(point[1])[len(self.Ss[step]):])
                            p_xprim=mq[0][a][x_prim]*p_y0 +mq[1][a][x_prim]*p_y1
                            
                            this_xprim_value= p_xprim * val_xprim
                            
                            this_action_value+=this_xprim_value
                            
                        tmp_q[p][a]=this_action_value
                        
                    tmp_best_action=np.argmax(tmp_q[p])
                    tmp_val=np.max(tmp_q[p])
                    
                    self.value_func[step][point]=tmp_val
                    self.action_func[step][point]=tmp_best_action
                    self.q_func[step][point]=tmp_q[p]
                            
        self.transition_kernel=TK
        
        return self.Ss,b_reachables,TK,self.q_func,self.value_func,self.action_func
 
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

    def utility_evaluator(self,x,exp_weights,exp_vorfaktoren,exp_util=True,utility_approx=False,apx_terms_num=2):
        '''
        It applies utility function on the input value. Utility function here should be some of exponentials.
        
        Parameters:
        -----------
        x: numpy array or float
            input value
        exp_weights: list of floats
            ordered list of weihts of exponential terms  
        exp_vorfaktoren: list of floats
            ordered lost of power coefficient of exponential terms
        
        Returns:
        -------
        The numpy array of transformed values by the utility function 
        '''
        ##### 4 x sigmoid(x-4)
        # if tailor of 4sig(x-4)
        if utility_approx==True:   
            weights=[1]*apx_terms_num
            e4=np.exp(4)
            denom=1+e4
            new_x=np.array(x)+4
            for i in range(apx_terms_num):

                if i==0:
                    util=np.array([1/denom]*len(x))
                elif i==1:
                    util+=e4*new_x/np.power(denom,2)
                elif i==2:
                    util+=e4*(e4-1)*np.power(new_x,2)/(2*np.power(denom,3))
                elif i==3:
                    util+=(e4-4*np.power(e4,2)+np.power(e4,3))*np.power(new_x,3)/(6*np.power(denom,4))
                elif i==4:
                    util+=e4*(-1+11*e4-11*np.power(e4,2)+np.power(e4,3))* np.power(new_x,4)/(24*np.power(denom,5))

            return 4 * util
        # if sum of exps
        elif (exp_util==True and utility_approx==False):
            terms_x=np.exp(np.matmul(np.array(exp_vorfaktoren).reshape(len(exp_vorfaktoren),1),np.array(x).reshape(1,len(x))))
            return np.dot(terms_x.T,np.array(exp_weights).reshape(len(exp_weights),1)).T
        
        # if excact 4sig(x-4)
        elif exp_util=='4sigmoid_x-4':
            new_x=np.array(x)+4   
            sig=1./(1+np.exp(-(new_x-4)))
            return (4*sig).reshape(1,len(sig))
        
        # if excact sig(x)
        elif exp_util=='sigmoid':
            new_x=np.array(x)  
            sig=1./(1+np.exp(-new_x))
            return sig.reshape(1,len(sig))


    def utility_evaluator_md(self,x,s,exp_weights,exp_vorfaktoren,exp_util=True,utility_approx=False,apx_terms_num=2):
        '''
        
        '''
        # make s for both states
        s2=s.copy()
        s2.extend(s)
        ######################### utility of S possible values
        # if approximating 2xsig(x)-1 by Tailor
        if utility_approx==True:   
           weights=[1]*apx_terms_num
           e4=np.exp(4)
           denom=1+e4
           new_s2=np.array(s2)+4
           for i in range(apx_terms_num):
               
               if i==0:
                   util=np.array([1/denom]*len(s2))
               elif i==1:
                   util+=e4*new_s2/np.power(denom,2)
               elif i==2:
                   util+=e4*(e4-1)*np.power(new_s2,2)/(2*np.power(denom,3))
               elif i==3:
                   util+=(e4-4*np.power(e4,2)+np.power(e4,3))*np.power(new_s2,3)/(6*np.power(denom,4))
               elif i==4:
                   util+=e4*(-1+11*e4-11*np.power(e4,2)+np.power(e4,3))* np.power(new_s2,4)/(24*np.power(denom,5))
           # no log, no exp.. these are redundant here        
           utility_s=4*util
        elif (exp_util==True and utility_approx==False):    
            # make all terms
            terms_s=np.exp(np.matmul(np.array(exp_vorfaktoren).reshape(len(exp_vorfaktoren),1),np.array(s2).reshape(1,len(s2))))
            # weighted some of them (in a column)
            utility_s=np.dot(terms_s.T,np.array(exp_weights).reshape(len(exp_weights),1)).T[0]
        
        elif exp_util=='4sigmoid_x-4':
            new_s2=np.array(s2)+4 
            sig=1./(1+np.exp(-(new_s2-4)))
            utility_s=4*sig
            
        # if excact sig(x)
        elif exp_util=='sigmoid':
            new_s2=np.array(s2)  
            sig=1./(1+np.exp(-new_s2))
            utility_s=sig
        ################################ utility of Mu points
        # value of points
        #print(utility_s)
        utility_x=np.dot(np.array(x),utility_s.reshape(len(utility_s),1))
        
        
        return utility_x
        
    def reset(self,initial_mu_state,initial_observation,initial_wealth):
        '''
        This function resets the agent to its belief about the initial wealth and set time-step to zero (begining of the simulation).

        Parameters
        ----------
        initial_mu_state : list of two floats
            the probability of being in each hidden state when the wealth is equal to initial_wealth
        initiative_observation : int
            The first x (observable state).
        initial_wealth: float
        Returns
        -------
        Noting. makes the initial internal_state and makes the agent ready for a new simulation.

        '''
        # set time-step=0
        self.current_internal_timeStep=0
        
        # define a variable for current x
        self.current_internal_x=initial_observation
        
        # set the initial Mu state
        self.current_internal_state=(initial_observation, tuple(initial_mu_state),0)
        
        #set initial wealth
        self.initial_wealth=initial_wealth
        
        # defien last action
        self.last_action=None
        
        
        # to load all possible actions in the 'naive_discrete' mode
        if self.agent_mode=='naive_discrete':
            self.summarized_reachables={}
            self.summarized_reachables_indexes={}
            
            self.summarized_transition_kernel={}
            #self.summarized_transition_kernel2={}
            self.summarized_value_func={}
            self.summarized_action_func={}
            self.summarized_q_func={}
            
            
            for step in range(self.max_time_step+1):
                
                #################### set variables
                
                self.summarized_reachables_indexes[step]=[]  
                self.summarized_reachables[step]=[]
                
                self.summarized_transition_kernel[step]={}
                #self.summarized_transition_kernel2[step]={}
                
                self.summarized_value_func[step]={}
                self.summarized_action_func[step]={}
                self.summarized_q_func[step]={}
                ##################### Fetch data
                
                # fetch reachable states and transition kernel of that step
                reachables_path=os.path.join(os.getcwd(),'B_state_space_discrete')
                reachables=np.load(os.path.join(reachables_path,'depth'+str(step)+'.npy'))
                
                # don't read any file for transition kernal from the last depth
                if step<self.max_time_step:
                    if self.saved_transition_kernel:
                        tk_path=os.path.join(os.getcwd(),'B_transition_kernel_discrete')
                        tk=pd.read_pickle(os.path.join(tk_path,'depth'+str(step)+'_'+str(step+1)+'.pkl'))
                    else:
                        tk=pd.DataFrame(self.transition_kernel[step])
                
                #################### state complete representation
                # for initial state
                if step==0:
                    
                    
                    first=tuple(self.myRound(self.nearest_grid_points((np.array(initial_mu_state)*10*self.partitioning_points_num).reshape(1,len(initial_mu_state)))[0],decimals=5))
                    
                    reachables=dict(zip(list(map(tuple,reachables[:,:2])),np.arange(len(reachables))))
                    
                    self.summarized_reachables_indexes[step]=[reachables[first] ]  
                    self.summarized_reachables[step]=[self.current_internal_state]
                else:
                    # make state by indexes
                    for rec in list(self.summarized_transition_kernel[step-1].keys()):
                        idx=self.summarized_transition_kernel[step-1][rec]
                        mu_state=reachables[idx]
                        
                        # make full representation of states
                        
                        x=rec[2]
                        state=tuple([x,tuple(mu_state),step])
                        state2=tuple([x,tuple(self.myRound(mu_state/self.partitioning_points_num,decimals=5)),step])
                        
                        # replace the index of next_state in the previous transition function by the complete representation of the state
                        self.summarized_transition_kernel[step-1][rec]=state2
                        # Do the same but for the float Mu space
                        #self.summarized_transition_kernel2[step-1][rec]=state2
                        # add the state to reachables
                        self.summarized_reachables[step].append(state2)
                        # add the index of the state
                        self.summarized_reachables_indexes[step].append(idx)
                        
                    
                
                ###################     update states data
                #print(self.summarized_reachables_indexes[step])
                for s,state_idx in enumerate(self.summarized_reachables_indexes[step]):
                    state=self.summarized_reachables[step][s]
                    #state2=tuple([state[0],tuple(self.myRound(np.array(state[1])/self.partitioning_points_num)),state[2]]   )
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
                            
                            for x_prime in range(self.num_of_observations):
                                
                                next_index=tk.loc[state_idx,str(a)+'_'+str(x_prime)]
                                
                                self.summarized_transition_kernel[step][tuple([state,a,x_prime])]= next_index
                                #self.summarized_transition_kernel2[step][tuple([state,a,x_prime])]= next_index
            

        
        return
    
    # simulation functs
    def do_action(self):
        
        '''
        This function returns the best action based-on: current time-step, current ointernal_state
        
        Returns
        -------
        best_action: int 
            Among actions' indexes (1-5)
        value_best_action: float
            value of doing that action at that (x,Mu,z) point
        current_internal_state: tuple 
            the current internal state which contains (last_observation, Mu_state, and time_step)
        '''
        # check wheather trial has ended or not 
        if self.current_internal_timeStep>=self.max_time_step:
            print('too much iterations!')
            return
        else:
            if self.agent_mode=='optimized_continious':
                best_action=self.action_func[self.current_internal_timeStep][self.current_internal_state]
                value_best_action=self.value_func[self.current_internal_timeStep][self.current_internal_state]
            # for naive_discrete mode
            else:
                best_action=self.summarized_action_func[self.current_internal_timeStep][self.current_internal_state]
                value_best_action=self.summarized_value_func[self.current_internal_timeStep][self.current_internal_state]
            
            
            
            # update last action
            self.last_action=best_action
            
            return best_action,value_best_action,self.current_internal_state
    
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
        current_internal_state: tuple of (observable_state, Mu_state, time_step )
            It is the current internal_state AFTER updating.
        '''
        
        
        if self.current_internal_timeStep>=self.max_time_step:
            print('too much iterations!')
            return
        else:      
            
            # update the current internal state
            if self.agent_mode=='optimized_continious':
                self.current_internal_state=self.transition_kernel[self.current_internal_timeStep][(self.current_internal_state,self.last_action,new_observation)]
            else:
                self.current_internal_state=self.summarized_transition_kernel[self.current_internal_timeStep][(self.current_internal_state,self.last_action,new_observation)]
            
            
            # increase time-step
            self.current_internal_timeStep=self.current_internal_timeStep+1
            
            # update current observable-state 
            self.current_internal_x=new_observation
            
            return self.current_internal_state
        
    # and functions to make all possible internal states
    def make_all_mu(self,partitioning_points_num,save_Mu=True, keep_all_steps=True):
        '''
        
        This function generates all possible Mu-space points (discrete PDF over S and Y).
        
        Parameters
        ----------
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the Mu_space.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}  
            
        save_Mu: Boolean, optional
            Save Mu space or not. The default is True.
            
        keep_all_steps: Boolean, optional
            Keep all steps' calculated Mu_space in memory and return them

        Returns
        -------
        if keep_all_steps is true: 
            universal_int_Mu: dict from time_step to list of all possible diceretized Mu_space at that time_step
            counts: a list containing number of points in Mu_space in time_steps
        else:
            None.
            

        '''
        counts=[]
        universal_int_Mu={}
        self.partitioning_points_num=partitioning_points_num
        # make universal Mu space

        # ############### Make Mu-Space
        
        # make possible wealth values
        self.Ss=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step,comparison_precision=100000)
        
        # make the path of saving the Mu-space. 
        # Mu_space we mean that a disceretized Mu-space (over y and s) which should calculated for
        # each time step (because possible S values depend on time_step)
        self.Mu_path=os.path.join(os.getcwd(),'B_all_Mu_spaces_discrete')
        
        
        
        # calculate Mu_space for each time_step    
            
        for step in range(self.max_time_step+1):
            
            # generate all possible probability distributions over possible s_values(wealth) and possible y-values (real state) with quantization step-size equal to 1/partitioning_points_num
            # note: the number of possible values of distribution is partitioning_points_num +1 (e.g: if partitioning_points_num=2 then possible probability values are : {0,0.5,1} ) 
            # note2: possible probability values are expressed in integer numbers in the base of b=partitioning_points_num+1 (e.g: if partitioning_points_num=2 then possible integer
            # probability values are : {0,1,2} ). This is because avoiding float variables, therefor more efficient memory allocation as well faster computionas.
            
            thisStep_int_Mu=self.chunk_mu([0,1],self.Ss[step],self.partitioning_points_num)
        
            # ############## Save Mu-Space
            if save_Mu==True:
                if not os.path.exists(self.Mu_path):
                    os.makedirs(self.Mu_path)
                file_path=os.path.join(self.Mu_path,'time_step_'+str(step)+'.npy')
                np.save(file_path,thisStep_int_Mu)
            
            ################# keep steps in memmory
            if keep_all_steps==True:
                counts.append(len(thisStep_int_Mu))
                universal_int_Mu[step]=thisStep_int_Mu
                
        if keep_all_steps==True:        
            return universal_int_Mu,counts
        else:
            return 
    
    def load_mu(self):
        '''
        Load all saved Mu_space points from disk (address is same as saving function) and make a universal variable of them.
        
        Returns:
        --------
        universal_int_Mu: a dict from time_steps to list of all possible Mu points in that step. 
        Mu_space expressed by integers to reduce the size, like 'make_all_mu()' .
        '''
        universal_int_Mu={}
        self.Mu_path=os.path.join(os.getcwd(),'B_all_Mu_spaces_discrete')
        for step in range(self.max_time_step+1):
            if not os.path.exists(self.Mu_path):
                print('directory not found!')
                return
            file_path=os.path.join(self.Mu_path,'time_step_'+str(step)+'.npy')
            thisStep_Mu=np.load(file_path,allow_pickle=True)
            universal_int_Mu[step]=thisStep_Mu
        return universal_int_Mu
    
    def load_internal_states_CN_discrete(self):
        '''
        Load all saved internal_states points from disk (address is same as saving function) and make a universal variable of them.
        
        Returns:
        --------
        universal_internal_state: a dict from time_steps to list of all possible internal_state points in that step. 
        internal_state_space expressed by integers to reduce the size, like 'make_all_internal_states_CN_discrete()'.
        States represented by ( x, Mu, time)
        '''
        universal_internal_state={}
        self.internal_state_path=os.path.join(os.getcwd(),'B_all_internal_spaces_discrete')
        for step in range(self.max_time_step+1):
            if not os.path.exists(self.internal_state_path):
                print('directory not found!')
                return
            file_path=os.path.join(self.internal_state_path,'time_step_'+str(step)+'.npy')
            thisStep_internal_state=np.load(file_path,allow_pickle=True)
            universal_internal_state[step]=thisStep_internal_state
        return universal_internal_state        
    def make_all_internal_states_CN_discrete(self,partitioning_points_num,save_state_space=True,keep_all_steps=True):
        '''
        
        This function generates all possible internal-states space points (x(observable part), Mu (PDF over wealths), time_step).
        Saving the x and time are redundant, becasue current x has no effect on the value of the next state (its effect has been included in the new Mu). Also, as the internal_state space recorded seperately for each time step,
        recording time is redundant. However, in sake of similarity to the original paper, and for better comparison in space complexity we used a repesentation like the papers. 
        
        Parameters
        ----------
        partitioning_points_num : int
            It determines the number of steps to chunk the interval of [0,1] in order to discreteize the Mu_space.
            Note: the number of points are 1 more than this number. e.g: partitioning_points_num=2 then, we have points ={0,50%,100%}  
            
        save_Mu: Boolean, optional
            Save internal_state space or not. The default is True.
            
        keep_all_steps: Boolean, optional
            Keep all steps' calculated internal_state space in memory and return them

        Returns
        -------
        if keep_all_steps is true: 
            universal_internal_state: dict from time_step to list of all possible diceretized state space at that time_step
            counts: a list containing number of points in state space in time_steps
        else:
            None.
            

        '''
        counts=[]
        universal_internal_state={}
        self.partitioning_points_num=partitioning_points_num
        # make universal Mu space

        # ############### Make Mu-Space
        
        # make possible wealth values
        self.Ss=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step,comparison_precision=100000)
        
        # make the path of saving the Mu-space. 
        # Mu_space we mean that a disceretized Mu-space (over y and s) which should calculated for
        # each time step (because possible S values depend on time_step)
        self.Mu_path=os.path.join(os.getcwd(),'B_all_internal_spaces_discrete')
        
        
        
        # calculate Mu_space for each time_step    
            
        for step in range(self.max_time_step+1):
            
            # generate all possible probability distributions over possible s_values(wealth) and possible y-values (real state) with quantization step-size equal to 1/partitioning_points_num
            # note: the number of possible values of distribution is partitioning_points_num +1 (e.g: if partitioning_points_num=2 then possible probability values are : {0,0.5,1} ) 
            # note2: possible probability values are expressed in integer numbers in the base of b=partitioning_points_num+1 (e.g: if partitioning_points_num=2 then possible integer
            # probability values are : {0,1,2} ). This is because avoiding float variables, therefor more efficient memory allocation as well faster computionas.
            
            thisStep_int_Mu=self.chunk_mu([0,1],self.Ss[step],self.partitioning_points_num)
            x=[0]*len(thisStep_int_Mu)
            x.extend([1]*len(thisStep_int_Mu))
            x=np.array(x).reshape(2*len(thisStep_int_Mu),1)
            thisStep_int_internal_state=np.tile(thisStep_int_Mu,(2,1))
            del(thisStep_int_Mu)
            thisStep_int_internal_state=np.hstack([x,thisStep_int_internal_state])
            
            del(x)
            gc.collect()
            
            time=np.array([step]*len(thisStep_int_internal_state)).reshape(len(thisStep_int_internal_state),1)
            thisStep_int_internal_state=np.hstack([thisStep_int_internal_state,time])
            
            
        
            # ############## Save Mu-Space
            if save_state_space==True:
                if not os.path.exists(self.Mu_path):
                    os.makedirs(self.Mu_path)
                file_path=os.path.join(self.Mu_path,'time_step_'+str(step)+'.npy')
                np.save(file_path,thisStep_int_internal_state)
            
            ################# keep steps in memmory
            if keep_all_steps==True:
                counts.append(len(thisStep_int_internal_state))
                universal_internal_state[step]=thisStep_int_internal_state
                
        if keep_all_steps==True:        
            return universal_internal_state,counts
        else:
            return 
    
    def chunk_mu(self,Y,S,partitioning_points_num):
        '''
        This function does two things together:
            first, chunks the interval [0,1] to a given number of discerete points (partitioning_points_num+1) 
            with the smallest point always equal to 0 and the biggest point=1.
            second, produces all combinations of the previous discerete values on a 2-D space of (Y x S), while the summation 
            of all values be equal to 1.
        By these operations, at the end, we have all of possible discretized PDFs over the Mu-space.
        
        Here, to avoid heavy process- and memory-consuming floating-point operations,we express the discrete points by
        integer numbers. These numbers can be seen as numbers in base of partitioning_points_num+1.

        Parameters
        ----------
        Y : list of integers
            index of un-observable states.
        S : list of numbers
            different possibe wealth values.
        partitioning_points_num : int
            number of chunks of [0,1]. number of points representing the interval is pdf_chunks_counts+1.

        Returns
        -------
        All possible discretized PDFs over Mu-space, expressed in integers in based-of pdf_chunks_counts+1.

        '''
        partitioning_points_num=partitioning_points_num
        pdf_values=np.arange(partitioning_points_num)
        
        ### initial D-PDF (Discrete-PDF)
        pdf=np.zeros(len(Y)*len(S),dtype=int)
        # probability of 1 for the first condition and zero for others
        pdf[0]=partitioning_points_num
        # add 200 as offset to place the related ASCII codes in a wide printable area  
        pdf=pdf+200
        # concatenate all coded characters to one string
        coded_pdf=''.join(chr(p) for p in pdf)
        
        # generate all possible combinations of discrete PDFs by a recursive function
        num=0
        combinations={}
        self.add_combinations(coded_pdf,combinations,[num])
        
        possible_PDF_dists=[[ord(point)-200 for point in PDFcomb] for PDFcomb in combinations.keys()]
        possible_PDF_dists=np.array(possible_PDF_dists)
        possible_int_dists=possible_PDF_dists[:]
        #possible_PDF_dists=possible_PDF_dists*(1./pdf_chunks_counts)
        return possible_int_dists
    
    def add_combinations(self,base,combs,num):
        '''
        It adds a given combination(base) to the 'combs' dictionary (if it was not redundant). Then, generates different
        combinations (based-on 'base') and calls itself to do this things on them. 
        Generating combinations performed by removing one unit from the first position and add it to each of the 
        other positions.By a dynamic programming way of thinking we can see it produce all valid combinations. 

        Parameters
        ----------
        base : String
            A sequence of characters which each of them represents a coded possible discretized PDF value .
        combs : Dict
            A dictionary which maps different coded Mu-values (D-PDF on MU-space) as keys to a number as value. 
        num : list of int
            

        Returns
        -------
        None.

        '''
        # remove redundant combinations
        if base in combs:
            return
        else:
            # add the new combination
            num[0]=num[0]+1
            combs[base]=num[0]
            
            # decode the base
            base_pdf=[ord(p)-200 for p in base]
            # making base's childs
            for candidate in range(1,len(base)):
                # making the next input for recursion
                next_pdf=base_pdf.copy()
                next_pdf[0]=next_pdf[0]-1
                next_pdf[candidate]=next_pdf[candidate]+1
    
                coded_pdf=''.join(chr(p+200) for p in next_pdf) 
                if next_pdf[0]<0:
                    pass
                else:
                    self.add_combinations('%s' % coded_pdf,combs,num)
        return
    
    
    # Discrete planning
    def nearest_grid_points(self,int_say_result):
        '''
        This function gets the result of say calculations in integer values and in base of 'num_of_PDF_chunks*10' and returns the nearest poit in Mu-space which has
        integer values in base-of 'num_of_PDF_chunks'. Here, distance defined as sum of distances from a grid-point in all dimenstions.
        

        Parameters
        ----------
        int_say_result : 2-D array of int
            result of SAY calculations.

        Returns
        -------
        int_say_result : 2-D array of int
            nearest dicretized point in Mu-space.

        '''
        # after dividing by 10, the sum of all values of a point is equal to num_of_PDF_chunks. Also, in all grid-points total number should be
        # equal to num_of_PDF_chunks ( but only integer numbers are permitted). So:
        # 1. we exclude how many discrete probability values that already exist
        # 2. if just considering case-1's values, then how many discerete poits remain to achieve total of num_of_PDF_chunks ( equal to 1 in base of num_of_PDF_chunk)
        # 3. sort less than 10 (less than 1 in base of num_of_PDF_chunk) residuals from big to small
        # 4. select n biggest residuals when n equals to remaining amount of integers to have a correct probability (i.e. n= ouput of case-2)
        # 5. round-up the biggests and floor the others
        for i,res in enumerate(int_say_result):
            b=res
            # 1
            c=np.array(b/10).astype(np.int8)
            #2
            d=self.partitioning_points_num-c.sum()
            if d==0: ############################################# change!
                int_say_result[i]=c
            else:
                #3
                reminders=np.array(b%10).astype(np.int8) 
                idx_rem=np.nonzero(reminders)[0]
                val_rem=reminders[idx_rem]
    
                idx_rem=idx_rem.reshape(1,len(idx_rem))
                val_rem=val_rem.reshape(1,len(val_rem))
                h=np.append(idx_rem,val_rem,axis=0)
    
                sort_dec_vals=h[:, np.argsort(-val_rem)][0][0]
                # 4
                idx_to_inc=sort_dec_vals[0:d]
                # 5
                c[idx_to_inc]=c[idx_to_inc]+1
                int_say_result[i]=c
        return int_say_result
    
    def discrete_state_expander2(self,max_time_step,Mu0,partitioning_points_num,
                                 r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=True,
                                 memory_threshold=1000000,rounding_prec_coeff=100000):
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
            'B_state_space_discrete'. The default is True.
            
        make_transition_kernel : Boolean, optional
            Make a mapping from indexes of this step's internal states to the index of next state's internal_states. Transition kernel
            is also a function of action and next observation. If this option is not set, the extender_function bypass the part that makes
            transition kernels. The default is True.
            
        save_transition_kernel : Boolean, optional 
            Save the made transition kernels of each time step in a seperate file in directory: 'B_transition_kernel_discrete'. If set to True,
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
            B_state_path=os.path.join(os.getcwd(),'B_state_space_discrete')
            if not os.path.exists(B_state_path):
                os.makedirs(B_state_path)
                
        #saving total probability over states of each time step separately
        if save_reachables:
            B_probs_path=os.path.join(os.getcwd(),'B_total_probabilities_discrete')
            if not os.path.exists(B_probs_path):
                os.makedirs(B_probs_path)
                       
        # saving Transition kernel of each time step seperately. Transition kernel is a dict contains keys: each (action, observation) pair and values: index
        # of internal states in depth i-th and the successive internal state in depth i+1-th 
        if save_transition_kernel:
            B_TK_path=os.path.join(os.getcwd(),'B_transition_kernel_discrete')
            if not os.path.exists(B_TK_path):
                os.makedirs(B_TK_path)
                
        # use disk in case of big state spaces
        if use_buffer:
            buffer_path=os.path.join(os.getcwd(),'B_buffer_discrete')
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
        
        # make possible wealth values
        self.Ss=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step,comparison_precision=rounding_prec_coeff)
        ################ make Q-kernels
                
        # q(x_prim, y_prim | x,y,a) while in our settign it is equal to q(x_prim,y_prim|y,a)
        # make Q-kernel, the probability of reaching each y_prim x_prim pair when the real state is y and doing action a
        q=self.make_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        # q(x_prim | x,y,a) while in our settign it is equal to q(x_prim|y,a)
        # make marginal-Q-kernel, the probability of getting observation x_prim, when the real state is y and doing action a
        mq=self.make_marginal_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)

        ##############################
        # expand state space for each time step
        
        for step in range(self.max_time_step):
            
            if step==0:
                
                current_mu=Mu0
                current_mu=np.array(current_mu)
                
                # save possible initial states 
                if save_reachables:
                    filename='depth0.npy'
                    filepath=os.path.join(B_state_path,filename)
                    np.save(filepath,Mu0)
                    
                # save size of possible initial states    
                reachable_counts.append(len(Mu0))
            
            
            
            ##### Do for all steps
            
            # save probability of being in each state
            if save_reachables:
                
                # make mu float 
                real_mu=self.myRound((np.array(current_mu)/self.partitioning_points_num),decimals=r_round)
                s_size=int(len(real_mu[0])/2)
                p_y0= np.sum(real_mu[:,:s_size],axis=1).reshape(len(real_mu),1)
                p_y1= np.sum(real_mu[:,s_size:],axis=1).reshape(len(real_mu),1)
                del(real_mu)
                gc.collect()
                probs=np.hstack([p_y0,p_y1])
                
                filepath=os.path.join(B_probs_path,'depth'+str(step)+'.npy')
                np.save(filepath,probs)
                
                del(p_y0)
                del(p_y1)
                del(probs)
                gc.collect()
                
            # fetch previous step's internal state's data
            Mu_size=len(current_mu)    # number of Mu- points
            prev_num=Mu_size
            ls2=len(current_mu[0])  # size ofMu-points
            ls=int(ls2/2)
            
            next_ls=len(self.Ss[step+1])
            next_ls2=int(next_ls  * 2)
            
            all_current_s=self.Ss[step]
            all_next_s=self.Ss[step+1]
            
            
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
                next_states=np.empty((self.num_of_actions,self.num_of_observations,Mu_size,next_ls2))
            elif mode==2:
                next_states=np.empty((self.num_of_observations,Mu_size,next_ls2))
            else:
                next_states=np.empty((Mu_size,next_ls2))
            
            # reachable states in the next step
            reachables=[]
            
            # make Mu float
            current_mu=self.myRound(np.array(current_mu)/self.partitioning_points_num,decimals=r_round)
            
            #decompose Mu-beliefs about being in each state (y=0 or 1)
            mus_y0=current_mu[:,0:ls]
            mus_y1=current_mu[:,ls:2*ls]
            mus_ys=[mus_y0,mus_y1]
       
            # Marginal Mu (Mu superscript Y in the paper, or Mu(dy,R)). This variable expresses the probability of being in each state(y)
            marginal_mus_y0=mus_y0.sum(axis=1)
            marginal_mus_y1=mus_y1.sum(axis=1)
            marginal_mus_ys=[marginal_mus_y0,marginal_mus_y1]
    
            for action in range(self.num_of_actions):
                a=action
                # in Beurele notation observabale parts of state (here observation) is x, and un-observable parts of the state are x
                for x_prime in range(self.num_of_observations):
                    
                    ########################################################   PSi ##############################
                    
                    ## prepare
                    
                    # C(x,y,a) while C depends only on y
                    #rewards/costs of doing action "a" when the real state is "y" 
                    y0=0
                    y1=1
                    y_prime0=0
                    y_prime1=1
                    c_y0=self.env.rewards[y0][action]
                    c_y1=self.env.rewards[y1][action]
                    c=[c_y0,c_y1]
                    z=np.power(self.env.discount_factor,step)
                    
                    ################### Calculate the denominator of SAY function
                
                    # It is equal to probaility of recieving observation x_prim, regardless of what are the wealth(s) or the state(y). 
                    # It calculate sum of probabilities of reaching each state (regardless of wealth level) (marginal_mus_y), while reaciving x_prime observation
                    say_denominator=marginal_mus_y0*mq[0][action][x_prime]+marginal_mus_y1*mq[1][action][x_prime]
                    #say_denominator=self.myRound(say_denominator)
                    
                    #################### Calculate the nomerator of the SAY function
                
                    # The dimensions of the Mu-space doesn't change with just one action and one observation.  Because in our experiment the reward function is deterministic, all of possible wealths of this step, will transfer to just one other value 
                    # based on state and action. So, the size of Mu-space remains constant in SAY calculator function (for doing only one action). Also, the number of possible distributions over Mu, is not a concern for this function: It maps all current possible values to
                    # continious values
        
                    ## allocate variables for results of nomerator calculations for each current state
        
                    # these arrays are here to represent Mu distribution of the next state (naturally over its own (next state's) wealth levels) 
                    for_y0=np.zeros((len(current_mu),len(all_next_s)*2))
                    for_y1=np.zeros((len(current_mu),len(all_next_s)*2))
                    next_mus=[for_y0,for_y1]
        
                    
                    
                    ## tmp_mus[0] for y=0 and tmp_mus[1] for y=1 calculations
                    tmp_mus=np.zeros((2,Mu_size,ls2))
                    
                    ##
                     # for each current state
                    for y,y_mus in enumerate(tmp_mus):
                        
                        # mus_ys is Mu-beliefs about being in a staet (y)
                        # probability of reaching y_prim=0
                        y_mus[:,:len(all_current_s)]=mus_ys[y]*q[y][action][y_prime0][x_prime]
                        # probability of reaching y_prim=1
                        y_mus[:,len(all_current_s):]=mus_ys[y]*q[y][action][y_prime1][x_prime]
        
                        # Dirac function: d(s+zc(x,y,a))
                        # Here, we compute the possible values of s for next Mu distribution         
                        next_possible_s=np.array(all_current_s)+z*c[y]
                        next_possible_s=self.myRound(next_possible_s,decimals=r_round)
        
                        #The calculated Mu distributions on the S-axis, are defined on the current S values, however these probabilities are for s+zc(y,a).
                        # so as the SAY function inputs are (x,a,x_prim), for each y we can map Mu(s) to be matched with next stage's possible values
        
                        # indexes of the next wealth levels which are the successors ( after current S-values recieving c(y,a) )
                        # Here, we used comparison method to check the equality of next time-spte's S-values and current S-values+ z*c 
                        next_s_indexes=np.empty(0,np.int32)
                        for ind,ns in enumerate(next_possible_s):
                            
                            next_related_index=np.where(np.abs(np.array(all_next_s)-ns)<(1/(rounding_prec_coeff*10)))[0][0]
                            next_s_indexes=np.append(next_s_indexes,next_related_index)
        
        
                        #assign Mu-probability to each possible S(=previous_s + c) point
                        # for S points in y_prim=0        
                        next_mus[y][:,next_s_indexes]=y_mus[:,:len(all_current_s)]
                        # for S points in y_prim=1
                        next_mus[y][:,next_s_indexes+len(all_next_s)]=y_mus[:,len(all_current_s):]
        
        
                    # sum of probabilities of next Mu-space for both conditions : p(mu|y0), p(mu|y1)
                    say_nomerator=next_mus[0]+next_mus[1]
                    #say_nomerator=self.myRound(say_nomerator)
                    # calculate final SAY result
                    # normalize the whole Mu-space dist. by dividing it by totoal probability of taking x_prim
                    
                    say_result=say_nomerator/say_denominator[:,None]  
                    #say_result=self.myRound(say_result,decimals=r_round)
                    
                    say_result=self.nearest_grid_points((say_result*self.partitioning_points_num*10).astype(np.int32))
                    
                    #######################################################################################################
                    
                    # fill the results based on the mode
                    # all together ( fill the action-&-observation related positions )
                    if (mode==1 or mode==0):
                        next_states[a,x_prime]=say_result
                        
                    # save for each action (fill the observation related positions) 
                    elif mode==2:
                        next_states[x_prime]=say_result
                        
                    # save for each action and observation ( fill the the whole variable, becasue it should represent 
                    # one unspecified action_observation pair). Then save that.
                    else:
                        # mode 3
                        next_states=say_result                      
                        # save the full_next_states in the buffer 
                        filename='a'+str(a)+'_xprime'+str(x_prime)+'.npy' # filename example: a2_yprime1.npy
                        filepath=os.path.join(buffer_path,filename)
                        np.save(filepath, next_states)
    
    
                    ############### find next_steps's unique reachable states
                    # remove redundant elements. By making all next states tuple and make them keys of a dict, then retrieve the keys as a list.
                    new_reachables=list((map(tuple,say_result)))
                    new_reachables=list(dict.fromkeys(new_reachables,0).keys())
                    
                    # accumulate all unique next states
                    # And, remove the records which are redundant between different sets of appended values 
                    if a==0 and x_prime==0:
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
                filepath=os.path.join(B_state_path,'depth'+str(step+1)+'.npy')
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
                for x_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        # read a file
                        filename='a'+str(a)+'_xprime'+str(x_prime)+'.npy'
                        filepath=os.path.join(buffer_path,filename)            
                        read=np.load(filepath,allow_pickle=True)
                        
                        # make index list for this action,observation pair
                        indexes=[]
                        
                        s_primes=read
                        for s_prime in s_primes:
                            # adding the index of unique reachable next_states for each for this current internal state under a and y_prime
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(x_prime)]=indexes
                        
            elif mode==2:
                for a in range(self.num_of_actions):
                    # read a file
                    filename='a'+str(a)+'.npy'
                    filepath=os.path.join(buffer_path,filename)            
                    read=np.load(filepath,allow_pickle=True)
                    
                    # the files contain data of y_prime =0 and 1
                    for x_prime in range(self.num_of_observations):
                        s_primes=read[x_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(x_prime)]=indexes
            
            # if all data were in one file            
            elif mode==1:
                # read a file
                filename='all.npy'           
                filepath=os.path.join(buffer_path,filename)            
                read=np.load(filepath,allow_pickle=True)
                
                # the file contains all data so it needs loop over all action and observations to seperate them
                for x_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        s_primes=read[a,x_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(x_prime)]=indexes
            
            # if not using the buffer
            elif mode==0:
                for x_prime in range(self.num_of_observations):
                    for a in range(self.num_of_actions):
                        # use the next_state variable which we had previously
                        s_primes=next_states[a,x_prime]
                        indexes=[]
                        for s_prime in s_primes:
                            indexes.append(reachables[tuple(s_prime)])
                        TK_indexes[str(a)+'_'+str(x_prime)]=indexes
            
            # if want to save the transition kernel
            # we use Pandas pickle write function
            if save_transition_kernel:
                filepath=os.path.join(B_TK_path,'depth'+str(step)+'_'+str(step+1)+'.pkl')
                pp=pd.DataFrame(TK_indexes)
                pp.to_pickle(filepath)
                
            # if not save the transition data on disk, then we should keep all of them in memory     
            else:
                all_TK.append(TK_indexes)
            
            
            ##### set prevous state for the next iteration
            current_mu=np.array(list(reachables.keys()))       
            
        # In current implementation, we have saved reachable states on the disk.
        # if don't want to write transition kernel on disk we should return it.
        if make_transition_kernel and not save_transition_kernel :
            return reachable_counts,all_TK
        else:
            return reachable_counts
        
    def discrete_value_iteration(self,partitioning_points_num,initial_wealth,exp_weights,exp_vorfaktoren,rounding_prec_coeff=100000,
                                 r_round=5,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=True,
                                 memory_threshold=1000000,keep_q_func=True,exp_util=True,utility_approx=False,apx_terms_num=2):
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
            'B_state_space_discrete'. The default is True.
            
        make_transition_kernel : Boolean, optional (used only in expander2)
            Make a mapping from indexes of this step's internal states to the index of next state's internal_states. Transition kernel
            is also a function of action and next observation. If this option is not set, the extender_function bypass the part that makes
            transition kernels. The default is True.
            
        save_transition_kernel : Boolean, optional (used in expander2, and in value_iteration for loading data)
            Save the made transition kernels of each time step in a seperate file in directory: 'B_transition_kernel_discrete'. If set to True,
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
        self.exp_vorfaktoren=exp_vorfaktoren
        self.exp_weights=exp_weights
        self.initial_wealth=initial_wealth
        self.exp_num=len(self.exp_vorfaktoren)
        # actual number is one extra point. i.e: self.partitioning_points_num=3 means valid points of {0,1,2,3} or {0%,33%,66%,100%}
        self.partitioning_points_num=partitioning_points_num
    
        
        # make emtpy dictionaries to store results
        self.keep_q=keep_q_func
        self.q_func={}
        self.value_func={}
        self.action_func={}
        
        ##### saving variables
        
        # save reachable states in each time step (each step's data separately )
        if save_reachables:
            B_state_path=os.path.join(os.getcwd(),'B_state_space_discrete')    # name of reachable states' folder 
            if not os.path.exists(B_state_path):
                os.makedirs(B_state_path)
        #saving total probability over states of each time step separately
        if save_reachables:
            B_probs_path=os.path.join(os.getcwd(),'B_total_probabilities_discrete')
            if not os.path.exists(B_probs_path):
                os.makedirs(B_probs_path)
                       
        
        # save transition kernel of each step in disk separately. A dict contains keys: each (action, observation) pair and values: index
        # of internal states in depth i-th and the successive internal state in depth i+1-th 
        if save_transition_kernel:
            B_TK_path=os.path.join(os.getcwd(),'B_transition_kernel_discrete')  # name of Transition kernels' folder
            if not os.path.exists(B_TK_path):
                os.makedirs(B_TK_path)
          
        # make possible wealth values
        self.Ss=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step,comparison_precision=rounding_prec_coeff)
        # q(x_prim | x,y,a) while in our settign it is equal to q(x_prim|y,a)
        # make marginal-Q-kernel, the probability of getting observation x_prim, when the real state is y and doing action a
        mq=self.make_marginal_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        #######################################################################
        #                   First Depth's Possible Points                     #
        #######################################################################
        
        Mu0=self.chunk_mu(self.env.states,[self.initial_wealth],self.partitioning_points_num)
                    
        
        #######################################################################
        #                   Forward search and expantion                      #
        #######################################################################
        # starting from the state0 and expand the decision tree until the end.
        # make reachable states in all steps and the transition kernel between them.
        
        if make_transition_kernel and not save_transition_kernel :
            in_memory=True
            self.step_sizes,self.transition_kernel=self.discrete_state_expander2(self.max_time_step,Mu0,self.partitioning_points_num,self.r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=False,use_buffer=use_buffer,memory_threshold=memory_threshold,rounding_prec_coeff=rounding_prec_coeff)
        else:
            in_memory=False
            self.step_sizes=self.discrete_state_expander2(self.max_time_step,Mu0,self.partitioning_points_num,self.r_round,save_reachables=True,make_transition_kernel=True,save_transition_kernel=True,use_buffer=use_buffer,memory_threshold=memory_threshold,rounding_prec_coeff=rounding_prec_coeff)
            
        #######################################################################
        #                        Backward evaluation                          #
        #######################################################################
        del(Mu0)
        gc.collect()
        
        
        # backwardly
                        
        for step in range(self.max_time_step,-1,-1):
                
            ############### filling last step
            if step==self.max_time_step:
                
                # fetch list of all possible states in the last time Step
                if save_reachables:
                    filename='depth'+str(step)+'.npy'
                    filepath=os.path.join(B_state_path,filename)
                    this_step=np.load(filepath,allow_pickle=True)
                else:
                    # not implememnted
                    pass
                this_step=self.myRound((this_step/self.partitioning_points_num),decimals=r_round)
                # calculate the value
                # with sum of exponentials utility func
                val=self.utility_evaluator_md(this_step,self.Ss[step],self.exp_weights,self.exp_vorfaktoren,
                                              exp_util,utility_approx,apx_terms_num)
                
                
                # fill the results
                # we don't feel q_func in this last step
                self.value_func[step]=val.copy()
                self.action_func[step]=np.array([-1]*len(val)).astype(np.int8)
                
                del(val)
                gc.collect()
                
            #################### filling other steps 
            else:
                # fetch list of all possible states in the last time Step
                if save_reachables:
                    filename='depth'+str(step)+'.npy'
                    filepath=os.path.join(B_probs_path,filename)
                    this_step_prob=np.load(filepath,allow_pickle=True)
                    #print(this_step_prob)
                    p_y0=np.array(this_step_prob[:,0]).reshape(len(this_step_prob),1)
                    p_y1=np.array(this_step_prob[:,1]).reshape(len(this_step_prob),1)
                else:
                    # not implememnted
                    pass
                
                if save_transition_kernel:
                    filename='depth'+str(step)+'_'+str(step+1)+'.pkl'
                    filepath=os.path.join(B_TK_path,filename)
                    tk=pd.read_pickle(filepath)
                else:
                    # not implemented
                    tk=pd.DataFrame(self.transition_kernel[step])
                    pass
    
                self.q_func[step]=np.array([[0]*self.num_of_actions]*len(tk)).astype(np.float64)
                
                
                for action in range(self.num_of_actions):
                    
                    
                    
                    for x_prime in range(self.num_of_observations):
                        
                        # make transition code to find the related column(index of the next states) in the saved transition kernels
                        a_x_code=str(action)+'_'+str(x_prime)
                         
                        # fetch related values of this action,y_prime transition, multiply thom with 1/|y| and accumulate them for each action
                        #tmp=self.myRound((1./self.num_of_observations)*self.value_func[step+1][tk.loc[:,a_x_code].to_list()],decimals=r_round)
                        
                        
                        # value of this internal state and action is equal to the value of the successive state (only its Mu part)
                        val_xprime=self.myRound(self.value_func[step+1][tk.loc[:,a_x_code].to_list()],decimals=r_round)
                        val_xprime=np.array(val_xprime).reshape(len(val_xprime),1)
                        p_xprime=mq[0][action][x_prime]*p_y0 +mq[1][action][x_prime]*p_y1
                        
                        this_xprime_value= p_xprime * val_xprime
                        
        
                        # add related value of each next observation to make the expected value of the action
                        self.q_func[step][:,action]+=this_xprime_value.reshape(-1)
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
    
    
    
    
    