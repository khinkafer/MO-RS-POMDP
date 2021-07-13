import json
import numpy as np


# Generalized Tiger Problem POMDP
# POMDP={S,A,T,R,Omega,O,Gamma}

class tiger_POMDP_env:
    
    states={0:"tiger_right",
            1:"tiger_left"}
    
    actions={"listen":0,
             "open_right_low":1,
             "open_left_low":2,
             "open_right_high":3,
             "open_left_high":4,}
    
    observations={0: "sound_right",
                  1: "soud_left"}
    def __init__(self,read_config,config_address,parameters=None):
        '''
        read_config determines wheteher we want to read environment parameters from a .json file (read_config==True) 
            or to take them directly from 'parameters' input (False) which is a dictionary 
          
        '''
        if read_config==True:
            
            params=self.__read_config__(config_address)
             
        elif read_config==True:
            params=parameters
        else:
            print('bad input')
            return
            
        # assigning parameters    
        self.pS_RL=params['state_change_r2l']
        self.pS_LR=params['state_change_l2r']
        self.pO_RL=params['false_observation_get_l_while_r']
        self.pO_LR=params['false_observation_get_r_while_l']
        self.c_0=params['reward_listen']
        self.c_l=params['reward_low_incorrect']
        self.c_h=params['reward_high_incorrect']
        self.r_l=params['reward_low_correct']
        self.r_h=params['reward_high_correct']
        self.gamma=params['discount_factor']
        self.initial_wealth=params['initial_wealth']
        self.discount_factor=self.gamma
        
        # creating transition function matrix
        
        self.transition_matrix=np.zeros((len(self.states),len(self.actions),len(self.states)))
        # set state changing probabilities 0.5 for 'open' actions. ( to start new trial with random position of tiger)
        self.transition_matrix[:]=0.5
        # set state changing probabities for 'listen' actions ([s1][0][s2])
        self.transition_matrix[0][0][0]=1-self.pS_RL
        self.transition_matrix[0][0][1]=self.pS_RL
        self.transition_matrix[1][0][0]=self.pS_LR
        self.transition_matrix[1][0][1]=1-self.pS_LR
       
        #creating reward matrix
        
        self.rewards=np.zeros((len(self.states),len(self.actions)))
        # listening cost
        self.rewards[:,0]=self.c_0
        # correct and low
        self.rewards[0,1]=self.rewards[1,2]=self.r_l
        # incorrect and low
        self.rewards[0,2]=self.rewards[1,1]=self.c_l
        # correct and high
        self.rewards[0,3]=self.rewards[1,4]=self.r_h
        # incorrect and high
        self.rewards[0,4]=self.rewards[1,3]=self.c_h
        
        # creating observation matrix
        
        self.observation_matrix=np.zeros((len(self.actions),len(self.states),len(self.observations)))
        # set observations of 'open' actions to 0.5 . Because, the 'open' actions terminate a trial and start a new one
        # so as the reward is not observable, the observation should be un-informative.
        self.observation_matrix[:]=0.5
        # ALL other actions:
        for i in range(1,len(self.actions)):
            self.observation_matrix[i][0][0]=0.8
            self.observation_matrix[i][0][1]=0.2
            self.observation_matrix[i][1][0]=0.2
            self.observation_matrix[i][1][1]=0.8


        
        # observations of 'listen' action ([0][state][oservation])
        # accurately observed Right
        self.observation_matrix[0][0][0]=1-self.pO_RL
        # noisy observation of Left while the state is Right 
        self.observation_matrix[0][0][1]=self.pO_RL
        # accurately observed Left
        self.observation_matrix[0][1][0]=self.pO_LR
        # noisy observation of Right while the state is Leftt 
        self.observation_matrix[0][1][1]=1-self.pO_LR
        self.time_step=0
        self.reset()
        return
    
    def __read_config__(self,address):
        with open(address,'r') as f:
            env_params=json.loads(f.read())
        p={}
        p['state_change_r2l']=env_params["state_transition_prob"]["right_to_left"]
        p['state_change_l2r']=env_params["state_transition_prob"]["left_to_right"]
        p['false_observation_get_l_while_r']=env_params["false_observation_prob"]["false_left"]
        p['false_observation_get_r_while_l']=env_params["false_observation_prob"]["false_right"]
        p['reward_listen']=env_params["reward"]["listen"]
        p['reward_low_incorrect']=env_params["reward"]["incorrect_open"]["low_risk"]
        p['reward_high_incorrect']=env_params["reward"]["incorrect_open"]["high_risk"]
        p['reward_low_correct']=env_params["reward"]["correct_open"]["low_risk"]
        p['reward_high_correct']=env_params["reward"]["correct_open"]["high_risk"]
        p['discount_factor']=env_params["discount_factor"]
        p['initial_wealth']=env_params["initial_wealth"]
        return p
    
    
    def step(self, action):
        #  calculate reward
        reward=self.rewards[self.current_state][self.actions[action]]
        # calculate next state
        r=np.random.uniform(0,1)
        next_state=self.current_state if r<self.transition_matrix[self.current_state][self.actions[action]][self.current_state] else 1-self.current_state
        # go to the next step
        self.current_state=next_state
        self.trial_time_step=self.trial_time_step+1
        self.time_step=self.time_step+1
        if action!='listen':
            self.trial_time_step=0
            
        # give the next state's observation
        r2=np.random.uniform(0,1)
        observation=self.current_state if r2<self.observation_matrix[self.actions[action]][self.current_state][self.current_state] else 1-self.current_state
        
        return self.time_step,self.trial_time_step,self.current_state,reward,observation
        
                   
    def reset(self):
        self.current_state=np.random.randint(2)
        self.trial_time_step=0
        