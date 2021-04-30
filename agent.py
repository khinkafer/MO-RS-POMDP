import numpy as np
from datetime import datetime
import sys
import gc
import os
from env import *



class Bauerle_Rieder_agent(object):
    
    def __init__(self,environment, num_of_Mu_chunks,max_iterations):
        
        self.env=environment
        
        # parameters
        
        # for each number of 'num_of_Mu_chunks', there are 1 more discerete points.
        # for e.g: by num_of_Mu_chunks=5, PDF values can be {0, 0.2, 0.4, 0.6, 0.8, 1}  
        self.num_of_PDF_chunks=num_of_Mu_chunks
        # time_step=0 means initial point, therefore time_step=n means decision-tree with n actions depth. 
        self.max_time_step=max_iterations
        # the wealth amount at the beginning
        self.initial_wealth=self.env.initial_wealth
        
        
        
        # results
        
        # value_function contains value of all internal-states (pairs of (x,Mu,z)) in each time-step. 
        # each of its elements represents values at a specific time-step: value_function[t0]= value of internal-states at time-step t0
        # each element is a 2-D array represents x0 and x1 :  value_function[t0][x0]= value of (x0,Mu,z) at t0 
        # each element in value_function[t][x] is a value number related to a specific Mu-value: value_function[t0][x0][Mu0]= value of (x0,Mu0,z) at t0 
        # Note 1: variable Z actually represents the time-step, so we can easilly see (x,Mu,z) as (x,Mu,t); so, z=discount_factor ^ t  
        # Note 2: values will be calculated for all possible Mu-values at a time-step and it restore in an ordered array 
        # with length equal to all possible Mu-values overally. So, to find no_None indexes in the value function (as well as
        # in the action_function) and having access to their actual Mu-value representation, the possible values in each time-step 
        # are stored in 'step_indexes' variable
        # Note 3: final time-step values is a 1-D array. because the final value is indipendant from x.( saving was redundant)
        self.value_function=[]
        # like as value_function, but for the best action of that state. best means actin with maximum reward.
        self.action_function=[]
        # indexes of possible Mu-values in each time-step. final step is fully complete and mapped with universal_Mu indexes, so not saved.
        self.step_indexes=[]
        
        # aux-parameters
        
        self.num_of_actions=len(self.env.actions)
        self.num_of_observable_states=len(self.env.observations)
        self.num_of_unbservable_states=len(self.env.states)
        self.num_of_observations=self.num_of_observable_states
        self.num_of_states=self.num_of_unbservable_states
        
        
        return
    
    def pre_planning_paration(self,make_and_save_Mu=True,save_Mu2index_chunks=True):
        '''
        
        This function generates all possible wealths levels(S), the Mu-space(PDF over S and Y) 
        and a mapping from different Mu-values to some indexes. 
        
        Parameters
        ----------
        make_and_save_Mu : Boolean, optional
            calculate and save Mu-space or just load it from a file. The default is True.
        save_Mu2index_chunks : Boolean, optional
            save chunked Mu-to-index mappings. The default is True.

        Returns
        -------
        None.

        '''
        
        # make universal Mu space

        # ############### Make Mu-Space
        
        # generate all possible s-values (wealth values) during the experiment with maximum trial equal to max_time_step
        self.S=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.max_time_step)
        
        # make the path of saving the universal Mu-space. 
        # By universal-Mu we mean that a disceretized Mu-space (over y and s) which covers all 
        # possible values of s during the eperiment when the maximum depth is max_time_step 
        path=os.path.join(os.getcwd(),'Universal_Mu')
        file_path=os.path.join(path,'time_step'+str(self.max_time_step)+'.npy')
        # path of save/load Mu-to-index dictionary chunked files
        self.Mu_chunks_path=os.path.join(os.getcwd(),'Universal_Mu_chunks')
        
        if make_and_save_Mu==True:
            # generate all possible probability distributions over possible s_values(wealth) and possible y-values (real state) with quantization step-size equal to 1/num_of_PDF_chunks
            # note: the number of possible values of distribution is num_of_PDF_chunks +1 (e.g: if num_of_PDF_chunks=2 then possible probability values are : {0,0.5,1} ) 
            # note2: possible probability values are expressed in integer numbers in the base of b=num_of_PDF_chunks+1 (e.g: if num_of_PDF_chunks=2 then possible integer
            # probability values are : {0,1,2} ). This is because avoiding float variables, therefor more efficient memory allocation as well faster computionas.
            self.universal_int_Mu=self.chunk_mu([0,1],self.S,self.num_of_PDF_chunks)
            
            # ############## Save/Load Mu-Space
    
            # save the universal Mu space           
            path=os.path.join(os.getcwd(),'Universal_Mu')
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(file_path,self.universal_int_Mu)
        else:
            
            # load pre-calculated universal_Mu
            self.universal_int_Mu=np.load(file_path)
        
        if save_Mu2index_chunks==True:
            
            # ############# Save dictionaries of {Mu-Space:indicator_number}
            #check if the directory exists and if not, make it
            if not os.path.exists(self.Mu_chunks_path):
                os.makedirs(self.Mu_chunks_path)
            # number of files to split the whole Mu-space. 
            # here I used (num_of_PDF_chunks * max_time_step) as a number proportionate to size of universal-Mu 
            # Note: the actual number of files is often one more than this number (num_of_files+1) 
            num_of_files=self.num_of_PDF_chunks * self.max_time_step
            
            # number of elements per file
            file_step_size=int(len(self.universal_int_Mu)/num_of_files)
            
            for i in range(num_of_files):
                # indicator is somehow an index of a specific probability dist. over Mu-space 
                indicator=np.arange(file_step_size*i,file_step_size*(i+1))
                # content of files are dictionaries with keys: different possible distributions over Mu, and values: index/indicator
                d=dict(zip(list(map(tuple,self.universal_int_Mu[file_step_size*i:file_step_size*(i+1)])),indicator))    
                # save data. In case of lack of memory: maybe there is a more effient model of saving data
                np.save(os.path.join(self.Mu_chunks_path,'Mu_chunk_'+str(i)+'.npy'),d)
            
            # Do things for the last chunk of Mu-points    
            indicator=np.arange(file_step_size*num_of_files,len(self.universal_int_Mu))
            d=dict(zip(list(map(tuple,self.universal_int_Mu[file_step_size*num_of_files:len(self.universal_int_Mu)])),indicator))
            np.save(os.path.join(self.Mu_chunks_path,'Mu_chunk_'+str(num_of_files)+'.npy'),d)
        return
    
    def value_iteration(self,utility_function='risk-neutral'):
        '''
        This function starts value iteration process backward (from last time-step to first step) and 
        in each step it adds the values and best action of all internal-states(x,Mu,z) in that time-step to 
        variables 'value_function' and 'action_function' respectively. It also adds the index of valid
        the Mu-values of that time-step to the 'step_indexes' variable. 

        Parameters
        ----------
        utility_function : String, optional
            Here, by determining the utility function, we can produce the risk-sensitivity. This variable will
            be passed to 'last_step_RS_value()' function, which applies te utility func. on the final wealths of
            the experiment. The default is 'risk-neutral'.

        Returns
        -------
        None.

        '''
        self.utility_function=utility_function
        # it used max_time_step and universal_int_Mu as a global variable
        for time_step in range(self.max_time_step,-1,-1):
            print('step:',time_step)
            
            if (time_step==self.max_time_step):
                # last step
                print('==== iterations')
                # fill the last element of value_function with an array contains: Utility(Value(each final internal-state))
                self.value_function.insert(0,self.last_step_RS_value())
                # The last step has no action ( we have A0, A1, .. An-1), so we fill it by None.
                self.action_function.insert(0,None)
                # All final states' Mu values are valid so saving the indexes was redundant
                self.step_indexes.insert(0,None)
            else:
                # any other time-step
                self.timeStep_value_calculator(time_step)
                print('------')
        return
    
    
    def generate_possible_wealths(self,cost_reward_values,initial_wealth,discount_factor,trials):
        '''
        
        It Generates all possible (unique) amounts of cost/reward combinations during the trials.S-dimension of Mu. 
         
        FOR NOW: it generates all possible combinations in all time-steps, but I should make it more efficient

        Parameters
        ----------
        cost_reward_values : array of numbers
            set of all possible cost and rewards of actions.
        discount_factor : float
            Gamma value (Beta in the paper).
        trials : int
            indicates the depth of decision-making and consequently number of possible wealth amounts .

        Returns
        -------
        an array includes all possible wealth levels until the specified trial.

        '''
        # add initial_wealth as the value of the first time-step. 
        values=np.array([initial_wealth])
        for t in range(trials+1):
            # if it is the first time-step, pass
            if t==0:
                pass
            else:
                tmp_values=np.empty(0)
                for val in cost_reward_values:
                    # next step values can be current ones + discounted value of each action, (except for the firts trial)
                    if t==1:
                        tmp=values + val
                    else:
                        tmp=values + val*discount_factor
                    tmp_values=np.append(tmp_values,tmp)
                values=np.unique(np.append(values,tmp_values))   
        return values
    
    def chunk_mu(self,Y,S,pdf_chunks_counts):
        '''
        This function does two things together:
            first, chunks the interval [0,1] to a given number of discerete points (pdf_chunks_counts+1) 
            with the smallest point always equal to 0 and the biggest point=1.
            second, produces all combinations of the previous discerete values on a 2-D space of (Y x S), while the summation 
            of all values be equal to 1.
        By these operations, at the end, we have all of possible discretized PDFs over the Mu-space.
        
        Here, to avoid heavy process- and memory-consuming floating-point operations,we express the discrete points by
        integer numbers. These numbers can be seen as numbers in base of pdf_chunks_counts+1.

        Parameters
        ----------
        Y : list of integers
            index of un-observable states.
        S : list of numbers
            different possibe wealth values.
        pdf_chunks_counts : int
            number of chunks of [0,1]. number of points representing the interval is pdf_chunks_counts+1.

        Returns
        -------
        All possible discretized PDFs over Mu-space, expressed in integers in based-of pdf_chunks_counts+1.

        '''
        pdf_points_num=pdf_chunks_counts+1
        pdf_values=np.arange(pdf_chunks_counts)
        
        ### initial D-PDF (Discrete-PDF)
        pdf=np.zeros(len(Y)*len(S),dtype=int)
        # probability of 1 for the first condition and zero for others
        pdf[0]=pdf_chunks_counts
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
            d=self.num_of_PDF_chunks-c.sum()
            if d==0:
                pass
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

    def say_calculator(self,x,action,x_prim,current_int_mu,z,current_possible_s):
        '''
        It applied the SAY updating rule to Mu-space.

        Parameters
        ----------
        x : int
            0 or 1 as the observable state (observation).
        action : int
            index of the action (among 1-5).
        x_prim : int
            0 or 1 as the next observable-state (nex observation).
        current_int_mu : 2-D array of int
            a reduced version of Mu-space points which contains only relevant points (regarding the possible S values 
            at this time-step).
        z : float
            accumulated discount factor for this time-step (discount_factor ^ time-step).
        current_possible_s : 1-D array of floats
            all possible S(wealth) values until this time_step.

        Returns
        -------
        2-D array of int
            It calculates the updated Mu-space based on (x',Mu,z,a), then calls 'nearest_grid_points()' 
            to return a dicretized Mu-point ( a valid point in the pre-designed grid on Mu-space) 

        '''
    
        # mus contains all possible current Mu-space combinations in this stage
        mus=current_int_mu*(1./self.num_of_PDF_chunks)
        #print('size of Mu :',sys.getsizeof(mus),'len of Mu:',len(current_int_mu))
        
        # decompose Mu-beliefs about being in each state (y=0 or 1)
        mus_y0=mus[:,0:int(len(mus[0])/2)]
        mus_y1=mus[:,int(len(mus[0])/2):len(mus[0])]
        mus_ys=[mus_y0,mus_y1]
        
        # Marginal Mu (Mu in power of Y in the paper, or Mu(dy,R)). This variable expresses the probability of being in each state(y)
        marginal_mus_y0=mus_y0.sum(axis=1)
        marginal_mus_y1=mus_y1.sum(axis=1)
        marginal_mus_ys=[marginal_mus_y0,marginal_mus_y1]
        
        # q(x_prim, y_prim | x,y,a) while in our settign it is equal to q(x_prim,y_prim|y,a)
        # make Q-kernel, the probability of reaching each y_prim x_prim pair when the real state is y and doing action a
        q=self.make_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        # q(x_prim | x,y,a) while in our settign it is equal to q(x_prim|y,a)
        # make marginal-Q-kernel, the probability of getting observation x_prim, when the real state is y and doing action a
        mq=self.make_marginal_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
    
        # C(x,y,a) while C depends only on y
        #rewards/costs of doing action a when the real state is y 
        y0=0
        y1=1
        y_prim0=0
        y_prim1=1
        c_y0=self.env.rewards[y0][action]
        c_y1=self.env.rewards[y1][action]
        c=[c_y0,c_y1]
        
        # Calculate the denominator of SAY function
        
        # It is equal to probaility of recieving observation x_prim, regardless of what are the wealth(s) or the state(y). 
        # It calculate sum of probabilities of reaching each state (regardless of wealth level) (marginal_mus_y), while reaciving x_prime observation
        say_denominator=marginal_mus_y0*mq[0][action][x_prim]+marginal_mus_y1*mq[1][action][x_prim]
        
        # calculate the nomerator of the SAY function
        # allocate variables for results of nomerator calculations for each current state
        for_y0=np.zeros((len(mus),len(self.S)*2))
        for_y1=np.zeros((len(mus),len(self.S)*2))
        next_mus=[for_y0,for_y1]
        
        # tmp_mus[0] for y=0 and tmp_mus[1] for y=1 calculations
        tmp_mus=np.zeros((2,len(mus),len(mus[0])))
        
        # for each current state
        for y,y_mus in enumerate(tmp_mus):
            
            # mus_ys is Mu-beliefs about being in a staet (y)
            # probability of reaching y_prim=0
            y_mus[:,:len(self.S)]=mus_ys[y]*q[y][action][y_prim0][x_prim]
            # probability of reaching y_prim=1
            y_mus[:,len(self.S):]=mus_ys[y]*q[y][action][y_prim1][x_prim]
            
            # Dirac function: d(s+zc(x,y,a))
            # Here, we compute the possible values of s for next Mu distribution
            next_possible_s=current_possible_s+z*c[y]
            
            #The calculated Mu dist. on axis S, are assigned to the current s values, however these probabilities are for s+zc(y,a)
            # so as the SAY function inputs are (x,a,x_prim), for each y we can rotate Mu(s) to be matched with next stage's possible values
            
            # indexes of S axis which has value in this SAY calculation
            current_s_indexes=np.where(np.isin(self.S,current_possible_s)==True)[0]
            # indexes of S axis which which would be the successor, after current S-values recieving c(y,a) 
            next_s_indexes=np.where(np.isin(self.S,next_possible_s)==True)[0]
            
            
            #tmp=np.zeros((len(mus),len(mus[0]))).astype(np.int8)
            #assign Mu-probability to each possible S(=previous_s + c) point
            # for S points in y_prim=0        
            next_mus[y][:,next_s_indexes]=y_mus[:,current_s_indexes]
            # for S points in y_prim=1
            next_mus[y][:,next_s_indexes+len(self.S)]=y_mus[:,current_s_indexes+len(self.S)]
            
            #next_mus[y]=tmp[:]
            
        
        # sum of probabilities of next Mu-space for both conditions : p(mu|y0), p(mu|y1)
        say_nomerator=next_mus[0]+next_mus[1]
        
        # calculate final SAY result
        # normalize the whole Mu-space dist. by dividing it by totoal probability of taking x_prim
        say_result=say_nomerator/say_denominator[:,None]   
        
        # mapping probabily dist. values (float number between 0 and 1) to integer numbers in base of the num_of_PDF_chunks*10 (continious integer values 
        # between 0 and num_of_PDF_chunks*10)
        # The reason is we want to perform calculations by integer values rather than float values (for memory and time efficiency).
        # So, we first multiply the continious values between 0 and 1 by num_of_PDF_chunks to have continious values between 0 and num_of_PDF_chunks.
        # Then, by multiplying by 10 and casting to integer, we have 10 times bigger integers which their ones position can be assumed as one digit after decimal point
        # while things remain integer yet. 
        # After finding the nearest correct point in Mu-space, we are using the values in base of num_of_PDF_chunks (not base of num_of_PDF_chunks*10) 
        int_say_result=(say_result*self.num_of_PDF_chunks*10).astype(np.int8)
        
        #return say_result
        return self.nearest_grid_points(int_say_result)

    def last_step_RS_value(self ):
        '''
        This function gets universal-Mu (possible Mus at the final stage of the task), as well as the type of utility function (utility function: a mapping of 
        S-values to different utility values). The default is risk-neutral which means utilityValues=S-Values. Any other utility function should be implemented here.
        The value calculated by Sum(belief_about_a_level_of_wealth x utility_value_of_that_level_of_wealth)[regardless of y]

        Returns
        -------
        values : 1-D float
            Returns an array in legnth of universal_Mu that elements represent
            the value of each Mu-point at the final time-step .

        '''
        # apply utility function on different wealth values
        if self.utility_function=='risk-neutral':
            utility_mapped_values=self.S
        else:
            pass
        
        # duplicate utility values to cover Y-dimension of the Mu-space
        utility_mapped_values=np.tile(utility_mapped_values,2)
        
        # value of a Mu-point calculated as (probability of being in that S, regardless of y- or Mu(s)) x (utility value of that amount of wealth-or U(s))
        # the formula in the papaer is: integral(integral(Us)) Mu(ds,dy) which is equal above formula
        # Note: Here we trasform the integer mu-values to the dicretized floating poits value between 0 and 1
        values=np.dot(self.universal_int_Mu/self.num_of_PDF_chunks,utility_mapped_values)
        return values  

    def mu_to_index(self,mu):
        '''
        This function gets an array of combinations of distributions over Mu-space and returns that points' indicator. It reads the previously 
        saved mappings of universal-Mu points to indicators which are located in 'universal_mu_chunks' directory. The files contain dictionary data-structure with
        keys= tuples of probabilities over Mu-space and values=an integer number as the indicator 

        Parameters
        ----------
        mu : 2-D array of int
            an array of Mu-points which we want to find their indexes.

        Returns
        -------
        1-D array of int
            the indexes of input points in the universal_Mu.

        '''
        # get number of all files in the related directory
        path, dirs, files = next(os.walk("./Universal_Mu_chunks"))
        file_count = len(files)
        
        # define a None vector to save final indicator numbers
        indicators=[None]*len(mu)
        # read all files one-by-one
        for f in range(file_count):
            map_file=np.load(os.path.join(self.Mu_chunks_path,'Mu_chunk_'+str(f)+'.npy'), allow_pickle=True).item()
            keys=map_file.keys()
            # for each mu point
            for i,res in enumerate(mu):
                if indicators[i]!=None:
                    # if it has been filled before, pass 
                    pass
                else:
                    r=tuple(res)
                    if (r in keys):            
                        indicators[i]=map_file[r]    
    
        return indicators
   
    def XA_value_calculator(self,x,action,current_possible_mu,time_step,current_possible_s):
        '''
        This function calculates the value of given (x, action). 
        Based-on current x and given action it: 
            1.finds the next Mu-space (SAY(Mu)) for each possible next observation (x'=0 and 1)
            2.finds the next (x',Mu',z') value
            3.based-on probability of being in each state(Mu(y)) and probability of recieving each observation (P(x'|x,y,a)),
            it returns the expected value of next time-step's value
            
        

        Parameters
        ----------
        x : int
            index of current x: 0 or 1.
        action : int
            index of selected action: 1 to 5.
        current_possible_mu : 2-D array of int
            all valid Mu-points until this time-step.
        time_step : int            
        current_possible_s : 1-D array of float
            possible accumulated wealths (s) during all previous steps (including current time-step).

        Returns
        -------
        current_XA_value : array of floats
            an array of expected values for each current Mu-point, based-on current x and the given action.

        '''
    
        next_step=time_step+1
        z=np.power(self.env.gamma,time_step)
        
        
        # value of the next internal-state (internal-state=(X,Mu,Z) )
        # this variable is the V(x_prim,say(mu),gamma*z) part of page 6 second equation.
        next_internalState_value_each_xPrim=np.array([np.empty(len(current_possible_mu)),np.empty(len(current_possible_mu))])
        
        # probability of being in each y
        y_probs=(np.sum(current_possible_mu[:,0:len(self.S)],axis=1)/self.num_of_PDF_chunks).reshape(len(current_possible_mu),1)
        y_probs=np.append(y_probs,(np.sum(current_possible_mu[:,len(self.S):len(self.S)*2],axis=1)/self.num_of_PDF_chunks).reshape(len(current_possible_mu),1),axis=1)
        
        # Q(x'|x,y,a).. which based-on the experiment design is equal to Q(x'|y,a)
        MQ=self.make_marginal_Q_kernel(self.env.transition_matrix,self.env.observation_matrix)
        
        # for each possible x_prim, calculate the values of the next internal-state 
        for i,x_prim in enumerate(list(self.env.observations.keys())):
            # VALUE of the next internal-state
            next_int_mu=self.say_calculator(x,action,x_prim,current_possible_mu,z,current_possible_s)
            
            next_indicators=np.array(self.mu_to_index(next_int_mu))
            
            if next_step==self.max_time_step:
                next_mu_xPrim_z_value=self.value_function[-1][next_indicators]
            else:
                
                next_mu_xPrim_z_value=self.value_function[time_step-self.max_time_step][x_prim][next_indicators]
            
        
            # PROBABILITY of the next internal-state
            
            # probability of transition to the next x' for being in y=0 or 1 and doing the given action
            y_a_to_xPrim_transitionKernel=np.array([MQ[0][action][x_prim],MQ[1][action][x_prim]])
            
            # actual probability of reaching the next x' for actual probability of being in y=0/1 and doing the given action
            xPrim_prob=np.dot(y_probs,y_a_to_xPrim_transitionKernel)
            
            # VALUE x PROBABILITY of each next x' (by current x,mu,z and a)
            next_internalState_value_each_xPrim[i]=next_mu_xPrim_z_value*xPrim_prob
        
        #sum of weighted (by probability) values of next x_prims, which in the value-iteration is the new value of current Q((x,mu,z),a) 
        current_XA_value=np.sum(next_internalState_value_each_xPrim,axis=0)
        
        return current_XA_value
            
    def timeStep_value_calculator(self,time_step):
        '''
        This function calculates the best value and best action of time-step for current x= 0 and 1.
        It computes the value of each (x,a) pair and for each x finds the best (high rewarded) action and its value.
        It adds the found value and action arrays (2-D array for (X x Mu) to 'value_function' and 'action_function' global variables)

        Parameters
        ----------
        time_step : int
            current time-step.

        Returns
        -------
        None. Just adds best values and best actions to 'value_function' and 'action_function' global variables.

        '''
        int_mus=self.universal_int_Mu.copy()
        # for test simulation
        #current_possible_s=generate_possible_wealths([-2,-1,-0.5,1,2],time_step)
        # the exact one
        current_possible_s=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,time_step)
        
        # find possible mu-points in this time_step and only consider them
        impossible_mus_indexes=np.where(np.isin(self.S,np.array(current_possible_s))==False)[0]
        impossible_mus_indexes=np.append(impossible_mus_indexes, impossible_mus_indexes+len(self.S))    
        int_mus[:,impossible_mus_indexes]=0
        int_mus=int_mus[np.where(int_mus.sum(axis=1)==self.num_of_PDF_chunks)].astype(np.int8)
        
        # indicators of this stage's Mu-space
        current_indicators=np.array(self.mu_to_index(int_mus))
        self.step_indexes.insert(0,current_indicators)
        
        # value of each (x,a) pair
        XA_vals=np.array([[np.empty(len(int_mus))]*self.num_of_actions]*self.num_of_observable_states)
        
        to_V=np.array([[None]*len(self.universal_int_Mu),[None]*len(self.universal_int_Mu)])
        to_A=np.array([[None]*len(self.universal_int_Mu),[None]*len(self.universal_int_Mu)])
        
        for x in range(len(XA_vals)):
            for action in range(len(XA_vals[x])):
                print('x:',x,'a:',action)
                XA_vals[x][action]=self.XA_value_calculator(x,action,int_mus,time_step,current_possible_s)
            
            best_values=np.max(XA_vals[x], axis=0)
            best_actions=np.argmax(XA_vals[x], axis=0)
            
            
            to_V[x][current_indicators]=best_values
            to_A[x][current_indicators]=best_actions
        self.value_function.insert(0,to_V)
        self.action_function.insert(0,to_A)
        return   

    def reset(self,y0_prob,initiative_observation):
        '''
        It reset the agent to its belief about the initial wealth and set time-step to zero (begining of the simulation)
        This function distributes the probability equally over different s-values of initial wealths. The difference is
        just applied to different idden states. This uniform distribution makes this function not a general one 
        but for our experiment it was even more than enough.

        Parameters
        ----------
        y0_prob : float
            primary belief of agent about being in each S-value in y0 hidden state. 
            1-y0_prob will be its belief about other state's s-values.
        initiative_observation : int
            The first x (observable state).
        Returns
        -------
        Noting. makes the agent ready for a new simulation.

        '''
        y0_int_prob=y0_prob*self.num_of_PDF_chunks
        if (y0_int_prob%1!=0):
            print('bad input value!')
            return
        else:
            # making a Mu-point-shape zero array
            self.current_internal_belief=np.zeros(len(self.S)*2)
            # set elements related to initial wealth value  
            self.current_internal_belief[np.where(self.S==self.initial_wealth)[0]]=y0_int_prob
            self.current_internal_belief[np.where(self.S==self.initial_wealth)[0]+len(self.S)]=self.num_of_PDF_chunks-y0_int_prob
            # set time-step=0
            self.current_internal_timeStep=0
            # define a variable for current x
            self.current_internal_x=initiative_observation
            self.last_action=None
        
            return
    
    def do_action(self):
        '''
        This function returns the best action based-on: current time-step, current observable-state, and current Mu
        
        Returns
        -------
        best_action: int 
            Among actions' indexes (1-5)
        value_of_action: float
            value of doing that action at that (x,Mu,z) point
        belief_at_action: 1-D array of float
            the current Mu point which leads to choosing this action (before doing the action) which is expressed as a PDF
        '''
        # check wheather trial has ended or not 
        if self.current_internal_timeStep>=self.max_time_step:
            print('too much iterations!')
            return
        else:
            
            # make current_internal_belief fit for mu_to_index function
            self.current_internal_belief=np.array([self.current_internal_belief]).reshape(-1)
            
            # index of current Mu
            self.current_internal_Mu_index=self.mu_to_index(np.array([self.current_internal_belief]))
            
            # choose best action and its value from 'value_function' and 'action_function' variables 
            best_action=self.action_function[self.current_internal_timeStep][self.current_internal_x][self.current_internal_Mu_index]
            value_of_action=self.value_function[self.current_internal_timeStep][self.current_internal_x][self.current_internal_Mu_index]
            belief_at_action=self.current_internal_belief
            belief_index_at_action=self.current_internal_Mu_index
            
            # record the choosen action to use in updating function
            self.last_action=best_action[0]

            return best_action[0],value_of_action[0],belief_at_action/self.num_of_PDF_chunks
        
    def update_agent(self,new_observation):
        '''
        This function takes the new observable state, and based-on current Mu, last observable-state and last action updates the : Mu, observable-state and time-step
        
        Parameters
        ----------
        new_oservation : int ( 0 or 1)
            the observation from environment, after the agent's action
        '''
        # check wheather trial has ended or not 
        if self.current_internal_timeStep>=self.max_time_step:
            print('too much iterations!')
            return
        else:     
            # possible S-values of this time-step, just to give to SAY-calculator() 
            next_step_possible_S=self.generate_possible_wealths(np.unique(self.env.rewards),self.initial_wealth,self.env.discount_factor,self.current_internal_timeStep)
            
            # update the current Mu
            self.current_internal_belief=self.say_calculator(x=self.current_internal_x,
                                action=self.last_action,
                                x_prim=new_observation,
                                current_int_mu=np.array([self.current_internal_belief]),
                                z=np.power(self.env.gamma,self.current_internal_timeStep),
                                current_possible_s=next_step_possible_S)
            
            # increase time-step
            self.current_internal_timeStep=self.current_internal_timeStep+1
            
            # update current observable-state 
            self.current_internal_x=new_observation
            return
        
        
        