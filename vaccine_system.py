import numpy as np 
import matplotlib.pyplot as plt 
from people_generation import * 
from Citizen import * 
from infection_rate import * 

class Government(object) : 
    """ Government Agent """
    def __init__(self, policy : str, priority_period : list, days : int) : 
        self.policy = policy 
        self.processed_requests = []  
        self.priority_period = np.array(priority_period)
        self.days = days 
        self.current_day = 0 
    
    def get_policy(self):
        return self.policy

    def reset(self) : 
        self.processed_requests = []    

    def decide(self, requests_by_risk, susceptible_num_by_risk, requests_by_priority, 
                susceptible_num_by_priority, citizen_num, vaccines_num, requests, setting) : 
        """ governemnt decides if should give vaccine injection 
            setting : 
                - "full_info" : government has full information of the lining requests
                - "streaming" : requests comes in order, government needs to make decision for coming requests and process the next request
        
        """

        if setting == "full_info" : 
            vaccinated_ids = []
            if self.policy == "Unif" : 
                vaccinated_ids = sampling(vaccines_num, requests)

            elif self.policy == "UP" :  
                
                for group_id in range(len(requests_by_priority)) : 
                    group_vaccine = int(vaccines_num * susceptible_num_by_priority[group_id] / citizen_num)
                    vaccinated_ids.append(sampling(group_vaccine, requests_by_priority[group_id]))

            elif self.policy == "SP" : 
                i = 0
                max_i = len(requests_by_priority) - 1
                while vaccines_num > 0:
                    if i > max_i: break

                    group_ids = requests_by_priority[i]
                    if vaccines_num <= len(group_ids):
                      vaccinated_ids.append(sampling(vaccines_num, group_ids))
                      vaccines_num = 0
                    else:
                      vaccinated_ids.append(group_ids)
                      vaccines_num = vaccines_num - len(group_ids)

                    i += 1
            
            elif self.policy == "MP" : 
                mixture_mask = (self.priority_period * self.days).astype(int) <= self.current_day 
                
                vac_group = []
                vac_num_by_group = []

                for pri_id, mixture_pri in enumerate(requests_by_priority) : 
                    if mixture_mask[pri_id] : 
                        vac_group.append(requests_by_priority[pri_id])
                        vac_num_by_group.append(susceptible_num_by_priority[pri_id])
                
                # if priority group vac num is positive 
                if np.sum(vac_num_by_group) > 0 : 
                    for group_id in range(len(vac_group)) : 
                        group_vaccine = int(vaccines_num * vac_num_by_group[group_id] / np.sum(vac_num_by_group))
                        vaccinated_ids.append(sampling(group_vaccine, vac_group[group_id]))
                        vaccines_num -= group_vaccine
                
                if vaccines_num > 0 : 
                    vac_group = []
                    vac_num_by_group = []

                    for pri_id, mixture_pri in enumerate(requests_by_priority) : 
                        if not mixture_mask[pri_id] : 
                            vac_group.append(requests_by_priority[pri_id])
                            vac_num_by_group.append(susceptible_num_by_priority[pri_id])
                    
                    for group_id in range(len(vac_group)) : 
                        group_vaccine = int(vaccines_num * vac_num_by_group[group_id] / np.sum(vac_num_by_group))
                        vaccinated_ids.append(sampling(group_vaccine, vac_group[group_id]))
                        vaccines_num -= group_vaccine
                
     
        elif setting == "streaming" : 
            if self.policy == "unif" : 
                vaccine_prob = vaccines_num / citizen_num 
                cur_decide = np.random.rand() < vaccine_prob
                return cur_decide
        
        self.days += 1 

        return vaccinated_ids 

class vaccine_system(object) : 
    def __init__(self, gov, citizen_num, vaccine_num, days, r_nau, gamma, risk_group_num, priority_group_num, setting="full_info") :
        self.gov = gov 
        self.citizen_num = citizen_num
        self.gamma = gamma 
        self.citizen_stats = []
        self.gen_citizen(self.citizen_num)

        self.ids = np.array(self.citizen_stats[:, self.col_indx["id"]], dtype=int)
        self.risk_group_num = risk_group_num 
        self.priority_group_num = priority_group_num 

        # generate original infected number 
        ori_infected_num = int(np.ceil(self.citizen_num * 0.006))
        ori_infected_ids = np.random.choice(self.ids, ori_infected_num, replace=False)
        
        self.citizens_i_status = self.citizen_stats[:, self.col_indx["infected"]]
        self.citizens_i_status[ori_infected_ids] = 1 
        self.citizen_stats[:, self.col_indx["request_vaccine"]][self.citizen_stats[:, self.col_indx["infected"]] == 0] = 1 
        
        self.gammas = self.citizen_stats[:, self.col_indx["gamma"]]
        self.requests = self.citizen_stats[:, self.col_indx["request_vaccine"]]
        self.death_risk = self.citizen_stats[:, self.col_indx["death_risk"]].astype(float)
        
        self.risk_group = self.citizen_stats[:, self.col_indx["risk"]]
        self.risk_matrix = self.create_risk_matrix(self.risk_group_num)
        
        self.priority_group = self.citizen_stats[:, self.col_indx["priority"]]
        
        self.status_to_idx = {"susceptible" : 0, "infected" : 1, "anti_body" : 2, "ill" : 3} 
        self.vaccine_num = vaccine_num
        self.vaccinated = []
        self.days = days 
        self.r_nau = r_nau 
        
        self.setting = setting
        self.current_day = 1

        self.risk_infected_id_by_group = {}
        self.risk_susceptible_id_by_group = {}
        self.risk_requests_by_group = {}
        self.risk_anti_body_by_group = {}

        self.pri_infected_id_by_group = {}
        self.pri_susceptible_id_by_group = {}
        self.pri_requests_by_group = {} 
        self.pri_anti_body_by_group = {}

        self.grouping(True, True, True)
    
    def reset(self, gov, citizen_num, vaccine_num, days, r_nau, setting="full_info") : 
        """ reset all parameters for new simulation """
        pass 

    def grouping(self, update_id=False, update_risk=False, update_priority=False) : 
        if update_risk : 
            self.risk_infected_num_by_group = []
            self.risk_susceptible_num_by_group = []

            for group_id in range(self.risk_group_num) : 
                self.risk_infected_num_by_group.append(np.sum((self.citizens_i_status == 1) * (self.risk_group == group_id)) )
                self.risk_infected_id_by_group[group_id] = self.ids[(self.risk_group == group_id) * (self.citizens_i_status == 1)] 
                
                self.risk_susceptible_num_by_group.append(np.sum((self.citizens_i_status == 0) * (self.risk_group == group_id)) )
                self.risk_susceptible_id_by_group[group_id] = self.ids[(self.risk_group == group_id) * (self.citizens_i_status == 0)] 

                self.risk_requests_by_group[group_id] = self.ids[(self.risk_group == group_id) * (self.requests == 1)]
                self.risk_anti_body_by_group[group_id] = self.ids[(self.risk_group == group_id) * (self.citizens_i_status == self.status_to_idx["anti_body"])]
            
            self.risk_infected_num_by_group = np.array(self.risk_infected_num_by_group)
            self.risk_susceptible_num_by_group = np.array(self.risk_susceptible_num_by_group)
            
        if update_priority : 
            self.pri_infected_num_by_group = []
            self.pri_susceptible_num_by_group = []

            for group_id in range(self.priority_group_num) : 
                self.pri_infected_num_by_group.append(np.sum((self.citizens_i_status == 1) * (self.priority_group == group_id)))
                self.pri_infected_id_by_group[group_id] = self.ids[(self.citizens_i_status == 1) * (self.priority_group == group_id) ]
                
                self.pri_susceptible_num_by_group.append(np.sum((self.citizens_i_status == 0) * (self.priority_group == group_id)))
                self.pri_susceptible_id_by_group[group_id] = self.ids[(self.citizens_i_status == 0) * (self.priority_group == group_id) ] 

                self.pri_requests_by_group[group_id] = self.ids[(self.priority_group == group_id) * (self.requests == 1)]
                self.pri_anti_body_by_group[group_id] = self.ids[(self.priority_group == group_id) * (self.citizens_i_status == self.status_to_idx["anti_body"])]

            self.pri_infected_num_by_group = np.array(self.pri_infected_num_by_group)
            self.pri_susceptible_num_by_group = np.array(self.pri_susceptible_num_by_group)

    """Added by Faye: Assining groups for different policies"""
    def gen_group_for_policies(self, age, occupation, essential):
        policy = self.gov.get_policy()
        # 0.15563900668028777 p(age being above 65)
        # 0.16551055 p(being in healthcare)
        # 0.39548794499999995  p(being essential worker)

        if policy[0] == "U" : 
            priority = 0 
        else : 
            if policy == "SP" or policy == "MP":
                #healthcare workers
                if occupation < len(occ_lst) and occ_lst[occupation] == 'Health Care and Social Assistance' : 
                    priority = 0
                #essential workers
                elif essential == 1: priority = 1 
                #seniors
                elif age > 55: priority = 2
                else: priority = 3

            elif policy == "pB": 
                #healthcare workers
                if essential == 0: priority = 0
                #essential workers and cannot wfh
                elif essential == 1: priority = 1 
                #seniors
                elif age > 55: priority = 2
                #essential workers and can wfh
                elif essential == 2: priority = 3
                else: priority  = 4
        
        return priority 

    def gen_citizen(self, citizen_num) : 
        """ generate virtual citizens """
        for i in range(citizen_num) : 
            info = Rn_Citizen()
            info["request_vaccine"] = False
            info["ill"] = False
            info["priority"] = self.gen_group_for_policies(info["age"], info["occupation"], info["essential"])
            info["id"] = i
            info["gamma"] = np.random.poisson(self.gamma)

            death_risk_params = ["age", "gender", "ethnicity", "weight", "height", "home_cat", "diabetes_type", "if_cancer"]
            death_risk_params = {key : value for key, value in info.items() if key in death_risk_params}
            info["death_risk"] = Citizen.get_sick_after_infection(**death_risk_params)
            
            info["risk"] = None 

            for risk_id, occupations in enumerate(occupation_by_risk) :     
                if info["occupation"] < len(occ_lst) and occ_lst[info["occupation"]] in occupations : 
                    info["risk"] = len(occupation_by_risk) - risk_id - 1

            if info["occupation"] == len(occ_lst) : 
                info["risk"] = 5

            self.citizen_stats.append(list(info.values()))
        
        self.citizen_stats = np.array(self.citizen_stats)
        self.col_indx = {key : idx for idx, key in enumerate(info.keys())}

        return 

    def recovery(self, re, ill_rate=0.1, method="resolve") : 
        """ simulate the recovery """

        # get the recovery index for infected people getting resolved 
        if method == "ratio" :  
            recovery_num = int(np.sum(self.citizens_i_status) * re) 
            new_recovery_idx = list(np.random.choice(np.arange(np.sum(self.citizens_i_status)), recovery_num, replace=False))
        elif method == "resolve" :
            gamma_mask = self.gammas == 0 
            infect_mask = self.citizens_i_status == 1 
            resolve_mask = gamma_mask * infect_mask
            new_resolve_idx = self.ids[resolve_mask]

            #print(type(self.death_risk[0]), self.death_risk[0])
            ill_mask = np.random.rand(len(self.death_risk)) < self.death_risk
            self.ill_idx = self.ids[resolve_mask * ill_mask]
        
        self.healthy_new_resolve_idx = np.setdiff1d(new_resolve_idx, self.ill_idx)

    def infection_spread(self, r_nau, infectious_period=5, random_mode=True) :
        """ simulate the infection """
        
        # compute infect parameter for each group 
        group_infect_param = np.matmul(self.risk_matrix, self.risk_infected_num_by_group) / self.citizen_num

        #print(group_infect_param)
        # compute infect num from each group's susceptible 
        group_newly_infected_num = self.risk_susceptible_num_by_group * group_infect_param
        #print(self.gov.get_policy(), "health worker : ", group_newly_infected_num[0])
        #print(group_infect_param)
        # sample infected id from each group 
        #print(self.risk_susceptible_id_by_group)
        group_newly_infected_id = [sampling(group_newly_infected_num[group_id], self.risk_susceptible_id_by_group[group_id]) for group_id in range(self.risk_group_num)]
        self.group_newly_infected_id = group_newly_infected_id


    def update(self) : 
        """" implements recovery, infection -> vaccine injection """
        # recover 
        self.recovery(0.1)
        
        # infection spread 
        noise = np.random.normal()
        noise = 1 if noise < -self.r_nau else noise  
        noisy_r_nau = self.r_nau + noise 
        self.infection_spread(noisy_r_nau, 5)
        
        # update from last day's stat 
        self.citizens_i_status[self.healthy_new_resolve_idx] = 2 
        self.citizens_i_status[self.ill_idx] = 3 
        #print(self.current_day, "recovery", "I : ", np.sum(self.citizens_i_status == 1), "U : ", self.citizen_num-np.sum(self.citizens_i_status == 1), "R : ", np.sum(self.citizens_i_status == 0))

        for group_id in range(self.risk_group_num) : 
            self.citizens_i_status[self.group_newly_infected_id[group_id]] = 1 
            self.requests[self.group_newly_infected_id[group_id]] = 0 
        
        zero_gamma_mask = self.gammas == 0 
        pos_gamma_mask = self.gammas > 0 
        
        infect_mask = self.citizens_i_status == 1 
        gamma_update_num = np.sum(zero_gamma_mask)

        self.gammas[zero_gamma_mask] = np.random.poisson(self.gamma, gamma_update_num)
        self.gammas[pos_gamma_mask * infect_mask] -= 1 
        
        self.grouping(False, True, True)
        #print(self.current_day, "infection", "I : ", np.sum(self.citizens_i_status == 1), "U : ", self.citizen_num-np.sum(self.citizens_i_status == 1), "R : ", np.sum(self.citizens_i_status == 0))

        # capacity_per_day = int(self.vaccine_num / self.days)
        effective_rate = 0.9
        effective_num = int(np.ceil(self.vaccine_num / self.days * effective_rate))

        # assume the effectiveness is immedately known 
        if np.sum(self.requests) > 0 : 
            # goverment implements policy    
            if self.setting == "full_info" :  
                if np.sum(self.requests) > effective_num :    
                    vaccinated_idx = self.gov.decide(self.risk_requests_by_group, self.risk_susceptible_num_by_group, self.pri_susceptible_id_by_group, \
                                            self.pri_susceptible_num_by_group, np.sum(self.requests), effective_num, self.ids[self.requests == 1], setting)

                    # remove vaccinated request of each group, add them into anti body  
                    # edited by Faye range(5) -> 
                    for group_id in range(len(vaccinated_idx)) : 
                        
                        self.requests[vaccinated_idx[group_id]] = 0 
                        self.citizens_i_status[vaccinated_idx[group_id]] = self.status_to_idx["anti_body"] 

                    self.grouping(False, True, True)

                else : 
                    vaccinated_idx = np.arange(np.sum(self.requests))
            elif self.setting == "streaming" : 
                vaccinated_idx = []
                idx = 0 
                while effective_num > 0 and idx < np.sum(self.requests): 
                    current_request = self.requests[idx]
                    cur_decide = self.gov.decide([current_request], np.sum(self.requests), effective_num, self.setting)
                    if cur_decide : 
                        vaccinated_idx.append(idx)
                        effective_num -= 1 
                    idx += 1 
        
        #print(self.current_day, "vaccine", "I : ", np.sum(self.citizens_i_status == 1), "U : ", self.citizen_num-np.sum(self.citizens_i_status == 1), "R : ", np.sum(self.citizens_i_status == 0))
        self.current_day += 1 


    def start(self) : 
        """ start simulation """
        infected_num = []
        uninfected_num = []
        requests_num = []
        anti_body_num = []
        ill_num = []
        
        group_infected_num = []
        group_uninfected_num = []
        group_requests_num = []
        group_anti_body_num = []
        group_ill_num = []

        # update for the given days 
        for i in range(self.days) : 
            self.update()
            infected_num.append(np.sum(self.citizens_i_status == 1))
            uninfected_num.append(self.citizen_num - np.sum(self.citizens_i_status == 1) - np.sum(self.citizens_i_status == 3))
            requests_num.append(np.sum(self.requests))
            anti_body_num.append(np.sum(self.citizens_i_status == 2))
            ill_num.append(np.sum(self.citizens_i_status == 3))
            
            for group_id in range(self.priority_group_num) : 
                group_infected_num.append(self.pri_infected_num_by_group)
                group_uninfected_num.append(self.citizen_num - np.sum( (self.priority_group == group_id) *\
                                        ( (self.citizens_i_status == 1) | (self.citizens_i_status == 3) ) ) )
                group_requests_num.append(np.sum( (self.priority_group == group_id) * (self.citizens_i_status == 0) ) )
                group_anti_body_num.append(np.sum( (self.priority_group == group_id) * (self.citizens_i_status == 2) ) )
                group_ill_num.append(np.sum( (self.priority_group == group_id) * (self.citizens_i_status == 3) ))

        #print(infected_num, ill_num)
        return infected_num, uninfected_num, requests_num, anti_body_num, ill_num,\
                group_infected_num, group_uninfected_num, group_requests_num, group_anti_body_num, group_ill_num
        
    def create_risk_matrix(self, row) : 
        """ create cross-group contact-risk matrix """
        mat = np.zeros((row, row))
        deltas = [2, 0.8, 0.6, 0.4, 0.3, 0.1]
        for i in range(row) : 
            delta = deltas[i]
            for j in range(i, row) : 
                if i == j : 
                    mat[i, j] = 1 * delta 
                else : 
                    mat[i, j] = (1 - 0.15 * (j - i)) * delta 
            
            #delta -= delta / row
        
        for i in range(row) : 
            for j in range(i) : 
                mat[i, j] = mat[j, i]
        
        return mat 
        
def sampling(target_num, id_to_sample) : 
        if len(id_to_sample) > np.ceil(target_num) : 
            sampled_id = np.random.choice(id_to_sample, int(np.ceil(target_num)), replace=False)
        else : 
            sampled_id = id_to_sample
        
        return sampled_id 

def simulate(sim_num = 1, sim_params=None) : 
    """ implements a whole run of simulation for a period """
    infected_nums = []
    uninfected_nums = []
    requests_nums = []
    anti_body_nums = []
    ill_nums = []

    group_infected_nums = []
    group_uninfected_nums = []
    group_requests_nums = []
    group_anti_body_nums = []
    group_ill_nums = []

    for i in range(sim_num) : 
        env = vaccine_system(**sim_params)
        infected_num, uninfected_num, requests_num, anti_body_num, ill_num,\
                group_infected_num, group_uninfected_num, group_requests_num, group_anti_body_num, group_ill_num = env.start() 

        infected_nums.append(infected_num)
        uninfected_nums.append(uninfected_num)
        requests_nums.append(requests_num)
        anti_body_nums.append(anti_body_num)
        ill_nums.append(ill_num)

        group_infected_nums.append(group_infected_num)
        group_uninfected_nums.append(group_uninfected_num)
        group_requests_nums.append(group_requests_num)
        group_anti_body_nums.append(group_anti_body_num)
        group_ill_nums.append(group_ill_num)

    stats = {}
    stats["infected_num"] = infected_nums
    stats["uninfected_num"] = uninfected_nums
    stats["request_num"] = requests_nums
    stats["anti_body_num"] = anti_body_nums
    stats["ill_num"] = ill_nums 

    group_stats = {}
    group_stats["infected_num"] = group_infected_nums
    group_stats["uninfected_num"] = group_uninfected_nums
    group_stats["request_num"] = group_requests_nums
    group_stats["anti_body_num"] = group_anti_body_nums
    group_stats["ill_num"] = group_ill_nums 


    return stats, group_stats

def collect_stats(data, axis, sample_mean=False) : 
    """ collect means and std """
    num = len(data)
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)

    mean_std = std / (num ** (1/2)) if sample_mean else std 
    #print(data) 
    #print(data, mean, "std : ", std)

    return mean, mean_std 

def main_sim(parameters : dict, sim_num : int, stats_type : str, stats_digit_form : str, popultaion : bool, metric : str, policy_compare : str) :
    """ run all simulations and plotting time-varying graphs 
        
        Arguments : 
        - parameters : make sure that only one of the keys correspond to list values, mainly for params comparison 
        - sim_num : simulation number 
        - stats : stats chosen for plots 
            - population : 
                - "basic" : output a) "infected_num", "uninfected_num", b) "request_num", "anti_body_num", "seriously_ill_num"
                - "vaccine_control " : output metric that measure the performance of vaccine : "infected_num" + "seriously_ill_num"
            - group : 
                - for better visualization, we provide arguments : 
                    - all_group with single metrics 
        - stats_digit_form : either number or percentage 
    """
    params_key = parameters.keys()
    days = parameters["days"] if parameters["days"] else 7 
    ori_citizen_num = parameters["citizen_num"]
    temp_parameters = parameters 

    policies = parameters.pop("policy")

    # select the params values for comparison 
    compare_key_num = 0 
    for param in params_key : 
        if type(parameters[param]) is list and len(parameters[param]) > 1 : 
            temp_param_values = parameters[param]
            temp_param_key = param 
            temp_parameters.pop(param)
            temp_params = temp_parameters
            compare_key_num += 1 
            break 
    
    # if not for comparison, create singleton list 
    if not compare_key_num : 
        temp_params = temp_parameters
        temp_param_key = list(temp_params.keys())[0]
        temp_param_values = [temp_params[temp_param_key]]

    # deploy plot structure 
    unit_width, unit_len = 6, 6
    
    if popultaion : 
        if not metric : 
            if stats_type == "basic" : 
                fig_width, fig_len = 2 * unit_width, len(temp_param_values) * unit_len
                fig, axes = plt.subplots(2, len(temp_param_values), figsize=(fig_len, fig_width))
            else : 
                fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
                fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
        
        else : 
            fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
            fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
    
    else : 
        fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
        fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
    
    priority_period = list(parameters["priority_period"])
    parameters.pop("priority_period")

    # compare param values 
    for idx, param_value in enumerate(temp_param_values) : 
        temp_params[temp_param_key] = param_value
        
        for policy in policies : 
            gov = Government(policy, priority_period, days)
            
            temp_params["gov"] = gov 
            stats, group_stats = simulate(sim_num=sim_num, sim_params=temp_params)

            if stats_digit_form == "number" : 
                ori_citizen_num = 1 
            
            for key, value in list(stats.items()) : 
                stats[key] = np.array(value) / ori_citizen_num
            
            for key, value in list(group_stats.items()) : 
                group_stats[key] = np.array(value) / ori_citizen_num

            x_label = "days" 
            y_label = "percentage" if stats_digit_form == "percent" else "num"
            
            if popultaion : 
                # show all metric, plot structure : 2 * (# of params of compare)
                if not metric : 
                    if stats_type == "basic" : 
                        labels = ["infected_num", "uninfected_num"]
                        datas = [stats[label] for label in labels]
                        
                        ax = axes[0][idx] if compare_key_num else axes[0]
                        plotting(datas, labels, days, ax=ax, title=(temp_param_key, param_value), x_label=x_label, y_label=y_label)

                        labels = ["request_num", "anti_body_num", "ill_num"]
                        datas = [stats[label] for label in labels]

                        ax = axes[1][idx] if compare_key_num else axes[1]

                        plotting(datas, labels, days, ax=ax, title=None, x_label=x_label, y_label=y_label)

                    elif stats_type == "vaccine_control" : 
                        labels = ["unsolved_num"]
                        datas = [stats["infected_num"] + stats["ill_num"]]
                        
                        ax = axes[idx] if compare_key_num else axes
                        sub_title = (temp_param_key, param_value) if compare_key_num else None  
                        plotting(datas, labels, days, ax=ax, title=sub_title, x_label=x_label, y_label=y_label)
                else : 
                    labels = [policy]
                    if metric == "vaccine_control" : 
                        datas = [stats["infected_num"] + stats["ill_num"]]
                        y_label = "unsolved"
                        
                    else : 
                        datas = [stats[metric]]
                        y_label = metric 

                    ax = axes[idx] if compare_key_num else axes
                    plotting(datas, labels, days, ax=ax, title=(temp_param_key, param_value), x_label=x_label, y_label=y_label, group=False)
            
            else : 

                if stats_type == "basic" : 
                    assert metric is not None 
                    datas = group_stats[metric]
                    labels = metric

                    ax = axes[idx] if compare_key_num else axes
                    
                    plotting(datas, labels, days, ax=ax, title=(temp_param_key, param_value), x_label=x_label, y_label=y_label, group=True)
            
    plt.show()
    return stats, group_stats

def plotting(datas, labels, days, ax, title, x_label, y_label, group=False) :  
    epochs = np.arange(1, days+1)

    if not group : 
        for idx, data in enumerate(datas) :    
            mean, std = collect_stats(data, 0)
            ax.plot(epochs, mean, label=labels[idx])
            ax.fill_between(epochs, mean-std, mean+std ,alpha=0.1)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if title : 
                param_key, param_value = title 
                ax.set_title(param_key + " : " + str(param_value))
            
            ax.legend()
    
    if group : 
        mean, std = collect_stats(datas, 0)
        group_num = mean.shape[1]
        for group_id in range(group_num) : 
            group_mean = mean[:, group_id]
            group_std = std[:, group_id]
            ax.plot(epochs, group_mean, label="group {}".format(group_id))
            ax.fill_between(epochs, group_mean-group_std, group_mean+group_std ,alpha=0.3)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if title : 
                param_key, param_value = title 
                ax.set_title(param_key + " : " + str(param_value) + ", metric : " + labels)
            
            ax.legend()

# env = vaccine_system(1000, 100, 7)
# infected_nums, uninfected_nums = env.simulate()

# datas = [infected_nums, uninfected_nums]
# labels = ["infected_num", "uninfected_num"]
#plotting(datas, labels)

# params = {"citizen_num" : 1000, "vaccine_num" : [100, 200, 300], "days" : 7}
# main_sim(params)

policy = "UP"
setting = "full_info"
priority_period = [0, 1/3, 2/3, 3/4]
days = 30
#gov = Government(policy, priority_period, days)
params = {"citizen_num" : 1000, "vaccine_num" : [50, 100, 200], 
            "days" : days, "r_nau" : 1.5, "gamma" : 10, "setting" : setting, 
            "risk_group_num" : 6, "priority_group_num" : 4, "policy" : ["Unif", "SP", "MP", "UP"], "priority_period" : priority_period}
main_sim(params, sim_num=1, stats_type="basic", stats_digit_form="number", popultaion=True, metric="vaccine_control", policy_compare=True)
