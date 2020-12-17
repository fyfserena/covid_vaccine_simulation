import numpy as np 
import matplotlib.pyplot as plt 

class Citizen(object) : 
    """ Citizen Agent """
    def __init__(self, info : dict) : 
        self.age = info["age"] 
        self.gender = info["gender"]
        self.occupation = info["occupation"] 
        self.infected = info["infected"] 
        self.gamma = info["gamma"]
        # self.ethnicity = info["ethnicity"] 
        # self.weight = info["weight"]
        # self.height = info["height"]
        # self.home_cat = info["home_cat"]
        # self.diabetes_type = info["diabetes_type"]
        # self.if_heart_disease = info["if_heart_disease"]
        # self.if_ckd = info["if_ckd"]
        # self.if_cancer = info["if_cancer"]

    def get_sick_after_infection(self) : 
        # bmi = self.weight/((self.height/100)**2)
        # if self.diabetes_type == 0: 
        #     b_type1, b_type2 = 0, 0
        # elif self.diabetes_type == 1:
        #     b_type1, b_type2 = 1, 0
        # else:
        #     b_type1, b_type2 = 0, 1

        # def death_female(age, bmi, ethrisk, homecat, b_type1, b_type2, b_stroke, b_ckd, b_cancer):
        #   Iethrisk = [0.1753,0.2157,0.3923,0.3772,0.1667,0.3130,0.2305]
        #   Ihomecat = [0,1.2843,0.3902]
        #   dage=age/10
        #   age_2 = pow(dage,3)*np.log(dage)
        #   age_1 = pow(dage,3)
        #   dbmi=bmi/10
        #   bmi_1 = pow(dbmi,.5)
        #   bmi_2 = pow(dbmi,.5)*np.log(dbmi)

        #   age_1 = age_1 - 115.5998
        #   age_2 = age_2 - 183.0383
        #   bmi_1 = bmi_1 - 1.6324
        #   bmi_2 = bmi_2 - 1.6001

        #   a=Iethrisk[ethrisk]+Ihomecat[homecat]+age_1 * 0.0535+age_2 * -0.0201 
        #   + bmi_1 * -19.7436+bmi_2 * 6.6649+b_stroke * (0.2915+0.2099)
        #   +b_type1 * 1.3918+ b_type2 * 1.8389+b_ckd * 1.5+b_cancer * 1.5
        #   +age_1 * b_type2 * -0.02006+age_2 * b_type2 * 0.0075

        #   return 100.0 * (1 - pow(0.999977290630341, np.exp(a)))

        # def death_male(age, bmi, ethrisk, homecat, b_type1, b_type2, b_stroke, b_ckd, b_cancer):
        #   Iethrisk = [0.4953,0.5357,0.7223,0.6972,0.4867,0.6330,0.5505]
        #   Ihomecat = [0,1.4545,0.4426]

        #   dage=age/10
        #   age_1 = dage
        #   age_2 = pow(dage,3)
        #   dbmi=bmi/10
        #   bmi_2 = pow(dbmi,-.5)*np.log(dbmi)
        #   bmi_1 = pow(dbmi,-.5)
    
        #   age_1 = age_1 - 4.7707
        #   age_2 = age_2 - 108.57944
        #   bmi_1 = bmi_1 - 0.61367
        #   bmi_2 = bmi_2 - 0.59929

        #   a = Iethrisk[ethrisk]+ Ihomecat[homecat]+age_1 * 1.45475+age_2 * -0.00282
        #   +bmi_1 * -22.0609+bmi_2 * -20.3035+b_stroke * (0.2126+0.25167)+ b_type1 * 1.7655
        #   + b_type2 * 1.5551+b_ckd * 1.5+b_cancer * 1.5+age_1 * b_type2 * -0.5325+age_2 * b_type2 * 0.00134
    
        #   return 100.0 * (1 - pow(0.999977290630341, np.exp(a)))

        # if not self.gender: return death_female(self.age, bmi, self.ethnicity, self.home_cat, b_type1, b_type2, self.if_heart_disease, self.if_ckd, self.if_cancer)
        # else: return death_male(self.age, bmi, self.ethnicity, self.home_cat, b_type1, b_type2, self.if_heart_disease, self.if_ckd, self.if_cancer)
        pass 

    def infect_after_infection(self) : 
        pass  

class Government(object) : 
    """ Government Agent """
    def __init__(self, policy : str) : 
        self.policy = policy 
        self.processed_requests = []    

    def reset(self) : 
        self.processed_requests = []    

    def decide(self, requests_by_group, susceptible_num_by_group, citizen_num, vaccines_num, setting) : 
        """ governemnt decides if should give vaccine injection 
            setting : 
                - "full_info" : government has full information of the lining requests
                - "streaming" : requests comes in order, government needs to make decision for coming requests and process the next request
        
        """

        # add current request to processed requests 
        #self.processed_requests.extend(requests)

        if setting == "full_info" : 
            if self.policy == "unif" :  
                vaccinated_ids = []
                for group_id in range(5) : 
                    group_vaccine = int(np.floor(vaccines_num * susceptible_num_by_group[group_id] / citizen_num))
                    vaccinated_ids.append(sampling(group_vaccine, requests_by_group[group_id]))
                
                return vaccinated_ids 
        
        elif setting == "streaming" : 
            if self.policy == "unif" : 
                vaccine_prob = vaccines_num / citizen_num 
                cur_decide = np.random.rand() < vaccine_prob
                return cur_decide
            
            

class vaccine_system(object) : 
    def __init__(self, gov, citizen_num, vaccine_num, days, r_nau, gamma, setting="full_info") :
        self.citizen_num = citizen_num
        self.gamma = gamma 
        self.citizen_stats = []
        self.gen_citizen(self.citizen_num)

        self.ids = np.array(self.citizen_stats[:, self.col_indx["id"]], dtype=int)
        
        # generate original infected number 
        ori_infected_num = int(np.ceil(self.citizen_num * 0.006))
        ori_infected_ids = np.random.choice(self.ids, ori_infected_num, replace=False)
        
        self.citizens_i_status = self.citizen_stats[:, self.col_indx["infected"]]
        self.citizens_i_status[ori_infected_ids] = 1 
        self.citizen_stats[:, self.col_indx["request_vaccine"]][self.citizen_stats[:, self.col_indx["infected"]] == 0] = 1 
        
        self.gammas = self.citizen_stats[:, self.col_indx["gamma"]]
        self.requests = self.citizen_stats[:, self.col_indx["request_vaccine"]]
        
        self.group = self.citizen_stats[:, self.col_indx["group"]]
        self.risk_matrix = self.create_risk_matrix(5)
        
        self.status_to_idx = {"susceptible" : 0, "infected" : 1, "anti_body" : 2, "ill" : 3} 
        self.vaccine_num = vaccine_num
        self.vaccinated = []
        self.days = days 
        self.r_nau = r_nau 
        self.gov = gov 
        self.setting = setting
        self.gov.reset() 

        self.grouping()
        self.setting = setting
        self.current_day = 1
        
    
    def reset(self, gov, citizen_num, vaccine_num, days, r_nau, setting="full_info") : 
        """ reset all parameters for new simulation """
        pass 

    def grouping(self, to_group_label=False) : 
        if not to_group_label : 
            self.id_by_group = {group_id : self.ids[self.group == group_id] for group_id in range(5)}

        self.infected_num_by_group = np.array([np.sum((self.citizens_i_status == 1) * (self.group == group_id)) for group_id in range(5)])
        self.infected_id_by_group = {group_id : self.ids[(self.group == group_id) * (self.citizens_i_status == 1)] for group_id in range(5)}
        
        self.susceptible_num_by_group = np.array([np.sum((self.citizens_i_status == 0) * (self.group == group_id)) for group_id in range(5)])
        self.susceptible_id_by_group = {group_id : self.ids[(self.group == group_id) * (self.citizens_i_status == 0)] for group_id in range(5)}

        self.requests_by_group = {group_id : self.ids[(self.group == group_id) * (self.requests == 1)] for group_id in range(5)}
        self.anti_body_by_group = {group_id : self.ids[(self.group == group_id) * (self.citizens_i_status == self.status_to_idx["anti_body"])] for group_id in range(5)}

    def gen_info(self) :
        """ generate information for virtual citizens """
        age = np.random.randint(10)
        gender = np.random.randint(2)
        occupation = np.random.randint(5)
        infected = False 
        gamma = np.random.poisson(self.gamma)

        return {"age" : age, "gender" : gender, "occupation" : occupation, 
                "infected" : infected, "gamma" : gamma, "request_vaccine" : False, 
                "ill" : False, "group" : occupation} 

    def gen_citizen(self, citizen_num) : 
        """ generate virtual citizens """
        for i in range(citizen_num) : 
            info = self.gen_info()
            info["id"] = i

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
            new_resolve_idx = self.ids[gamma_mask * infect_mask]
        
        # compute the ill id from new_
        if len(new_resolve_idx) > 0 : 
            ill_num = int(np.ceil(ill_rate * len(new_resolve_idx)))
            ill_idx = list(np.random.choice(range(len(new_resolve_idx)), ill_num, replace=False))
        else : 
            ill_idx = []

        # filter out the healthy id from new resolved case, update ills 
        mask = np.ones(len(new_resolve_idx), dtype = bool)

        mask[ill_idx] = False
        self.ill_idx = new_resolve_idx[~mask]

        healthy_new_resolve_idx = new_resolve_idx[mask]
        self.healthy_new_resolve_idx = healthy_new_resolve_idx

    def infection_spread(self, r_nau, infectious_period=5, random_mode=True) :
        """ simulate the infection """
        if random_mode : 
            # compute infect parameter for each group 
            group_infect_param = np.matmul(self.risk_matrix, self.infected_num_by_group) / self.citizen_num

            # compute infect num from each group's susceptible 
            group_newly_infected_num = self.susceptible_num_by_group * group_infect_param
            
            # sample infected id from each group 
            group_newly_infected_id = [sampling(group_newly_infected_num[group_id], self.susceptible_id_by_group[group_id]) for group_id in range(5)]
            self.group_newly_infected_id = group_newly_infected_id

        else : 
            new_infected_num = np.sum(self.citizens_i_status) * r_nau / infectious_period 


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

        for group_id in range(5) : 
            self.citizens_i_status[self.group_newly_infected_id[group_id]] = 1 
            self.requests[self.group_newly_infected_id[group_id]] = 0 
        
        zero_gamma_mask = self.gammas == 0 
        pos_gamma_mask = self.gammas > 0 
        
        infect_mask = self.citizens_i_status == 1 
        gamma_update_num = np.sum(zero_gamma_mask)

        self.gammas[zero_gamma_mask] = np.random.poisson(self.gamma, gamma_update_num)
        self.gammas[pos_gamma_mask * infect_mask] -= 1 
        
        self.grouping(to_group_label=True)
        #print(self.current_day, "infection", "I : ", np.sum(self.citizens_i_status == 1), "U : ", self.citizen_num-np.sum(self.citizens_i_status == 1), "R : ", np.sum(self.citizens_i_status == 0))

        # capacity_per_day = int(self.vaccine_num / self.days)
        effective_rate = 0.9
        effective_num = int(self.vaccine_num / self.days * effective_rate)

        # assume the effectiveness is immedately known 
        if np.sum(self.requests) > 0 : 
            # goverment implements policy    
            if self.setting == "full_info" :  
                if np.sum(self.requests) > effective_num :    
                    vaccinated_idx = self.gov.decide(self.requests_by_group, self.susceptible_num_by_group, np.sum(self.requests), effective_num, self.setting)

                    # remove vaccinated request of each group, add them into anti body  
                    for group_id in range(5) : 
                        self.requests[vaccinated_idx[group_id]] = 0 
                        self.citizens_i_status[vaccinated_idx[group_id]] = self.status_to_idx["anti_body"] 

                    self.grouping(to_group_label=True)

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
            
            group_infected_num.append(self.infected_num_by_group)
            group_uninfected_num.append([self.citizen_num - np.sum( (self.group == group_id) *\
                                        ( (self.citizens_i_status == 1) | (self.citizens_i_status == 3) ) ) for group_id in range(5)])
            group_requests_num.append([np.sum( (self.group == group_id) * (self.citizens_i_status == 0) ) for group_id in range(5)])
            group_anti_body_num.append([np.sum( (self.group == group_id) * (self.citizens_i_status == 2) ) for group_id in range(5)])
            group_ill_num.append([np.sum( (self.group == group_id) * (self.citizens_i_status == 3) ) for group_id in range(5)])

        return infected_num, uninfected_num, requests_num, anti_body_num, ill_num,\
                group_infected_num, group_uninfected_num, group_requests_num, group_anti_body_num, group_ill_num
        
    def create_risk_matrix(self, row) : 
        """ create cross-group contact-risk matrix """
        mat = np.zeros((row, row))
        delta = 0.4
        for i in range(row) : 
            for j in range(i, row) : 
                if i == j : 
                    mat[i, j] = 1 * delta 
                else : 
                    mat[i, j] = (1 - 0.25 * (j - i)) * delta 
            
            delta -= delta / row
        
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
    #print(data, mean, std)

    return mean, mean_std 

def main_sim(parameters : dict, sim_num : int, stats_type : str, stats_digit_form : str, popultaion : bool, metric : str) :
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
        if stats_type == "basic" : 
            fig_width, fig_len = 2 * unit_width, len(temp_param_values) * unit_len
            fig, axes = plt.subplots(2, len(temp_param_values), figsize=(fig_len, fig_width))
        else : 
            fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
            fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
    
    else : 
        fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
        fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
    
    # compare param values 
    for idx, param_value in enumerate(temp_param_values) : 
        temp_params[temp_param_key] = param_value
        stats, group_stats = simulate(sim_num=sim_num, sim_params=temp_params)

        if stats_digit_form == "percent" :  
            for key, value in list(stats.items()) : 
                stats[key] = np.array(value) / ori_citizen_num
            
            for key, value in list(group_stats.items()) : 
                group_stats[key] = np.array(value) / ori_citizen_num

        x_label = "days" 
        y_label = "percentage" if stats_digit_form == "percent" else "num"
        
        if popultaion : 

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

            if stats_type == "basic" : 
                assert metric is not None 
                datas = group_stats[metric]
                labels = metric

                ax = axes[idx] if compare_key_num else axes
                
                plotting(datas, labels, days, ax=ax, title=(temp_param_key, param_value), x_label=x_label, y_label=y_label, group=True)
            
    plt.show()

def plotting(datas, labels, days, ax, title, x_label, y_label, group=False) :  
    epochs = np.arange(1, days+1)

    if not group : 
        for idx, data in enumerate(datas) :    
            mean, std = collect_stats(data, 0)
            ax.plot(epochs, mean, label=labels[idx])
            ax.fill_between(epochs, mean-std, mean+std ,alpha=0.3)
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

policy = "unif"
setting = "full_info"
params = {"citizen_num" : 1000, "vaccine_num" : [2, 5, 10], "days" : 100, "r_nau" : 1.5, "gov" : Government(policy), "gamma" : 20, "setting" : setting}
main_sim(params, sim_num=10, stats_type="basic", stats_digit_form="number", popultaion=True, metric="infected_num")

#plotting(uninfected_nums, "uninfected_num")