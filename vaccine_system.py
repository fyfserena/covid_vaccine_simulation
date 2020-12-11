import numpy as np 
import matplotlib.pyplot as plt 

class Citizen(object) : 
    """ Citizen Agent """
    def __init__(self, info : dict) : 
        self.age = info["age"] 
        self.gender = info["gender"]
        self.occupation = info["occupation"] 
        self.infected = False 
        self.gamma = np.random.poisson(5)

    def get_sick_after_infection(self) : 
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

    def decide(self, requests, citizen_num, vaccines_num, setting) : 
        """ governemnt decides if should give vaccine injection 
            setting : 
                - "full_info" : government has full information of the lining requests
                - "streaming" : requests comes in order, government needs to make decision for coming requests and process the next request
        
        """

        # add current request to processed requests 
        self.processed_requests.extend(requests)

        if setting == "full_info" : 
            if self.policy == "unif" : 
                if type(requests) == list :     
                    vaccinated_idx = np.random.choice(np.arange(len(requests)), vaccines_num, replace=False)
                    
                    return vaccinated_idx
        
        elif setting == "streaming" : 
            if self.policy == "unif" : 
                vaccine_prob = vaccines_num / citizen_num 
                cur_decide = np.random.rand() < vaccine_prob
                return cur_decide
            
            

class vaccine_system(object) : 
    def __init__(self, gov, citizen_num, vaccine_num, days, r_nau, setting="full_info") :
        self.citizen_num = citizen_num
        self.all_citizens = self.gen_citizen(self.citizen_num)
        self.citizens = self.all_citizens[0]
        self.infected_citizens = self.all_citizens[1]
        self.uninfected_citizens = self.all_citizens[2]
        self.requests = list(self.uninfected_citizens)
        np.random.shuffle(self.requests)
        self.vaccine_num = vaccine_num
        self.vaccinated = []
        self.days = days 
        self.r_nau = r_nau 
        self.gov = gov 
        self.gammas = np.random.poisson(5, len(self.infected_citizens))
        self.setting = setting
        self.ills = []

    def gen_info(self) :
        """ generate information for virtual citizens """
        age = np.random.randint(10)
        gender = np.random.randint(2)
        occupation = np.random.randint(5)
        infected = np.random.rand(1) < 0.06 

        return {"age" : age, "gender" : gender, "occupation" : occupation, "infected" : infected}

    def gen_citizen(self, citizen_num) : 
        """ generate virtual citizens """

        infected_citizens = []
        uninfected_citizens = []
        citizens = []

        for i in range(citizen_num) : 
            info = self.gen_info()
            citizen_ = Citizen(info)
            
            if info["infected"] : 
                infected_citizens.append(citizen_)
            else : 
                uninfected_citizens.append(citizen_)
            
            citizens.append(citizen_)

        return citizens, infected_citizens, uninfected_citizens

    def recovery(self, re, ill_rate=0.1, method="resolve") : 
        """ simulate the recovery """

        # get the recovery index for infected people getting resolved 
        if method == "ratio" :  
            recovery_num = int(len(self.infected_citizens) * re) 
            new_recovery_idx = list(np.random.choice(np.arange(len(self.infected_citizens)), recovery_num, replace=False))
        elif method == "resolve" :
            recovery_num = 0 
            new_recovery_idx = []
            for idx, citizen_ in enumerate(self.infected_citizens) : 
                if citizen_.gamma == 0 : 
                    recovery_num += 1 
                    new_recovery_idx.append(idx)
                    citizen_.gamma = np.random.poisson(5)

        self.infected_citizens = np.array(self.infected_citizens)
        
        # get the ill index from the new recovered 
        if len(new_recovery_idx) > 0 : 
            ill_num = int(ill_rate * len(new_recovery_idx))
            ill_idx = list(np.random.choice(new_recovery_idx, ill_num, replace=False))
        else : 
            ill_idx = []

        # filter out the new recovered 
        mask = np.ones(len(self.infected_citizens), dtype=bool)
        mask[new_recovery_idx] = False
        new_recovery_citizen = self.infected_citizens[~mask]
        
        # filter out the illed 
        mask[new_recovery_idx] = True 
        mask[ill_idx] = False 
        ill_new_recovery = self.infected_citizens[~mask]

        # filter out the healthy new recovered 
        mask[new_recovery_idx] = False 
        mask[ill_idx] = True 
        healthy_new_recovery = self.infected_citizens[~mask]

        # filter out the left infected 
        mask[ill_idx] = False 
        self.infected_citizens = self.infected_citizens[mask] 
        self.infected_citizens = list(self.infected_citizens)

        # requests and unifected add back the healty recovered 
        self.requests.extend(healthy_new_recovery)
        self.uninfected_citizens.extend(new_recovery_citizen)

        # record ill people 
        self.ills.extend(ill_new_recovery)
        #print(len(self.ills), len(new_recovery_idx)) 

    def infection_spread(self, r_nau, infectious_period=5, random_mode=True) :
        """ simulate the infection """
        discount_factor = 0 
        while not discount_factor : 
            discount_factor = np.random.poisson(10)

        new_infected_num = 0 
        if random_mode : 
            for citizen_ in self.infected_citizens : 
                if citizen_.gamma > 0 : 
                    new_infected_num += r_nau / infectious_period
                    citizen_.gamma -= 1  

            # old_gammas = self.gammas[self.gammas > 0]  
            # new_infected_num = np.sum(old_gammas) * (r_nau / infectious_period)

            # old_gammas -= 1 
            # new_gammas_len = len(self.gammas[self.gammas == 0])
            # new_gammas = np.random.poisson(infectious_period, new_gammas_len)
            # self.gammas = np.append(old_gammas, new_gammas)

        else : 
            new_infected_num = len(self.infected_citizens) * r_nau / infectious_period 

        new_infected_num = int(new_infected_num) 
        available_num_for_infect = len(self.requests)
        
        if available_num_for_infect >= new_infected_num :
            #print(new_infected_num)
            new_infected_citizen = list(np.random.choice(self.requests, new_infected_num, replace=False))
        else : 
            new_infected_citizen = list(np.random.choice(self.requests, available_num_for_infect, replace=False))
        
        self.infected_citizens.extend(new_infected_citizen)

        temp_uninfected_citizen = [c for c in self.uninfected_citizens if c not in new_infected_citizen]
        self.uninfected_citizens = temp_uninfected_citizen

        temp_requests = [c for c in self.requests if c not in new_infected_citizen]
        self.requests = temp_requests


    def update(self) : 
        """" implements recovery -> infection -> vaccine injection """
        # recover 
        self.recovery(0.1)
        #print("recovery", "I : ", len(self.infected_citizens), "U : ", len(self.uninfected_citizens), "R : ", len(self.requests))

        # infection spread 
        noise = np.random.normal()
        noise = 1 if noise < -self.r_nau else noise  
        noisy_r_nau = self.r_nau + noise 
        self.infection_spread(noisy_r_nau, 5)
        #print("infection", "I : ", len(self.infected_citizens), "U : ", len(self.uninfected_citizens), "R : ", len(self.requests))

        capacity_per_day = int(self.vaccine_num / self.days)
        effective_rate = 0.9
        effective_num = int(capacity_per_day * effective_rate)

        # assume the effectiveness is immedately known 
        if len(self.requests) > 0 : 
            # goverment implements policy 
            if self.setting == "full_info" : 
                if len(self.requests) > effective_num :         
                    vaccinated_idx = self.gov.decide(self.requests, len(self.requests), effective_num, self.setting)
                else : 
                    vaccinated_idx = np.arange(len(self.requests))
            elif self.setting == "streaming" : 
                vaccinated_idx = []
                idx = 0 
                while effective_num > 0 and idx < len(self.requests): 
                    current_request = self.requests[idx]
                    cur_decide = self.gov.decide([current_request], len(self.requests), effective_num, self.setting)
                    if cur_decide : 
                        vaccinated_idx.append(idx)
                        effective_num -= 1 
                    idx += 1 

            # vaccinated_num = len(vaccinated_idx)
            mask = np.ones(len(self.requests), dtype=bool)
            mask[vaccinated_idx] = False 
            
            new_vaccinated = np.array(self.requests)[~mask]
            self.requests = np.array(self.requests)[mask] 
            self.vaccinated.extend(new_vaccinated) 
            self.requests = list(self.requests)
        
        #print("vaccine", "I : ", len(self.infected_citizens), "U : ", len(self.uninfected_citizens), "R : ", len(self.requests))


    def start(self) : 
        """ start simulation """
        infected_num = []
        uninfected_num = []
        requests_num = []
        anti_body_num = []
        ill_num = []

        # update for the given days 
        for i in range(self.days) : 
            self.update()
            infected_num.append(len(self.infected_citizens))
            uninfected_num.append(len(self.uninfected_citizens))
            requests_num.append(len(self.requests))
            anti_body_num.append(len(self.uninfected_citizens) - len(self.requests))
            ill_num.append(len(self.ills))
        
        return infected_num, uninfected_num, requests_num, anti_body_num, ill_num

    def reset(self) : 
        """ reset all parameters for new simulation """
        self.all_citizens = self.gen_citizen(self.citizen_num)
        self.citizens = self.all_citizens[0]
        self.infected_citizens = self.all_citizens[1]
        self.uninfected_citizens = self.all_citizens[2]
        self.requests = list(self.uninfected_citizens)
        np.random.shuffle(self.requests)
        self.vaccinated = []
        self.ills = []
        self.gov.reset() 
        
    def simulate(self, sim_num = 1) : 
        """ implements a whole run of simulation for a period """
        infected_nums = []
        uninfected_nums = []
        requests_nums = []
        anti_body_nums = []
        ill_nums = []

        for i in range(sim_num) : 
            infected_num, uninfected_num, requests_num, anti_body_num, ill_num = self.start() 
            infected_nums.append(infected_num)
            uninfected_nums.append(uninfected_num)
            requests_nums.append(requests_num)
            anti_body_nums.append(anti_body_num)
            ill_nums.append(ill_num)
            self.reset() 

        return infected_nums, uninfected_nums, requests_nums, anti_body_nums, ill_nums 

def collect_stats(data, axis, sample_mean=False) : 
    """ collect means and std """
    num = len(data)
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)

    mean_std = std / (num ** (1/2)) if sample_mean else std 
    #print(data, mean, std)

    return mean, mean_std 

def main_sim(parameters : dict, sim_num : int, stats : str, stats_form : str) :
    """ run all simulations and plotting time-varying graphs 
        
        Arguments : 
        - parameters : make sure that only one of the keys correspond to list values, mainly for params comparison 
        - sim_num : simulation number 
        - stats : stats chosen for plots 
            - "basic" : output a) "infected_num", "uninfected_num", b) "request_num", "anti_body_num", "seriously_ill_num"
            - "vaccine_control " : output metric that measure the performance of vaccine : "infected_num" + "seriously_ill_num"
        - stats_form : either number or percentage 
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
    
    if stats == "basic" : 
        fig_width, fig_len = 2 * unit_width, len(temp_param_values) * unit_len
        fig, axes = plt.subplots(2, len(temp_param_values), figsize=(fig_len, fig_width))
    else : 
        fig_width, fig_len = 1 * unit_width, len(temp_param_values) * unit_len
        fig, axes = plt.subplots(1, len(temp_param_values), figsize=(fig_len, fig_width))
    
    # compare param values 
    for idx, param_value in enumerate(temp_param_values) : 
        temp_params[temp_param_key] = param_value
        env = vaccine_system(**temp_params)
        infected_nums, uninfected_nums, requests_nums, anti_body_nums, ill_nums = env.simulate(sim_num=100)

        if stats_form == "percent" :  
            infected_nums, uninfected_nums, requests_nums, anti_body_nums, ill_nums =\
                list(map(lambda x : np.array(x) / ori_citizen_num, [infected_nums, uninfected_nums, requests_nums, anti_body_nums, ill_nums]))

        x_label = "days" 
        y_label = "percentage" if stats_form == "percent" else "num"
        
        if stats == "basic" : 
            datas = [infected_nums, uninfected_nums]
            labels = ["infected_num", "uninfected_num"]

            ax = axes[0][idx] if compare_key_num else axes[0]
            plotting(datas, labels, days, ax=ax, title=(temp_param_key, param_value), x_label=x_label, y_label=y_label)

            datas = [requests_nums, anti_body_nums, ill_nums]
            labels = ["request_num", "anti_body_num", "seriously_ill_num"]

            ax = axes[1][idx] if compare_key_num else axes[1]
            sub_title = (temp_param_key, param_value) if compare_key_num else None  
            plotting(datas, labels, days, ax=ax, title=sub_title, x_label=x_label, y_label=y_label)
        elif stats == "vaccine_control" : 
            datas = [np.array(infected_nums) + np.array(ill_nums)]
            labels = ["unsolved_num"]

            ax = axes[idx] if compare_key_num else axes
            sub_title = (temp_param_key, param_value) if compare_key_num else None  
            plotting(datas, labels, days, ax=ax, title=sub_title, x_label=x_label, y_label=y_label)
            
    plt.show()

def plotting(datas, labels, days, ax, title, x_label, y_label) :  
    epochs = np.arange(1, days+1)

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

# env = vaccine_system(1000, 100, 7)
# infected_nums, uninfected_nums = env.simulate()

# datas = [infected_nums, uninfected_nums]
# labels = ["infected_num", "uninfected_num"]
#plotting(datas, labels)

# params = {"citizen_num" : 1000, "vaccine_num" : [100, 200, 300], "days" : 7}
# main_sim(params)

policy = "unif"
setting = ["full_info", "streaming"]
params = {"citizen_num" : 1000, "vaccine_num" : 500, "days" : 14, "r_nau" : 1.5, "gov" : Government(policy), "setting" : setting}
main_sim(params, sim_num=100, stats="vaccine_control", stats_form="percent")

#plotting(uninfected_nums, "uninfected_num")