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
        self.citizens = []    

    def reset(self) : 
        self.citizens = []

    def decide(self, requests, vaccines_num) : 
        self.citizens.extend(requests)

        if self.policy == "unif" : 
            if type(requests) == list :     
                vaccinated_idx = np.random.choice(np.arange(len(requests)), vaccines_num, replace=False)
                
                return vaccinated_idx

class vaccine_system(object) : 
    def __init__(self, gov, citizen_num, vaccine_num, days, r_nau) :
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
            if len(self.requests) > effective_num : 
                # goverment implements policy 
                vaccinated_idx = self.gov.decide(self.requests, effective_num)
            else : 
                vaccinated_idx = np.arange(len(self.requests))

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

        for i in range(self.days) : 
            self.update()
            infected_num.append(len(self.infected_citizens))
            uninfected_num.append(len(self.uninfected_citizens))
            requests_num.append(len(self.requests))
            anti_body_num.append(len(self.uninfected_citizens) - len(self.requests))
            ill_num.append(len(self.ills))
        
        return infected_num, uninfected_num, requests_num, anti_body_num, ill_num

    def reset(self) : 
        """ reset all parameters """
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

def collect_stats(data, axis) : 
    """ collect means and std """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)

    #print(data, mean, std)

    return mean, std 

def main_sim(parameters : dict) :
    """ run all simulations and plotting time-varying graphs 
        parameters : make sure that one and only one of the keys correspond to list values, mainly for params comparison 
    """
    params_key = parameters.keys()
    days = parameters["days"] if parameters["days"] else 7 
    temp_parameters = parameters 

    # select the params values for comparison 
    for param in params_key : 
        if type(parameters[param]) is list and len(parameters[param]) > 1 : 
            temp_param_values = parameters[param]
            temp_param_key = param 
            temp_parameters.pop(param)
            temp_params = temp_parameters
            break 

    fig, axes = plt.subplots(2, len(temp_param_values), figsize=(12, 8))
    
    # compare param values 
    for idx, param_value in enumerate(temp_param_values) : 
        temp_params[temp_param_key] = param_value
        env = vaccine_system(**temp_params)
        infected_nums, uninfected_nums, requests_nums, anti_body_nums, ill_nums = env.simulate(sim_num=100)

        datas = [infected_nums, uninfected_nums]
        labels = ["infected_num", "uninfected_num"]
        plotting(datas, labels, days, ax=axes[0][idx], title=(temp_param_key, param_value))

        datas = [requests_nums, anti_body_nums, ill_nums]
        labels = ["request_num", "anti_body_num", "seriously_ill_num"]
        plotting(datas, labels, days, ax=axes[1][idx], title=(temp_param_key, param_value))

    plt.show()

def plotting(datas, labels, days, ax, title) :  
    epochs = np.arange(1, days+1)

    for idx, data in enumerate(datas) :    
        mean, std = collect_stats(data, 0)
        ax.plot(epochs, mean, label=labels[idx])
        ax.fill_between(epochs, mean-std, mean+std ,alpha=0.3)
        ax.set_xlabel("days")
        ax.set_ylabel("numbers")
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
params = {"citizen_num" : 1000, "vaccine_num" : 500, "days" : 7, "r_nau" : [1.5, 2.4, 2.8], "gov" : Government(policy)}
main_sim(params)

#plotting(uninfected_nums, "uninfected_num")