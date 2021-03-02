import numpy as np 

class Citizen(object) : 
    """ Citizen Agent """
    def __init__(self, info : dict) : 
        pass 

    # @staticmethod
    def get_sick_after_infection(age, gender, ethnicity, weight, height, home_cat, diabetes_type, if_cancer) :  
        bmi = 703 * weight/((height/100)**2)
        if diabetes_type == 0: 
            b_type1, b_type2 = 0, 0
        elif diabetes_type == 1:
            b_type1, b_type2 = 1, 0
        else:
            b_type1, b_type2 = 0, 1
            
        def death_female(age, bmi, ethrisk, homecat, b_type1, b_type2, b_cancer):
            Iethrisk = [0.1753,0.2157,0.3923,0.3772,0.1667,0.3130,0.2305]
            Ihomecat = [0,1.2843,0.3902]
            dage=age/10
            age_2 = pow(dage,3)*np.log(dage)
            age_1 = pow(dage,3)
            dbmi=bmi/10
            bmi_1 = pow(dbmi,.5)
            bmi_2 = pow(dbmi,.5)*np.log(dbmi)

            age_1 = age_1 - 115.5998
            age_2 = age_2 - 183.0383
            bmi_1 = bmi_1 - 1.6324
            bmi_2 = bmi_2 - 1.6001

            a=Iethrisk[ethrisk]+Ihomecat[homecat]+age_1 * 0.0535+age_2 * -0.0201 
            + bmi_1 * -19.7436+bmi_2 * 6.6649 + b_type1 * 1.3918 + b_type2 * 1.8389
            +b_cancer * 1.5 + age_1 * b_type2 * -0.02006 + age_2 * b_type2 * 0.0075

            risk = (1 - pow(0.999977290630341, np.exp(a)))
            
            return risk 

        def death_male(age, bmi, ethrisk, homecat, b_type1, b_type2, b_cancer):
            Iethrisk = [0.4953,0.5357,0.7223,0.6972,0.4867,0.6330,0.5505]
            Ihomecat = [0,1.4545,0.4426]

            dage=age/10
            age_1 = dage
            age_2 = pow(dage,3)
            dbmi=bmi/10
            bmi_2 = pow(dbmi,-.5)*np.log(dbmi)
            bmi_1 = pow(dbmi,-.5)
      
            age_1 = age_1 - 4.7707
            age_2 = age_2 - 108.57944
            bmi_1 = bmi_1 - 0.61367
            bmi_2 = bmi_2 - 0.59929

            a = Iethrisk[ethrisk]+ Ihomecat[homecat]+age_1 * 1.45475+age_2 * -0.00282
            +bmi_1 * -22.0609+bmi_2 * -20.3035 + b_type1 * 1.7655+ b_type2 * 1.5551
            + b_cancer * 1.5 + age_1 * b_type2 * -0.5325 + age_2 * b_type2 * 0.00134

            risk = (1 - pow(0.999977290630341, np.exp(a)))
            
            return risk 

        if not gender: return death_female(age, bmi, ethnicity, home_cat, b_type1, b_type2, if_cancer)
        else: return death_male(age, bmi, ethnicity, home_cat, b_type1, b_type2, if_cancer)

    def infect_after_infection(self) : 
        pass  