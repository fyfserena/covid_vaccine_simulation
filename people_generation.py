import numpy as np 
import pandas as pd 

ESSENTIAL = ['Agriculture, Forestry, Fishing and Hunting','Mining','Utilities',
'Transportation and Warehousing','Health Care and Social Assistance', 'Educational Services', 'Total, All Government']
YOUTH = 18 
SENIOR = 55

lst = [['Under 5 years',266631,254624],['5 to 9 years',229618,217270],['10 to 14 years',245178,236052],['15 to 19 years',222707,217427],
        ['20 to 24 years',143722,269500],['25 to 29 years',366875,389344],['30 to 34 years',360411,366919],
        ['35 to 39 years',297649,300039],['40 to 44 years',255745,284887],['45 to 49 years',246703,267670],
        ['50 to 54 years',244377,268550],['55 to 59 years',246442,271675],['60 to 64 years',218664,261958],
        ['65 to 69 years',173352,218078],['70 to 74 years',142924,185532],['75 to 79 years',91804,135936],
        ['80 and older',121302,212266]]

df = pd.DataFrame(lst, columns =['Groups', 'Men', 'Women'])
age_M= df[["Men"]]/(np.sum(df['Men'].values))
age_F= df[["Women"]]/(np.sum(df['Women'].values))
age_m = np.array(age_M.Men.values)
cdf_m = np.cumsum(age_m)
age_f = np.array(age_F.Women.values)
cdf_f = np.cumsum(age_f)
outcome = np.array(df["Groups"].values)
a = pd.DataFrame([[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60],[60,55],[65,70],[70,75],[75,80],[80,100]])
outcomes = pd.merge(pd.DataFrame(outcome),a,on = a.index)
outcomes = outcomes.rename(columns={"0_x":'Groups','0_y':'Min',1:"Max"})
outcomes = outcomes[["Groups","Min","Max"]]

cancer_m = [74.6,40.0,47.6,71.6,122.6,193.6,273.2,311.2,461.6,766.4,1438.0,2214.4,2881.4,3106.6,2567.4,2095.2,2833.4]
cancer_m = pd.merge(pd.DataFrame(cancer_m),pd.DataFrame(df["Men"]),on=pd.DataFrame(df["Men"]).index)
cancer_m["Prob"] = cancer_m[0]/cancer_m["Men"]
c_m = np.array(cancer_m["Prob"].values)
cancer_f = [62.0,32.8,38.2,61.6,152.2,297.8,448.0,630.8,957.8,1345.8,1853.6,2222.0,2522.6,2644.0,2330.0,2005.0,3428]
cancer_f = pd.merge(pd.DataFrame(cancer_f),pd.DataFrame(df["Women"]),on=pd.DataFrame(df["Women"]).index)
cancer_f["Prob"] = cancer_f[0]/cancer_f["Women"]
c_f = np.array(cancer_f["Prob"].values)

#race = np.array(['White','Black or African American','Hispanic',"Asian",'American Indian & Alaska Native','Two or More Races'])
race = np.array([5,3,0,2,1,6])
race_pct = np.array([0.552,0.14,0.193,0.086,0.002,0.027])
race = race[np.argsort(race_pct)[::-1]]
race_pct.sort()
race_pct = race_pct[::-1]
race_pct = race_pct.cumsum()

height_m = [['Under 5 years',34,46],['5 to 9 years',45,50],['10 to 14 years',55,64],['15 to 19 years',67,71],
        ['20 to 24 years',67,71],['25 to 29 years',67,71],['30 to 34 years',67,71],
        ['35 to 39 years',67,71],['40 to 44 years',67,71],['45 to 49 years',67,71],
        ['50 to 54 years',67,71],['55 to 59 years',67,71],['60 to 64 years',67,69],
        ['65 to 69 years',67,69],['70 to 74 years',67,69],['75 to 79 years',67,69],
        ['80 and older',67,69]]

weight_m = [['Under 5 years',28,46],['5 to 9 years',46,56],['10 to 14 years',68,112],['15 to 19 years',134,155],
        ['20 to 24 years',155,10],['25 to 29 years',184,208],['30 to 34 years',184,210],
        ['35 to 39 years',186,212],['40 to 44 years',186,212],['45 to 49 years',186,212],
        ['50 to 54 years',190,210],['55 to 59 years',190,210],['60 to 64 years',184,194],
        ['65 to 69 years',184,192],['70 to 74 years',184,192],['75 to 79 years',182,190],
        ['80 and older',182,190]]

height_f = [['Under 5 years',34,45],['5 to 9 years',45,50],['10 to 14 years',54,63],['15 to 19 years',63,66],
        ['20 to 24 years',63,66],['25 to 29 years',63,66],['30 to 34 years',63,66],
        ['35 to 39 years',63,66],['40 to 44 years',61,65],['45 to 49 years',61,65],
        ['50 to 54 years',61,65],['55 to 59 years',61,65],['60 to 64 years',59,64],
        ['65 to 69 years',59,64],['70 to 74 years',59,64],['75 to 79 years',59,64],
        ['80 and older',59,64]]

weight_f = [['Under 5 years',27,46],['5 to 9 years',46,58],['10 to 14 years',72,109],['15 to 19 years',109,130],
        ['20 to 24 years',130,160],['25 to 29 years',155.5,179.5],['30 to 34 years',155.5,179.5],
        ['35 to 39 years',155.5,179.5],['40 to 44 years',164,192],['45 to 49 years',164,192],
        ['50 to 54 years',165,192],['55 to 59 years',164,192],['60 to 64 years',153,178],
        ['65 to 69 years',153,178],['70 to 74 years',153,178],['75 to 79 years',153,178],
        ['80 and older',153,178]]

h_f= pd.DataFrame(height_f, columns =['Groups', 'Min','Max'])
w_f= pd.DataFrame(weight_f, columns =['Groups', 'Min','Max'])
h_m= pd.DataFrame(height_m, columns =['Groups', 'Min','Max'])
w_m= pd.DataFrame(weight_m, columns =['Groups', 'Min','Max'])

home_cat_f = pd.DataFrame([['65 to 69 years',3231],['70 to 74 years',6463.8],['75 to 79 years',16159.5],['80 and older',38782.8]])
home_cat_m = pd.DataFrame([['65 to 69 years',1256.85],['70 to 74 years',2513.7],['75 to 79 years',6284.25],['80 and older',15082.2]])
home_f = pd.merge(df[['Groups','Women']],home_cat_f, left_on = df['Groups'],right_on =home_cat_f[0], how= 'outer')
home_f['PCT'] = home_f[1]/home_f['Women']
home_m = pd.merge(df[['Groups','Men']],home_cat_m, left_on = df['Groups'],right_on =home_cat_m[0], how= 'outer')
home_m['PCT'] = home_m[1]/home_m['Men']
home_f = home_f[['Groups','PCT']].fillna(0)
home_m = home_m[['Groups','PCT']].fillna(0)

prob_child_homeless = 18653/df.sum(axis = 1)[1:5].sum()
prob_adult_homeless = 38689/df.sum(axis = 1)[5:13].sum()

diabetes_data = [['18-24',7000],['25-44',76000],['45-64',229000],['65+',195000]]
prob_diabetes_18_24 = 7000/df.sum(axis = 1)[3:5].sum()
prob_diabetes_25_44 = 76000/df.sum(axis = 1)[5:9].sum()
prob_diabetes_45_64 = 229000/df.sum(axis = 1)[9:13].sum()
prob_diabetes_65 = 195000/df.sum(axis = 1)[13:17].sum()

female = pd.DataFrame([0,0,0,29200,168900,291000,265000,212000,212000,198500,198500,171000,121800,55200,19000,7980,3420])
male = pd.DataFrame([0,0,0,24400,157300,292000,292000,241000,241000,208500,208500,165000,119000,53500,22200,11340,4860])
occ_f = pd.merge(df[['Groups',"Women"]],female, on=df.index)
occ_f['PCT'] = occ_f[0]/occ_f["Women"]
occ_f = occ_f[['Groups',"PCT"]]
occ_m = pd.merge(df[['Groups',"Men"]],male, on=df.index)
occ_m['PCT'] = occ_m[0]/occ_m["Men"]
occ_m = occ_m[['Groups',"PCT"]]
occ_lst = np.array(['Agriculture, Forestry, Fishing and Hunting','Mining','Utilities','Construction','Manufacturing','Wholesale Trade','Retail Trade',
'Transportation and Warehousing','Information','Finance and Insurance','Real Estate and Rental and Leasing','Professional and Technical Services',
'Management of Companies and Enterprises','Administrative and Waste Services','Educational Services','Health Care and Social Assistance',
'Arts, Entertainment, and Recreation','Accommodation and Food Services','Other Services, Ex. Public Admin','Total, All Government','Other'])
oc_pct = np.array([0.0000851,0.00000826,0.003649073,0.034802815,0.017155205,0.031722573,0.081533429,0.028225172,0.043838021,0.077196539,0.0308852,0.094403477,
0.016386731,0.055627316,0.042599246,0.165510547,0.02051269,0.085815999,0.04107194,0.126763074,0.0022075974])

occ_lst = occ_lst[np.argsort(oc_pct)[::-1]]
oc_pct.sort()
oc_pct = oc_pct[::-1]
oc_pct = oc_pct.cumsum()

def Rn_Citizen():
    if np.random.rand()<= np.sum(df['Men'].values)/(np.sum(df['Men'].values) + np.sum(df['Women'].values)):
        gender = 1
        u = np.random.rand()
        for i in range(len(outcome)):
            if u<cdf_m[i]:
                break
        age = np.ceil(np.random.uniform(outcomes.iloc[i].Min,outcomes.iloc[i].Max))
        cancer = (1 if c_m[i] >= np.random.rand() else 0)
        height = np.random.uniform(h_m.iloc[i].Min,h_m.iloc[i].Max)
        weight = np.random.uniform(w_m.iloc[i].Min,w_m.iloc[i].Max)
        home_cat = (1 if home_m.PCT[i] >= np.random.rand() else 0)
        emp = (1 if occ_m.PCT[i] >= np.random.rand() else 0)
  
    else:
        gender = 0
        u = np.random.rand()
        for i in range(len(outcome)):
            if u<cdf_f[i]:
                break
        age = np.ceil(np.random.uniform(outcomes.iloc[i].Min,outcomes.iloc[i].Max))
        cancer = (1 if c_f[i] >= np.random.rand() else 0)
        height = np.random.uniform(h_f.iloc[i].Min,h_f.iloc[i].Max)
        weight = np.random.uniform(w_f.iloc[i].Min,w_f.iloc[i].Max)
        home_cat = (1 if home_f.PCT[i] >= np.random.rand() else 0)
        emp = (1 if occ_f.PCT[i] >= np.random.rand() else 0)
    
    if home_cat == 0:
        if i in range(1,5):
            home_cat  = 2 if prob_child_homeless>=np.random.rand() else 0
        elif i in range(5,13):
            home_cat  = 2 if prob_adult_homeless>=np.random.rand() else 0
    
    if i in range(3,5):
        diabetes = 1 if prob_diabetes_18_24>=np.random.rand() else 0
    elif i in range(5,9):
        diabetes = 1 if prob_diabetes_25_44>=np.random.rand() else 0
    elif i in range(9,13):
        diabetes = 1 if prob_diabetes_45_64>=np.random.rand() else 0
    elif i in range(13,17):
        diabetes = 1 if prob_diabetes_65>=np.random.rand() else 0
    else:
        diabetes = 0

    u = np.random.rand()
    for i in range(len(race)):
        if u<=race_pct[i]:
            break
    ethnicity = race[i]
    essential = False 

    if emp == 1 and (age > YOUTH and age < SENIOR):
        u = np.random.rand()
        for i in range(len(occ_lst)):
            if u<=oc_pct[i]:
                break

        if oc_pct[i] in ESSENTIAL : 
            essential = True 
        
        occupation = i
        
    else:
        occupation = 'unemployed'
        occupation = len(occ_lst)

    # added by faye
    #age = 19 if age < 19 else age
    
    return {"age":age, "gender": gender, "infected":0, 
            "ethnicity": ethnicity , "weight":weight, "height":height, 
            "home_cat":home_cat, "diabetes_type":diabetes, "if_cancer":cancer,
            'occupation':occupation, "essential" : essential}
    
    


Rn_Citizen()
