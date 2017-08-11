# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:32:36 2016

jwink 11/17/2016
"""

# import things
import hddm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
import os.path as osp

# set save directory
save_dir = 'M:\EyeTracker.51\Analysis\Ajinomoto\HDDM_models\HDDM_savedModels'

# set data paths
data_path = 'M:/EyeTracker.51/Data/Ajinomoto/Concatenated/Option_Coding/Active/all_data.csv'
#data_path = '/Volumes/Huettel/EyeTracker.51/Data/Ajinomoto/Concatenated/Option_Coding/Active/all_data.csv'
rep_data_path = 'M:/EyeTracker.51/Data/Ajinomoto_Replication/Concatenated/Option_Coding/Active/all_data_rep.csv'
#rep_data_path = '/Volumes/Huettel/EyeTracker.51/Data/Ajinomoto_Replication/Concatenated/Option_Coding/Active/all_data_rep.csv'

# import original and replication data
data = hddm.load_csv(data_path,low_memory=False)
rep_data = hddm.load_csv(rep_data_path,low_memory=False)

# remove non-food trials
data = data[data.Phase == 'food']
rep_data = rep_data[rep_data.Phase == 'food']

# remove trials with missing data
data = data[pd.notnull(data['d_HealthRank'])]
rep_data = rep_data[pd.notnull(rep_data['d_HealthRank'])]

# remove extreme differences in ratings
data = data[data.d_WantRate != 4]
data = data[data.d_WantRate != -4]
data = data[data.d_HealthRate != 4]
data = data[data.d_HealthRate != -4]
data = data[data.d_TasteRate != 4]
data = data[data.d_TasteRate != -4]
rep_data = rep_data[rep_data.d_WantRate != 4]
rep_data = rep_data[rep_data.d_WantRate != -4]
rep_data = rep_data[rep_data.d_HealthRate != 4]
rep_data = rep_data[rep_data.d_HealthRate != -4]
rep_data = rep_data[rep_data.d_TasteRate != 4]
rep_data = rep_data[rep_data.d_TasteRate != -4]

# recode left choices to healthy choices
# make new column called ChoiceHealthy - needs to be adjusted to leave in NaN values where the data do not exist
data['ChoiceHealthy'] = np.where(((data['d_HealthRank'] > 0) & (data['Choice_Left'] == 1)) | 
                                 ((data['d_HealthRank'] < 0) & (data['Choice_Left'] == 0)),1,0)
rep_data['ChoiceHealthy'] = np.where(((rep_data['d_HealthRank'] > 0) & (rep_data['Choice_Left'] == 1)) | 
                                 ((rep_data['d_HealthRank'] < 0) & (rep_data['Choice_Left'] == 0)),1,0)
# change time to seconds
data['Response_Time'] = data['Response_Time']/1000
rep_data['Response_Time'] = rep_data['Response_Time']/1000

data['healthAdvantage'] = data['d_HealthRate']
data['healthAdvantage'].ix[data['l_HealthRank'] > data['r_HealthRank']] = data['healthAdvantage'][data['l_HealthRank'] > data['r_HealthRank']].multiply(-1,axis='index')
data['healthAdvantage'].ix[data['healthAdvantage'] == -0] = 0
rep_data['healthAdvantage'] = rep_data['d_HealthRate']
rep_data['healthAdvantage'].ix[rep_data['l_HealthRank'] > rep_data['r_HealthRank']] = rep_data['healthAdvantage'][rep_data['l_HealthRank'] > rep_data['r_HealthRank']].multiply(-1,axis='index')
rep_data['healthAdvantage'].ix[rep_data['healthAdvantage'] == -0] = 0

data['tasteAdvantage'] = data['d_TasteRate']
data['tasteAdvantage'].ix[data['l_HealthRank'] > data['r_HealthRank']] = data['tasteAdvantage'][data['l_HealthRank'] > data['r_HealthRank']].multiply(-1,axis='index')
data['tasteAdvantage'].ix[data['tasteAdvantage'] == -0] = 0
rep_data['tasteAdvantage'] = rep_data['d_TasteRate']
rep_data['tasteAdvantage'].ix[rep_data['l_HealthRank'] > rep_data['r_HealthRank']] = rep_data['tasteAdvantage'][rep_data['l_HealthRank'] > rep_data['r_HealthRank']].multiply(-1,axis='index')
rep_data['tasteAdvantage'].ix[rep_data['tasteAdvantage'] == -0] = 0

data['wantAdvantage'] = data['d_WantRate']
data['wantAdvantage'].ix[data['l_HealthRank'] > data['r_HealthRank']] = data['wantAdvantage'][data['l_HealthRank'] > data['r_HealthRank']].multiply(-1,axis='index')
data['wantAdvantage'].ix[data['wantAdvantage'] == -0] = 0
rep_data['wantAdvantage'] = rep_data['d_WantRate']
rep_data['wantAdvantage'].ix[rep_data['l_HealthRank'] > rep_data['r_HealthRank']] = rep_data['wantAdvantage'][rep_data['l_HealthRank'] > rep_data['r_HealthRank']].multiply(-1,axis='index')
rep_data['wantAdvantage'].ix[rep_data['wantAdvantage'] == -0] = 0

# look at these tmp variables to validate healthy-referenced coding of ratings
#tmp1 = data[['Left_Item','Right_Item','l_HealthRate','r_HealthRate','l_HealthRank','r_HealthRank','d_HealthRate','healthAdvantage']]
#tmp2 = data[['Left_Item','Right_Item','l_TasteRate','r_TasteRate','l_TasteRank','r_TasteRank','d_TasteRate','tasteAdvantage']]
#tmp3 = data[['Left_Item','Right_Item','l_WantRate','r_WantRate','l_WantRank','r_WantRank','d_WantRate','wantAdvantage']]

data['LastFixatedChosen'] = data['Choice_Left'] == data['lastFix_Left']
rep_data['LastFixatedChosen'] = rep_data['Choice_Left'] == rep_data['lastFix_Left']

# change column names
data.rename(columns={'Px': 'subj_idx', 'Response_Time': 'rt', 'ChoiceHealthy': 'response'}, inplace=True)
rep_data.rename(columns={'Px': 'subj_idx', 'Response_Time': 'rt', 'ChoiceHealthy': 'response'}, inplace=True)
# databse with fewer columns; use first version with HDDM Regression models
#df = data[['subj_idx','rt','Choice_Left','d_WantRate','d_TasteRate', 'd_HealthRate','d_HealthRank','alt_WantRate','response','Condition']]
df = data[['subj_idx','rt','Choice_Left','d_WantRate','d_TasteRate', 'd_HealthRate','healthAdvantage','tasteAdvantage','wantAdvantage','response','Condition','LastFixatedChosen']]
rep_df = rep_data[['subj_idx','rt','Choice_Left','d_WantRate','d_TasteRate', 'd_HealthRate','healthAdvantage','tasteAdvantage','wantAdvantage','response','Condition','LastFixatedChosen']]

# remove weird negative values in healthAdvantage; visualize the distribution of these
# by plotting the histogram of the original data: data['healthAdvantage'].hist()
df = df[df.healthAdvantage >= 0]
rep_df = rep_df[rep_df.healthAdvantage >= 0]

df_lastFixChosen = df[df.LastFixatedChosen == True]
df_lastFixNotChosen = df[df.LastFixatedChosen == False]
rep_df_lastFixChosen = rep_df[rep_df.LastFixatedChosen == True]
rep_df_lastFixNotChosen = rep_df[rep_df.LastFixatedChosen == False]


# adding a column of ones to run regression model 
#df['ones'] = pd.Series(1,index=df.index)
#creating a taste prime column
#df['TastePrime']= pd.Series(0, index=df.index)
#df['TastePrime'][df['Condition'] == 'taste'] = 1
#creating a health prime column
#df['HealthPrime']= pd.Series(0, index=df.index)
#df['HealthPrime'][df['Condition'] == 'health'] = 1
#creating a taste prime column
#df['ControlPrime']= pd.Series(0, index=df.index)
#df['ControlPrime'][df['Condition'] == 'control'] = 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# start hddm modeling

# insert the following line after hddm.HDDM instantiation line to save traces to the dbname
#full_model.mcmc(dbname='traces.db', db='pickle')

# set number of samples per model and burn-in rate
n_samples = 5000
n_burns = 1000

validity_model = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 'z':['d_WantRate'], 't':['d_WantRate'], 'a':['d_WantRate']}, bias = True)
validity_model.find_starting_values()
validity_model.sample(n_samples, burn=n_burns)

validity_model2 = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['wantAdvantage'], 'z':['wantAdvantage'], 't':['wantAdvantage'], 'a':['wantAdvantage']}, bias = True)
validity_model2.find_starting_values()
validity_model2.sample(n_samples, burn=n_burns)

full_model = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
full_model.find_starting_values()
full_model.sample(n_samples, burn=n_burns)

# only take data from trials where taste rating advantage is 0
df_dT0 = df[df.tasteAdvantage == 0]
full_model_dT0 = hddm.HDDM(df_dT0, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
full_model_dT0.find_starting_values()
full_model_dT0.sample(n_samples, burn=n_burns)

# view condition effects across all unique levels of taste rating advantage
full_model_all_dT = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['Condition','tasteAdvantage'], 'z':['Condition','tasteAdvantage'], 't':['Condition','tasteAdvantage'], 'a':['Condition','tasteAdvantage']}, bias = True)
full_model_all_dT.find_starting_values()
full_model_all_dT.sample(n_samples, burn=n_burns)

conditionByRatings_model = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['Condition','healthAdvantage','tasteAdvantage']}, bias = True)
conditionByRatings_model.find_starting_values()
conditionByRatings_model.sample(n_samples, burn=n_burns)

# THIS ONE FOR DISSERTATION CHAPTER 3
full_model_rep = hddm.HDDM(rep_df, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
full_model_rep.find_starting_values()
full_model_rep.sample(n_samples, burn=n_burns)

full_model_rep_all_dT = hddm.HDDM(rep_df, p_outlier = 0.05, depends_on={'v': ['Condition','tasteAdvantage'], 'z':['Condition','tasteAdvantage'], 't':['Condition','tasteAdvantage'], 'a':['Condition','tasteAdvantage']}, bias = True)
full_model_rep_all_dT.find_starting_values()
full_model_rep_all_dT.sample(n_samples, burn=n_burns)


# split models by whether the last fixated item was chosen or not
lfChosen_model = hddm.HDDM(df_lastFixChosen, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
lfChosen_model.find_starting_values()
lfChosen_model.sample(n_samples, burn=n_burns)

lfUnchosen_model = hddm.HDDM(df_lastFixNotChosen, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
lfUnchosen_model.find_starting_values()
lfUnchosen_model.sample(n_samples, burn=n_burns)

# now with the replication data
lfChosen_model_rep = hddm.HDDM(rep_df_lastFixChosen, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
lfChosen_model_rep.find_starting_values()
lfChosen_model_rep.sample(n_samples, burn=n_burns)

lfUnchosen_model_rep = hddm.HDDM(rep_df_lastFixNotChosen, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
lfUnchosen_model_rep.find_starting_values()
lfUnchosen_model_rep.sample(n_samples, burn=n_burns)

#############    full_model breakdown of posteriors    #############

v_taste, v_health, v_control = full_model.nodes_db.node[[ "v(taste)", "v(health)", "v(control)"]]
hddm.analyze.plot_posterior_nodes([v_taste, v_health, v_control])
np.savetxt('v_taste.txt',v_taste.trace(),delimiter=',')
np.savetxt('v_health.txt',v_health.trace(),delimiter=',')
np.savetxt('v_control.txt',v_control.trace(),delimiter=',')
v_taste_rep, v_health_rep = full_model_rep.nodes_db.node[[ "v(taste)", "v(health)"]]
hddm.analyze.plot_posterior_nodes([v_taste_rep, v_health_rep])
np.savetxt('v_taste_rep.txt',v_taste_rep.trace(),delimiter=',')
np.savetxt('v_health_rep.txt',v_health_rep.trace(),delimiter=',')
  
t_taste, t_health, t_control = full_model.nodes_db.node[[ "t(taste)", "t(health)", "t(control)"]]
hddm.analyze.plot_posterior_nodes([t_taste, t_health, t_control])
np.savetxt('t_taste.txt',t_taste.trace(),delimiter=',')
np.savetxt('t_health.txt',t_health.trace(),delimiter=',')
np.savetxt('t_control.txt',t_control.trace(),delimiter=',')
t_taste_rep, t_health_rep = full_model_rep.nodes_db.node[[ "t(taste)", "t(health)"]]
hddm.analyze.plot_posterior_nodes([t_taste_rep, t_health_rep])
np.savetxt('t_taste_rep.txt',t_taste_rep.trace(),delimiter=',')
np.savetxt('t_health_rep.txt',t_health_rep.trace(),delimiter=',')

z_taste, z_health, z_control = full_model.nodes_db.node[[ "z(taste)", "z(health)", "z(control)"]]
hddm.analyze.plot_posterior_nodes([z_taste, z_health, z_control])
np.savetxt('z_taste.txt',z_taste.trace(),delimiter=',')
np.savetxt('z_health.txt',z_health.trace(),delimiter=',')
np.savetxt('z_control.txt',z_control.trace(),delimiter=',')
z_taste_rep, z_health_rep = full_model_rep.nodes_db.node[[ "z(taste)", "z(health)"]]
hddm.analyze.plot_posterior_nodes([z_taste_rep, z_health_rep]) 
np.savetxt('z_taste_rep.txt',z_taste_rep.trace(),delimiter=',')
np.savetxt('z_health_rep.txt',z_health_rep.trace(),delimiter=',')

a_taste, a_health, a_control = full_model.nodes_db.node[[ "a(taste)", "a(health)", "a(control)"]]
hddm.analyze.plot_posterior_nodes([a_taste, a_health, a_control])
np.savetxt('a_taste.txt',a_taste.trace(),delimiter=',')
np.savetxt('a_health.txt',a_health.trace(),delimiter=',')
np.savetxt('a_control.txt',a_control.trace(),delimiter=',')
a_taste_rep, a_health_rep = full_model_rep.nodes_db.node[[ "a(taste)", "a(health)"]]
hddm.analyze.plot_posterior_nodes([a_taste_rep, a_health_rep])
np.savetxt('a_taste_rep.txt',a_taste_rep.trace(),delimiter=',')
np.savetxt('a_health_rep.txt',a_health_rep.trace(),delimiter=',')


#############    full_model_dT0 at dTaste = 0 breakdown of posteriors    #############

v_taste_dT0, v_health_dT0, v_control_dT0 = full_model_all_dT.nodes_db.node[[ "v(taste)", "v(health)", "v(control)"]]
hddm.analyze.plot_posterior_nodes([v_taste_dT0, v_health_dT0, v_control_dT0])
(v_health_dT0.trace() > v_taste_dT0.trace()).mean()
np.savetxt('v_taste_dT0.txt',v_taste_dT0.trace(),delimiter=',')
np.savetxt('v_health_dT0.txt',v_health_dT0.trace(),delimiter=',')
np.savetxt('v_control_dT0.txt',v_control_dT0.trace(),delimiter=',')
  
z_taste_dT0, z_health_dT0, z_control_dT0 = full_model_dT0.nodes_db.node[[ "z(taste)", "z(health)", "z(control)"]]
hddm.analyze.plot_posterior_nodes([z_taste_dT0, z_health_dT0, z_control_dT0])
np.savetxt('z_taste_dT0.txt',z_taste_dT0.trace(),delimiter=',')
np.savetxt('z_health_dT0.txt',z_health_dT0.trace(),delimiter=',')
np.savetxt('z_control_dT0.txt',z_control_dT0.trace(),delimiter=',')


#############    validity_model breakdown of posteriors    #############

v_n3Want, v_n2Want, v_n1Want, v_0Want, v_p1Want, v_p2Want, v_p3Want = validity_model2.nodes_db.node[[ "v(-3.0)", "v(-2.0)", "v(-1.0)", "v(0.0)", "v(1.0)", "v(2.0)", "v(3.0)"]]
hddm.analyze.plot_posterior_nodes([v_n3Want, v_n2Want, v_n1Want, v_0Want, v_p1Want, v_p2Want, v_p3Want])
#np.savetxt('v_taste_dT0.txt',v_taste.trace(),delimiter=',')
#np.savetxt('v_health_dT0.txt',v_health.trace(),delimiter=',')
#np.savetxt('v_control_dT0.txt',v_control.trace(),delimiter=',')
  
t_n3Want, t_n2Want, t_n1Want, t_0Want, t_p1Want, t_p2Want, t_p3Want = validity_model2.nodes_db.node[[ "t(-3.0)", "t(-2.0)", "t(-1.0)", "t(0.0)", "t(1.0)", "t(2.0)", "t(3.0)"]]
hddm.analyze.plot_posterior_nodes([t_n3Want, t_n2Want, t_n1Want, t_0Want, t_p1Want, t_p2Want, t_p3Want])
#np.savetxt('t_taste_dT0.txt',t_taste.trace(),delimiter=',')
#np.savetxt('t_health_dT0.txt',t_health.trace(),delimiter=',')
#np.savetxt('t_control_dT0.txt',t_control.trace(),delimiter=',')

z_n3Want, z_n2Want, z_n1Want, z_0Want, z_p1Want, z_p2Want, z_p3Want = validity_model2.nodes_db.node[[ "z(-3.0)", "z(-2.0)", "z(-1.0)", "z(0.0)", "z(1.0)", "z(2.0)", "z(3.0)"]]
hddm.analyze.plot_posterior_nodes([z_n3Want, z_n2Want, z_n1Want, z_0Want, z_p1Want, z_p2Want, z_p3Want])
#np.savetxt('z_taste_dT0.txt',z_taste.trace(),delimiter=',')
#np.savetxt('z_health_dT0.txt',z_health.trace(),delimiter=',')
#np.savetxt('z_control_dT0.txt',z_control.trace(),delimiter=',')

a_n3Want, a_n2Want, a_n1Want, a_0Want, a_p1Want, a_p2Want, a_p3Want = validity_model2.nodes_db.node[[ "a(-3.0)", "a(-2.0)", "a(-1.0)", "a(0.0)", "a(1.0)", "a(2.0)", "a(3.0)"]]
hddm.analyze.plot_posterior_nodes([a_n3Want, a_n2Want, a_n1Want, a_0Want, a_p1Want, a_p2Want, a_p3Want])
#np.savetxt('a_taste_dT0.txt',a_taste.trace(),delimiter=',')
#np.savetxt('a_health_dT0.txt',a_health.trace(),delimiter=',')
#np.savetxt('a_control_dT0.txt',a_control.trace(),delimiter=',')


#############    full_model_all_dT breakdown of posteriors    #############

v_taste_dTn3, v_taste_dTn2, v_taste_dTn1, v_taste_dT0, v_taste_dTp1, v_taste_dTp2, v_taste_dTp3, = full_model_all_dT.nodes_db.node[[ "v(taste.-3.0)", "v(taste.-2.0)", "v(taste.-1.0)","v(taste.0.0)","v(taste.1.0)","v(taste.2.0)","v(taste.3.0)",]]
v_health_dTn3, v_health_dTn2, v_health_dTn1, v_health_dT0, v_health_dTp1, v_health_dTp2, v_health_dTp3, = full_model_all_dT.nodes_db.node[[ "v(health.-3.0)", "v(health.-2.0)", "v(health.-1.0)","v(health.0.0)","v(health.1.0)","v(health.2.0)","v(health.3.0)",]]
v_control_dTn3, v_control_dTn2, v_control_dTn1, v_control_dT0, v_control_dTp1, v_control_dTp2, v_control_dTp3, = full_model_all_dT.nodes_db.node[[ "v(control.-3.0)", "v(control.-2.0)", "v(control.-1.0)","v(control.0.0)","v(control.1.0)","v(control.2.0)","v(control.3.0)",]]

np.savetxt('v_taste_dTn3.txt',v_taste_dTn3.trace(),delimiter=',')
np.savetxt('v_taste_dTn2.txt',v_taste_dTn2.trace(),delimiter=',')
np.savetxt('v_taste_dTn1.txt',v_taste_dTn1.trace(),delimiter=',')
np.savetxt('v_taste_dT0.txt',v_taste_dT0.trace(),delimiter=',')
np.savetxt('v_taste_dTp1.txt',v_taste_dTp1.trace(),delimiter=',')
np.savetxt('v_taste_dTp2.txt',v_taste_dTp2.trace(),delimiter=',')
np.savetxt('v_taste_dTp3.txt',v_taste_dTp3.trace(),delimiter=',')

np.savetxt('v_health_dTn3.txt',v_health_dTn3.trace(),delimiter=',')
np.savetxt('v_health_dTn2.txt',v_health_dTn2.trace(),delimiter=',')
np.savetxt('v_health_dTn1.txt',v_health_dTn1.trace(),delimiter=',')
np.savetxt('v_health_dT0.txt',v_health_dT0.trace(),delimiter=',')
np.savetxt('v_health_dTp1.txt',v_health_dTp1.trace(),delimiter=',')
np.savetxt('v_health_dTp2.txt',v_health_dTp2.trace(),delimiter=',')
np.savetxt('v_health_dTp3.txt',v_health_dTp3.trace(),delimiter=',')

np.savetxt('v_control_dTn3.txt',v_control_dTn3.trace(),delimiter=',')
np.savetxt('v_control_dTn2.txt',v_control_dTn2.trace(),delimiter=',')
np.savetxt('v_control_dTn1.txt',v_control_dTn1.trace(),delimiter=',')
np.savetxt('v_control_dT0.txt',v_control_dT0.trace(),delimiter=',')
np.savetxt('v_control_dTp1.txt',v_control_dTp1.trace(),delimiter=',')
np.savetxt('v_control_dTp2.txt',v_control_dTp2.trace(),delimiter=',')
np.savetxt('v_control_dTp3.txt',v_control_dTp3.trace(),delimiter=',')

hddm.analyze.plot_posterior_nodes([v_taste_dTn3, v_health_dTn3, v_control_dTn3])
hddm.analyze.plot_posterior_nodes([v_taste_dTn2, v_health_dTn2, v_control_dTn2])
hddm.analyze.plot_posterior_nodes([v_taste_dTn1, v_health_dTn1, v_control_dTn1])
hddm.analyze.plot_posterior_nodes([v_taste_dT0, v_health_dT0, v_control_dT0])
hddm.analyze.plot_posterior_nodes([v_taste_dTp1, v_health_dTp1, v_control_dTp1])
hddm.analyze.plot_posterior_nodes([v_taste_dTp2, v_health_dTp2, v_control_dTp2])
hddm.analyze.plot_posterior_nodes([v_taste_dTp3, v_health_dTp3, v_control_dTp3])

z_taste_dTn3, z_taste_dTn2, z_taste_dTn1, z_taste_dT0, z_taste_dTp1, z_taste_dTp2, z_taste_dTp3, = full_model_all_dT.nodes_db.node[[ "z(taste.-3.0)", "z(taste.-2.0)", "z(taste.-1.0)","z(taste.0.0)","z(taste.1.0)","z(taste.2.0)","z(taste.3.0)",]]
z_health_dTn3, z_health_dTn2, z_health_dTn1, z_health_dT0, z_health_dTp1, z_health_dTp2, z_health_dTp3, = full_model_all_dT.nodes_db.node[[ "z(health.-3.0)", "z(health.-2.0)", "z(health.-1.0)","z(health.0.0)","z(health.1.0)","z(health.2.0)","z(health.3.0)",]]
z_control_dTn3, z_control_dTn2, z_control_dTn1, z_control_dT0, z_control_dTp1, z_control_dTp2, z_control_dTp3, = full_model_all_dT.nodes_db.node[[ "z(control.-3.0)", "z(control.-2.0)", "z(control.-1.0)","z(control.0.0)","z(control.1.0)","z(control.2.0)","z(control.3.0)",]]


np.savetxt('z_taste_dTn3.txt',z_taste_dTn3.trace(),delimiter=',')
np.savetxt('z_taste_dTn2.txt',z_taste_dTn2.trace(),delimiter=',')
np.savetxt('z_taste_dTn1.txt',z_taste_dTn1.trace(),delimiter=',')
np.savetxt('z_taste_dT0.txt',z_taste_dT0.trace(),delimiter=',')
np.savetxt('z_taste_dTp1.txt',z_taste_dTp1.trace(),delimiter=',')
np.savetxt('z_taste_dTp2.txt',z_taste_dTp2.trace(),delimiter=',')
np.savetxt('z_taste_dTp3.txt',z_taste_dTp3.trace(),delimiter=',')

np.savetxt('z_health_dTn3.txt',z_health_dTn3.trace(),delimiter=',')
np.savetxt('z_health_dTn2.txt',z_health_dTn2.trace(),delimiter=',')
np.savetxt('z_health_dTn1.txt',z_health_dTn1.trace(),delimiter=',')
np.savetxt('z_health_dT0.txt',z_health_dT0.trace(),delimiter=',')
np.savetxt('z_health_dTp1.txt',z_health_dTp1.trace(),delimiter=',')
np.savetxt('z_health_dTp2.txt',z_health_dTp2.trace(),delimiter=',')
np.savetxt('z_health_dTp3.txt',z_health_dTp3.trace(),delimiter=',')

np.savetxt('z_control_dTn3.txt',z_control_dTn3.trace(),delimiter=',')
np.savetxt('z_control_dTn2.txt',z_control_dTn2.trace(),delimiter=',')
np.savetxt('z_control_dTn1.txt',z_control_dTn1.trace(),delimiter=',')
np.savetxt('z_control_dT0.txt',z_control_dT0.trace(),delimiter=',')
np.savetxt('z_control_dTp1.txt',z_control_dTp1.trace(),delimiter=',')
np.savetxt('z_control_dTp2.txt',z_control_dTp2.trace(),delimiter=',')
np.savetxt('z_control_dTp3.txt',z_control_dTp3.trace(),delimiter=',')

hddm.analyze.plot_posterior_nodes([v_taste_dTn3, v_health_dTn3, v_control_dTn3])
hddm.analyze.plot_posterior_nodes([v_taste_dTn2, v_health_dTn2, v_control_dTn2])
hddm.analyze.plot_posterior_nodes([v_taste_dTn1, v_health_dTn1, v_control_dTn1])
hddm.analyze.plot_posterior_nodes([v_taste_dT0, v_health_dT0, v_control_dT0])
hddm.analyze.plot_posterior_nodes([v_taste_dTp1, v_health_dTp1, v_control_dTp1])
hddm.analyze.plot_posterior_nodes([v_taste_dTp2, v_health_dTp2, v_control_dTp2])
hddm.analyze.plot_posterior_nodes([v_taste_dTp3, v_health_dTp3, v_control_dTp3])

hddm.analyze.plot_posterior_nodes([z_taste_dTn3, z_health_dTn3])
hddm.analyze.plot_posterior_nodes([z_taste_dTn2, z_health_dTn2])
hddm.analyze.plot_posterior_nodes([z_taste_dTn1, z_health_dTn1])
hddm.analyze.plot_posterior_nodes([z_taste_dT0, z_health_dT0])
hddm.analyze.plot_posterior_nodes([z_taste_dTp1, z_health_dTp1])
hddm.analyze.plot_posterior_nodes([z_taste_dTp2, z_health_dTp2])
hddm.analyze.plot_posterior_nodes([z_taste_dTp3, z_health_dTp3])

# the followinglines get traces for all params of the replication model
v_taste_dTn3, v_taste_dTn2, v_taste_dTn1, v_taste_dT0, v_taste_dTp1, v_taste_dTp2, v_taste_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "v(taste.-3.0)", "v(taste.-2.0)", "v(taste.-1.0)","v(taste.0.0)","v(taste.1.0)","v(taste.2.0)","v(taste.3.0)",]]
v_health_dTn3, v_health_dTn2, v_health_dTn1, v_health_dT0, v_health_dTp1, v_health_dTp2, v_health_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "v(health.-3.0)", "v(health.-2.0)", "v(health.-1.0)","v(health.0.0)","v(health.1.0)","v(health.2.0)","v(health.3.0)",]]

z_taste_dTn3, z_taste_dTn2, z_taste_dTn1, z_taste_dT0, z_taste_dTp1, z_taste_dTp2, z_taste_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "z(taste.-3.0)", "z(taste.-2.0)", "z(taste.-1.0)","z(taste.0.0)","z(taste.1.0)","z(taste.2.0)","z(taste.3.0)",]]
z_health_dTn3, z_health_dTn2, z_health_dTn1, z_health_dT0, z_health_dTp1, z_health_dTp2, z_health_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "z(health.-3.0)", "z(health.-2.0)", "z(health.-1.0)","z(health.0.0)","z(health.1.0)","z(health.2.0)","z(health.3.0)",]]

a_taste_dTn3, a_taste_dTn2, a_taste_dTn1, a_taste_dT0, a_taste_dTp1, a_taste_dTp2, a_taste_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "a(taste.-3.0)", "a(taste.-2.0)", "a(taste.-1.0)","a(taste.0.0)","a(taste.1.0)","a(taste.2.0)","a(taste.3.0)",]]
a_health_dTn3, a_health_dTn2, a_health_dTn1, a_health_dT0, a_health_dTp1, a_health_dTp2, a_health_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "a(health.-3.0)", "a(health.-2.0)", "a(health.-1.0)","a(health.0.0)","a(health.1.0)","a(health.2.0)","a(health.3.0)",]]

t_taste_dTn3, t_taste_dTn2, t_taste_dTn1, t_taste_dT0, t_taste_dTp1, t_taste_dTp2, t_taste_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "t(taste.-3.0)", "t(taste.-2.0)", "t(taste.-1.0)","t(taste.0.0)","t(taste.1.0)","t(taste.2.0)","t(taste.3.0)",]]
t_health_dTn3, t_health_dTn2, t_health_dTn1, t_health_dT0, t_health_dTp1, t_health_dTp2, t_health_dTp3, = full_model_rep_all_dT.nodes_db.node[[ "t(health.-3.0)", "t(health.-2.0)", "t(health.-1.0)","t(health.0.0)","t(health.1.0)","t(health.2.0)","t(health.3.0)",]]


np.savetxt('a_taste_dTn3.txt',a_taste_dTn3.trace(),delimiter=',')
np.savetxt('a_taste_dTn2.txt',a_taste_dTn2.trace(),delimiter=',')
np.savetxt('a_taste_dTn1.txt',a_taste_dTn1.trace(),delimiter=',')
np.savetxt('a_taste_dT0.txt',a_taste_dT0.trace(),delimiter=',')
np.savetxt('a_taste_dTp1.txt',a_taste_dTp1.trace(),delimiter=',')
np.savetxt('a_taste_dTp2.txt',a_taste_dTp2.trace(),delimiter=',')
np.savetxt('a_taste_dTp3.txt',a_taste_dTp3.trace(),delimiter=',')

np.savetxt('a_health_dTn3.txt',a_health_dTn3.trace(),delimiter=',')
np.savetxt('a_health_dTn2.txt',a_health_dTn2.trace(),delimiter=',')
np.savetxt('a_health_dTn1.txt',a_health_dTn1.trace(),delimiter=',')
np.savetxt('a_health_dT0.txt',a_health_dT0.trace(),delimiter=',')
np.savetxt('a_health_dTp1.txt',a_health_dTp1.trace(),delimiter=',')
np.savetxt('a_health_dTp2.txt',a_health_dTp2.trace(),delimiter=',')
np.savetxt('a_health_dTp3.txt',a_health_dTp3.trace(),delimiter=',')

np.savetxt('t_taste_dTn3.txt',t_taste_dTn3.trace(),delimiter=',')
np.savetxt('t_taste_dTn2.txt',t_taste_dTn2.trace(),delimiter=',')
np.savetxt('t_taste_dTn1.txt',t_taste_dTn1.trace(),delimiter=',')
np.savetxt('t_taste_dT0.txt',t_taste_dT0.trace(),delimiter=',')
np.savetxt('t_taste_dTp1.txt',t_taste_dTp1.trace(),delimiter=',')
np.savetxt('t_taste_dTp2.txt',t_taste_dTp2.trace(),delimiter=',')
np.savetxt('t_taste_dTp3.txt',t_taste_dTp3.trace(),delimiter=',')

np.savetxt('t_health_dTn3.txt',t_health_dTn3.trace(),delimiter=',')
np.savetxt('t_health_dTn2.txt',t_health_dTn2.trace(),delimiter=',')
np.savetxt('t_health_dTn1.txt',t_health_dTn1.trace(),delimiter=',')
np.savetxt('t_health_dT0.txt',t_health_dT0.trace(),delimiter=',')
np.savetxt('t_health_dTp1.txt',t_health_dTp1.trace(),delimiter=',')
np.savetxt('t_health_dTp2.txt',t_health_dTp2.trace(),delimiter=',')
np.savetxt('t_health_dTp3.txt',t_health_dTp3.trace(),delimiter=',')



##########################################################################
# regression model
regdf = data[['subj_idx','rt','healthAdvantage','tasteAdvantage','response','Condition']]
regdf.dropna(axis='index',how='any',inplace=True)
# remove row 7926 where px 25's rt is 0.000841, 
# and row 17907 where px 58's rt is 0.000874,  
# and row 29909 where px 117's rt is 0.000834,  
# and row 36781 where px 136's rt is 0.000834
# these values screw up the regression model
regdf.drop(7926, inplace=True)
regdf.drop(17907, inplace=True)
regdf.drop(29909, inplace=True)
regdf.drop(36781, inplace=True)

rep_regdf = rep_data[['subj_idx','rt','healthAdvantage','tasteAdvantage','response','Condition']]
rep_regdf.dropna(axis='index',how='any',inplace=True)
rep_regdf = rep_regdf[rep_regdf['rt'] > 0.2]


#regdf[['healthAdvantage']] = regdf[['healthAdvantage']].astype(int)
#regdf[['tasteAdvantage']] = regdf[['tasteAdvantage']].astype(int)
#regdf[['subj_idx']] = regdf[['subj_idx']].astype(int)

driftRegression = hddm.models.HDDMRegressor(regdf, 'v ~ healthAdvantage*tasteAdvantage', depends_on={'v': 'Condition'}, p_outlier = 0.05)
driftRegression.sample(n_samples, burn=n_burns)

rep_driftRegression = hddm.models.HDDMRegressor(rep_regdf, 'v ~ healthAdvantage*tasteAdvantage', depends_on={'v': 'Condition'}, p_outlier = 0.05)
rep_driftRegression.sample(n_samples, burn=n_burns)

driftRegression.print_stats(fname='driftRegressionStats.txt')

driftRegression_noise = hddm.models.HDDMRegressor(regdf, 'v ~ healthAdvantage*tasteAdvantage', depends_on={'v': 'Condition'}, include = 'all', p_outlier = 0.05)
driftRegression_noise.sample(n_samples, burn=n_burns)



v_healthAdv, v_tasteAdv, v_healthTaste = driftRegression.nodes_db.node[[ "v_healthAdvantage", "v_tasteAdvantage", "v_healthAdvantage:tasteAdvantage"]]
hddm.analyze.plot_posterior_nodes([v_tasteAdv, v_healthAdv, v_healthTaste])
plt.title('Regression Betas on Drift Rate')
plt.savefig('regressionBetas.pdf')
np.savetxt('v_healthAdvBeta.txt',v_healthAdv.trace(),delimiter=',')
np.savetxt('v_tasteAdvBeta.txt',v_tasteAdv.trace(),delimiter=',')
np.savetxt('v_healthTasteInteractionBeta.txt',v_healthTaste.trace(),delimiter=',')

v_healthIntercept, v_tasteIntercept, v_controlIntercept = driftRegression.nodes_db.node[[ "v_Intercept(health)", "v_Intercept(taste)", "v_Intercept(control)"]]
hddm.analyze.plot_posterior_nodes([v_tasteIntercept, v_healthIntercept, v_controlIntercept])
plt.title('Effect of Prime on Drift Rate Intercepts')
plt.savefig('PrimeOnIntercepts.pdf')
np.savetxt('v_healthIntercept.txt',v_healthIntercept.trace(),delimiter=',')
np.savetxt('v_tasteIntercept.txt',v_tasteIntercept.trace(),delimiter=',')
np.savetxt('v_controlIntercept.txt',v_controlIntercept.trace(),delimiter=',')

(v_healthIntercept.trace() > v_controlIntercept.trace()).mean()



v_healthIntercept_noise, v_tasteIntercept_noise, v_controlIntercept_noise = driftRegression_noise.nodes_db.node[[ "v_Intercept(health)", "v_Intercept(taste)", "v_Intercept(control)"]]
hddm.analyze.plot_posterior_nodes([v_tasteIntercept_noise, v_healthIntercept_noise, v_controlIntercept_noise])
plt.title('Effect of Prime on Drift Rate Intercepts')
(v_healthIntercept_noise.trace() > v_tasteIntercept_noise.trace()).mean()


# replication regression model results

v_healthAdv, v_tasteAdv, v_healthTaste = rep_driftRegression.nodes_db.node[[ "v_healthAdvantage", "v_tasteAdvantage", "v_healthAdvantage:tasteAdvantage"]]
hddm.analyze.plot_posterior_nodes([v_tasteAdv, v_healthAdv, v_healthTaste])
plt.title('Replication: Regression Betas on Drift Rate')
plt.savefig('regressionBetas.pdf')
np.savetxt('v_healthAdvBeta.txt',v_healthAdv.trace(),delimiter=',')
np.savetxt('v_tasteAdvBeta.txt',v_tasteAdv.trace(),delimiter=',')
np.savetxt('v_healthTasteInteractionBeta.txt',v_healthTaste.trace(),delimiter=',')

v_healthIntercept, v_tasteIntercept = rep_driftRegression.nodes_db.node[[ "v_Intercept(health)", "v_Intercept(taste)"]]
hddm.analyze.plot_posterior_nodes([v_tasteIntercept, v_healthIntercept])
plt.title('Replication: Effect of Prime on Drift Rate Intercepts')
plt.savefig('PrimeOnIntercepts.pdf')
np.savetxt('v_healthIntercept.txt',v_healthIntercept.trace(),delimiter=',')
np.savetxt('v_tasteIntercept.txt',v_tasteIntercept.trace(),delimiter=',')




# For each subject, I want to plot the relationship between their z and v parameters. 
# First let's save the z and v traces for each participant

# make a list of param/subj labels for pulling traces
v_list = ['v(control)','v(health)','v(taste)','v_std','v_subj(control).5.0','v_subj(control).8.0','v_subj(control).9.0','v_subj(control).17.0','v_subj(control).19.0','v_subj(control).21.0','v_subj(control).24.0','v_subj(control).25.0','v_subj(control).29.0','v_subj(control).32.0','v_subj(control).34.0','v_subj(control).36.0','v_subj(control).38.0',
          'v_subj(control).39.0','v_subj(control).40.0','v_subj(control).41.0','v_subj(control).64.0','v_subj(control).90.0','v_subj(control).92.0','v_subj(control).93.0','v_subj(control).102.0','v_subj(control).107.0','v_subj(control).110.0','v_subj(control).112.0','v_subj(control).113.0','v_subj(control).114.0','v_subj(control).116.0','v_subj(control).117.0',
          'v_subj(control).118.0','v_subj(control).126.0','v_subj(control).129.0','v_subj(control).131.0','v_subj(control).136.0','v_subj(control).137.0','v_subj(control).141.0','v_subj(control).143.0','v_subj(control).145.0','v_subj(health).2.0','v_subj(health).4.0','v_subj(health).6.0','v_subj(health).7.0','v_subj(health).10.0','v_subj(health).13.0',
          'v_subj(health).14.0','v_subj(health).15.0','v_subj(health).16.0','v_subj(health).23.0','v_subj(health).27.0','v_subj(health).28.0','v_subj(health).30.0','v_subj(health).33.0','v_subj(health).35.0','v_subj(health).43.0','v_subj(health).45.0','v_subj(health).62.0','v_subj(health).94.0','v_subj(health).96.0','v_subj(health).97.0','v_subj(health).100.0',
          'v_subj(health).101.0','v_subj(health).105.0','v_subj(health).108.0','v_subj(health).109.0','v_subj(health).115.0','v_subj(health).119.0','v_subj(health).120.0','v_subj(health).124.0','v_subj(health).125.0','v_subj(health).133.0','v_subj(health).134.0','v_subj(health).135.0','v_subj(health).140.0','v_subj(health).142.0','v_subj(health).144.0',
          'v_subj(health).146.0','v_subj(taste).1.0','v_subj(taste).20.0','v_subj(taste).22.0','v_subj(taste).37.0','v_subj(taste).46.0','v_subj(taste).47.0','v_subj(taste).48.0','v_subj(taste).49.0','v_subj(taste).50.0','v_subj(taste).51.0','v_subj(taste).53.0','v_subj(taste).55.0','v_subj(taste).57.0','v_subj(taste).58.0','v_subj(taste).60.0','v_subj(taste).61.0',
          'v_subj(taste).63.0','v_subj(taste).65.0','v_subj(taste).91.0','v_subj(taste).95.0','v_subj(taste).99.0','v_subj(taste).103.0','v_subj(taste).104.0','v_subj(taste).106.0','v_subj(taste).111.0','v_subj(taste).121.0','v_subj(taste).122.0','v_subj(taste).123.0','v_subj(taste).127.0','v_subj(taste).128.0','v_subj(taste).130.0','v_subj(taste).132.0','v_subj(taste).138.0','v_subj(taste).139.0']
z_list = ['z(control)','z(health)','z(taste)','z_std','z_subj(control).5.0','z_subj(control).8.0','z_subj(control).9.0','z_subj(control).17.0','z_subj(control).19.0','z_subj(control).21.0','z_subj(control).24.0','z_subj(control).25.0','z_subj(control).29.0','z_subj(control).32.0','z_subj(control).34.0','z_subj(control).36.0','z_subj(control).38.0','z_subj(control).39.0',
          'z_subj(control).40.0','z_subj(control).41.0','z_subj(control).64.0','z_subj(control).90.0','z_subj(control).92.0','z_subj(control).93.0','z_subj(control).102.0','z_subj(control).107.0','z_subj(control).110.0','z_subj(control).112.0','z_subj(control).113.0','z_subj(control).114.0','z_subj(control).116.0','z_subj(control).117.0','z_subj(control).118.0','z_subj(control).126.0',
          'z_subj(control).129.0','z_subj(control).131.0','z_subj(control).136.0','z_subj(control).137.0','z_subj(control).141.0','z_subj(control).143.0','z_subj(control).145.0','z_subj(health).2.0','z_subj(health).4.0','z_subj(health).6.0','z_subj(health).7.0','z_subj(health).10.0','z_subj(health).13.0','z_subj(health).14.0','z_subj(health).15.0','z_subj(health).16.0','z_subj(health).23.0',
          'z_subj(health).27.0','z_subj(health).28.0','z_subj(health).30.0','z_subj(health).33.0','z_subj(health).35.0','z_subj(health).43.0','z_subj(health).45.0','z_subj(health).62.0','z_subj(health).94.0','z_subj(health).96.0','z_subj(health).97.0','z_subj(health).100.0','z_subj(health).101.0','z_subj(health).105.0','z_subj(health).108.0','z_subj(health).109.0','z_subj(health).115.0',
          'z_subj(health).119.0','z_subj(health).120.0','z_subj(health).124.0','z_subj(health).125.0','z_subj(health).133.0','z_subj(health).134.0','z_subj(health).135.0','z_subj(health).140.0','z_subj(health).142.0','z_subj(health).144.0','z_subj(health).146.0','z_subj(taste).1.0','z_subj(taste).20.0','z_subj(taste).22.0','z_subj(taste).37.0','z_subj(taste).46.0','z_subj(taste).47.0',
          'z_subj(taste).48.0','z_subj(taste).49.0','z_subj(taste).50.0','z_subj(taste).51.0','z_subj(taste).53.0','z_subj(taste).55.0','z_subj(taste).57.0','z_subj(taste).58.0','z_subj(taste).60.0','z_subj(taste).61.0','z_subj(taste).63.0','z_subj(taste).65.0','z_subj(taste).91.0','z_subj(taste).95.0','z_subj(taste).99.0','z_subj(taste).103.0','z_subj(taste).104.0','z_subj(taste).106.0',
          'z_subj(taste).111.0','z_subj(taste).121.0','z_subj(taste).122.0','z_subj(taste).123.0','z_subj(taste).127.0','z_subj(taste).128.0','z_subj(taste).130.0','z_subj(taste).132.0','z_subj(taste).138.0','z_subj(taste).139.0']
v_list_rep = ['v(health)','v(taste)','v_std','v_subj(health).4.0','v_subj(health).6.0','v_subj(health).8.0','v_subj(health).11.0','v_subj(health).13.0','v_subj(health).15.0',
              'v_subj(health).17.0','v_subj(health).21.0','v_subj(health).23.0','v_subj(health).25.0','v_subj(health).28.0','v_subj(health).30.0',
              'v_subj(health).32.0','v_subj(health).34.0','v_subj(health).36.0','v_subj(health).38.0','v_subj(health).40.0','v_subj(health).42.0',
              'v_subj(taste).1.0','v_subj(taste).3.0','v_subj(taste).5.0','v_subj(taste).7.0','v_subj(taste).9.0','v_subj(taste).10.0','v_subj(taste).12.0',
              'v_subj(taste).14.0','v_subj(taste).16.0','v_subj(taste).18.0','v_subj(taste).22.0','v_subj(taste).24.0','v_subj(taste).26.0','v_subj(taste).27.0',
              'v_subj(taste).29.0','v_subj(taste).31.0','v_subj(taste).35.0','v_subj(taste).37.0','v_subj(taste).39.0','v_subj(taste).41.0']
z_list_rep = ['z(health)','z(taste)','z_std','z_subj(health).4.0','z_subj(health).6.0','z_subj(health).8.0','z_subj(health).11.0','z_subj(health).13.0','z_subj(health).15.0',
              'z_subj(health).17.0','z_subj(health).21.0','z_subj(health).23.0','z_subj(health).25.0','z_subj(health).28.0','z_subj(health).30.0',
              'z_subj(health).32.0','z_subj(health).34.0','z_subj(health).36.0','z_subj(health).38.0','z_subj(health).40.0','z_subj(health).42.0',
              'z_subj(taste).1.0','z_subj(taste).3.0','z_subj(taste).5.0','z_subj(taste).7.0','z_subj(taste).9.0','z_subj(taste).10.0','z_subj(taste).12.0',
              'z_subj(taste).14.0','z_subj(taste).16.0','z_subj(taste).18.0','z_subj(taste).22.0','z_subj(taste).24.0','z_subj(taste).26.0','z_subj(taste).27.0',
              'z_subj(taste).29.0','z_subj(taste).31.0','z_subj(taste).35.0','z_subj(taste).37.0','z_subj(taste).39.0','z_subj(taste).41.0']

# now let's save each trace to eventually throw into R
for item in v_list:
    tmp = full_model.nodes_db.node[item]
    np.savetxt((item + '.txt'),tmp.trace(),delimiter=',')

for item in v_list_rep:
    tmp = full_model_rep.nodes_db.node[item]
    np.savetxt((item + '.rep.txt'),tmp.trace(),delimiter=',')
    
for item in z_list:
    tmp = full_model.nodes_db.node[item]
    np.savetxt((item + '.txt'),tmp.trace(),delimiter=',')

for item in z_list_rep:
    tmp = full_model_rep.nodes_db.node[item]
    np.savetxt((item + '.rep.txt'),tmp.trace(),delimiter=',')

#tmp = full_model_rep.nodes_db.node[[v_list]]



# Which model is better for primary and replication samples? Bias or drift rate?

full_model_primeDrift = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['Condition','tasteAdvantage','healthAdvantage']})
full_model_primeDrift.find_starting_values()
full_model_primeDrift.sample(n_samples, burn=n_burns)

full_model_primeDrift_rep = hddm.HDDM(rep_df, p_outlier = 0.05, depends_on={'v': ['Condition','tasteAdvantage','healthAdvantage']})
full_model_primeDrift_rep.find_starting_values()
full_model_primeDrift_rep.sample(n_samples, burn=n_burns)

full_model_primeBias = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['tasteAdvantage','healthAdvantage'], 'z':['Condition']}, bias = True)
full_model_primeBias.find_starting_values()
full_model_primeBias.sample(n_samples, burn=n_burns)

full_model_primeBias_rep = hddm.HDDM(rep_df, p_outlier = 0.05, depends_on={'v': ['tasteAdvantage','healthAdvantage'], 'z':['Condition']}, bias = True)
full_model_primeBias_rep.find_starting_values()
full_model_primeBias_rep.sample(n_samples, burn=n_burns)



############## Winkle's Poop

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Split by whether the last fixated option was chosen or not
#lastFixChosen_m = hddm.HDDM(df_lastFixChosen, p_outlier = 0.05, depends_on={'v': ['Condition'], 'z':['Condition'], 't':['Condition'], 'a':['Condition']}, bias = True)
#lastFixChosen_m.find_starting_values()
#lastFixChosen_m.sample(5000,burn=300)
#
#t_taste, t_health, t_control = lastFixChosen_m.nodes_db.node[[ "t(taste)", "t(health)", "t(control)"]]
#a_taste, a_health, a_control = lastFixChosen_m.nodes_db.node[[ "a(taste)", "a(health)", "a(control)"]]
#v_taste, v_health, v_control = lastFixChosen_m.nodes_db.node[[ "v(taste)", "v(health)", "v(control)"]]
#z_taste, z_health, z_control = lastFixChosen_m.nodes_db.node[[ "z(taste)", "z(health)", "z(control)"]]
#hddm.analyze.plot_posterior_nodes([t_taste, t_health, t_control])
#hddm.analyze.plot_posterior_nodes([a_taste, a_health, a_control])
#hddm.analyze.plot_posterior_nodes([v_taste, v_health, v_control])
#hddm.analyze.plot_posterior_nodes([z_taste, z_health, z_control])


## 0. Base model
#basemodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate']}, include = 'all', bias = True)
#basemodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate']}, bias = True)
#basemodel.find_starting_values()
#basemodel.sample(n_samples, burn=n_burns, db='sqlite', dbname = 'model.db') #dbname='traces.db', db='pickle') 
#basemodel.sample(5000, burn=300)
##basemodel.save(osp.join(save_dir,'base_vOnWant_n2000b400'))
#
#
### 1. Does the prime affect drift rate?
#vmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate','TastePrime','HealthPrime','ControlPrime']}, include = 'all', bias = True)
#vmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate','TastePrime','HealthPrime','ControlPrime']}, bias = True)
#vmodel.find_starting_values()
#vmodel.sample(n_samples, burn=n_burns, db='sqlite', dbname = 'model.db')
#vmodel.sample(5000, burn=300)
##model.save(osp.join(save_dir,'v_onPrime_n5000b1000'))
#
#v_taste_n3,v_taste_n2,v_taste_n1,v_taste_0,v_taste_p1,v_taste_p2, v_taste_p3 = vmodel.nodes_db.node[[ "v(0.0.0.0.1.0.-3.0)", "v(0.0.0.0.1.0.-2.0)", "v(0.0.0.0.1.0.-1.0)","v(0.0.0.0.1.0.0.0)","v(0.0.0.0.1.0.1.0)","v(0.0.0.0.1.0.2.0)","v(0.0.0.0.1.0.3.0)"]]
#hddm.analyze.plot_posterior_nodes([v_taste_n3,v_taste_n2,v_taste_n1,v_taste_0,v_taste_p1,v_taste_p2, v_taste_p3])
#
#v_health_n3,v_health_n2,v_health_n1,v_health_0,v_health_p1,v_health_p2, v_health_p3 = vmodel.nodes_db.node[[ "v(0.0.1.0.0.0.-3.0)", "v(0.0.1.0.0.0.-2.0)", "v(0.0.1.0.0.0.-1.0)","v(0.0.1.0.0.0.0.0)","v(0.0.1.0.0.0.1.0)","v(0.0.1.0.0.0.2.0)","v(0.0.1.0.0.0.3.0)"]]
#hddm.analyze.plot_posterior_nodes([v_health_n3,v_health_n2,v_health_n1,v_health_0,v_health_p1,v_health_p2, v_health_p3])
#
#v_control_n3,v_control_n2,v_control_n1,v_control_0,v_control_p1,v_control_p2, v_control_p3 = vmodel.nodes_db.node[[ "v(1.0.0.0.0.0.-3.0)", "v(1.0.0.0.0.0.-2.0)", "v(1.0.0.0.0.0.-1.0)","v(1.0.0.0.0.0.0.0)","v(1.0.0.0.0.0.1.0)","v(1.0.0.0.0.0.2.0)","v(1.0.0.0.0.0.3.0)"]]
#hddm.analyze.plot_posterior_nodes([v_control_n3,v_control_n2,v_control_n1,v_control_0,v_control_p1,v_control_p2, v_control_p3])
#
#
### 2. Does the prime affect non-decision time?
#tmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 't':['TastePrime','HealthPrime','ControlPrime']}, include = 'all', bias = True)
#tmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 't':['TastePrime','HealthPrime','ControlPrime']}, bias = True)
#tmodel.find_starting_values()
#tmodel.sample(n_samples, burn=n_burns, db='sqlite', dbname = 'model.db')
#tmodel.sample(5000, burn=300)
##model.save(osp.join(save_dir,'t_onPrime_n5000b1000'))
#
#t_taste, t_health, t_control = tmodel.nodes_db.node[[ "t(1.0.0)", "t(0.1.0)", "t(0.0.1)"]]
#hddm.analyze.plot_posterior_nodes([t_taste, t_health, t_control])
#
#
#
### 3. Does the prime affect starting bias?
#zmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 'z':['TastePrime','HealthPrime','ControlPrime']}, include = 'all', bias = True)
#zmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 'z':['TastePrime','HealthPrime','ControlPrime']}, bias = True)
#zmodel.find_starting_values()
#zmodel.sample(n_samples, burn=n_burns, db='sqlite', dbname = 'model.db')
#zmodel.sample(5000, burn=300)
##model.save(osp.join(save_dir,'z_onPrime_n5000b1000'))
#
#z_taste, z_health, z_control = zmodel.nodes_db.node[[ "z(1.0.0)", "z(0.1.0)", "z(0.0.1)"]]
#hddm.analyze.plot_posterior_nodes([z_taste, z_health, z_control])
#print "P(Health > Taste) = ", (z_health.trace() > z_taste.trace()).mean()
#print "P(Health > Control) = ", (z_health.trace() > z_control.trace()).mean()
#
#
#
### 4. Does prime affect non-decision time and starting bias simultaneously?
#tzmodel = hddm.HDDM(df, p_outlier = 0.05, depends_on={'v': ['d_WantRate'], 'z':['TastePrime','HealthPrime','ControlPrime'], 't':['TastePrime','HealthPrime','ControlPrime']}, bias = True)
#tzmodel.find_starting_values()
#tzmodel.sample(5000,burn=300)
#
#tz_z_taste, tz_z_health, tz_z_control = zmodel.nodes_db.node[[ "z(0.0.1)", "z(0.1.0)", "z(1.0.0)"]]
#hddm.analyze.plot_posterior_nodes([tz_z_taste, tz_z_health, tz_z_control])
#
#tz_t_taste, tz_t_health, tz_t_control = tzmodel.nodes_db.node[["t(0.0.1)", "t(0.1.0)", "t(1.0.0)"]]
#hddm.analyze.plot_posterior_nodes([tz_t_taste, tz_t_health, tz_t_control])






#model.plot_posteriors(['a', 't', 'v', 'a_std'])
#v_taste, v_health, v_control = model_v.nodes_db.node[[ "v(0.0.0.0.1.0.3.0)", "v(0.0.1.0.0.0.3.0)", "v(1.0.0.0.0.0.3.0)"]]
#hddm.analyze.plot_posterior_nodes ([v_taste, v_health, v_control])
#plt.title('Effect of Prime on Drift Rate')
#plt.savefig('PrimeOnv.pdf')
#
#z_data = hddm.utils.post_pred_gen(model)
#v_data = hddm.utils.post_pred_gen(model_v)