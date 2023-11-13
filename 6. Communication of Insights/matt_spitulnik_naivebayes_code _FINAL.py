# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:12:21 2023

@author: spitum1
"""
#import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statistics import mean
import random as rd
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
import numpy as np
import re
import inflect
wordPlur=inflect.engine()
import gensim
from gensim.parsing.preprocessing import remove_stopwords
gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

home_dir='C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project'

#import the data
orgDataDF=pd.read_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/players_with_busts3.csv')

#pull out only the base columns/data I'll need
baseDataDF=orgDataDF[['PERSON_ID','First Name','Last Name','Suffix1','Suffix2','Strengths','Weaknesses','Bust','Bust2','Bust3','Bust4']]

#create the dataframes that are just strengths (str), just weaknesses (saw), or both strengths and weaknesses (saw)
baseDataDF_str_pre=baseDataDF[(baseDataDF['Strengths'].isna()==False)]
baseDataDF_str_pre=baseDataDF_str_pre[['Strengths','Bust','Bust2','Bust3','Bust4']]
baseDataDF_str_pre=baseDataDF_str_pre.rename(columns={'Strengths':'text', 'Bust': 'Bust1'})
#bust 1 df
baseDataDF_str_pre_1=baseDataDF_str_pre[['text','Bust1']]
baseDataDF_str_pre_1=baseDataDF_str_pre_1[(baseDataDF_str_pre_1['Bust1']=='N')|(baseDataDF_str_pre_1['Bust1']=='Y')]
#bust 2 df
baseDataDF_str_pre_2=baseDataDF_str_pre[['text','Bust2']]
baseDataDF_str_pre_2=baseDataDF_str_pre_2[(baseDataDF_str_pre_2['Bust2']=='N')|(baseDataDF_str_pre_2['Bust2']=='Y')]
#bust 3 df
baseDataDF_str_pre_3=baseDataDF_str_pre[['text','Bust3']]
baseDataDF_str_pre_3=baseDataDF_str_pre_3[(baseDataDF_str_pre_3['Bust3']=='N')|(baseDataDF_str_pre_3['Bust3']=='Y')]
#bust 4 df
baseDataDF_str_pre_4=baseDataDF_str_pre[['text','Bust4']]
baseDataDF_str_pre_4=baseDataDF_str_pre_4[(baseDataDF_str_pre_4['Bust4']=='N')|(baseDataDF_str_pre_4['Bust4']=='Y')]

baseDataDF_wkn_pre=baseDataDF[(baseDataDF['Weaknesses'].isna()==False)]
baseDataDF_wkn_pre=baseDataDF_wkn_pre[['Weaknesses','Bust','Bust2','Bust3','Bust4']]
baseDataDF_wkn_pre=baseDataDF_wkn_pre.rename(columns={'Weaknesses':'text', 'Bust': 'Bust1'})
#bust 1 df
baseDataDF_wkn_pre_1=baseDataDF_wkn_pre[['text','Bust1']]
baseDataDF_wkn_pre_1=baseDataDF_wkn_pre_1[(baseDataDF_wkn_pre_1['Bust1']=='N')|(baseDataDF_wkn_pre_1['Bust1']=='Y')]
#bust 2 df
baseDataDF_wkn_pre_2=baseDataDF_wkn_pre[['text','Bust2']]
baseDataDF_wkn_pre_2=baseDataDF_wkn_pre_2[(baseDataDF_wkn_pre_2['Bust2']=='N')|(baseDataDF_wkn_pre_2['Bust2']=='Y')]
#bust 3 df
baseDataDF_wkn_pre_3=baseDataDF_wkn_pre[['text','Bust3']]
baseDataDF_wkn_pre_3=baseDataDF_wkn_pre_3[(baseDataDF_wkn_pre_3['Bust3']=='N')|(baseDataDF_wkn_pre_3['Bust3']=='Y')]
#bust 4 df
baseDataDF_wkn_pre_4=baseDataDF_wkn_pre[['text','Bust4']]
baseDataDF_wkn_pre_4=baseDataDF_wkn_pre_4[(baseDataDF_wkn_pre_4['Bust4']=='N')|(baseDataDF_wkn_pre_4['Bust4']=='Y')]

#need to create the saw DF in a way that will preserve the original indexes for later cleaning
baseDataDF_saw_pre=baseDataDF.drop(i for i in baseDataDF.index if pd.isna(baseDataDF.loc[i,'Strengths']))
for i in baseDataDF_saw_pre.index:
    if pd.isna(baseDataDF_saw_pre.loc[i,'Weaknesses']):
        baseDataDF_saw_pre.loc[i,'text']=baseDataDF_saw_pre.loc[i,'Strengths']
    else:
        baseDataDF_saw_pre.loc[i,'text']=baseDataDF_saw_pre.loc[i,'Strengths']+baseDataDF_saw_pre.loc[i,'Weaknesses']

baseDataDF_saw_pre=baseDataDF_saw_pre[['text','Bust','Bust2','Bust3','Bust4']]
baseDataDF_saw_pre=baseDataDF_saw_pre.rename(columns={'Bust': 'Bust1'})
#bust 1 df
baseDataDF_saw_pre_1=baseDataDF_saw_pre[['text','Bust1']]
baseDataDF_saw_pre_1=baseDataDF_saw_pre_1[(baseDataDF_saw_pre_1['Bust1']=='N')|(baseDataDF_saw_pre_1['Bust1']=='Y')]
#bust 2 df
baseDataDF_saw_pre_2=baseDataDF_saw_pre[['text','Bust2']]
baseDataDF_saw_pre_2=baseDataDF_saw_pre_2[(baseDataDF_saw_pre_2['Bust2']=='N')|(baseDataDF_saw_pre_2['Bust2']=='Y')]
#bust 3 df
baseDataDF_saw_pre_3=baseDataDF_saw_pre[['text','Bust3']]
baseDataDF_saw_pre_3=baseDataDF_saw_pre_3[(baseDataDF_saw_pre_3['Bust3']=='N')|(baseDataDF_saw_pre_3['Bust3']=='Y')]
#bust 4 df
baseDataDF_saw_pre_4=baseDataDF_saw_pre[['text','Bust4']]
baseDataDF_saw_pre_4=baseDataDF_saw_pre_4[(baseDataDF_saw_pre_4['Bust4']=='N')|(baseDataDF_saw_pre_4['Bust4']=='Y')]

df_list=['baseDataDF_str_pre_1','baseDataDF_str_pre_2','baseDataDF_str_pre_3','baseDataDF_str_pre_4','baseDataDF_wkn_pre_1','baseDataDF_wkn_pre_2','baseDataDF_wkn_pre_3','baseDataDF_wkn_pre_4','baseDataDF_saw_pre_1','baseDataDF_saw_pre_2','baseDataDF_saw_pre_3','baseDataDF_saw_pre_4']

def run_models(DF_list,vect,NB,cross):
    tempDFInfo=pd.DataFrame()
    for DF in DF_list:
        #get last 7 of the dataframe used for later naming convention
        L7=DF[len(DF)-9:len(DF)-2]
        L3=DF[len(DF)-5:len(DF)-2]
        L1=DF[len(DF)-1:]
        #establish if the vectorizer will be binary or not
        if NB=='BN':
            use_binary=True
        else:
            use_binary=False
        #establish the type of vectorizer to use
        if vect=='TF':
            vect_tool=TfidfVectorizer(input='content',binary=use_binary)
        else:
            vect_tool=CountVectorizer(input='content',binary=use_binary)
        #create the DTM
        vect_DTM=vect_tool.fit_transform(eval(DF)['text'])
        #get the feature list
        ColNames=vect_tool.get_feature_names_out()
        #create the dataframe
        globals()[str((f'{vect}_{L7}_{NB}_DF_{L1}'))]=pd.DataFrame(vect_DTM.toarray(),columns=ColNames)
        
        #establish the type of Naive Bayes model to use
        if NB=='BN':
            NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        else:
            NB_model=MultinomialNB()
        #now use cross validation to get the accuracies and build a DF listing them all
        tempScores=cross_val_score(NB_model,eval(str((f'{vect}_{L7}_{NB}_DF_{L1}'))),eval(DF).iloc[:,1],cv=cross)
        tempModName=str(f'{vect}_{L7}_{NB}_{L1}')
        tempMeanScore=mean(tempScores)
        TimeForDF=[tempModName,tempMeanScore]
        TimeForDF.extend(tempScores)
        TempScoreInfo = pd.DataFrame([TimeForDF])
        TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
        for c in range(0,len(tempScores)):
            TempScoreInfo=TempScoreInfo.rename(columns={c+2:c+1})
        CombFreqInfo=[tempDFInfo,TempScoreInfo]
        tempDFInfo=pd.concat(CombFreqInfo)    
    globals()[str((f'{vect}_{L3}_{NB}_acc'))]=tempDFInfo.reset_index(drop=True)
    return

run_models(df_list,'CV','MN',10)
run_models(df_list,'CV','BN',10)
run_models(df_list,'TF','MN',10)
run_models(df_list,'TF','BN',10)

'''######This version of the cleaning function took too long.
#create function for cleaning the data
def clean_DFs(DF_list):
    global df_list_cle
    df_list_cle=[]
    for DF in DF_list:
        tempDF_lowcon=eval(DF).copy(deep=True)
        for i in tempDF_lowcon.index:
            tempDF_lowcon.loc[i,'text']=contractions.fix(tempDF_lowcon.loc[i,'text'].lower())
        #lemmetize and make everything singular
        tempDF_lemsing=tempDF_lowcon.copy(deep=True)
        for i in tempDF_lemsing.index:
            tempRev=nltk.word_tokenize(tempDF_lemsing.loc[i,'text'])
            tempRevTags=nltk.pos_tag(tempRev)
            tempRevNew=[]
            for ct in tempRevTags:
                tagList=re.match(r'V',ct[1]) or re.match(r'JJ',ct[1])
                if tagList:
                    tempRevNew.append(lem.lemmatize(ct[0],'v'))
                else:
                    tempRevNew.append(ct[0])
            for p in range(0,len(tempRevNew)):
                if wordPlur.singular_noun(tempRevNew[p]) == False:
                    continue
                else:
                    tempRevNew[p]=wordPlur.singular_noun(tempRevNew[p])
            tempDF_lemsing.loc[i,'text']=' '.join(tempRevNew[z] for z in range(0,len(tempRevNew)))
        #remove gensim stop words (sw)
        tempDF_sw=tempDF_lemsing.copy(deep=True)
        for i in tempDF_sw.index:
        	tempDF_sw.loc[i,'text']=remove_stopwords(tempDF_sw.loc[i,'text'])
        #remove the players name if it appears in the review
        tempDF_pla=tempDF_sw.copy(deep=True)
        for i in tempDF_pla.index:
            if baseDataDF.loc[i,'First Name'] in tempDF_pla.loc[i,'text'] or baseDataDF.loc[i,'Last Name'] in tempDF_pla.loc[i,'text']:
                tempDF_pla.loc[i,'text']=re.sub(baseDataDF.loc[i,'Last Name'],'',tempDF_pla.loc[i,'text'])
                tempDF_pla.loc[i,'text']=re.sub(baseDataDF.loc[i,'First Name'],'',tempDF_pla.loc[i,'text'])  
        #remove words that contain numbers/other characters, are 3 characters or less, or more then 13 characters
        tempDF_num=tempDF_pla.copy(deep=True)
        for i in tempDF_num.index:
            tempRev=nltk.word_tokenize(tempDF_num.loc[i,'text'])
            tempRev_2=[]
            for h in tempRev:
                if not (re.search(r'[^A-Za-z]+', h)) and len(str(h))>3 and len(str(h))<=13:
                    tempRev_2.append(h)
            tempDF_num.loc[i,'text']=' '.join(tempRev_2[r] for r in range(0,len(tempRev_2)))
        L_3=DF[:len(DF)-5]
        L1=DF[len(DF)-1:]
        globals()[str((f'{L_3}cle_{L1}'))]=tempDF_num
        df_list_cle.append(str((f'{L_3}cle_{L1}')))
    return df_list_cle'''

#create (hopefully) more efficient function for cleaning the data
def clean_DFs(DF_list):
    global df_list_cle
    df_list_cle=[]
    for DF in DF_list:
        tempDF=eval(DF)
        for i in tempDF.index:
            #make everything lowercase and fix contractions
            tmpText=contractions.fix(tempDF.loc[i,'text'].lower())
            #remove the players name if it appears in the review
            tmpText=re.sub(baseDataDF.loc[i,'Last Name'].lower(),'',tmpText)
            tmpText=re.sub(baseDataDF.loc[i,'First Name'].lower(),'',tmpText)
            #remove words that contain numbers/other characters
            tempTxt=re.sub(r"[^A-Za-z\-]", " ", tmpText).split()
            #lemmetize and make everything singular
            tempRevTags=nltk.pos_tag(tempTxt)
            tempRevNew=[]
            for ct in tempRevTags:
                tagList=re.match(r'V',ct[1]) or re.match(r'JJ',ct[1])
                if tagList:
                    tempRevNew.append(lem.lemmatize(ct[0],'v'))
                else:
                    tempRevNew.append(ct[0])
            for p in range(0,len(tempRevNew)):
                if wordPlur.singular_noun(tempRevNew[p]) == False:
                    continue
                else:
                    tempRevNew[p]=wordPlur.singular_noun(tempRevNew[p])
            #remove stop words, or words that are 3 characters or less, or more then 13 characters
            tempTxtNew=[word for word in tempRevNew if word not in gensim_stopwords and len(word)>3 and len(word)<=13]
            tempDF.loc[i,'text']=' '.join(r for r in tempTxtNew)
        L_3=DF[:len(DF)-5]
        L1=DF[len(DF)-1:]
        globals()[str((f'{L_3}cle_{L1}'))]=tempDF
        df_list_cle.append(str((f'{L_3}cle_{L1}')))
    return df_list_cle
    
clean_DFs(df_list)

run_models(df_list_cle,'CV','MN',10)
run_models(df_list_cle,'CV','BN',10)
run_models(df_list_cle,'TF','MN',10)
run_models(df_list_cle,'TF','BN',10)

#now create a funciton that will be used to remove words that appear in too frequent of pre-draft analysis or not enough of it ######this didn't feel efficient enough either, so did the function below
'''def freq_removal(high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=eval(f'{h}_{i}_cle_{j}_DF_{num}')
                    for c in DF.columns:
                        if DF[c][DF[c]!=0].count()/len(DF) > high:
                            DF=DF.drop(c,axis=1)
                        elif DF[c][DF[c]!=0].count()/len(DF) < low:
                            DF=DF.drop(c,axis=1)    
                    globals()[str((f'{h}_{i}_fre_{j}_DF_{num}'))]=DF
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    tempScores=cross_val_score(NB_model,eval(str((f'{h}_{i}_fre_{j}_DF_{num}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'{h}_{i}_fre_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo)    
    globals()[str(('fre_acc'))]=tempDFInfo.reset_index(drop=True)
    return'''

def freq_removal(high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=eval(f'{h}_{i}_cle_{j}_DF_{num}')
                    hmark=len(DF)*high
                    lmark=len(DF)*low
                    tempDF=pd.DataFrame()
                    for c in DF.columns:
                        non_z_ct=(DF[c] != 0).sum()  
                        if non_z_ct > lmark and non_z_ct < hmark:
                            tempDF[c]=DF[c]

                    globals()[str((f'{h}_{i}_fre_{j}_DF_{num}'))]=tempDF
                        
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    tempScores=cross_val_score(NB_model,eval(str((f'{h}_{i}_fre_{j}_DF_{num}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'{h}_{i}_fre_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo)    
    globals()[str(('fre_acc'))]=tempDFInfo.reset_index(drop=True)
    return

freq_removal(.99,.01,10)

#now create a function that will remove correlated ######this didn't feel efficient enough either, so did the function below
'''def corr_removal(step, high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=eval(f'{h}_{i}_{step}_{j}_DF_{num}')
                    DF2=DF.copy(deep=True)
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    temp_DF['factor']=temp_DF[temp_DF.columns[1]].factorize()[0]
                    for c in DF.columns:
                        for co in DF. columns:
                            if c in DF2.columns and co in DF2.columns:
                                if c != co:
                                    if DF[c].corr(DF[co])>high:
                                        if DF[c].corr(temp_DF['factor']) > DF[co].corr(temp_DF['factor']):
                                            DF2=DF2.drop(co,axis=1)
                                        else:
                                            DF2=DF2.drop(c,axis=1)
                                    elif DF[c].corr(DF[co])<low:
                                        if DF[c].corr(temp_DF['factor']) < DF[co].corr(temp_DF['factor']):
                                            DF2=DF2.drop(co,axis=1)
                                        else:
                                            DF2=DF2.drop(c,axis=1)
                                
                    globals()[str((f'{h}_{i}_cor_{j}_DF_{num}'))]=DF2
                    
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    tempScores=cross_val_score(NB_model,eval(str((f'{h}_{i}_cor_{j}_DF_{num}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'{h}_{i}_cor_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo) 
    globals()[str((f'cor_{step}_acc'))]=tempDFInfo.reset_index(drop=True)
    return'''

def corr_removal(step, high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=eval(f'{h}_{i}_{step}_{j}_DF_{num}')
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    temp_DF['factor']=temp_DF[temp_DF.columns[1]].factorize()[0]
                    tmpList=[]
                    for c in DF.columns:
                        for co in DF. columns:
                            if c!= co:
                                if c not in tmpList and co not in tmpList:
                                    if DF[c].corr(DF[co])>high:
                                        if DF[c].corr(temp_DF['factor']) > DF[co].corr(temp_DF['factor']):
                                            tmpList.append(co)
                                        else:
                                            tmpList.append(c)
                                    elif DF[c].corr(DF[co])<low:
                                        if DF[c].corr(temp_DF['factor']) < DF[co].corr(temp_DF['factor']):
                                            tmpList.append(co)
                                        else:
                                            tmpList.append(c)
                    DF2=DF[DF.columns.difference(tmpList)]
                            
                                
                    globals()[str((f'{h}_{i}_cor_{j}_DF_{num}_no{step}'))]=DF2
                    
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    tempScores=cross_val_score(NB_model,eval(str((f'{h}_{i}_cor_{j}_DF_{num}_no{step}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'{h}_{i}_cor_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo) 
    globals()[str(('cor_acc'))]=tempDFInfo.reset_index(drop=True)
    return

#corr_removal('pre',.75,-.75,10) #this took a long time to finish, trying the other DFs
#corr_removal('cle',.75,-.75,10) #this took a long time to finish, trying the other DFs
corr_removal('fre',.75,-.75,10)
cor_acc.to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/cor_acc_fre.csv',index=False,header=True)

tempDFInfo=pd.DataFrame()
DF=eval('TF_saw_fre_BN_DF_1')
temp_DF=eval('baseDataDF_saw_pre_1')
temp_DF['factor']=temp_DF[temp_DF.columns[1]].factorize()[0]
tmpList=[]
for c in DF.columns:
    for co in DF. columns:
        if c!= co:
            if c not in tmpList and co not in tmpList:
                if DF[c].corr(DF[co])>high:
                    if DF[c].corr(temp_DF['factor']) > DF[co].corr(temp_DF['factor']):
                        tmpList.append(co)
                    else:
                        tmpList.append(c)
                elif DF[c].corr(DF[co])<low:
                    if DF[c].corr(temp_DF['factor']) < DF[co].corr(temp_DF['factor']):
                        tmpList.append(co)
                    else:
                        tmpList.append(c)
DF2=DF[DF.columns.difference(tmpList)]
                

##################################
######Now try this work on bigrams
##################################
def run_models_bi(DF_list,vect,NB,cross):
    tempDFInfo=pd.DataFrame()
    for DF in DF_list:
        #get last 7 of the dataframe used for later naming convention
        L7=DF[len(DF)-9:len(DF)-2]
        L3=DF[len(DF)-5:len(DF)-2]
        L1=DF[len(DF)-1:]
        #establish if the vectorizer will be binary or not
        if NB=='BN':
            use_binary=True
        else:
            use_binary=False
        #establish the type of vectorizer to use
        if vect=='TF':
            vect_tool=TfidfVectorizer(input='content',binary=use_binary,ngram_range=(2, 2))
        else:
            vect_tool=CountVectorizer(input='content',binary=use_binary,ngram_range=(2, 2))
        #create the DTM
        vect_DTM=vect_tool.fit_transform(eval(DF)['text'])
        #get the feature list
        ColNames=vect_tool.get_feature_names_out()
        #create the dataframe
        globals()[str((f'BI_{vect}_{L7}_{NB}_DF_{L1}'))]=pd.DataFrame(vect_DTM.toarray(),columns=ColNames)
        
        #establish the type of Naive Bayes model to use
        if NB=='BN':
            NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        else:
            NB_model=MultinomialNB()
        #now use cross validation to get the accuracies and build a DF listing them all
        tempScores=cross_val_score(NB_model,eval(str((f'BI_{vect}_{L7}_{NB}_DF_{L1}'))),eval(DF).iloc[:,1],cv=cross)
        tempModName=str(f'BI_{vect}_{L7}_{NB}_{L1}')
        tempMeanScore=mean(tempScores)
        TimeForDF=[tempModName,tempMeanScore]
        TimeForDF.extend(tempScores)
        TempScoreInfo = pd.DataFrame([TimeForDF])
        TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
        for c in range(0,len(tempScores)):
            TempScoreInfo=TempScoreInfo.rename(columns={c+2:c+1})
        CombFreqInfo=[tempDFInfo,TempScoreInfo]
        tempDFInfo=pd.concat(CombFreqInfo)
        
        eval(str((f'BI_{vect}_{L7}_{NB}_DF_{L1}'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{vect}_{L7}_{NB}_DF_{L1}'))+'.csv',index=False,header=True)
        del globals()[str((f'BI_{vect}_{L7}_{NB}_DF_{L1}'))]
    globals()[str((f'BI_{vect}_{L3}_{NB}_acc'))]=tempDFInfo.reset_index(drop=True)
    
    eval(str((f'BI_{vect}_{L3}_{NB}_acc'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{vect}_{L3}_{NB}_acc'))+'.csv',index=False,header=True)
    return

run_models_bi(df_list,'CV','MN',10)
run_models_bi(df_list,'CV','BN',10)
run_models_bi(df_list,'TF','MN',10)
run_models_bi(df_list,'TF','BN',10)

run_models_bi(df_list_cle,'CV','MN',10)
run_models_bi(df_list_cle,'CV','BN',10)
run_models_bi(df_list_cle,'TF','MN',10)
run_models_bi(df_list_cle,'TF','BN',10)

#create the function for removing frequent and not frequent words with bigrams instead
def freq_removal_bi(high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=pd.read_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{h}_{i}_cle_{j}_DF_{num}'))+'.csv')
                    hmark=len(DF)*high
                    lmark=len(DF)*low
                    tempDF=pd.DataFrame()
                    for c in DF.columns:
                        non_z_ct=(DF[c] != 0).sum()  
                        if non_z_ct > lmark and non_z_ct < hmark:
                            tempDF[c]=DF[c]
                        
                    globals()[str((f'BI_{h}_{i}_fre_{j}_DF_{num}'))]=tempDF
                        
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    tempScores=cross_val_score(NB_model,eval(str((f'BI_{h}_{i}_fre_{j}_DF_{num}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'BI_{h}_{i}_fre_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo)
                    
                    eval(str((f'BI_{h}_{i}_fre_{j}_DF_{num}'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{h}_{i}_fre_{j}_DF_{num}'))+'.csv',index=False,header=True)
                    del globals()[str((f'BI_{h}_{i}_fre_{j}_DF_{num}'))]
    globals()[str(('BI_fre_acc'))]=tempDFInfo.reset_index(drop=True)
    eval(str(('BI_fre_acc'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str(('BI_fre_acc'))+'.csv',index=False,header=True)
    return

freq_removal_bi(.99,.01,10)

#bigram correlation removal
def corr_removal_bi(step, high, low, cross):
    tempDFInfo=pd.DataFrame()
    for i in ['str','wkn','saw']:
        for h in ['CV','TF']:
            for j in ['MN','BN']:
                for num in range(1,5):
                    DF=pd.read_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{h}_{i}_{step}_{j}_DF_{num}'))+'.csv')
                    DF2=DF.copy(deep=True)
                    temp_DF=eval(f'baseDataDF_{i}_pre_{num}')
                    temp_DF['factor']=temp_DF[temp_DF.columns[1]].factorize()[0]
                    tmpList=[]
                    for c in DF.columns:
                        for co in DF. columns:
                            if c!= co:
                                if c not in tmpList and co not in tmpList:
                                    if DF[c].corr(DF[co])>high:
                                        if DF[c].corr(temp_DF['factor']) > DF[co].corr(temp_DF['factor']):
                                            tmpList.append(co)
                                        else:
                                            tmpList.append(c)
                                    elif DF[c].corr(DF[co])<low:
                                        if DF[c].corr(temp_DF['factor']) < DF[co].corr(temp_DF['factor']):
                                            tmpList.append(co)
                                        else:
                                            tmpList.append(c)
                                            
                    DF2=DF[DF.columns.difference(tmpList)]
                    globals()[str((f'BI_{h}_{i}_cor_{j}_DF_{num}_no{step}'))]=DF2
                    
                    #establish the type of Naive Bayes model to use
                    if j=='BN':
                        NB_model=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
                    else:
                        NB_model=MultinomialNB()
                    #now use cross validation to get the accuracies and build a DF listing them all
                    tempScores=cross_val_score(NB_model,eval(str((f'BI_{h}_{i}_cor_{j}_DF_{num}_no{step}'))),temp_DF[temp_DF.columns[1]],cv=cross)
                    tempModName=str(f'BI_{h}_{i}_cor_{j}_{num}')
                    tempMeanScore=mean(tempScores)
                    TimeForDF=[tempModName,tempMeanScore]
                    TimeForDF.extend(tempScores)
                    TempScoreInfo = pd.DataFrame([TimeForDF])
                    TempScoreInfo=TempScoreInfo.rename(columns={0:'Model',1:'MeanScore'})
                    for cl in range(0,len(tempScores)):
                        TempScoreInfo=TempScoreInfo.rename(columns={cl+2:cl+1})
                    CombFreqInfo=[tempDFInfo,TempScoreInfo]
                    tempDFInfo=pd.concat(CombFreqInfo) 
    
                    eval(str((f'BI_{h}_{i}_cor_{j}_DF_{num}_no{step}'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str((f'BI_{h}_{i}_cor_{j}_DF_{num}_no{step}'))+'.csv',index=False,header=True)
                    del globals()[str((f'BI_{h}_{i}_cor_{j}_DF_{num}_no{step}'))]
    globals()[str(('BI_cor_acc'))]=tempDFInfo.reset_index(drop=True)
    eval(str(('BI_cor_acc'))).to_csv(r'C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+str(('BI_cor_acc'))+'.csv',index=False,header=True)
    return

#corr_removal_bi('pre',.75,-.75,10) #this took a long time to finish, trying the other DFs
#corr_removal_bi('cle',.75,-.75,10) #this took a long time to finish, trying the other DFs
corr_removal_bi('fre',.75,-.75,10)

##########################################
#####Now Viz Work#########################
##########################################

###create a bar graph that shows how the number of columns changeed in each step for each label and str/wkn/saw
col_saw_1=[len(CV_saw_pre_BN_DF_1.columns),len(CV_saw_cle_BN_DF_1.columns),len(CV_saw_fre_BN_DF_1.columns),len(CV_saw_cor_BN_DF_1_nofre.columns)]
col_str_1=[len(CV_str_pre_BN_DF_1.columns),len(CV_str_cle_BN_DF_1.columns),len(CV_str_fre_BN_DF_1.columns),len(CV_str_cor_BN_DF_1_nofre.columns)]
col_wkn_1=[len(CV_wkn_pre_BN_DF_1.columns),len(CV_wkn_cle_BN_DF_1.columns),len(CV_wkn_fre_BN_DF_1.columns),len(CV_wkn_cor_BN_DF_1_nofre.columns)]

col_saw_2=[len(CV_saw_pre_BN_DF_2.columns),len(CV_saw_cle_BN_DF_2.columns),len(CV_saw_fre_BN_DF_2.columns),len(CV_saw_cor_BN_DF_2_nofre.columns)]
col_str_2=[len(CV_str_pre_BN_DF_2.columns),len(CV_str_cle_BN_DF_2.columns),len(CV_str_fre_BN_DF_2.columns),len(CV_str_cor_BN_DF_2_nofre.columns)]
col_wkn_2=[len(CV_wkn_pre_BN_DF_2.columns),len(CV_wkn_cle_BN_DF_2.columns),len(CV_wkn_fre_BN_DF_2.columns),len(CV_wkn_cor_BN_DF_2_nofre.columns)]

col_saw_3=[len(CV_saw_pre_BN_DF_3.columns),len(CV_saw_cle_BN_DF_3.columns),len(CV_saw_fre_BN_DF_3.columns),len(CV_saw_cor_BN_DF_3_nofre.columns)]
col_str_3=[len(CV_str_pre_BN_DF_3.columns),len(CV_str_cle_BN_DF_3.columns),len(CV_str_fre_BN_DF_3.columns),len(CV_str_cor_BN_DF_3_nofre.columns)]
col_wkn_3=[len(CV_wkn_pre_BN_DF_3.columns),len(CV_wkn_cle_BN_DF_3.columns),len(CV_wkn_fre_BN_DF_3.columns),len(CV_wkn_cor_BN_DF_3_nofre.columns)]

col_saw_4=[len(CV_saw_pre_BN_DF_4.columns),len(CV_saw_cle_BN_DF_4.columns),len(CV_saw_fre_BN_DF_4.columns),len(CV_saw_cor_BN_DF_4_nofre.columns)]
col_str_4=[len(CV_str_pre_BN_DF_4.columns),len(CV_str_cle_BN_DF_4.columns),len(CV_str_fre_BN_DF_4.columns),len(CV_str_cor_BN_DF_4_nofre.columns)]
col_wkn_4=[len(CV_wkn_pre_BN_DF_4.columns),len(CV_wkn_cle_BN_DF_4.columns),len(CV_wkn_fre_BN_DF_4.columns),len(CV_wkn_cor_BN_DF_4_nofre.columns)]
step_labels=['Pre_Clean','Post_Clean','Freq_Removal','Cor_Removal']
bust_labels=['bust 1','bust 2','bust 3', 'bust 4' ,'bust 1','bust 2','bust 3', 'bust 4','bust 1','bust 2','bust 3', 'bust 4','bust 1','bust 2','bust 3', 'bust 4']
X_axis =np.arange(0,8,2)
x_labels=np.arange(-.75,7.25,.5)

fig, ax1 = plt.subplots(figsize=(20,15))
plt.xticks(x_labels, bust_labels,fontsize=16)
plt.yticks(fontsize=18)
plt.xlabel("Bust Label/Text Type/Step In Cleaning",fontsize=24)
plt.ylabel("Total Columns (# Of Words In Vocab)",fontsize=24)
saw_plot=plt.bar(X_axis - 0.75, col_saw_1, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis - 0.25, col_saw_2, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis + 0.25, col_saw_3, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis + 0.75, col_saw_4, 0.4, label = 'saw',edgecolor='black',color='b')

str_plot=plt.bar(X_axis - 0.85, col_str_1, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis - 0.35, col_str_2, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis + 0.35, col_str_3, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis + 0.85, col_str_4, 0.2, label = 'str',edgecolor='black',color='r')

wkn_plot=plt.bar(X_axis - 0.65, col_wkn_1, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis - 0.15, col_wkn_2, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis + 0.15, col_wkn_3, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis + 0.65, col_wkn_4, 0.2, label = 'wkn',edgecolor='black',color='y')

plt.axvspan(-1,1,facecolor='r',alpha=.06,zorder=-100)
plt.axvspan(1,3,facecolor='b',alpha=.06,zorder=-100)
plt.axvspan(3,5,facecolor='tab:orange',alpha=.06,zorder=-100)
plt.axvspan(5,7,facecolor='g',alpha=.06,zorder=-100)

for i in X_axis:
    plt.text(i-.75,col_saw_1[int(i/2)],round(col_saw_1[int(i/2)],2),fontsize=14,ha='center')
for i in X_axis:
    plt.text(i-.25,col_saw_2[int(i/2)],round(col_saw_2[int(i/2)],2),fontsize=14,ha='center')
for i in X_axis:
    plt.text(i+.25,col_saw_3[int(i/2)],round(col_saw_3[int(i/2)],2),fontsize=14,ha='center')
for i in X_axis:
    plt.text(i+.75,col_saw_4[int(i/2)],round(col_saw_4[int(i/2)],2),fontsize=14,ha='center')

for i in X_axis:
    plt.text(i-.85,col_str_1[int(i/2)]/2,round(col_str_1[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i-.35,col_str_2[int(i/2)]/2,round(col_str_2[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i+.35,col_str_3[int(i/2)]/2,round(col_str_3[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i+.85,col_str_4[int(i/2)]/2,round(col_str_4[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    
for i in X_axis:
    plt.text(i-.65,col_wkn_1[int(i/2)]/2,round(col_wkn_1[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i-.15,col_wkn_2[int(i/2)]/2,round(col_wkn_2[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i+.15,col_wkn_3[int(i/2)]/2,round(col_wkn_3[int(i/2)],2),fontsize=14,ha='center',rotation=90)
for i in X_axis:
    plt.text(i+.65,col_wkn_4[int(i/2)]/2,round(col_wkn_4[int(i/2)],2),fontsize=14,ha='center',rotation=90)

plt.legend(handles=[saw_plot,str_plot,wkn_plot],fontsize=20)
plt.title('Total # Of Columns Per NB Model Type', pad=20, fontsize=30)
ax2=ax1.twiny()
plt.xticks([.165,.37,.625,.85], step_labels,fontsize=18)
plt.savefig('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/images/all_column_counts.png')

#figure out how many columns were dropped between steps
colDif_DF=pd.DataFrame()
tmpIndex=0
for i in ['saw','str','wkn']:
    for h in ['1','2','3','4']:
        tmp_list=f'col_{i}_{h}'
        colDif_DF.loc[tmpIndex,'analysis']=tmp_list
        colDif_DF.loc[tmpIndex,'Raw Col Total']=eval(tmp_list)[0]
        colDif_DF.loc[tmpIndex,'Raw-Step1 Diff']=eval(tmp_list)[0]-eval(tmp_list)[1]           
        colDif_DF.loc[tmpIndex,'% Change1']=round((eval(tmp_list)[0]-eval(tmp_list)[1])/eval(tmp_list)[0],3)
        colDif_DF.loc[tmpIndex,'Step1 Total']=eval(tmp_list)[1]
        colDif_DF.loc[tmpIndex,'Step1-2 Diff']=eval(tmp_list)[1]-eval(tmp_list)[2]            
        colDif_DF.loc[tmpIndex,'% Change2']=round((eval(tmp_list)[1]-eval(tmp_list)[2])/eval(tmp_list)[1],3)
        colDif_DF.loc[tmpIndex,'Step2 Total']=eval(tmp_list)[2]
        colDif_DF.loc[tmpIndex,'Step2-3 Diff']=eval(tmp_list)[2]-eval(tmp_list)[3]            
        colDif_DF.loc[tmpIndex,'% Change3']=round((eval(tmp_list)[2]-eval(tmp_list)[3])/eval(tmp_list)[2],3)
        colDif_DF.loc[tmpIndex,'Final Total']=eval(tmp_list)[3]
        tmpIndex=tmpIndex+1

#Add the mean for each column at the end
colDif_DF.iloc[:,1:].groupby(['% Change1']).mean()
tmp_list=colDif_DF.iloc[:,1:].mean(axis=0)
colDif_DF.loc[12,'analysis']='Mean Change'
for i in range(0,10):
    colDif_DF.iloc[12,i+1]=round(tmp_list[i],3)
    
######now the bigrams
#since the bigram DFs were all exported automaticall to save space, need to import just the length of the columns
for n in ['1','2','3','4']:
    for i in ['saw','str','wkn']:
        for s in ['pre','cle','fre','cor']:
            tmpDF=f'BI_CV_{i}_{s}_BN_DF_{n}'
            if s=='cor':
                tmpDF=tmpDF+'_nofre'
            globals()[str((f'colnum_{i}_{s}_{n}'))]=len(pd.read_csv(f'{home_dir}/backup files/backup data/{tmpDF}.csv', nrows=0).columns)


BI_col_saw_1=[colnum_saw_pre_1,colnum_saw_cle_1,colnum_saw_fre_1,colnum_saw_cor_1]
BI_col_str_1=[colnum_str_pre_1,colnum_str_cle_1,colnum_str_fre_1,colnum_str_cor_1]
BI_col_wkn_1=[colnum_wkn_pre_1,colnum_wkn_cle_1,colnum_wkn_fre_1,colnum_wkn_cor_1]

BI_col_saw_2=[colnum_saw_pre_2,colnum_saw_cle_2,colnum_saw_fre_2,colnum_saw_cor_2]
BI_col_str_2=[colnum_str_pre_2,colnum_str_cle_2,colnum_str_fre_2,colnum_str_cor_2]
BI_col_wkn_2=[colnum_wkn_pre_2,colnum_wkn_cle_2,colnum_wkn_fre_2,colnum_wkn_cor_2]

BI_col_saw_3=[colnum_saw_pre_3,colnum_saw_cle_3,colnum_saw_fre_3,colnum_saw_cor_3]
BI_col_str_3=[colnum_str_pre_3,colnum_str_cle_3,colnum_str_fre_3,colnum_str_cor_3]
BI_col_wkn_3=[colnum_wkn_pre_3,colnum_wkn_cle_3,colnum_wkn_fre_3,colnum_wkn_cor_3]

BI_col_saw_4=[colnum_saw_pre_4,colnum_saw_cle_4,colnum_saw_fre_4,colnum_saw_cor_4]
BI_col_str_4=[colnum_str_pre_4,colnum_str_cle_4,colnum_str_fre_4,colnum_str_cor_4]
BI_col_wkn_4=[colnum_wkn_pre_4,colnum_wkn_cle_4,colnum_wkn_fre_4,colnum_wkn_cor_4]
step_labels=['Pre_Clean','Post_Clean','Freq_Removal','Cor_Removal']
bust_labels=['bust 1','bust 2','bust 3', 'bust 4' ,'bust 1','bust 2','bust 3', 'bust 4','bust 1','bust 2','bust 3', 'bust 4','bust 1','bust 2','bust 3', 'bust 4']
X_axis =np.arange(0,8,2)
x_labels=np.arange(-.75,7.25,.5)

fig, ax1 = plt.subplots(figsize=(20,15))
plt.xticks(x_labels, bust_labels,fontsize=16)
plt.yticks(fontsize=18)
plt.xlabel("Bust Label/Text Type/Step In Cleaning",fontsize=24)
plt.ylabel("Total Columns (# Of Words In Vocab)",fontsize=24)
saw_plot=plt.bar(X_axis - 0.75, BI_col_saw_1, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis - 0.25, BI_col_saw_2, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis + 0.25, BI_col_saw_3, 0.4, label = 'saw',edgecolor='black',color='b')
plt.bar(X_axis + 0.75, BI_col_saw_4, 0.4, label = 'saw',edgecolor='black',color='b')

str_plot=plt.bar(X_axis - 0.85, BI_col_str_1, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis - 0.35, BI_col_str_2, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis + 0.35, BI_col_str_3, 0.2, label = 'str',edgecolor='black',color='r')
plt.bar(X_axis + 0.85, BI_col_str_4, 0.2, label = 'str',edgecolor='black',color='r')

wkn_plot=plt.bar(X_axis - 0.65, BI_col_wkn_1, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis - 0.15, BI_col_wkn_2, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis + 0.15, BI_col_wkn_3, 0.2, label = 'wkn',edgecolor='black',color='y')
plt.bar(X_axis + 0.65, BI_col_wkn_4, 0.2, label = 'wkn',edgecolor='black',color='y')

plt.axvspan(-1,1,facecolor='r',alpha=.06,zorder=-100)
plt.axvspan(1,3,facecolor='b',alpha=.06,zorder=-100)
plt.axvspan(3,5,facecolor='tab:orange',alpha=.06,zorder=-100)
plt.axvspan(5,7,facecolor='g',alpha=.06,zorder=-100)

for i in X_axis:
    if i < 4:
        plt.text(i-.75,BI_col_saw_1[int(i/2)],round(BI_col_saw_1[int(i/2)],2),fontsize=14,ha='center')
    else:
        plt.text(i-.75,BI_col_saw_1[int(i/2)]*6,'saw:' +'\n' + str(round(BI_col_saw_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i < 4:
        plt.text(i-.25,BI_col_saw_2[int(i/2)],round(BI_col_saw_2[int(i/2)],2),fontsize=14,ha='center')
    else:
        plt.text(i-.25,BI_col_saw_1[int(i/2)]*6,'saw:' +'\n' + str(round(BI_col_saw_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i < 4:
        plt.text(i+.25,BI_col_saw_3[int(i/2)],round(BI_col_saw_3[int(i/2)],2),fontsize=14,ha='center')
    else:
        plt.text(i+.25,BI_col_saw_1[int(i/2)]*6,'saw:' +'\n' + str(round(BI_col_saw_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i < 4:
        plt.text(i+.75,BI_col_saw_4[int(i/2)],round(BI_col_saw_4[int(i/2)],2),fontsize=14,ha='center')
    else:
        plt.text(i+.75,BI_col_saw_1[int(i/2)]*6,'saw:' +'\n' + str(round(BI_col_saw_1[int(i/2)],2)),fontsize=14,ha='center')

for i in X_axis:
    if i < 4:
        plt.text(i-.85,BI_col_str_1[int(i/2)]/2,round(BI_col_str_1[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i-.75,BI_col_saw_1[int(i/2)]*3.75,'str:' +'\n' + str(round(BI_col_str_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i < 4:
        plt.text(i-.35,BI_col_str_2[int(i/2)]/2,round(BI_col_str_2[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i-.25,BI_col_saw_1[int(i/2)]*3.75,'str:' +'\n' + str(round(BI_col_str_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i <4:
        plt.text(i+.35,BI_col_str_3[int(i/2)]/2,round(BI_col_str_3[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i+.25,BI_col_saw_1[int(i/2)]*3.75,'str:' +'\n' + str(round(BI_col_str_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i <4:
        plt.text(i+.85,BI_col_str_4[int(i/2)]/2,round(BI_col_str_4[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i+.75,BI_col_saw_1[int(i/2)]*3.75,'str:' +'\n' + str(round(BI_col_str_1[int(i/2)],2)),fontsize=14,ha='center')
    
for i in X_axis:
    if i <4:
        plt.text(i-.65,BI_col_wkn_1[int(i/2)]/2,round(BI_col_wkn_1[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i-.75,BI_col_saw_1[int(i/2)]*1.5,'wkn:' +'\n' + str(round(BI_col_wkn_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i <4:
        plt.text(i-.15,BI_col_wkn_2[int(i/2)]/2,round(BI_col_wkn_2[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i-.25,BI_col_saw_1[int(i/2)]*1.5,'wkn:' +'\n' + str(round(BI_col_wkn_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i <4:
        plt.text(i+.15,BI_col_wkn_3[int(i/2)]/2,round(BI_col_wkn_3[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i+.25,BI_col_saw_1[int(i/2)]*1.5,'wkn:' +'\n' + str(round(BI_col_wkn_1[int(i/2)],2)),fontsize=14,ha='center')
for i in X_axis:
    if i <4:
        plt.text(i+.65,BI_col_wkn_4[int(i/2)]/2,round(BI_col_wkn_4[int(i/2)],2),fontsize=14,ha='center',rotation=90)
    else:
        plt.text(i+.75,BI_col_saw_1[int(i/2)]*1.5,'wkn:' +'\n' + str(round(BI_col_wkn_1[int(i/2)],2)),fontsize=14,ha='center')

plt.legend(handles=[saw_plot,str_plot,wkn_plot],fontsize=20)
plt.title('Total # Of Columns Per NB Model Type (Bigrams)', pad=20, fontsize=30)
ax2=ax1.twiny()
plt.xticks([.165,.37,.625,.85], step_labels,fontsize=18)
plt.tight_layout()
plt.savefig(f'{home_dir}/images/BI_all_column_counts.png')

#figure out how many columns were dropped between steps
BI_colDif_DF=pd.DataFrame()
tmpIndex=0
for i in ['saw','str','wkn']:
    for h in ['1','2','3','4']:
        tmp_list=f'BI_col_{i}_{h}'
        BI_colDif_DF.loc[tmpIndex,'analysis']=tmp_list
        BI_colDif_DF.loc[tmpIndex,'Raw Col Total']=eval(tmp_list)[0]
        BI_colDif_DF.loc[tmpIndex,'Raw-Step1 Diff']=eval(tmp_list)[0]-eval(tmp_list)[1]           
        BI_colDif_DF.loc[tmpIndex,'% Change1']=round((eval(tmp_list)[0]-eval(tmp_list)[1])/eval(tmp_list)[0],3)
        BI_colDif_DF.loc[tmpIndex,'Step1 Total']=eval(tmp_list)[1]
        BI_colDif_DF.loc[tmpIndex,'Step1-2 Diff']=eval(tmp_list)[1]-eval(tmp_list)[2]            
        BI_colDif_DF.loc[tmpIndex,'% Change2']=round((eval(tmp_list)[1]-eval(tmp_list)[2])/eval(tmp_list)[1],3)
        BI_colDif_DF.loc[tmpIndex,'Step2 Total']=eval(tmp_list)[2]
        BI_colDif_DF.loc[tmpIndex,'Step2-3 Diff']=eval(tmp_list)[2]-eval(tmp_list)[3]            
        BI_colDif_DF.loc[tmpIndex,'% Change3']=round((eval(tmp_list)[2]-eval(tmp_list)[3])/eval(tmp_list)[2],3)
        BI_colDif_DF.loc[tmpIndex,'Final Total']=eval(tmp_list)[3]
        tmpIndex=tmpIndex+1

#Add the mean for each column at the end
BI_colDif_DF.iloc[:,1:].groupby(['% Change1']).mean()
tmp_list=BI_colDif_DF.iloc[:,1:].mean(axis=0)
BI_colDif_DF.loc[12,'analysis']='Mean Change'
for i in range(0,10):
    BI_colDif_DF.iloc[12,i+1]=round(tmp_list[i],3)
    
####now work on accuracy visuals    
#combine pre clean and clean accuracies
all_pre_acc=pd.concat([CV_pre_BN_acc,CV_pre_MN_acc,TF_pre_BN_acc,TF_pre_MN_acc]).reset_index(drop=True)
all_cle_acc=pd.concat([CV_cle_BN_acc,CV_cle_MN_acc,TF_cle_BN_acc,TF_cle_MN_acc]).reset_index(drop=True)

###now combine all single term accuracies for every model into one DF
tot_acc_list=[all_pre_acc,all_cle_acc,fre_acc,cor_acc]
tot_acc_DF=pd.concat(tot_acc_list).reset_index(drop=True)

#add a column that specifies
for i in tot_acc_DF.index:
    tmpMod=tot_acc_DF.loc[i,'Model']
    text=tmpMod[3:6]
    clean=tmpMod[7:10]
    num=tmpMod[len(tmpMod)-1:]
    tot_acc_DF.loc[i,['Text','Clean Step','Label']]=text,clean,num

agg_tot_acc=tot_acc_DF.iloc[:,[1,12,13,14]].groupby(['Clean Step','Text','Label']).mean()
agg_lab_acc=agg_tot_acc.groupby(['Clean Step','Label']).mean() #label 4 strongest across all 4 cleaning steps, pre: 58, cle: 58, fre: 58, cor: 59
agg_tot_acc.groupby(['Label']).mean()

####plot the accuracy labels across the 4 cleaning steps/labels
clean_labels=[agg_lab_acc.index[i][0] for i in range(0,len(agg_lab_acc.index))]
bust_labels= [agg_lab_acc.index[i][0]+agg_lab_acc.index[i][1] for i in range(0,len(agg_lab_acc.index))]
acc_scores=[round(agg_lab_acc['MeanScore'][i],2) for i in range(0,len(agg_lab_acc.index))]
x_axis_range=np.arange(0,16,1)

fig, ax = plt.subplots(figsize=(20,15))
plt.bar(x_axis_range, acc_scores,color='g')
for i in range(len(clean_labels)):
    plt.text(i,acc_scores[i]/2, acc_scores[i], fontsize=18,ha='center')
plt.xticks(x_axis_range,bust_labels, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Clean Step/Label",fontsize=24)
plt.ylabel("Accuracies",fontsize=24)
plt.title('Single Term Accuracies Per Cleaning Step and Label', pad=20, fontsize=30)
plt.tight_layout()
plt.savefig('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/images/single_term_accuracies.png')

#change data label column to int
tot_acc_DF['Label']=tot_acc_DF['Label'].astype('int')
#now figure out the highest label 4 model overall
max(tot_acc_DF['MeanScore'][tot_acc_DF['Label']==4])
label4_top=tot_acc_DF[tot_acc_DF['MeanScore']==max(tot_acc_DF['MeanScore'][tot_acc_DF['Label']==4])]
label3_top=tot_acc_DF[tot_acc_DF['MeanScore']==max(tot_acc_DF['MeanScore'][tot_acc_DF['Label']==3])]
label2_top=tot_acc_DF[tot_acc_DF['MeanScore']==max(tot_acc_DF['MeanScore'][tot_acc_DF['Label']==2])]
label1_top=tot_acc_DF[tot_acc_DF['MeanScore']==max(tot_acc_DF['MeanScore'][tot_acc_DF['Label']==1])]

#now create confusion matrices for the top models from each label
def conf_create():
    acc_list=[]
    for num in range(1,5):
        tmpTop=eval(f'label{num}_top').iloc[0,0]
        tmpTop=tmpTop.replace(f'_{num}',f'_DF_{num}')
        text=tmpTop[3:6]
        clean=tmpTop[7:10]
        NB_mod=tmpTop[11:13]
        if clean=='cor':
            tmpTop=tmpTop+'_nofre'
        if clean=='fre' or clean=='cor':
            base_clean='cle'
        else:
            base_clean=clean
        tmpDF=eval(tmpTop).copy(deep=True)
        tmpDF.insert(loc=0, column='LABEL', value=eval(f'baseDataDF_{text}_{base_clean}_{num}')[f'Bust{num}'].reset_index(drop=True))
        #####now use these DFs to create NB models
        rd.seed(200)
        Train_DF, Test_DF = train_test_split(tmpDF, test_size=0.3)
        #labels need to be removed from the test data, first save them to lists for testing later
        Test_Labels_Test_DF=Test_DF["LABEL"]
        #now remove the columns
        Test_DF=Test_DF.drop(["LABEL"],axis=1)
        #labels also removed from the training data in order to train the models
        Train_Labels_Train_DF=Train_DF["LABEL"]
        #now remove the columns
        Train_DF=Train_DF.drop(["LABEL"],axis=1)
        #create the modelors
        if NB_mod=='BN':
            tmpModelor=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        else:
            tmpModelor=MultinomialNB()
        #train the models
        tmpModelor.fit(Train_DF,Train_Labels_Train_DF)
        #make the predictions
        tmpModelor_Pred=tmpModelor.predict(Test_DF)
        #create confusion matrices
        tmp_matrix=confusion_matrix(Test_Labels_Test_DF,tmpModelor_Pred)
        ####create heatmap version of the confusion matrix to display them
        #first sentiment countvectorizer
        fig, ax = plt.subplots(figsize=(20,12))
        sns.heatmap(tmp_matrix, annot=True, annot_kws={"size": 20}, fmt='g', ax=ax, cbar=False)
        ax.set_xlabel('Predicted labels',fontsize=24)
        ax.set_ylabel('True labels',fontsize=24) 
        ax.set_title(f'{tmpTop} Confusion Matrix',fontsize=30) 
        ax.xaxis.set_ticklabels(['N', 'Y'],fontsize=18) 
        ax.yaxis.set_ticklabels(['N', 'Y'],fontsize=18)
        plt.savefig(f'{home_dir}/images/{tmpTop}.png')

        #calculate how accurate overall each model was
        tmpMod_acc=(tmp_matrix[0,0]+tmp_matrix[1,1])/len(Test_Labels_Test_DF)
        acc_list.append({tmpTop:tmpMod_acc})
    return(acc_list)

top_label_acc=conf_create()


#########now bigram work
#reimport bigram accuracies if needed
for i in ['cle','pre']:
    tmpDFInfo=pd.DataFrame()
    for h in ['CV','TF']:
        for j in ['MN','BN']:
            tmpStr=f'BI_{h}_{i}_{j}_acc'
            tmpAcc=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/'+tmpStr+'.csv')
            CombFreqInfo=[tmpDFInfo,tmpAcc]
            tmpDFInfo=pd.concat(CombFreqInfo)
            globals()[str((f'BI_all_{i}_acc'))]=tmpDFInfo.reset_index(drop=True)
    
BI_fre_acc=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/BI_fre_acc.csv')
BI_cor_acc=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/backup files/backup data/BI_cor_acc.csv')

###now combine accuracies for every model into one DF
BI_tot_acc_list=[BI_all_pre_acc,BI_all_cle_acc,BI_fre_acc,BI_cor_acc]
BI_tot_acc_DF=pd.concat(BI_tot_acc_list).reset_index(drop=True)

#add a column that specifies
for i in BI_tot_acc_DF.index:
    tmpMod=BI_tot_acc_DF.loc[i,'Model']
    text=tmpMod[6:9]
    clean=tmpMod[10:13]
    num=tmpMod[len(tmpMod)-1:]
    BI_tot_acc_DF.loc[i,['Text','Clean Step','Label']]=text,clean,num

BI_agg_tot_acc=BI_tot_acc_DF.iloc[:,[1,12,13,14]].groupby(['Clean Step','Text','Label']).mean()
BI_agg_lab_acc=BI_agg_tot_acc.groupby(['Clean Step','Label']).mean() #label 4 strongest across all 4 cleaning steps, pre: 58, cle: 57, fre: 59, cor: 59
BI_agg_tot_acc.groupby(['Label']).mean()

####plot the accuracy labels across the 4 cleaning steps/labels
clean_labels=[BI_agg_lab_acc.index[i][0] for i in range(0,len(BI_agg_lab_acc.index))]
bust_labels= [BI_agg_lab_acc.index[i][0]+BI_agg_lab_acc.index[i][1] for i in range(0,len(BI_agg_lab_acc.index))]
acc_scores=[round(BI_agg_lab_acc['MeanScore'][i],2) for i in range(0,len(BI_agg_lab_acc.index))]
x_axis_range=np.arange(0,16,1)

fig, ax = plt.subplots(figsize=(20,15))
plt.bar(x_axis_range, acc_scores,color='g')
for i in range(len(clean_labels)):
    plt.text(i,acc_scores[i]/2, acc_scores[i], fontsize=18,ha='center')
plt.xticks(x_axis_range,bust_labels, fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Clean Step/Label",fontsize=24)
plt.ylabel("Accuracies",fontsize=24)
plt.title('Single Term Accuracies Per Cleaning Step and Label (Bigrams)', pad=20, fontsize=30)
plt.tight_layout()
plt.savefig('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/images/bigram_accuracies.png')

#change data label column to int
BI_tot_acc_DF['Label']=BI_tot_acc_DF['Label'].astype('int')
#now figure out the highest label 4 model overall
max(BI_tot_acc_DF['MeanScore'][BI_tot_acc_DF['Label']==4])
BI_label4_top=BI_tot_acc_DF[BI_tot_acc_DF['MeanScore']==max(BI_tot_acc_DF['MeanScore'][BI_tot_acc_DF['Label']==4])]
BI_label3_top=BI_tot_acc_DF[BI_tot_acc_DF['MeanScore']==max(BI_tot_acc_DF['MeanScore'][BI_tot_acc_DF['Label']==3])]
BI_label2_top=BI_tot_acc_DF[BI_tot_acc_DF['MeanScore']==max(BI_tot_acc_DF['MeanScore'][BI_tot_acc_DF['Label']==2])]
BI_label1_top=BI_tot_acc_DF[BI_tot_acc_DF['MeanScore']==max(BI_tot_acc_DF['MeanScore'][BI_tot_acc_DF['Label']==1])]

#now create confusion matrices for the top models from each label
def bi_conf_create():
    acc_list=[]
    for num in range(1,5):
        tmpTop=eval(f'BI_label{num}_top').iloc[0,0]
        tmpTop=tmpTop.replace(f'_{num}',f'_DF_{num}')
        text=tmpTop[6:9]
        clean=tmpTop[10:13]
        NB_mod=tmpTop[14:16]
        if clean=='cor':
            tmpTop=tmpTop+'_nofre'
        if clean=='fre' or clean=='cor':
            base_clean='cle'
        else:
            base_clean=clean
        tmpDF=pd.read_csv(f'{home_dir}/backup files/backup data/{tmpTop}.csv')
        tmpDF.insert(loc=0, column='LABEL', value=eval(f'baseDataDF_{text}_{base_clean}_{num}')[f'Bust{num}'].reset_index(drop=True))
        #####now use these DFs to create NB models
        rd.seed(200)
        Train_DF, Test_DF = train_test_split(tmpDF, test_size=0.3)
        #labels need to be removed from the test data, first save them to lists for testing later
        Test_Labels_Test_DF=Test_DF["LABEL"]
        #now remove the columns
        Test_DF=Test_DF.drop(["LABEL"],axis=1)
        #labels also removed from the training data in order to train the models
        Train_Labels_Train_DF=Train_DF["LABEL"]
        #now remove the columns
        Train_DF=Train_DF.drop(["LABEL"],axis=1)
        #create the modelors
        if NB_mod=='BN':
            tmpModelor=BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        else:
            tmpModelor=MultinomialNB()
        #train the models
        tmpModelor.fit(Train_DF,Train_Labels_Train_DF)
        #make the predictions
        tmpModelor_Pred=tmpModelor.predict(Test_DF)
        #create confusion matrices
        tmp_matrix=confusion_matrix(Test_Labels_Test_DF,tmpModelor_Pred)
        ####create heatmap version of the confusion matrix to display them
        #first sentiment countvectorizer
        fig, ax = plt.subplots(figsize=(20,12))
        sns.heatmap(tmp_matrix, annot=True, annot_kws={"size": 20}, fmt='g', ax=ax, cbar=False)
        ax.set_xlabel('Predicted labels',fontsize=24)
        ax.set_ylabel('True labels',fontsize=24) 
        ax.set_title(f'{tmpTop} Confusion Matrix',fontsize=30) 
        ax.xaxis.set_ticklabels(['N', 'Y'],fontsize=18) 
        ax.yaxis.set_ticklabels(['N', 'Y'],fontsize=18)
        plt.savefig(f'{home_dir}/images/{tmpTop}.png')

        #calculate how accurate overall each model was
        tmpMod_acc=(tmp_matrix[0,0]+tmp_matrix[1,1])/len(Test_Labels_Test_DF)
        acc_list.append({tmpTop:tmpMod_acc})
    return(acc_list)

BI_top_label_acc=bi_conf_create()

############################Export all accuracies
tot_acc_DF.to_csv(f'{home_dir}/tot_acc_DF.csv')
BI_tot_acc_DF.to_csv(f'{home_dir}/BI_tot_acc_DF.csv')


####################For presentation
#############combine all accuracies in one
tot_acc_DF=tot_acc_DF.rename(columns={1: "Run1", 2: "Run2", 3: "Run3", 4: "Run4", 5: "Run5", 6: "Run6", 7: "Run7", 8: "Run8", 9: "Run9", 10: "Run10"})
BI_tot_acc_DF=BI_tot_acc_DF.rename(columns={1: "Run1", 2: "Run2", 3: "Run3", 4: "Run4", 5: "Run5", 6: "Run6", 7: "Run7", 8: "Run8", 9: "Run9", 10: "Run10"})
BI_Sing_ACC_LIST=[tot_acc_DF,BI_tot_acc_DF]
BI_Sing_ACC=pd.concat(BI_Sing_ACC_LIST)
max(BI_Sing_ACC['MeanScore'])
BI_Sing_ACC_Top_Overall=BI_Sing_ACC[BI_Sing_ACC['MeanScore']==max(BI_Sing_ACC['MeanScore'][BI_Sing_ACC['Label']==4])]

BI_agg_tot_acc.groupby(['Label']).mean()
Label_Total_Overall=BI_Sing_ACC[['MeanScore','Label']].groupby(['Label']).mean()
Clean_Total_Overall=BI_Sing_ACC[['MeanScore','Clean Step']].groupby(['Clean Step']).mean()

#Plot mean scores for cleaning Steps
clean_labels=list(Clean_Total_Overall.index)
acc_scores=[Clean_Total_Overall.values[i][0] for i in range(0,len(Clean_Total_Overall.values))]
x_axis_range=np.arange(0,4,1)

fig, ax = plt.subplots(figsize=(20,15))
plt.bar(x_axis_range, acc_scores,color='g')
for i in range(len(clean_labels)):
    plt.text(i,acc_scores[i]/2, round(acc_scores[i],2), fontsize=30,ha='center')
plt.xticks(x_axis_range,clean_labels, fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Clean Step",fontsize=30)
plt.ylabel("Accuracies",fontsize=30)
plt.title('Average Accuracy During Each Cleaning Step', pad=20, fontsize=40)
plt.tight_layout()
plt.savefig('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/images/average_accuracies_cleaning_steps.png')

#Plot mean scores for different labels
clean_labels=list(Label_Total_Overall.index)
acc_scores=[Label_Total_Overall.values[i][0] for i in range(0,len(Label_Total_Overall.values))]
x_axis_range=np.arange(0,4,1)

fig, ax = plt.subplots(figsize=(20,15))
plt.bar(x_axis_range, acc_scores,color='g')
for i in range(len(clean_labels)):
    plt.text(i,acc_scores[i]/2, round(acc_scores[i],2), fontsize=30,ha='center')
plt.xticks(x_axis_range,clean_labels, fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Label",fontsize=30)
plt.ylabel("Accuracies",fontsize=30)
plt.title('Average Accuracy For Each Label', pad=20, fontsize=40)
plt.tight_layout()
plt.savefig('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/images/average_accuracies_labels.png')