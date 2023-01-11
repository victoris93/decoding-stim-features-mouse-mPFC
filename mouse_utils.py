import scipy.io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
import sys
from os import mkdir
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from b2b_regression import *
import re
from sklearn.preprocessing import StandardScaler
from cross_b2b_utils import *
from matplotlib.lines import Line2D


rules = {
            "1":"ORI",
            "2":"SF",
            "3":"SF",
            "4":"ORI",
            "5":"SF",
            "6":"SF"
        }

rules_region = {
            "7":"SF",
            "8":"ORI",
            "9":"ORI",
            "10":"SF",
            "11":"ORI",
        }

def remove_regressors_from_raw_predictors(raw_predictors, T5_regressors, regressors):
    mask=np.full(len(T5_regressors), False, dtype=bool)
    mask[[T5_regressors.index(i) for i in regressors]] = True 
    raw_predictors = raw_predictors[:,mask]
    return raw_predictors

def make_predictors(df, mouse, session, cross_session_analysis = False, type = float, dummify_vars = None, region  = None, reference_labels = None):
    df_mouse_session = df[(df["mouse"] == mouse) & (df["session"] == session)]
    if dummify_vars != None:
        predictors = pd.get_dummies(df_mouse_session, columns=dummify_vars)
    predictors = predictors[predictors["Cell"] == "n0"].drop(labels = "Cell", axis = 1)
    predictors = predictors.drop(labels = ["RunningSpeed", "CellResponse", "mouse", "session", "StimulusID"], axis = 1)
    predictor_labels = predictors.columns
    raw_predictors = predictors.to_numpy(dtype= float)
    if region == None:
        mouse_name = "mouse_%s" % mouse
    else:
        mouse_name = "mouse_%s_region%s" %(mouse, region)
    if cross_session_analysis == False:
        np.save(arr = raw_predictors, file = os.getcwd() + "/b2b/raw_predictors/T5/raw_predictors_%s_T5.npy" % mouse_name)
        predictors = StandardScaler(with_mean = False).fit_transform(raw_predictors)
    else:
        reference_predictors = np.load(os.getcwd() + "/b2b/raw_predictors/T5/raw_predictors_%s_T5.npy" % mouse_name)
        reference_predictors = remove_regressors_from_raw_predictors(reference_predictors, reference_labels, predictor_labels)
        scaler = StandardScaler(with_mean = False).fit(reference_predictors)
        predictors = scaler.transform(raw_predictors)
    return predictors, predictor_labels

def get_cell_response(df, mouse, session, type = float):
    df_mouse_session = df[(df["mouse"] == mouse) & (df["session"] == session)]
    if pd.isna(df_mouse_session["CellResponse"]).any() == True:
        na_indices = np.where(pd.isna(df[(df["mouse"] == mouse) & (df["session"] == session) & (df["Cell"] == "n0")]["CellResponse"]) == True)[0]
        df_mouse_session = df_mouse_session.dropna(subset = "CellResponse")
    else:
        na_indices = np.array([None])
    CellResponse= df_mouse_session.pivot(columns = "Cell", values = "CellResponse", index = "RunningSpeed").reset_index()
    CellResponse = CellResponse.drop("RunningSpeed", axis = 1).to_numpy(dtype= float)
    return CellResponse, na_indices

def get_deltaR(model, regressor_labels, indep_vars, dep_var, save = False, file = None):
    dR = pd.DataFrame(columns = list(regressor_labels))
    dR.loc[0] = score_knockout(model, indep_vars, dep_var)
    dRlong = dR.melt(ignore_index = False).reset_index()
    dRlong.columns = ["index", "Regressor", "deltaR"]
    if save == True:
        dRlong.to_csv(file)
    return dRlong
    
def get_S(model, regressor_labels, save = False, file = None):
    S = pd.DataFrame(columns = list(regressor_labels))
    S.loc[0] = model.S_
    SLong = S.melt(ignore_index = False).reset_index()
    SLong.columns = ["index", "Regressor", "S"]
    if save == True:
        SLong.to_csv(file)
    return SLong
    
def calculate_p_value(values, value):
    mean = values.mean()
    std = values.std()
    z = (value - mean)/std

    p = 2 * (1 - scipy.stats.norm(0, 1).cdf(abs(z)))
    return p

def add_p_value_to_result_df(mice, deltaR_results, S_results, regressors_labels, session, cross_session_analysis = False,region = None, save = False, file_dir = None):

    for index, mouse in enumerate(mice):
        if region != None:
            mouse_label = "mouse%s_region%s" % (mouse, region)
            rule = rules_region[str(mouse)]
        else:
            mouse_label = "mouse%s" % mouse
            rule = rules[str(mouse)]

        deltaRp_values = []
        Sp_values = []
        deltaR_significant = []
        S_significant = []
        if cross_session_analysis == False:
            deltaR_name = "deltaR"
            S_name = "S"
        else:
            deltaR_name = "%s_vs_T5_deltaR" % session
            S_name = "%s_vs_T5_S" % session
        reference_session = "T5"
        
        for regressor in regressors_labels:
            deltaR_true = deltaR_results[index][deltaR_results[index]["Regressor"] == regressor][deltaR_name].values[0]
            S_true = S_results[index][S_results[index]["Regressor"] == regressor][S_name].values[0]
            deltaR_distr = np.load(os.getcwd() + "/b2b/shuffled_regressors/%s/shuffled_dR_%s_%s_%s.npy" %(mouse_label, mouse_label, reference_session, regressor))
            S_distr = np.load(os.getcwd() + "/b2b/shuffled_regressors/%s/shuffled_S_%s_%s_%s.npy" %(mouse_label, mouse_label, reference_session, regressor))
            p_deltaR = round(calculate_p_value(deltaR_distr, deltaR_true), 4)
            p_S = round(calculate_p_value(S_distr, S_true), 4)
            deltaRp_values.append(p_deltaR)
            Sp_values.append(p_S)

            if p_deltaR <= 0.05:
                sig = True
            else:
                sig = False
            deltaR_significant.append(sig)

            if p_S <= 0.05:
                sig = True
            else:
                sig = False
            S_significant.append(sig)

        deltaR_results[index]["p-value"] = np.asarray(deltaRp_values)
        deltaR_results[index]["Sig."] = np.asarray(deltaR_significant)
        S_results[index]["p-value"] = np.asarray(Sp_values)
        S_results[index]["Sig."] = np.asarray(S_significant)
        if save == True:
            if cross_session_analysis == True:
                prefix = "cross_"
            else:
                prefix = ""
            deltaR_results[index].to_csv(file_dir + "/%sdeltaR_mouse_%s_rule_%s_%s.csv" %(prefix, mouse, rule, session))
            S_results[index].to_csv(file_dir + "/%sS_mouse_%s_rule_%s_%s.csv" %(prefix, mouse, rule, session))

    return (deltaR_results, S_results)

def plot_b2b_result(result_df, result_name, mouse, rule, session, save = False, stats = False, region = None, ymax = None, file_dir = None):
    if region != None:
        mouse_label = "mouse_%s_region%s" % (str(mouse), str(region))
        plot_title = "Mouse %s, Region %s" %(mouse, region)

    else:
        mouse_label = "mouse_%s" % str(mouse)
        plot_title = "Mouse %s" %(mouse)

    markers = {True: "*", False: "o"}
    sizes = {True: 500, False: 40}

    sns.set(rc={'figure.figsize':(10,6)})
    sns.set_style("whitegrid")

    g1_data = result_df[~result_df["Regressor"].str.contains("Stimulus")]
    g2_data = result_df[result_df["Regressor"].str.contains('StimulusSF')]
    g3_data = result_df[result_df["Regressor"].str.contains('StimulusORI')]

    max_y = max(max(g1_data[result_name]), max(g2_data[result_name]), max(g3_data[result_name]))
    if ymax != None:
        if ymax < max_y:
            ymax = max_y

    if stats == False:
        g1 = sns.scatterplot(x = g1_data["Regressor"], y = g1_data[result_name])
        g1_data[result_name].plot()
        g2 = sns.scatterplot(x = g2_data["Regressor"], y = g2_data[result_name],  marker="H") #palette = "coolwarm",
        g2_data[result_name].plot()
        g3 = sns.scatterplot(x = g3_data["Regressor"], y = g3_data[result_name], marker="D") #, palette = "PiYG_r"
        g3_data[result_name].plot()
        g1.set(ylim=(None, ymax))
        plt.legend([],[], frameon=False)
    else:
        g1 = sns.scatterplot(x = g1_data["Regressor"], y = g1_data[result_name], style = g1_data["Sig."], markers = markers, size = g1_data["Sig."], size_order= [True, False], sizes = sizes)
        g1.legend(labels=['S','NS'])
        g1_data[result_name].plot()
        g2 = sns.scatterplot(x = g2_data["Regressor"], y = g2_data[result_name],style = g2_data["Sig."],markers = markers, size = g2_data["Sig."], size_order= [True, False], sizes = sizes) #palette = "coolwarm",
        g2_data[result_name].plot()
        g3 = sns.scatterplot(x = g3_data["Regressor"], y = g3_data[result_name], style = g3_data["Sig."],markers = markers, size = g3_data["Sig."], size_order= [True, False], sizes = sizes) #, palette = "PiYG_r"
        g3_data[result_name].plot()
        g1.set(ylim=(None, ymax))
        legend_elements = [Line2D([0], [0], marker='.', color='w', label='NS',
                        markerfacecolor='black', markersize=10), 
            Line2D([0], [0], marker='*', color='w', label='Sig',
                        markerfacecolor='black', markersize=20)]
        plt.legend(handles=legend_elements, loc=1)

    plt.tick_params(axis='x', rotation=90, labelsize = 15)
    plt.tick_params(axis='y', labelsize = 15)
    plt.xlabel("Regressor", fontsize = 15)
    plt.ylabel(result_name, fontsize = 15)
    plt.title(f'{result_name}. {plot_title}, Rule: {rule}, {session}.', fontsize = 15)

    if save == True:
        filename = file_dir + f"{result_name}_{mouse_label}_rule_{rule}_{session}.png"
        plt.savefig(filename, bbox_inches = 'tight')

    plt.show()
    plt.clf()


def get_mouse_shuff_model_stats(mouse, regressors, region = None):
    if region != None:
        mouse_label = "mouse%s_region%s" % (str(mouse), str(region))
    else:
        mouse_label = "mouse%s" % str(mouse)
    dir_path = os.getcwd() + "/b2b/shuffled_regressors/" + mouse_label
    print(dir_path)
    regressor_dict_dR = dict.fromkeys(regressors)
    regressor_dict_S = dict.fromkeys(regressors)
    for regressor in regressors:
        shuff_dR = np.load(dir_path + "/" + "shuffled_dR_" + mouse_label + "_T5_%s.npy" % regressor)
        shuff_S = np.load(dir_path + "/" + "shuffled_S_" + mouse_label + "_T5_%s.npy" % regressor)
        regressor_dict_dR[regressor] = shuff_dR
        regressor_dict_S[regressor] = shuff_S
    return [regressor_dict_dR, regressor_dict_S]

def plot_distributions(mice, true_models_dR, true_models_S, regressors, rule, region = None, save = False, file_dir = None):
    for index, mouse in enumerate(mice):
        if region != None:
            mouse_label = "mouse%s_region%s" %(mouse, region)
            mouse_title = "Mouse %s, Region %s, Rule %s" %(mouse, region, rule)
        else:
            mouse_label = "mouse%s" % mouse
            mouse_title = "Mouse %s, Rule %s" %(mouse, rule)
        path = os.getcwd() + "/figures/shuffled_regressors/" + mouse_label

        for regressor in regressors:
            shuff_regressor_dict_dR = get_mouse_shuff_model_stats(mouse, region = region, regressors= regressors)[0]
            shuff_regressor_dict_S = get_mouse_shuff_model_stats(mouse, region = region,regressors= regressors)[1]

            fig = plt.figure(figsize=(7,6))
            fig.patch.set_facecolor('white')

            deltaR_dist = np.asarray(shuff_regressor_dict_dR[regressor]).squeeze()
            S_dist = np.asarray(shuff_regressor_dict_S[regressor]).squeeze()

            sns.histplot(deltaR_dist, color = "moccasin")
            sns.histplot(S_dist, color = "palegreen")

            deltaR_true = true_models_dR[index][true_models_dR[index]["Regressor"] == regressor]["deltaR"].values[0]
            p_deltaR = str(round(calculate_p_value(deltaR_dist, deltaR_true), 4))

            S_true = true_models_S[index][true_models_S[index]["Regressor"] == regressor]["S"].values[0]
            p_S = str(round(calculate_p_value(S_dist, S_true), 2))
            p_deltaR, p_S
            
            plt.axvline(x = deltaR_true, ymax = 0.5, c = "saddlebrown")
            plt.axvline(x = S_true, ymax = 0.5, c = "darkgreen")

            fig.legend(labels=['deltaR, p = %s' % p_deltaR,'S, p = %s' % p_S], bbox_to_anchor=(0, 0.35, 0.91, 0.5))
            plt.xlabel("Effect size")
            plt.title("%s. Distributions of shuffled S & deltaR: %s" % (mouse_title, regressor))
            plt.show()
            if save == True:
                fig.savefig(fname = file_dir + "/shuffled_regressors/%s/shuffled_%s_T5_%s.png" % (mouse_label, mouse_label, regressor), bbox_inches = 'tight')


# For cross-session analysis:

def get_cross_deltaR(G_optim, H_optim, regressor_labels, session, indep_vars, dep_var, save = False, file = None):
    dR = pd.DataFrame(columns = list(regressor_labels))
    dR.loc[0] = cross_score_knockout(G_optim, H_optim, indep_vars, dep_var)
    dRlong = dR.melt(ignore_index = False).reset_index()
    dRlong.columns = ["index", "Regressor", "%s_vs_T5_deltaR" % session]
    if save == True:
        dRlong.to_csv(file)
    return dRlong

def get_cross_S(G_optim, regressor_labels, session, indep_vars, dep_var, save = False, file = None):
    S = pd.DataFrame(columns = list(regressor_labels))
    S_score, _ = cross_fit_H(G_optim, indep_vars, dep_var)
    S.loc[0] = S_score
    SLong = S.melt(ignore_index = False).reset_index()
    SLong.columns = ["index", "Regressor", "%s_vs_T5_S" % session]
    if save == True:
        SLong.to_csv(file)
    return SLong
    
def calculate_p_value(values, value):
    mean = values.mean()
    std = values.std()
    z = (value - mean)/std
    p = 2 * (1 - scipy.stats.norm(0, 1).cdf(abs(z)))
    return p