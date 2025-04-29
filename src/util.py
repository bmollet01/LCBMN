import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ast
import json
from scipy.stats import wilcoxon
from math import log

def string_to_list(str):
    try:
        return ast.literal_eval(str)
    except (SyntaxError, ValueError):
        # If the string is not a well-formed list, return it as is
        return str

###### Figure 3 : Capacity evaluation ######

def plot_capacity_evaluation(task, save=False):
    xgb_est = pd.read_csv(f'./data/{task}/xgb_est.csv')
    datasets = xgb_est['Dataset'].unique()

    result_amn = pd.read_csv(f'./data/{task}/method_comparison.csv')
    result_amn = result_amn.loc[result_amn['method']=='AMN']
    mean_AMN = result_amn.groupby('Dataset').mean()
    sns.set_theme()
    sns.set(rc={'figure.figsize':(15,12)}, font_scale=3)

    # Plot the mean lines and store the line objects for the legend
    lines = sns.lineplot(data=xgb_est, x='niter', y='mean', hue='Dataset')
    line_handles, _ = lines.get_legend_handles_labels()

    # Plot std and horizontal lines, and create custom legend handles
    for i, dataset in enumerate(datasets):
        category_data = xgb_est[xgb_est['Dataset'] == dataset]
        color = sns.color_palette()[i]
        plt.fill_between(category_data['niter'], category_data['mean'] - category_data['std'],
                        category_data['mean'] + category_data['std'], alpha=0.2, color=color)

        plt.axhline(y=mean_AMN.loc[dataset].values[0],
                    color=color,
                    linestyle='--')
    # Visual settings
    plt.xscale('log')
    plt.xlabel('$N_{est}$')
    if task=="regression":
        plt.ylabel('$Q^2$')
        plt.xlim(5, 100)
    else:
        plt.ylabel('$Acc$')
    plt.ylim(0.65, 1)
    plt.legend(handles=line_handles, labels=datasets.tolist(), loc='upper left', prop={'size': 23})
    if save:
        plt.savefig(f"./figures/Figure-3_{task}.jpg", dpi=600)

def plot_nestimator(task, save=False):
    data = pd.read_csv(f"./data/{task}/capacity.csv",index_col=0)
    data['capacity'] = data['capacity'].apply(lambda x: log(x))
    datasets = data['dataset'].unique()
    sns.set_theme()
    sns.set(rc={'figure.figsize':(15,12)}, font_scale=3)
    palette_color = {datasets[0]:"sandybrown", datasets[1]:"steelblue", datasets[2]:"seagreen"}
    sns.boxplot(data=data, x="dataset", y="capacity", palette=palette_color, width=0.45)
    
    plt.xticks(ticks=[0, 1, 2], labels=datasets)
    plt.xticks(fontsize=30)
    plt.ylabel("Capacity : $Log(N_{est})$")
    plt.xlabel(None)
    if save:
        plt.savefig(f"./figures/Figure-3_{task}_nest.jpg", dpi=600)

###### Figure 4 : Performance comparison ######

def wilcoxon_test(df, score):
    """
    Perform Wilcoxon signed-rank test for each pair of methods in the dataframe.
    """

    methods = df['method'].unique()
    results = []
    
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            
            # Get scores for each method
            scores1 = df[df['method'] == method1][score]
            scores2 = df[df['method'] == method2][score]
            
            # Perform Wilcoxon test
            stat, p = wilcoxon(scores1, scores2)
            
            results.append({'method1': method1, 'method2': method2, 'statistic': stat, 'p-value': p})
    
    return pd.DataFrame(results)

def plot_boxplots_with_wilcoxon(df, wilcoxon_results, order, score, task, save=None):
    """
    Plot boxplots of scores by method and add Wilcoxon test statistics as annotations.
    """

    plt.figure(figsize=(10, 8))
    sns.set(style="darkgrid", font_scale=2)
    # Create a boxplot
    custom_palette = sns.color_palette()[:2] + ['indianred']
    ax = sns.boxplot(x='method', y=score, data=df, order=order, palette=custom_palette)
    
    # Add Wilcoxon test statistics as annotations
    methods = df['method'].unique()
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = order[i]
            method2 = order[j]
            
            # Get the Wilcoxon statistic
            result = wilcoxon_results[(wilcoxon_results['method1'] == method1) & (wilcoxon_results['method2'] == method2)]
            if result.empty:
                result = wilcoxon_results[(wilcoxon_results['method1'] == method2) & (wilcoxon_results['method2'] == method1)]
            
            stat = result['statistic'].values[0]
            p_value = result['p-value'].values[0]
            
            # Add annotation
            y_max = max(df[df['method'] == method1][score].max(), df[df['method'] == method2][score].max())
            x1, x2 = i, j
            y, h, col = y_max + 0.05+(i+j)*0.1, 0.005, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            # ax.text((x1+x2)*.5, y+h, f'W={stat:.2f}, p={p_value:.2e}', ha='center', va='bottom', color=col)
            ax.text((x1+x2)*.5, y+h, f'p={p_value:.2e}', ha='center', va='bottom', color=col)
    
    # ax.set_title('Boxplots of Scores by Method with Wilcoxon Test Statistics')
    ax.set_xticklabels(["XGBoost", "Perceptron", "AMN"])
    ax.set_xlabel(None)
    if task=="regression":
        ax.set_ylabel("$Q^2$")
        ax.set_xticklabels(["XGBoost", "Linear Reg", "AMN"])
    if save:
        plt.savefig(f"./figures/Figure-4_{task}", dpi=600)
    plt.show()

def comparison_method(task, save=False):
    if task=="regression":
        order = ["xgb", "linear reg", "AMN"]
        score = "Q2"
        data = pd.read_csv("./data/regression/method_comparison.csv")
    elif task=="classification":
        order = ['xgb', 'perceptron', 'AMN']
        score = "Acc"
        data = pd.read_csv("./data/classification/Figure-4_method_comparison.csv")

    # Perform wilcoxon test
    wilcoxon_results = wilcoxon_test(data, score)
    # Plot the results
    plot_boxplots_with_wilcoxon(data, wilcoxon_results, order, score, task, save=save)


###### Figure 6 : Ablation ######

def ablation(task, save=False):
    list_untrainable_dim = [1, 10, 20, 30, 40, 50, 100, 150, 200, 300, 500, 700]
    list_hidden_dim = [1, 10, 25, 50, 100, 250, 500, 700]

    data = open(f'./data/{task}/ablation.json')
    list_replicats_clas_sub = json.load(data)

    big_dict_ab_class = list_replicats_clas_sub[0] # summary dict
    list_replicats_clas_sub = list_replicats_clas_sub[1:] #remove first element

    for replica in list_replicats_clas_sub:
        for dataset in replica:
            for i in range(len(replica[dataset])):
                for fold in replica[dataset][i][0]:
                    big_dict_ab_class[dataset][i][0].append(fold)


        
    result_amn_clas = pd.read_csv(f'./data/{task}/AMN_hidden_dim.csv', index_col=0)
    result_amn_clas = result_amn_clas.applymap(string_to_list)

    dict_clas_amn = {}
    for dataset in result_amn_clas.columns:
        dict_clas_amn[dataset] = []
        for i in result_amn_clas[dataset]:
            dict_clas_amn[dataset].append(i)



    for dataset_name in  big_dict_ab_class.keys():
        # Get stats for the plot
        avg_AMN = [np.average(i) for i in dict_clas_amn[dataset_name]]
        std_AMN = [np.std(i) for i in dict_clas_amn[dataset_name]]
        avg_ANN = [np.average(i) for i in big_dict_ab_class[dataset_name]]
        std_ANN = [np.std(i) for i in big_dict_ab_class[dataset_name]]
        
        # Plot comparaison NN +  untrainable NN
        sns.set_theme()
        sns.set(rc={'figure.figsize':(25,12)}, font_scale=4)
        plt.figure()

        # PLot avg
        sns.lineplot(x=list_hidden_dim, y=avg_ANN, color="orange", label='NN + linear function')
        sns.lineplot(x=list_untrainable_dim, y=avg_AMN, color="blue", label='NN + bacterium')
        # Plot std:
        plt.fill_between(list_hidden_dim, [a - b for a, b in zip(avg_ANN, std_ANN)],
                        [a + b for a, b in zip(avg_ANN, std_ANN)], alpha=0.2, color="orange")
        plt.fill_between(list_untrainable_dim, [a - b for a, b in zip(avg_AMN, std_AMN)],
                    [a + b for a, b in zip(avg_AMN, std_AMN)], alpha=0.2, color="blue")


        plt.xlabel("Number of neurones in NN's hidden layer")
        if task=="regression":
            plt.ylabel("$Q^2$")
        else:
            plt.ylabel("$Acc$")
        plt.title(dataset_name)
        plt.ylim(0.5,1)
        plt.legend()
        plt.plot()
        plt.savefig(f"./figures/Figure-6_{dataset_name}_abl_{task}.png")

###### Figure 7 : Substitution ######

def substitution(task, save=False):
    list_hidden_dim = [1, 10, 25, 50, 100, 250, 500, 1000]
    bounds = [0,1000]
    data = open(f'./data/{task}/substitution.json')
    list_replicats_clas_sub = json.load(data)

    big_dict_sub_class = list_replicats_clas_sub[0] # summary dict
    list_replicats_clas_sub = list_replicats_clas_sub[1:] #remove first element

    for replica in list_replicats_clas_sub:
        for dataset in replica:
            for i in range(len(replica[dataset])):
                for fold in replica[dataset][i][0]:
                    big_dict_sub_class[dataset][i][0].append(fold)




    result_amn_reg = pd.read_csv(f'./data/{task}/method_comparison.csv')

    result_amn_reg = result_amn_reg.loc[result_amn_reg['method']=='AMN']
    mean_AMN_reg = result_amn_reg.groupby('Dataset').mean()
    std_AMN_reg = result_amn_reg.groupby('Dataset').std()


    for dataset_name in  big_dict_sub_class.keys():
        # Get stats for the plot
        #avg_AMN = [np.average(i) for i in dict_AMN[dataset_name]]
        #std_AMN = [np.std(i) for i in dict_AMN[dataset_name]]
        avg_ANN = [np.average(i) for i in big_dict_sub_class[dataset_name]]
        std_ANN = [np.std(i) for i in big_dict_sub_class[dataset_name]]
        ### Plot comparaison AMN ANN

        sns.set_theme()
        sns.set(rc={'figure.figsize':(25,12)}, font_scale=4)
        plt.figure()

        # Plot NN + AMN with 200 neurones in hidden layer
        plt.axhline(y=mean_AMN_reg.loc[dataset_name][0], xmin=0.05, xmax=0.95, color="blue", linestyle='--', label='NN + AMN')
        plt.fill_between(bounds, mean_AMN_reg.loc[dataset_name][0] - std_AMN_reg.loc[dataset_name][0],
                        mean_AMN_reg.loc[dataset_name][0] + std_AMN_reg.loc[dataset_name][0], alpha=0.2, color="blue")
        sns.lineplot(x=list_hidden_dim, y=avg_ANN, color="orange", label='ANN')

        # Plot std:
        # category_data_AMN = dict_AMN[dataset_name]
        category_data_ANN = big_dict_sub_class[dataset_name]
        #plt.fill_between(list_hidden_dim, [a - b for a, b in zip(avg_AMN, std_AMN)],
        #                [a + b for a, b in zip(avg_AMN, std_AMN)], alpha=0.2, label=f'Std Dev - {dataset_name}', color="blue")
        plt.fill_between(list_hidden_dim, [a - b for a, b in zip(avg_ANN, std_ANN)],
                        [a + b for a, b in zip(avg_ANN, std_ANN)], alpha=0.2, label=f'Std Dev - {dataset_name}', color="orange")
        plt.xlabel("Number of neurones in the non-trainable ANN")
        if task=="regression":
            plt.ylabel("$Q^2$")
        else:
            plt.ylabel("$Acc$")
        
        plt.title(dataset_name)
        plt.ylim(0,1)
        plt.plot()
        if save:
            plt.savefig(f"./figures/Figure-7_{dataset_name}_sub_{task}.png")