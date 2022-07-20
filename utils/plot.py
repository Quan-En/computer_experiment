"""
Several plot functions

Version: 2022-02-20
Author: Quan-En, Li

In this program have: 
- function: 
    PlotCandidateSet(candidate_set: list, title: str="", xy_labels: list=[r"$f_1$",r"$f_2$"]) -> null
    PlotScatter(y_list: list, title: str, pareto_set: list, test_y_list: list=None) -> null
    PlotContourAnd3D(x_list, y_list, titles, single_colorbar=True) -> null
    PlotContour(x_list, y_list, titles, single_colorbar=True) -> null
    PlotContourWithParetoSet(x_list, y_list, titles, pareto_x_list, single_colorbar=True) -> null
    Plot3D(x_list, y_list, titles, single_colorbar=True) -> null
    ScatterPoints(w, level_num=3, range_list=[[-4.,4.], [-4.,4.]]) -> null
    PlotHistory(ei_value_list, time_list, cr_list) -> null
"""

import numpy as np
import pandas as pd
from math import ceil
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from utils.utils import collect_efficient_solutions
plt.rcParams.update({'font.family':'Times New Roman', 'font.size': 15})

def PlotCandidateSet(candidate_set: list, title: str="", xy_labels: list=[r"$f_1$",r"$f_2$"]):
    pateto_set = collect_efficient_solutions(candidate_set)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.row_stack(candidate_set)[:,0], np.row_stack(candidate_set)[:,1], marker=".", c="darkgray", s=60)
    ax.scatter(np.row_stack(pateto_set)[:,0], np.row_stack(pateto_set)[:,1], marker=".", c="blue", s=60, label="Pareto front")
    # ax.scatter(np.row_stack(pateto_set)[:,0], np.row_stack(pateto_set)[:,1], marker="+", c="blue", s=100, label="True Pareto front")
    ax.set_title(title, size=20)
    ax.set_xlabel(xy_labels[0], size=15)
    ax.set_ylabel(xy_labels[1], size=15)
    ax.legend()
    fig.tight_layout()
    plt.show()

def PlotScatter(pareto_set: list, y_list: list=None, test_y_list: list=None, title: str=""):
    fig, ax = plt.subplots(figsize=(7, 5))

    # True pareto front
    ax.scatter(np.row_stack(pareto_set)[:, 0], np.row_stack(pareto_set)[:, 1], marker=".", s=100, facecolors="gray")

    # Current pareto front
    if y_list:ax.scatter(*y_list, marker="+", s=75, color="blue")

    # Added new point
    if test_y_list: ax.scatter(*test_y_list, marker="*", s=100, facecolors="red", edgecolors="black")

    # Set labels and title
    ax.set_title(title, size=20)
    ax.set_xlabel(r"$y_1(x)$", size=15)
    ax.set_ylabel(r"$y_2(x)$", size=15)

    fig.tight_layout()
    plt.show()

def PlotContourAnd3D(x_list, y_list, titles, single_colorbar=True):
    """ Plot contour-plot and 3D-plot for single objective function """

    # Set up a figure
    num_plot_row = len(y_list)
    fig = plt.figure(figsize=(6 * 2, 5.5 * num_plot_row))
    
    # Set (min, max) values for single colorbar
    if single_colorbar:
        min_value = np.stack(y_list).min()
        max_value = np.stack(y_list).max()
        levels = np.linspace(min_value - 0.01, max_value + 0.01, 100)

    for index, y in enumerate(y_list):

        # Add plot
        ax1 = fig.add_subplot(num_plot_row, 2, (2 * index) + 1)

        # Plot surface
        if single_colorbar:
            surf = ax1.contourf(
                x_list[0], x_list[1], y,
                levels=levels,
                cmap="RdGy", alpha=0.95,
            )
        else:
            surf = ax1.contourf(
                x_list[0], x_list[1], y,
                cmap="RdGy", alpha=0.95,
            )

        # Add title, labels and colorbar
        ax1.set_xlabel(r"$x_1$", size=15)
        ax1.set_ylabel(r"$x_2$", size=15)
        ax1.set_title(titles[index], size=20)
        if not single_colorbar: fig.colorbar(surf, shrink=0.5, aspect=20)

        # Add plot
        ax2 = fig.add_subplot(num_plot_row, 2, (2 * index) + 2, projection="3d")

        # Plot surface
        if single_colorbar:
            surf = ax2.plot_surface(
                X=x_list[0], Y=x_list[1], Z=y,
                vmin=min_value, vmax=max_value,
                rstride=1, cstride=1, cmap="RdGy", linewidth=0, antialiased=False,
            )
        else:
            surf = ax2.plot_surface(
                X=x_list[0], Y=x_list[1], Z=y,
                rstride=1, cstride=1, cmap="RdGy", linewidth=0, antialiased=False,
            )
        # Add title, labels and colorbar
        ax2.set_xlabel(r"$x_1$", size=15)
        ax2.set_ylabel(r"$x_2$", size=15)
        ax2.set_title(titles[index], size=20)
        if not single_colorbar: fig.colorbar(surf, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    
    if single_colorbar:
        
        cmap = plt.get_cmap("RdGy", len(levels))
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([1, 0.1, 0.03, 0.6])
        fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
    
    plt.show()

def PlotContour(x_list, y_list, titles, single_colorbar=True, by_row=True):
    """ Plot contour-plot for two objective function """

    # Set up a figure
    num_plot = len(y_list)
    half_num_plot = ceil(num_plot / 2)

    # Two columns: each width 6
    if by_row:
        num_plot_row = (num_plot // 2) + (num_plot % 2 != 0)
        fig = plt.figure(figsize=(6 * 2, 5.5 * num_plot_row))
    else:
        num_plot_col = (num_plot // 2) + (num_plot % 2 != 0)
        fig = plt.figure(figsize=(6 * num_plot_col, 5.5 * 2))
    
    # Set (min, max) values for single colorbar
    if single_colorbar:
        min_value = np.stack(y_list).min()
        max_value = np.stack(y_list).max()
        levels = np.linspace(min_value - 0.01, max_value + 0.01, 100)

    for index, y in enumerate(y_list):

        # Add plot
        if by_row:
            ax = fig.add_subplot(num_plot_row, 2, index + 1)
        else:
            if index % 2 == 0:
                ax = fig.add_subplot(2, num_plot_col, ceil(index / 2) + 1)
            else:
                ax = fig.add_subplot(2, num_plot_col, ceil(index / 2) + half_num_plot)
            # if index < half_num_plot:
            #     ax = fig.add_subplot(2, num_plot_col, 2*index + 1)
            # else:
            #     ax = fig.add_subplot(2, num_plot_col, 2*(index - half_num_plot + 1))

        # Plot surface
        if single_colorbar:
            surf = ax.contourf(
                x_list[0], x_list[1], y,
                levels=levels,
                cmap="RdGy", alpha=0.95,
            )
        else:
            surf = ax.contourf(
                x_list[0], x_list[1], y,
                cmap="RdGy", alpha=0.95,
            )

        # Add title, labels and colorbar
        ax.set_xlabel(r"$x_1$", size=15)
        ax.set_ylabel(r"$x_2$", size=15)
        ax.set_title(titles[index], size=20)
        if not single_colorbar: fig.colorbar(surf, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    
    if single_colorbar:
        
        cmap = plt.get_cmap("RdGy", len(levels))
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if by_row:
            cbar_ax = fig.add_axes([1, 0.1, 0.03, 0.6])
            fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
        else:
            cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.6])
            fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
    
    plt.show()

def PlotContourWithParetoSet(x_list, y_list, titles, pareto_x_list, single_colorbar=True):
    """ Plot contour-plot for two objective function """

    # Set up a figure
    num_plot = len(y_list)
    half_num_plot = ceil(num_plot / 2)

    # Two columns: each width 6
    num_plot_row = (num_plot // 2) + (num_plot % 2 != 0)
    fig = plt.figure(figsize=(6 * 2, 5.5 * num_plot_row))
    
    # Set (min, max) values for single colorbar
    if single_colorbar:
        min_value = np.stack(y_list).min()
        max_value = np.stack(y_list).max()
        levels = np.linspace(min_value - 0.01, max_value + 0.01, 100)

    for index, y in enumerate(y_list):

        # Add plot
        ax = fig.add_subplot(num_plot_row, 2, index + 1)

        # Plot surface
        if single_colorbar:
            surf = ax.contourf(
                x_list[0], x_list[1], y,
                levels=levels,
                cmap="RdGy", alpha=0.9,
            )
        else:
            surf = ax.contourf(
                x_list[0], x_list[1], y,
                cmap="RdGy", alpha=0.9,
            )

        # Scatter plot
        ax.scatter(*pareto_x_list[index//2].T, marker=".", s=100, c="blue", label="Pareto front")

        # Add title, labels and colorbar
        ax.set_xlabel(r"$x_1$", size=15)
        ax.set_ylabel(r"$x_2$", size=15)
        ax.set_title(titles[index], size=20)
        if not single_colorbar: fig.colorbar(surf, shrink=0.5, aspect=20)
    
    plt.legend() # prop={'size': 12}
    plt.tight_layout()
    
    if single_colorbar:
        
        cmap = plt.get_cmap("RdGy", len(levels))
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([1, 0.1, 0.03, 0.6])
        fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
    
    plt.show()

def Plot3D(x_list, y_list, titles, single_colorbar=True, by_row=True):
    """ Plot 3D-plot for two objective function """

    # Set up a figure
    num_plot = len(y_list)
    half_num_plot = ceil(num_plot / 2)
    # Two columns: each width 6
    if by_row:
        num_plot_row = (num_plot // 2) + (num_plot % 2 != 0)
        fig = plt.figure(figsize=(6 * 2, 5.5 * num_plot_row))
    else:
        num_plot_col = (num_plot // 2) + (num_plot % 2 != 0)
        fig = plt.figure(figsize=(6 * num_plot_col, 5.5 * 2))

    # Set (min, max) values for single colorbar
    if single_colorbar:
        min_value = np.stack(y_list).min()
        max_value = np.stack(y_list).max()

    for index, y in enumerate(y_list):

        # Add plot
        if by_row:
            ax = fig.add_subplot(num_plot_row, 2, index + 1, projection="3d")
        else:
            if index % 2 == 0:
                ax = fig.add_subplot(2, num_plot_col, ceil(index / 2) + 1, projection="3d")
            else:
                ax = fig.add_subplot(2, num_plot_col, ceil(index / 2) + half_num_plot, projection="3d")

        # Plot surface
        if single_colorbar:
            surf = ax.plot_surface(
                X=x_list[0], Y=x_list[1], Z=y,
                vmin=min_value, vmax=max_value,
                rstride=1, cstride=1, cmap="RdGy", linewidth=0, antialiased=False,
            )
        else:
            surf = ax.plot_surface(
                X=x_list[0], Y=x_list[1], Z=y,
                rstride=1, cstride=1, cmap="RdGy", linewidth=0, antialiased=False,
            )

        # Add title, labels and colorbar
        ax.set_xlabel(r"$x_1$", size=15)
        ax.set_ylabel(r"$x_2$", size=15)
        ax.set_title(titles[index], size=20)
        if not single_colorbar: fig.colorbar(surf, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    
    if single_colorbar:
        levels = np.linspace(min_value - 0.01, max_value + 0.01, 100)
        cmap = plt.get_cmap("RdGy", len(levels))
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if by_row:
            cbar_ax = fig.add_axes([1, 0.1, 0.03, 0.6])
            fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
        else:
            cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.6])
            fig.colorbar(sm, cax=cbar_ax, shrink=0.5)
    
    plt.show()

def ScatterPoints(w, level_num=3, range_list=[[-4.,4.], [-4.,4.]]):
    """Scatter plot of sample at each qualitative (`level_num` levels)"""

    z = w[:, 0] # z start from 1
    x = w[:, 1:]

    fig, axs = plt.subplots(ncols=level_num, figsize=(2 + 5 * level_num, 5))

    for i in range(level_num):
        z_cond = z == (i+1)
        axs[i].scatter(x[z_cond, 0], x[z_cond, 1], marker=".", s=80, color="gray")
        axs[i].set_xlabel(r"$x_1$", size=15)
        axs[i].set_ylabel(r"$x_2$", size=15)
        axs[i].set_title("Points of z =" + str(i+1), size=17)
        axs[i].set_xlim(range_list[0][0], range_list[0][1])
        axs[i].set_ylim(range_list[1][0], range_list[1][1])

    plt.tight_layout()
    plt.show()

def PlotHistory(ei_value_list, time_list, cr_list):
    """line plot of EI-value, time cost and contribution rate"""
    # Mean
    mean_list = [
        np.mean(sub_list, axis=0)
        for sub_list in [ei_value_list, time_list, cr_list]
    ]

    # Quantile
    quantile_list = [
        np.percentile(np.row_stack(sub_list), q=(2.5, 97.5), axis=0)
        for sub_list in [ei_value_list, time_list, cr_list]
    ]
    
    # Plot
    fig, axs = plt.subplots(ncols=3, figsize=(16, 4))

    for i, sub_list in enumerate([ei_value_list, time_list, cr_list]):
        for single_list in sub_list:
            axs[i].plot(single_list, c="lightgray")
        axs[i].plot(quantile_list[i][0], c="red")
        axs[i].plot(quantile_list[i][1], c="red")
        axs[i].plot(mean_list[i], c="blue")
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        
    axs[0].set_title("History of EHVI")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("EI value")

    axs[1].set_title("History of time cost")
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("time (sec)")

    axs[2].set_title("History of contribution rate (CR)")
    axs[2].set_ylabel("CR value")
    axs[2].set_xlabel("step")

    fig.tight_layout()
    plt.show()

def PlotNoiseEstimation(dict_list, step=20, suptitle=""):

    noise_arr_list = []
    for sub_dict in dict_list:
        noise_arr = np.array([sub_dict["parm"][j]["noise_sigma"] for j in range(step)])
        noise_arr_list.append(noise_arr)

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5))

    for sub_arr in noise_arr_list:
        axs[0].plot(sub_arr[:,0], c="lightgray")
        axs[1].plot(sub_arr[:,1], c="lightgray")

    axs[0].plot(np.quantile(np.array([arr[:,0] for arr in noise_arr_list]), q=0.975, axis=0), c="red", label="97.5%")
    axs[0].plot(np.quantile(np.array([arr[:,0] for arr in noise_arr_list]), q=0.5, axis=0), c="blue", label="median")
    axs[0].plot(np.quantile(np.array([arr[:,0] for arr in noise_arr_list]), q=0.025, axis=0), c="red", label="2.5%")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xlabel("step", size=15)
    axs[0].set_ylabel("value", size=15)
    axs[0].set_title(r"Estimation of $\sigma_1$", size=17)

    axs[1].plot(np.quantile(np.array([arr[:,1] for arr in noise_arr_list]), q=0.975, axis=0), c="red", label="97.5%")
    axs[1].plot(np.quantile(np.array([arr[:,1] for arr in noise_arr_list]), q=0.5, axis=0), c="blue", label="median")
    axs[1].plot(np.quantile(np.array([arr[:,1] for arr in noise_arr_list]), q=0.025, axis=0), c="red", label="2.5%")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_xlabel("step", size=15)
    axs[1].set_ylabel("value", size=15)
    axs[1].set_title(r"Estimation of $\sigma_2$", size=17)

    plt.suptitle(suptitle, size=18)
    plt.tight_layout()
    plt.legend()
    plt.show()

def PlotCorrEstimation(dict_list, step=20, suptitle=""):

    TaskCorr_arr_list = []
    for sub_dict in dict_list:
        corr_arr = np.array([sub_dict["parm"][j]["TaskCovar"][0,1] / np.sqrt(np.product(np.diag(sub_dict["parm"][j]["TaskCovar"]))) for j in range(step)])
        TaskCorr_arr_list.append(corr_arr)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sub_arr in TaskCorr_arr_list:ax.plot(sub_arr, c="lightgray")

    ax.plot(np.quantile(np.array(TaskCorr_arr_list), q=0.975, axis=0), c="red", label="97.5%")
    ax.plot(np.quantile(np.array(TaskCorr_arr_list), q=0.5, axis=0), c="blue", label="median")
    ax.plot(np.quantile(np.array(TaskCorr_arr_list), q=0.025, axis=0), c="red", label="2.5%")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("step", size=15)
    ax.set_ylabel("value", size=15)
    ax.set_title("Estimation of correlation (Task)", size=17)

    plt.suptitle(suptitle, size=18)
    plt.tight_layout()
    plt.legend()
    plt.show()

def PlotDataStepCorr(dist_list, start_index, end_index, suptitle=""):

    StepCorr_arr_list = []
    for step in range(start_index, end_index):
        sub_corr_arr = np.array([
            np.corrcoef(sub_dict["output_list"][0][:step], sub_dict["output_list"][1][:step])[0,1]
            for sub_dict in dist_list
        ])
        StepCorr_arr_list.append(sub_corr_arr)


    StepCorr_arr = np.column_stack(StepCorr_arr_list)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sub_arr in StepCorr_arr:ax.plot(sub_arr, c="lightgray")

    ax.plot(np.quantile(StepCorr_arr, q=0.975, axis=0), c="red", label="97.5%")
    ax.plot(np.quantile(StepCorr_arr, q=0.5, axis=0), c="blue", label="median")
    ax.plot(np.quantile(StepCorr_arr, q=0.025, axis=0), c="red", label="2.5%")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("step", size=15)
    ax.set_ylabel("value", size=15)
    ax.set_title("Correlation (current data)", size=17)

    plt.suptitle(suptitle, size=18)
    plt.tight_layout()
    plt.legend()
    plt.show()

def PlotTrueCorr(dict_list, title_note=r"(with noise $\sigma_1=\sigma_2=0.15$)"):

    overall_true_corr_arr = np.array([
    np.corrcoef(np.concatenate((sub_dict["z1y1"], sub_dict["z2y1"])), np.concatenate((sub_dict["z1y2"], sub_dict["z2y2"])))[0,1]
    for sub_dict in dict_list
    ])
    z1_true_corr_arr = np.array([
        np.corrcoef(sub_dict["z1y1"], sub_dict["z1y2"])[0,1]
        for sub_dict in dict_list
    ])
    z2_true_corr_arr = np.array([
        np.corrcoef(sub_dict["z2y1"], sub_dict["z2y2"])[0,1]
        for sub_dict in dict_list
    ])
    corr = np.concatenate((overall_true_corr_arr, z1_true_corr_arr, z2_true_corr_arr))
    corr_type = np.array(
        ["overall"] * len(overall_true_corr_arr) + \
        ["q=1"] * len(z1_true_corr_arr) + \
        ["q=2"] * len(z2_true_corr_arr)\
    )
    corr_df = pd.DataFrame({"corr_type":corr_type, "corr":corr})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.boxplot(x="corr_type", y="corr", data=corr_df, showfliers = False)
    ax = sns.swarmplot(x="corr_type", y="corr", data=corr_df, color=".25")

    ax.set_xlabel("Correlation type")
    ax.set_xlabel("Correlation value")
    ax.set_title("True surface correlation\n" + title_note)

    plt.show()

def PlotHistoryQuantile(cr_lists, model_name_list, color_list=["red", "green"], title="Contribution rate", y_label="CR value", step_num=40, y_lim=None):
    fig, ax = plt.subplots(figsize=(10,6))

    for i, sub_list in enumerate(cr_lists): 
        ax.fill_between(
            x=np.array(range(step_num+1)),
            y1=np.quantile(sub_list, q=0.025, axis=0)[:step_num+1],
            y2=np.quantile(sub_list, q=0.975, axis=0)[:step_num+1],
            alpha=0.2,
            color=color_list[i],
            label="[0.025, 0.975] quantile of " + model_name_list[i]
        )    
        ax.plot(np.quantile(sub_list, q=0.025, axis=0)[:step_num+1], linestyle='-.', c=color_list[i])
        ax.plot(np.quantile(sub_list, q=0.5, axis=0)[:step_num+1], label="Median of " + model_name_list[i], c=color_list[i])
        ax.plot(np.quantile(sub_list, q=0.975, axis=0)[:step_num+1], linestyle='-.', c=color_list[i])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Step index", size=15)
    ax.set_ylabel(y_label, size=15)
    ax.set_title(title, size=15)
    
    ax.set_xlim(0, step_num)
    if y_lim: ax.set_ylim(*y_lim)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    plt.legend()
    plt.tight_layout()
    plt.show()

def PlotAllHistoryQuantile(cr_lists:list, time_lists:list, eiv_lists:list, step_num_list:list, model_name_list:list, color_list:list, **kwargs):
    fig, axs = plt.subplots(ncols=3, figsize=(18,4))


    for ax_idx, att in enumerate([cr_lists, time_lists, eiv_lists]):
        for i, sub_list in enumerate(att): 
            axs[ax_idx].fill_between(
                x=np.array(range(step_num_list[ax_idx]+1)),
                y1=np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[ax_idx]+1],
                y2=np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[ax_idx]+1],
                alpha=0.2,
                color=color_list[i],
                label="[0.025, 0.975] quantile of " + model_name_list[i] if model_name_list else "[0.025, 0.975] quantile"
            )    
            axs[ax_idx].plot(np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[ax_idx]+1], linestyle='-.', c=color_list[i])
            axs[ax_idx].plot(np.quantile(sub_list, q=0.5, axis=0)[:step_num_list[ax_idx]+1], label="Median of " + model_name_list[i] if model_name_list else "Median", c=color_list[i])
            axs[ax_idx].plot(np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[ax_idx]+1], linestyle='-.', c=color_list[i])

    for i in range(3):
        axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[i].set_xlim(0, step_num_list[i])

    axs[0].set_xlabel("Step index", size=20, fontweight="bold")
    axs[1].set_xlabel("Step index", size=20, fontweight="bold")
    axs[2].set_xlabel("Step index", size=20, fontweight="bold")

    axs[0].set_ylabel("CR value", size=20, fontweight="bold")
    axs[1].set_ylabel("Seconds", size=20, fontweight="bold")
    axs[2].set_ylabel("Max " + r"$EI_{\mathcal{H}}$", size=20, fontweight="bold")

    axs[0].set_title("(a)", size=25, fontweight="bold")
    axs[1].set_title("(b)", size=25, fontweight="bold")
    axs[2].set_title("(c)", size=25, fontweight="bold")

    if "cr_lim" in kwargs:axs[0].set_ylim(*kwargs["cr_lim"])
    if "time_lim" in kwargs:axs[1].set_ylim(*kwargs["time_lim"])
    if "ei_lim" in kwargs:axs[2].set_ylim(*kwargs["ei_lim"])
    
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    plt.legend()
    plt.tight_layout()
    plt.show()

def PlotHisQuan(cr_lists:list, time_lists:list, eiv_lists:list, step_num_list:list, model_name_list:list, color_list:list, **kwargs):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18,10))

    # CR
    for i, sub_list in enumerate(cr_lists): 
        axs[0,0].fill_between(
            x=np.array(range(step_num_list[0]+1)),
            y1=np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[0]+1],
            y2=np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[0]+1],
            alpha=0.2,
            color=color_list[i],
            label="[0.025, 0.975] quantile of " + model_name_list[i] if model_name_list else "[0.025, 0.975] quantile"
        )    
        axs[0,0].plot(np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[0]+1], linestyle='-.', c=color_list[i])
        axs[0,0].plot(np.quantile(sub_list, q=0.5, axis=0)[:step_num_list[0]+1], label="Median of " + model_name_list[i] if model_name_list else "Median", c=color_list[i])
        axs[0,0].plot(np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[0]+1], linestyle='-.', c=color_list[i])

    # Time
    for i, sub_list in enumerate(time_lists): 
        axs[0,1].fill_between(
            x=np.array(range(step_num_list[1]+1)),
            y1=np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[1]+1],
            y2=np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[1]+1],
            alpha=0.2,
            color=color_list[i],
            label="[0.025, 0.975] quantile of " + model_name_list[i] if model_name_list else "[0.025, 0.975] quantile"
        )    
        axs[0,1].plot(np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[1]+1], linestyle='-.', c=color_list[i])
        axs[0,1].plot(np.quantile(sub_list, q=0.5, axis=0)[:step_num_list[1]+1], label="Median of " + model_name_list[i] if model_name_list else "Median", c=color_list[i])
        axs[0,1].plot(np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[1]+1], linestyle='-.', c=color_list[i])

    # EI value
    for i, sub_list in enumerate(eiv_lists): 
        axs[1,0].fill_between(
            x=np.array(range(step_num_list[2]+1)),
            y1=np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[2]+1],
            y2=np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[2]+1],
            alpha=0.2,
            color=color_list[i],
            label="[0.025, 0.975] quantile of " + model_name_list[i] if model_name_list else "[0.025, 0.975] quantile"
        )    
        axs[1,0].plot(np.quantile(sub_list, q=0.025, axis=0)[:step_num_list[2]+1], linestyle='-.', c=color_list[i])
        axs[1,0].plot(np.quantile(sub_list, q=0.5, axis=0)[:step_num_list[2]+1], label="Median of " + model_name_list[i] if model_name_list else "Median", c=color_list[i])
        axs[1,0].plot(np.quantile(sub_list, q=0.975, axis=0)[:step_num_list[2]+1], linestyle='-.', c=color_list[i])

    # For legend
    for i, sub_list in enumerate(cr_lists): 
        axs[1,1].fill_between(
            x=[], y1=[], y2=[], alpha=0.2, color=color_list[i],
            label="[0.025, 0.975] quantile of " + model_name_list[i] if model_name_list else "[0.025, 0.975] quantile"
        )    
        axs[1,1].plot([], label="Median of " + model_name_list[i] if model_name_list else "Median", c=color_list[i])
    axs[1,1].axis("off")

    axs[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0,1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1,0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0,0].set_xlim(0, step_num_list[0])
    axs[0,1].set_xlim(0, step_num_list[1])
    axs[1,0].set_xlim(0, step_num_list[2])


    axs[0,0].set_xlabel("Step index", size=20, fontweight="bold")
    axs[0,1].set_xlabel("Step index", size=20, fontweight="bold")
    axs[1,0].set_xlabel("Step index", size=20, fontweight="bold")

    axs[0,0].set_ylabel("CR value", size=23, fontweight="bold")
    axs[0,1].set_ylabel("Seconds", size=23, fontweight="bold")
    axs[1,0].set_ylabel("Max " + r"$EI_{\mathcal{H}}$", size=23, fontweight="bold")

    axs[0,0].set_title("(a)", size=25, fontweight="bold")
    axs[0,1].set_title("(b)", size=25, fontweight="bold")
    axs[1,0].set_title("(c)", size=25, fontweight="bold")

    if "cr_lim" in kwargs:axs[0,0].set_ylim(*kwargs["cr_lim"])
    if "time_lim" in kwargs:axs[0,1].set_ylim(*kwargs["time_lim"])
    if "ei_lim" in kwargs:axs[1,0].set_ylim(*kwargs["ei_lim"])
    
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.8))
    axs[1,1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()