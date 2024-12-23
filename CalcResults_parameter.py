import os
import pandas as pd
import numpy as np
import json
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt
from log_distance_measures.config import EventLogIDs, discretize_to_hour, AbsoluteTimestampType
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from log_distance_measures.work_in_progress import work_in_progress_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from Utility import split_data_by_case
# import pm4py
# from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
# from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
# from pm4py.algo.evaluation.generalization import evaluator as generalization_evaluator
# from pm4py.algo.evaluation.behavioral_profile import evaluator as behavioral_profile_evaluator

def get_subfolder_names(folder_path):
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

folder_path = 'SimulatedLogs'
subfolder_names = get_subfolder_names(folder_path)

event_log_ids = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
    case="case_id",
    activity="activity",
    start_time="start_time",
    end_time="end_time",
    resource="resource"
)

event_log_ids_test = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
    case="case:concept:name",
    activity="concept:name",
    start_time="start_time",
    end_time="end_time",
    resource="org:resource"
)

def compute_confidence_interval(data, confidence=0.95):
    """
    Computes the confidence interval for a given list of data using the t-score method.
    
    Parameters:
        data (list or array): A list or array of numerical data points.
        confidence (float): The confidence level for the interval (default is 0.95 for 95%).
        
    Returns:
        tuple: The mean and the confidence interval as (mean, (lower_bound, upper_bound)).
    """
    if len(data) < 2:
        raise ValueError("At least two data points are required to compute a confidence interval.")
    
    # Compute mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation
    
    # Sample size and standard error
    n = len(data)
    standard_error = std_dev / np.sqrt(n)
    
    # Degrees of freedom and t-score
    df = n - 1
    t_score = t.ppf((1 + confidence) / 2, df)
    
    # Compute margin of error and confidence interval
    margin_of_error = t_score * standard_error
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    
    return mean, confidence_interval

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h

# Initialize lists to store distances
fourier_distances = {
    "AED": [],
    "CED": [],
    "RED": [],
    "CAR": [],
    "CTD": [],
    "CWD": [],
    "WIPD": [],
}

names = {"4_False": "Long-Term 4 days",
            "7_False": "Long-Term 7 days",
            "7_True": "Short-Term 7 days",
            "10_False": "Long-Term 10 days"}

plot_names = {"AcademicCredentials": "AC_CRE",
              "BPI_Challenge_2012": "BP_FUL",
              "BPI_Challenge_2012_ACTIVITIES": "BP_FIL",
              "Hospital_Billing_Event_Log": "HBP",
              "LoanApp" : "LOAN",
              "Procure2Pay": "P2P",}

results = []

folders = []

for subfolder in subfolder_names:
    if(subfolder == "skip"):
        continue

    
    test_event_log = pd.read_csv(f"{folder_path}/{subfolder}/log_test.csv")
    test_event_log[event_log_ids_test.start_time] = pd.to_datetime(test_event_log[event_log_ids_test.start_time], format='ISO8601')
    test_event_log[event_log_ids_test.end_time] = pd.to_datetime(test_event_log[event_log_ids_test.end_time], format='ISO8601')

    full_test_log = pd.read_csv(f"{folder_path}/{subfolder}/log_entire.csv")
    full_test_log[event_log_ids_test.start_time] = pd.to_datetime(full_test_log[event_log_ids_test.start_time], format='ISO8601')
    full_test_log[event_log_ids_test.end_time] = pd.to_datetime(full_test_log[event_log_ids_test.end_time], format='ISO8601')

    plottest = []
    plotentire = []

    folders = get_subfolder_names(f"{folder_path}/{subfolder}")
    for folder in folders:
        if(folder == "skip"):
            continue
        fourier_distances = {
            "AED": [],
            "CED": [],
            "RED": [],
            "CAR": [],
            "CTD": [],
            "CWD": [],
            "WIPD": [],
        }
        
        for i in range(1, 11):
            print("Folder path")
            print(f"{folder_path}/{subfolder}/{folder}/log_{i}.csv")
            fourier_event_log = pd.read_csv(f"{folder_path}/{subfolder}/{folder}/log_{i}.csv")
            fourier_event_log[event_log_ids.start_time] = pd.to_datetime(fourier_event_log[event_log_ids.start_time], format='ISO8601')
            fourier_event_log[event_log_ids.end_time] = pd.to_datetime(fourier_event_log[event_log_ids.end_time], format='ISO8601')
            train_fourier_event_log, test_fourier_event_log = split_data_by_case(fourier_event_log, 0.8, "case_id", timestamp=event_log_ids.start_time)

            fourier_distances["CAR"].append(case_arrival_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids))
            fourier_distances["CWD"].append(circadian_workforce_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids))
            fourier_distances["WIPD"].append(work_in_progress_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids, window_size=pd.Timedelta(hours=1)))
            fourier_distances["AED"].append(absolute_event_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids, discretize_type=AbsoluteTimestampType.BOTH, discretize_event=discretize_to_hour))
            fourier_distances["CED"].append(circadian_event_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids, discretize_type=AbsoluteTimestampType.BOTH))
            fourier_distances["RED"].append(relative_event_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids, discretize_type=AbsoluteTimestampType.BOTH, discretize_event=discretize_to_hour))
            fourier_distances["CTD"].append(cycle_time_distribution_distance(test_event_log, event_log_ids_test, test_fourier_event_log, event_log_ids, bin_size=pd.Timedelta(hours=1)))
        
        # Calculate mean and 95% confidence interval for each distance measure
        for key in fourier_distances:
            mean, margin_of_error = mean_confidence_interval(fourier_distances[key])
            results.append({
                "parameter": folder,
                "distance_type": key,
                "mean": mean,
                "margin_of_error": margin_of_error
            })

        with open(f"{folder_path}/{subfolder}/results.json", 'w') as json_file:
            json.dump(results, json_file, indent=4)

        min_start_times_fourier = test_fourier_event_log.groupby('case_id', as_index=False)['start_time'].min()
        min_start_times_fourier['start_date'] = min_start_times_fourier['start_time'].dt.date
        cases_per_date_fourier = min_start_times_fourier.groupby('start_date').size()

        plottest.append({folder: cases_per_date_fourier})

        min_start_times_fourier = fourier_event_log.groupby('case_id', as_index=False)['start_time'].min()
        min_start_times_fourier['start_date'] = min_start_times_fourier['start_time'].dt.date
        cases_per_date_fourier = min_start_times_fourier.groupby('start_date').size()

        plotentire.append({folder: cases_per_date_fourier})



   
    
    min_start_times_test = test_event_log.groupby('case:concept:name', as_index=False)['start_time'].min()

    min_start_times_test['start_date'] = min_start_times_test['start_time'].dt.date

    cases_per_date_test = min_start_times_test.groupby('start_date').size()
    plt.figure(figsize=(10, 6))


    for plot_dict in plottest:
        for key in plot_dict:
            # Extract the part before ".json" and split by underscores
            parts = key.split(".")[0].split("_")
            # Get the last two parts
            parameter_part = "_".join(parts[-2:])
            print(parameter_part)
            plt.plot(plot_dict[key].index, plot_dict[key].values, label=f'{names[parameter_part]}')

    # for i in range(len(plottest)):
    #     plt.plot(plottest[i].index, plottest[i].values, label=f'{names[i]}')
    #plt.plot(cases_per_date_fourier.index, cases_per_date_fourier.values, label='Fourier')
    plt.plot(cases_per_date_test.index, cases_per_date_test.values, label='Test Data')

    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title('Number of Cases per Date')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{folder_path}/{subfolder}/number_of_cases_per_test.png')

    min_start_times_test = full_test_log.groupby('case:concept:name', as_index=False)['start_time'].min()

    min_start_times_test['start_date'] = min_start_times_test['start_time'].dt.date

    cases_per_date_test = min_start_times_test.groupby('start_date').size()

    plt.figure(figsize=(10, 6))

    #plt.plot(cases_per_date_fourier.index, cases_per_date_fourier.values, label='Fourier')

    for plot_dict in plotentire:
        for key in plot_dict:
            # Extract the part before ".json" and split by underscores
            parts = key.split(".")[0].split("_")
            # Get the last two parts
            parameter_part = "_".join(parts[-2:])
            print(parameter_part)
            plt.plot(plot_dict[key].index, plot_dict[key].values, label=f'{names[parameter_part]}')

    plt.plot(cases_per_date_test.index, cases_per_date_test.values, label='Real Log')


    json_file_path = f"{folder_path}/{subfolder}/info.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    vertical_line_date = pd.to_datetime(data['first_date_test'], utc=True)

    plt.axvline(pd.to_datetime(vertical_line_date), color='r', linestyle='--', label='Begin Test Data')


    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    title = f'Case Arrivals by Date for {plot_names[subfolder]}'
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f'{folder_path}/{subfolder}/number_of_cases_per_entire.png')
    plt.savefig(f'{folder_path}/{subfolder}/number_of_cases_per_entire.svg')






# min_start_times_simod = test_simod_event_log.groupby('case_id', as_index=False)['start_time'].min()
# min_start_times_fourier = test_fourier_event_log.groupby('case_id', as_index=False)['start_time'].min()
# min_start_times_test = test_event_log.groupby('case:concept:name', as_index=False)['start_time'].min()

# min_start_times_simod['start_date'] = min_start_times_simod['start_time'].dt.date
# min_start_times_fourier['start_date'] = min_start_times_fourier['start_time'].dt.date
# min_start_times_test['start_date'] = min_start_times_test['start_time'].dt.date

# cases_per_date_simod = min_start_times_simod.groupby('start_date').size()
# cases_per_date_fourier = min_start_times_fourier.groupby('start_date').size()
# cases_per_date_test = min_start_times_test.groupby('start_date').size()
# plt.figure(figsize=(10, 6))

# plt.plot(cases_per_date_simod.index, cases_per_date_simod.values, label='Simod')
# plt.plot(cases_per_date_fourier.index, cases_per_date_fourier.values, label='Fourier')
# plt.plot(cases_per_date_test.index, cases_per_date_test.values, label='Test')

# plt.xlabel('Date')
# plt.ylabel('Number of Cases')
# plt.title('Number of Cases per Date')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show(block=False)

# min_start_times_simod = simod_event_log.groupby('case_id', as_index=False)['start_time'].min()
# min_start_times_fourier = fourier_event_log.groupby('case_id', as_index=False)['start_time'].min()
# min_start_times_test = full_test_log.groupby('case:concept:name', as_index=False)['start_time'].min()

# min_start_times_simod['start_date'] = min_start_times_simod['start_time'].dt.date
# min_start_times_fourier['start_date'] = min_start_times_fourier['start_time'].dt.date
# min_start_times_test['start_date'] = min_start_times_test['start_time'].dt.date

# cases_per_date_simod = min_start_times_simod.groupby('start_date').size()
# cases_per_date_fourier = min_start_times_fourier.groupby('start_date').size()
# cases_per_date_test = min_start_times_test.groupby('start_date').size()

# plt.figure(figsize=(10, 6))

# plt.plot(cases_per_date_simod.index, cases_per_date_simod.values, label='Simod')
# plt.plot(cases_per_date_fourier.index, cases_per_date_fourier.values, label='Fourier')
# plt.plot(cases_per_date_test.index, cases_per_date_test.values, label='Test')

# plt.xlabel('Date')
# plt.ylabel('Number of Cases')
# plt.title('Number of Cases per Date')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show(block=False)

# #print number of unique cases
# print("Number of unique cases in simod", len(min_start_times_simod))
# print("Number of unique cases in fourier", len(min_start_times_fourier))
# print("Number of unique cases in test", len(min_start_times_test))


# x = input()


# # Load your event logs
# test_log = pm4py.read_xes('test_log.xes')
# simulated_log = pm4py.read_xes('simulated_log.xes')

# # Discover process models
# test_model, test_initial_marking, test_final_marking = pm4py.discover_petri_net_alpha(test_log)
# sim_model, sim_initial_marking, sim_final_marking = pm4py.discover_petri_net_alpha(simulated_log)

# # Replay fitness
# fitness = replay_fitness_evaluator.apply(test_log, test_model, test_initial_marking, test_final_marking)
# print(f"Replay Fitness: {fitness['average_fitness']}")

# # Precision
# precision = precision_evaluator.apply(test_log, test_model, test_initial_marking, test_final_marking)
# print(f"Precision: {precision}")

# # Generalization
# generalization = generalization_evaluator.apply(test_log, test_model, test_initial_marking, test_final_marking)
# print(f"Generalization: {generalization}")

# # Behavioral Profile Similarity
# similarity = behavioral_profile_evaluator.apply(test_log, simulated_log)
# print(f"Behavioral Profile Similarity: {similarity}")