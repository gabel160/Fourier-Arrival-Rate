import pm4py
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

def filter_log_by_activity(event_log, activity_name):
    return event_log[event_log["concept:name"] == activity_name]

def filter_log_by_resource(event_log, resource_name):
    return event_log[event_log["org:resource"] == resource_name]

def import_xes(file_path):
    return pm4py.read_xes(file_path)

def export_xes(event_log, file_path):
    pm4py.write_xes(event_log, file_path)

def prepare_dataset_timestamp(dataset):
    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'])
    dataset = dataset.set_index('time:timestamp')
    dataset = dataset.sort_index()
    return dataset

def average_time_diff_by_resource(dataset, activity_name):
    dataset = dataset.sort_values('time:timestamp')
    dataset['time_diff'] = dataset.groupby('case:concept:name')['time:timestamp'].diff()
    dataset["time_diff"] = dataset["time_diff"].dt.total_seconds() / 3600
    dataset = dataset[dataset['concept:name'] == activity_name]
    grouped = dataset.groupby('org:resource')['time_diff'].mean()
    mean_value = grouped.mean()
    grouped = pd.concat([grouped, pd.Series([mean_value], index=['mean'])])
    print("Differenz zu Max:" + str(((grouped.max() - grouped.mean()) / grouped.mean()) * 100))
    print("Differenz zu Min:" + str(((grouped.min() - grouped.mean()) / grouped.mean()) * 100))
    return grouped

def count_activity_by_resource(dataset, activity_name):
    dataset = filter_log_by_activity(dataset, activity_name)
    dataset = dataset.groupby('org:resource').size()
    mean_value = dataset.mean()
    dataset = pd.concat([dataset, pd.Series([mean_value], index=['mean'])])
    return dataset

def split_data_by_case(dataset, train_percentage, case_id_name='case:concept:name', timestamp = 'time:timestamp'):
    """
    Splits the dataset into training and test sets based on the given percentage,
    ensuring that cases are not split between the sets.

    Parameters:
    dataset (pd.DataFrame): The dataset to split.
    train_percentage (float): The percentage of data to use for training (e.g., 0.8 for 80%).

    Returns:
    tuple: A tuple containing the training set and the test set.
    """
    # Group the dataset by the "case:concept:name" column
    grouped = dataset.groupby(case_id_name)
    
    # Get the unique case names
    case_names = sorted(grouped.groups.keys(), key=lambda case: grouped.get_group(case)[timestamp].iloc[0])
    
    # Calculate the number of cases for the training set
    train_size = int(len(case_names) * train_percentage)
    
    # Split the case names into training and test sets
    train_case_names = list(case_names)[:train_size]
    test_case_names = list(case_names)[train_size:]
    
    # Create the training and test sets by concatenating the grouped data
    train_set = pd.concat([grouped.get_group(case) for case in train_case_names])
    test_set = pd.concat([grouped.get_group(case) for case in test_case_names])
    
    return train_set, test_set

def group_resources_by_activity(dataset):
    # Group by activity and resource
    activity_resource_groups = dataset.groupby(['concept:name', 'org:resource']).size().reset_index(name='count')

    # Create a mapping of resources to the activities they work on
    resource_to_activities = defaultdict(set)
    for _, row in activity_resource_groups.iterrows():
        activity = row['concept:name']
        resource = row['org:resource']
        resource_to_activities[resource].add(activity)

    # Group resources into classes based on the activities they work on
    activity_to_resources = defaultdict(list)
    for resource, activities in resource_to_activities.items():
        activity_tuple = tuple(sorted(activities))  # Sort to ensure consistent grouping
        activity_to_resources[activity_tuple].append(resource)

    filtered_activity_to_resources = {activities: resources for activities, resources in activity_to_resources.items() if len(resources) > 1}

    return filtered_activity_to_resources

def map_resource_to_class(resource, resource_classes):
    for activities, resources in resource_classes.items():
        if resource in resources:
            return activities
    return None

def calculate_activities_per_day(dataset, resource_classes):
    # Map each resource to its resource class
    dataset['resource_class'] = dataset['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

    # Convert timestamp to date
    dataset['date'] = dataset['time:timestamp'].dt.date

    # Group by date and resource class to count activities
    activities_per_day = dataset.groupby(['date', 'resource_class']).size().reset_index(name='activity_count')

    return activities_per_day

# def calculate_specific_activity_per_day(dataset, resource_classes, activity_name):
#     # Filter dataset to include only the specified activity
#     dataset = dataset[dataset['concept:name'] == activity_name]

#     # Map each resource to its resource class
#     dataset['resource_class'] = dataset['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

#     # Convert timestamp to date
#     dataset['date'] = dataset['time:timestamp'].dt.date

#     # Group by date and resource class to count activities
#     activities_per_day = dataset.groupby(['date', 'resource_class']).size().reset_index(name='activity_count')

#     return activities_per_day

# def calculate_specific_activity_per_day(dataset, resource_classes, activity_name, start_date, end_date):
#     # Filter dataset to include only the specified activity
#     dataset = dataset[dataset['concept:name'] == activity_name]

#     # Map each resource to its resource class
#     dataset['resource_class'] = dataset['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

#     # Convert timestamp to date
#     dataset['date'] = dataset['time:timestamp'].dt.date

#     # Group by date and resource class to count activities
#     activities_per_day = dataset.groupby(['date', 'resource_class']).size().reset_index(name='activity_count')

#     # Create a date range from start_date to end_date
#     date_range = pd.date_range(start=start_date, end=end_date)

#     # Create a DataFrame with all combinations of dates and resource classes
#     all_combinations = pd.MultiIndex.from_product([date_range, resource_classes.keys()], names=['date', 'resource_class']).to_frame(index=False)

#     # Ensure the 'date' columns are of the same type
#     all_combinations['date'] = all_combinations['date'].dt.date

#     # Merge the activities_per_day with all_combinations to fill missing dates/resource classes with 0
#     activities_per_day = all_combinations.merge(activities_per_day, on=['date', 'resource_class'], how='left').fillna(0)

#     return activities_per_day

def calculate_specific_activity_per_day(dataset, resource_classes, activity_name, start_date, end_date):
    # Filter dataset to include only the specified activity
    dataset = dataset[dataset['concept:name'] == activity_name]

    # Map each resource to its resource class
    dataset['resource_class'] = dataset['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

    # Convert timestamp to date
    dataset['date'] = dataset['time:timestamp'].dt.date

    # Group by date and resource class to count activities
    activities_per_day = dataset.groupby(['date', 'resource_class']).size().reset_index(name='activity_count')

    # Create a date range from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date)

    # Filter resource classes to only include those present in the filtered dataset
    relevant_resource_classes = activities_per_day['resource_class'].unique()

    # Create a DataFrame with all combinations of dates and relevant resource classes
    all_combinations = pd.MultiIndex.from_product([date_range, relevant_resource_classes], names=['date', 'resource_class']).to_frame(index=False)

    # Ensure the 'date' columns are of the same type
    all_combinations['date'] = all_combinations['date'].dt.date

    # Merge the activities_per_day with all_combinations to fill missing dates/resource classes with 0
    activities_per_day = all_combinations.merge(activities_per_day, on=['date', 'resource_class'], how='left').fillna(0)

    return activities_per_day


def plot_activities_per_day(activities_per_day):
    # Plot the data
    plt.figure(figsize=(12, 8))
    for resource_class in activities_per_day['resource_class'].unique():
        class_data = activities_per_day[activities_per_day['resource_class'] == resource_class]
        plt.plot(class_data['date'], class_data['activity_count'], label=f'Class: {resource_class}')

    plt.xlabel('Date')
    plt.ylabel('Number of Activities')
    plt.title('Number of Activities Executed Per Day for Each Resource Class')
    plt.legend()
    plt.show(block=False)

    # random_classes = np.random.choice(activities_per_day['resource_class'].unique(), 5, replace=False)

    # # Plot the data for the selected random resource classes
    # plt.figure(figsize=(12, 8))
    # for resource_class in random_classes:
    #     class_data = activities_per_day[activities_per_day['resource_class'] == resource_class]
    #     plt.plot(class_data['date'], class_data['activity_count'], label=f'Class: {resource_class}')

    # plt.xlabel('Date')
    # plt.ylabel('Number of Activities')
    # plt.title('Number of Activities Executed Per Day for 5 random Resource Classes')
    # plt.legend()
    # plt.show(block=False)

def rename_resource_classes(activities_per_day):
    # Create a mapping from resource class names to numbers
    resource_class_mapping = {name: idx for idx, name in enumerate(activities_per_day['resource_class'].unique())}
    
    # Apply the mapping to the resource_class column
    activities_per_day['resource_class'] = activities_per_day['resource_class'].map(resource_class_mapping)
    
    return activities_per_day, resource_class_mapping

def calculate_activity_executions_per_day(dataset, activity_name, resource_classes):
    # Filter the dataset for the specified activity
    filtered_data = dataset[dataset['concept:name'] == activity_name]

    # Map each resource to its class
    filtered_data['resource_class'] = filtered_data['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

    filtered_data['date'] = filtered_data['time:timestamp'].dt.date

    # Group by date and resource class, then count the number of executions
    executions_per_day = filtered_data.groupby(['date', 'resource_class']).size().reset_index(name='execution_count')

    total_executions = executions_per_day['execution_count'].sum()
    print(f"Total executions: {total_executions}")

    return executions_per_day

def calculate_activity_executions_per_day_DateAndCount(dataset, activity_name, resource_classes, start_date, end_date):
    # Filter the dataset for the specified activity
    filtered_data = dataset[dataset['concept:name'] == activity_name]

    # Map each resource to its class
    filtered_data['resource_class'] = filtered_data['org:resource'].apply(lambda x: map_resource_to_class(x, resource_classes))

    filtered_data['date'] = filtered_data['time:timestamp'].dt.date

    # Group by date and resource class, then count the number of executions
    executions_per_day = filtered_data.groupby(['date', 'resource_class']).size().reset_index(name='execution_count')

    # Create a date range from start_date to end_date
    all_dates = pd.date_range(start=start_date, end=end_date).date

    # Create a DataFrame with all dates and merge with executions_per_day
    all_dates_df = pd.DataFrame(all_dates, columns=['date'])
    executions_per_day = all_dates_df.merge(executions_per_day, on='date', how='left').fillna(0)

    # Sum the execution counts across all resource classes for each date
    executions_per_day = executions_per_day.groupby('date')['execution_count'].sum().reset_index()

    # Rename the columns
    executions_per_day.columns = ['time:timestamp', 'count']

    return executions_per_day

def plot_activity_executions_per_day(execution_per_day):
    plt.figure(figsize=(12, 8))
    for resource_class in execution_per_day['resource_class'].unique():
        class_data = execution_per_day[execution_per_day['resource_class'] == resource_class]
        plt.plot(class_data['date'], class_data['execution_count'], label=f'Class: {resource_class}')

    plt.xlabel('Date')
    plt.ylabel('Execution Count')
    plt.title('Daily Execution Count of Activity "W_completeren aanvraag" for Each Resource Class')
    plt.legend()
    plt.show(block=False)
    
def plot_daily_case_count(dataset, start_date, end_date):
    dataframe = dataset
    dataframe['time:timestamp'] = pd.to_datetime(dataframe['time:timestamp'])
    dataframe['date'] = dataframe['time:timestamp'].dt.date
    case_counts = dataframe.groupby('date')['case:concept:name'].nunique(dropna=False)
        # Create a date range from the minimum to the maximum date in the dataset
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Reindex the case_counts series to include all dates in the date range, filling missing values with 0
    case_counts = case_counts.reindex(date_range, fill_value=0)
    print(case_counts.head(20))
    return case_counts
    plt.figure(figsize=(10, 6))
    case_counts.plot(kind='line')
    plt.title('Daily Case Count')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.grid(True)
    plt.show(block=False)


# Assuming 'execution_per_day' is the DataFrame with the execution count

def compute_and_plot_correlation(ts1, ts2, name):
    # Ensure the time series are aligned
    ts2["test"] = ts1
    timeseries1 = ts2["count"]
    timeseries2 = ts2["test"]

    pearson_corr, _ = pearsonr(timeseries1, timeseries2)
    print(f"Pearson correlation: {pearson_corr:.3f}")

    plt.figure(figsize=(10, 6))
    print("timeseries1")
    print(timeseries1)
    print("timeseries2")
    print(timeseries2)
    plt.scatter(timeseries1, timeseries2)
    plt.title(f'Scatter Plot of Count vs Count of ' + name + f'\nPearson Correlation: {pearson_corr:.3f}')
    plt.xlabel("Count")
    plt.ylabel("Number of Cases")
    plt.grid(True)
    plt.show(block=False)

def plot_scatter_with_activity_count(timeseries, activities_per_day, name):
    # Ensure the 'date' column in activities_per_day is of datetime type
    activities_per_day['date'] = pd.to_datetime(activities_per_day['date'])

    # Merge the timeseries with activities_per_day on the date
    merged_data = activities_per_day.merge(timeseries, left_on='date', right_index=True, how='inner')

    merged_data = merged_data.rename(columns={'case:concept:name': 'count'})

    print("Merged Data")
    print(merged_data)

    # Plot the scatter plot
    fig, ax = plt.subplots()
    for resource_class in merged_data['resource_class'].unique():
        subset = merged_data[merged_data['resource_class'] == resource_class]
        ax.scatter(subset['activity_count'], subset['count'], label=resource_class)

        # Calculate Pearson correlation coefficient for each resource class
        pearson_corr, _ = pearsonr(subset['count'], subset['activity_count'])
        ax.annotate(f'{resource_class} (r={pearson_corr:.2f})', 
                    xy=(subset['count'].mean(), subset['activity_count'].mean()), 
                    textcoords='offset points', 
                    xytext=(0,10), 
                    ha='center')

    ax.set_xlabel('Count of Activities')
    ax.set_ylabel('Activity Count')
    ax.legend(title='Resource Class')
    plt.title(f'Scatter Plot of Activity Counts {name}')
    fig.canvas.manager.set_window_title(f'Scatter Plot - {name}')
    plt.show(block=False)

def filter_only_start_activities(dataset, case_id_name='case:concept:name', lifecycle_name='lifecycle:transition'):
    # Filter only start activities
    start_activities = dataset.sort_values(by=['time:timestamp'])
    start_activities = dataset[dataset[lifecycle_name] == 'start']
    start_activities = start_activities.drop_duplicates(subset=[case_id_name], keep='first')
    return start_activities