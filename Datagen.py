from Utility import *
from prophet import Prophet
import pm4py
import sys
import os
import json
from fractions import Fraction
from datetime import timedelta

class DatasetInfo:
    def __init__(self, name, num_cases_train, num_cases_test, first_date_train, last_date_train, first_date_test, last_date_test, days_train, days_test):
        self.name = name
        self.num_cases_train = num_cases_train
        self.num_cases_test = num_cases_test
        self.first_date_train = first_date_train
        self.last_date_train = last_date_train
        self.first_date_test = first_date_test
        self.last_date_test = last_date_test
        self.days_train = days_train
        self.days_test = days_test


    def get_name(self):
        return self.name

    def get_path(self):
        return self.path

    def get_num_classes(self):
        return self.num_classes

    def get_num_samples(self):
        return self.num_samples

    def to_dict(self):
        return {
            'name': self.name,
            'num_cases_train': self.num_cases_train,
            'num_cases_test': self.num_cases_test,
            'first_date_train': self.first_date_train,
            'last_date_train': self.last_date_train,
            'first_date_test': self.first_date_test,
            'last_date_test': self.last_date_test,
            'days_train': self.days_train,
            'days_test': self.days_test
        }
    
    def save_to_json(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, default=str)



def fourier(dataset, threshold, extended_days, path, flag=False):
    print("dataset")
    print(dataset)
    print("dataset length")
    print(len(dataset))
    fft_result = np.fft.rfft(dataset)
    freqs = np.fft.rfftfreq(len(dataset))
    periods = 1 / freqs[1:]
    magnitude = np.abs(fft_result)

    filtered_fft = fft_result.copy()
    if flag:
        filtered_fft[freqs < threshold] = 0
    else:
        filtered_fft[freqs > threshold] = 0

    filtered_dataset = np.fft.irfft(filtered_fft)
    filtered_dataset = np.where(filtered_dataset < 1, 1, filtered_dataset)

    plt.subplot(2, 1, 2)
    plt.plot(periods, magnitude[1:], label='Original')
    plt.plot(periods, np.abs(filtered_fft)[1:], linestyle='--', label='Filtered LT4')
    plt.title('Days per Cycle (inverted)')
    plt.xlabel('Period (days per cycle)')
    plt.ylabel('Magnitude')
    plt.xscale('log')  # Logarithmic scale for better visualization
    plt.grid(True)
    plt.axvline(x=7, color='r', linestyle='--', label="7 days")
    plt.axvline(x=30, color='g', linestyle='--', label="30 days")
    plt.axvline(x=90, color='b', linestyle='--', label="90 days")
    # Optional: Highlight specific periods
    # highlight_periods = [7, 30, 90]  # Weekly, monthly, and quarterly
    # for period in highlight_periods:
    #     plt.axvline(x=period, color='r', linestyle='--', label=f'{period} days')

    plt.legend(fontsize='small')

    time_array = np.arange(len(dataset))
    plt.subplot(2, 1, 1)
    # Plot original and reconstructed signals
    plt.plot(time_array, dataset, label='Original')
    if(len(time_array) != len(filtered_dataset)):
        time_array = time_array[:-1]
    plt.plot(time_array, filtered_dataset, label='Reconstructed LT4', linestyle='--')
    #plt.axhline(y=average, color='r', linestyle='--', label='Average')
    plt.title('Original vs Reconstructed Event Count')
    plt.xlabel('Date')
    plt.ylabel('Event Count')
    plt.legend(fontsize='small')

    plt.tight_layout()
    plt.title("Arrival Rate for BP_FUL")
    plt.savefig(f'{path}number_of_cases_per_test.png')
    plt.savefig(f'{path}number_of_cases_per_test.svg')
    # model = ExponentialSmoothing(filtered_dataset, trend='add', seasonal=None)
    # future_values = model.fit().forecast(extended_days)
    # print("future values")
    # print(future_values)
    # model = ARIMA(filtered_dataset, order=(1, 1, 1)).fit()

    # # Forecast future values
    # future_values = model.forecast(steps=20)
    next_date = dataset.index[-1] + timedelta(days=1)
    df = pd.DataFrame({'ds': pd.date_range(start=next_date, periods=len(filtered_dataset), freq='D'), 'y': filtered_dataset})
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=extended_days)
    future['ds'] = future['ds'].dt.tz_localize(None)  # Remove timezone
    forecast = model.predict(future)

    length = len(dataset) + extended_days

    extended_dataset = np.concatenate((filtered_dataset, forecast['yhat'].values))
    extended_dataset = np.where(extended_dataset < 1, 1, extended_dataset)


    # extended_dataset = np.concatenate((filtered_dataset,) * 2)
    # extended_dataset = extended_dataset[len(dataset):length]

    return extended_dataset

def cases_in_dataset(dataset):
    return len(dataset['case:concept:name'].unique())

def first_date_in_dataset(dataset):
    return dataset['time:timestamp'].min()

def last_date_in_dataset(dataset):
    return dataset['time:timestamp'].max()

def days_in_dataset(dataset):
    return (dataset['time:timestamp'].max() - dataset['time:timestamp'].min()).days

def transform_from_life_cycle_to_single_event(dataset):

    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'])

    if(len(dataset["lifecycle:transition"].unique()) == 1):
        pivot = dataset.copy()
        pivot['start_time'] = pivot['time:timestamp']
        pivot['end_time'] = pivot['time:timestamp']
        pivot = pivot.drop(columns=['time:timestamp'])
        return pivot

    dataset['occurrence'] = dataset.groupby(['concept:name', "lifecycle:transition", "org:resource", "case:concept:name"]).cumcount()

    pivot = dataset.pivot(index=['concept:name', "org:resource", "case:concept:name", 'occurrence'], columns='lifecycle:transition', values='time:timestamp').reset_index()

    pivot = pivot.drop(columns="occurrence")

    pivot.columns.name = None
    pivot = pivot.rename(columns={"start": "start_time", "complete": "end_time", "START": "start_time", "COMPLETE": "end_time"})
    pivot['start_time'].fillna(pivot['end_time'], inplace=True)
    pivot['end_time'].fillna(pivot['start_time'], inplace=True)


    invalid_rows = pivot[pivot['end_time'] < pivot['start_time']]
    
    # Print the invalid rows
    pivot = pivot.drop(invalid_rows.index)
    #pivot = pivot.dropna(subset=['start_time', 'end_time'])
    pivot.sort_values(by=['start_time'], inplace=True)




    return pivot

def str_to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python datagen.py <path_to_xes_file> <parameter>")
    #     sys.exit(1)

    xes_file_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(xes_file_path))[0]
    parameter = Fraction(sys.argv[2])
    #Flag represents whether to keep the frequencies above or below the threshold, in this case True means we keep the frequencies below the threshold
    flag = str_to_bool(sys.argv[3])

    #Create a folder to store the output files
    if not os.path.exists(f"output/{base_name}_{parameter.denominator}_{flag}"):
        os.makedirs(f"output/{base_name}_{parameter.denominator}_{flag}")

    #The transformed datasets are needed for SIMOD

    output_train_transformed_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_TRAIN_TRANSFORMED_PARAMETER_{parameter.denominator}_{flag}.csv"
    output_test_transformed_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_TEST_TRANSFORMED_PARAMETER_{parameter.denominator}_{flag}.csv"
    output_entire_transformed_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_ENTIRE_TRANSFORMED_PARAMETER_{parameter.denominator}_{flag}.csv"
    output_train_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_TRAIN_PARAMETER_{parameter.denominator}_{flag}.xes"
    output_test_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_TEST_PARAMETER_{parameter.denominator}_{flag}.xes"
    output_arrival_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_ARRIVALRATE_{parameter.denominator}_{flag}.txt"
    output_information_filename = f"output/{base_name}_{parameter.denominator}_{flag}/{base_name}_INFORMATION_{parameter.denominator}_{flag}.json"

    dataset = import_xes(xes_file_path)

    dataset['time:timestamp'] = pd.to_datetime(dataset['time:timestamp'])
    dataset['date'] = dataset['time:timestamp'].dt.date
    unique_dates_count = dataset['date'].nunique()
    print(f"Number of unique dates: {unique_dates_count}")

    sorted_dataset = dataset.sort_values(by=['time:timestamp'])
    entire = transform_from_life_cycle_to_single_event(sorted_dataset)
    entire.to_csv(output_entire_transformed_filename, index=False)
    train, test = split_data_by_case(sorted_dataset, 0.8)
    datasetInfo = DatasetInfo(base_name, cases_in_dataset(train), cases_in_dataset(test), first_date_in_dataset(train), last_date_in_dataset(train), first_date_in_dataset(test), last_date_in_dataset(test), days_in_dataset(train), days_in_dataset(test))
    datasetInfo.save_to_json(output_information_filename)
    pm4py.write_xes(train, output_train_filename)
    pm4py.write_xes(test, output_test_filename)
    trainTransformed = transform_from_life_cycle_to_single_event(train)
    trainTransformed.to_csv(output_train_transformed_filename, index=False)
    testTransformed = transform_from_life_cycle_to_single_event(test)
    testTransformed.to_csv(output_test_transformed_filename, index=False)
    train = train.sort_values(by=['time:timestamp'])
    train = train.drop_duplicates(subset=['case:concept:name'], keep='first')
    #train = train[train["lifecycle:transition"].str.lower() == "start"]
    train['time:timestamp'] = pd.to_datetime(train['time:timestamp'])
    train = train.set_index('time:timestamp')
    train = train.resample('D').size()
    data = fourier(train, parameter, 1000, f'output/{base_name}_{parameter.denominator}_{flag}/', flag)
    print("Cases in test: ", cases_in_dataset(test))
    print("First date in test: ", first_date_in_dataset(test))
    print("Days in test: ", days_in_dataset(test))

    data_str = ','.join(data.astype(str))


    with open(output_arrival_filename, 'w') as file:
        file.write(data_str)



if __name__ == "__main__":
    main()