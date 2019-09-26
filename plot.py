import numpy as np

def open_master_data(file):
    master_data = np.loadtxt(file, encoding='utf-8')
    return master_data

def open_csv_data(file):
    csv_data = np.loadtxt(file, delimiter=',', skiprows=1, encoding='utf-8')
    return csv_data

if __name__ == "__main__":
    open_master_data('plot_data/190716.txt')
    csv_file = open_csv_data('plot_data/long_needle.csv')
    print(csv_file)
    