import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import csv

def plot(master_file_path, csv_file_path):
    master_data = np.loadtxt(master_file_path, encoding='utf-8')
    csv_data = np.loadtxt(csv_file_path, delimiter=',', skiprows=1, encoding='utf-8')

    csv_schedule = csv_data.astype(int)
    start_time = datetime(csv_schedule[0,0], csv_schedule[0,1], csv_schedule[0,2], csv_schedule[0,3], csv_schedule[0,4], 00)
    row_size = csv_data.shape[0] -1 
    end_time = datetime(csv_schedule[row_size,0], csv_schedule[row_size,1], csv_schedule[row_size,2], csv_schedule[row_size,3], csv_schedule[row_size,4], 59)

    sorted_csv_data = []
    for s_index in tqdm(range(csv_schedule.shape[0])):
        index = s_index - 1
        target_time = datetime(csv_schedule[index,0], csv_schedule[index,1], csv_schedule[index,2], csv_schedule[index,3], csv_schedule[index,4], csv_schedule[index,5])
        new_row = [csv_schedule[index,0], csv_schedule[index,1], csv_schedule[index,2], csv_schedule[index,3], csv_schedule[index,4], csv_schedule[index,5], (target_time - start_time).total_seconds(), csv_data[index,14]]
        sorted_csv_data.append(new_row)
   
    sorted_master_data = []
    for row in tqdm(master_data): 
        row_schedule = row.astype(int)
        target_time = datetime(row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6])
        if start_time <= target_time <= end_time:
            if row[8] == 0:
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), 0])
            else:
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[8]/15.379])
    
    x1 = np.array(sorted_csv_data)[:,6]
    x2 = np.array(sorted_master_data)[:,6]
    y1 = np.array(sorted_csv_data)[:,7]
    y2 = np.array(sorted_master_data)[:,7]
    x3 = []
    y3 = []

    for c_row in tqdm(sorted_csv_data):
        for m_row in sorted_master_data:
            if c_row[6] == m_row[6]:
                x3.append(c_row[7])
                y3.append(m_row[7])

    r = np.corrcoef(np.array(x3), np.array(y3))[0,1]

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(x1, y1)
    ax1.set_title('Time Change for tilt-long analog data')
    ax1.set_xlabel('t[s]')
    ax1.set_ylabel('tilt-long value')

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(x2, y2)
    ax2.set_title('Time Change for tilt-long digital data')
    ax2.set_xlabel('t[s]')
    ax2.set_ylabel('tilt-long value[arc-sec]')

    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.scatter(np.array(x3), np.array(y3))
    ax3.set_title('compare about analog and digital data(r=' + str(r) + ')')
    ax3.set_xlabel('tilt-long value')
    ax3.set_ylabel('tilt-long value[arc-sec]')

    plt.show()


if __name__ == "__main__":
    plot("C:\\Users\\Yusei\\D58-pictures\\190716.txt", 'plot_data/long-needle.csv')
