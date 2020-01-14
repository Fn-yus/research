from decimal import Decimal, getcontext, ROUND_HALF_UP
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def plot(master_file_path, csv_file_path, target):
    master_data = np.loadtxt(master_file_path, encoding='utf-8')
    csv_data    = np.loadtxt(csv_file_path, delimiter=',', skiprows=1, encoding='utf-8')

    csv_schedule = csv_data.astype(int)
    start_time   = datetime(csv_schedule[0,0], csv_schedule[0,1], csv_schedule[0,2], csv_schedule[0,3], csv_schedule[0,4], 00)
    end_time     = datetime(csv_schedule[-1,0], csv_schedule[-1,1], csv_schedule[-1,2], csv_schedule[-1,3], csv_schedule[-1,4], 59)

    sorted_csv_data = []
    for index, csv_row in enumerate(csv_schedule):
        target_time = datetime(csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5])
        new_row = []
        if "needle" in target.lower():
            new_row = [csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5], (target_time - start_time).total_seconds(), csv_data[index,14]]
        elif "bubble" in target.lower():
            new_row = [csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5], (target_time - start_time).total_seconds(), csv_data[index,6]]
        sorted_csv_data.append(new_row)
   
    sorted_master_data = []
    experiment_data    = []
    for row in master_data: 
        row_schedule = row.astype(int)
        target_time  = datetime(row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6])
        if start_time <= target_time <= end_time:
            if "long" in target.lower():
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[8]/15.379])
            elif "cross" in target.lower():
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[9]/17.107])

        experiment_start_time = datetime(2019, 7, 16, 13, 21, 00)
        long_end_time         = datetime(2019, 7, 16, 13, 58, 00)
        cross_start_time      = datetime(2019, 7, 16, 15, 16, 00)
        experiment_end_time   = datetime(2019, 7, 16, 15, 50, 00)
        if experiment_start_time <= target_time <= long_end_time or cross_start_time <= target_time <= experiment_end_time:
            experiment_data.append([(target_time - experiment_start_time).total_seconds(), row[7], row[8]/15.379, row[9]/17.107])
        elif long_end_time <= target_time <= cross_start_time:
            experiment_data.append([(target_time - experiment_start_time).total_seconds(), None, None, None])

    x1 = np.array(sorted_csv_data)[:,6]
    x2 = np.array(sorted_master_data)[:,6]
    y1 = np.array(sorted_csv_data)[:,7]
    y2 = np.array(sorted_master_data)[:,7]
    x3 = []
    y3 = []
    x4 = np.array(experiment_data)[:,0]
    y4_1 = np.array(experiment_data)[:,1]
    y4_2 = np.array(experiment_data)[:,2]
    y4_3 = np.array(experiment_data)[:,3]    

    for c_row in sorted_csv_data:
        for m_row in sorted_master_data:
            if c_row[6] == m_row[6]:
                x3.append(c_row[7])
                y3.append(m_row[7])

    r              = np.corrcoef(np.array(x3), np.array(y3))[0,1]
    (a, b, sa, sb) = __least_square(np.array(x3), np.array(y3))

    graph_target = target.replace("-", "_")        #返り値例 "long_needle" 
    target_sort  = target.replace(target[-7:], "")  #返り値は "long" か "cross"

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6 = plt.figure()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(x1, y1)
    ax1.set_title('tilt-{} analog data'.format(graph_target))
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('tilt-{} value'.format(graph_target))
    ax1.grid(axis='y')

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(x2, y2)
    ax2.set_title('tilt-{} digital data'.format(target_sort))
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('tilt-{} value[arc-sec]'.format(target_sort))
    ax2.grid(axis='y')

    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.scatter(np.array(x3), np.array(y3))
    ax3.plot(np.array(x3), (a*np.array(x3)+b), color="red", label="y = ({}±{})x + ({}±{})".format(a, sa, b, sb))
    ax3.legend()
    ax3.set_title('Compare about analog and digital data(r={})'.format(r))
    ax3.set_xlabel('tilt-{} value'.format(graph_target))
    ax3.set_ylabel('tilt-{} value [arc-sec]'.format(target_sort))
    ax3.grid(axis='both')

    ax4 = fig4.add_subplot(2, 1, 1)
    ax4.plot(x4, y4_1)
    ax4.set_ylabel('Gravity [mGal]')
    ax4.grid(axis='y')

    ax5 = fig4.add_subplot(2, 1, 2)
    ax5.plot(x4, y4_2)
    ax5.plot(x4, y4_3)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel('tilt value [arc-sec]')
    ax5.grid(axis='y')
    ax5.legend(['tilt-long', 'tilt-cross'])

    ax6 = fig5.add_subplot(1, 1, 1)
    ax6.plot(np.arange(len(x3)), x3)

    ax7 = fig6.add_subplot(1, 1, 1)
    ax7.plot(np.arange(len(y3)), y3)

    plt.show()

def __least_square(x, y):
    xy = x * y
    square_x = np.square(x)
    square_y = np.square(y)
    N = len(x)

    a = (N * sum(xy) - (sum(x) * sum(y)))/(N * sum(square_x) - (sum(x) ** 2))
    b = (sum(square_x) * sum(y) - sum(x) * sum(xy))/(N * sum(square_x) - (sum(x) ** 2))

    sy = Decimal(((sum(np.square(a * x + b - y)))/(N - 2)) ** 0.5)

    getcontext().prec     = 1
    getcontext().rounding = ROUND_HALF_UP
    sa = sy * Decimal(((N/(N * sum(square_x) - (sum(x) ** 2))) ** 0.5))
    sb = sy * Decimal(((sum(square_x)/(N * sum(square_x) - (sum(x) ** 2))) ** 0.5))

    getcontext().prec = 28
    a = Decimal(a).quantize(sa)
    b = Decimal(b).quantize(sb)

    return float(a), float(b), float(sa), float(sb)
