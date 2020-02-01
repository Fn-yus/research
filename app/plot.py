from decimal import Decimal, getcontext, ROUND_HALF_UP
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import inverse

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
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[8]/15.379, row[7]])
            elif "cross" in target.lower():
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[9]/17.107, row[7]])

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
        tmp_tilt = []
        for m_row in sorted_master_data:
            if c_row[6] == m_row[6]:
                tmp_tilt.append(m_row[7])
            if m_row == sorted_master_data[-1]:
                x3.append(c_row[7])
                y3.append(np.mean(tmp_tilt))
                tmp_tilt = [] # 初期化

    r              = np.corrcoef(np.array(x3), np.array(y3))[0,1]
    (a, b, sa, sb) = __least_square(np.array(x3), np.array(y3))

    coefficient_data = [a, b, sa, sb]    
    m_dict = inverse.main(np.array(sorted_master_data), np.array(sorted_csv_data), np.array(coefficient_data))

    [A, B, C, D], [dA, dB, dC, dD] = __calculate_error(m_dict['analog_m'], m_dict['d_analog_m'])
    [E, F, G, H], [dE, dF, dG, dH] = __calculate_error(m_dict['digital_m'], m_dict['d_digital_m'])
    y5 = np.array([A + B * csv_list[6] + C * (a * csv_list[7] + b) + D * ((a * csv_list[7] + b) ** 2) for csv_list in sorted_csv_data])
    y6 = np.array([E + F * master_list[6] + G * master_list[7] + H * (master_list[7] ** 2) for master_list in sorted_master_data])

    graph_target = target.replace("-", "_")        #返り値例 "long_needle" 
    target_sort  = target.replace(target[-7:], "")  #返り値は "long" か "cross"

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6 = plt.figure()
    fig7 = plt.figure()
    fig8 = plt.figure()
    fig9 = plt.figure(figsize=[19.2, 4.8])
    fig10 = plt.figure(figsize=[19.2, 4.8])

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

    ax4_1 = fig4.add_subplot(2, 1, 1)
    ax4_1.plot(x4, y4_1)
    ax4_1.set_ylabel('Gravity [mGal]')
    ax4_1.grid(axis='y')

    ax4_2 = fig4.add_subplot(2, 1, 2)
    ax4_2.plot(x4, y4_2)
    ax4_2.plot(x4, y4_3)
    ax4_2.set_xlabel('t [s]')
    ax4_2.set_ylabel('tilt value [arc-sec]')
    ax4_2.grid(axis='y')
    ax4_2.legend(['tilt-long', 'tilt-cross'])

    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.plot(np.arange(len(x3)), x3)

    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.plot(np.arange(len(y3)), y3)

    ax7 = fig7.add_subplot(1, 1, 1)
    ax7.plot(x1, y5) # label="y={}+{}t+{}x(t)+{}x(t)^2".format(A, B, C, D))
    # ax8.legend()
    if "long" in target.lower():
        ax7.set_ylim(6200, 6400)
    elif "cross" in target.lower():
        ax7.set_ylim(6300, 6500)
    ax7.grid(axis='both')

    ax8 = fig8.add_subplot(1, 1, 1)
    ax8.plot(x2, y6) # label="y={}+{}t+{}x(t)+{}x(t)^2".format(E, F, G, H))
    # ax9.legend()
    if "long" in target.lower():
        ax8.set_ylim(6200, 6400)
    elif "cross" in target.lower():
        ax8.set_ylim(6300, 6500)
    ax8.grid(axis='both')

    fig9.suptitle(target.replace("-", "(").capitalize() + ")", fontsize=20, fontweight='black')

    ax9_1 = fig9.add_subplot(1, 3, 1)
    y9_1 = a * np.array(sorted_csv_data)[:,7] + b
    ax9_1.scatter(x1, y9_1)
    ax9_1.set_xlabel("t [s]")
    ax9_1.set_ylabel("x(t) [arc-sec]")
    ax9_1.set_xlim(-100, 2125)
    if "long" in target.lower():
        ax9_1.set_ylim(-100, 120)
    elif "cross" in target.lower():
        ax9_1.set_ylim(-120, 120)
    ax9_1.grid(axis='both')

    ax9_2 = fig9.add_subplot(1, 3, 2)
    y9_2 = np.array(sorted_master_data)[:,8] * 1000
    ax9_2.plot(x2, y9_2, color='gray', alpha=0.5, label='g_obs(t)')
    ax9_2.scatter(x1, y5, alpha=0.7, label='g_cal(t)')
    ax9_2.plot(x1, A + x1 * B, color='red', alpha=0.5, label='a: {} ± {}, \nb: {} ± {}'.format(A, dA, B, dB))
    ax9_2.set_ylabel("g(t) [μGal]")
    if "long" in target.lower():
        ax9_2.set_xlim(-100, 2400)
        ax9_2.set_ylim(6230, 6400)
    elif "cross" in target.lower():
        ax9_2.set_xlim(-100, 2100)
        ax9_2.set_ylim(6340, 6500)
    ax9_2.grid(axis='both')
    ax9_2.legend(loc='upper left')

    ax9_3 = fig9.add_subplot(1, 3, 3)
    ax9_3.scatter(y9_1, C * y9_1 + D * (y9_1 ** 2), label='c: {} ± {}, \nd: {} ± {}'.format(C, dC, D, dD))
    ax9_3.set_xlabel("x(t) [arc-sec]")
    ax9_3.set_ylabel("g(t) [μGal]")
    ax9_3.set_xlim(-120, 120)
    ax9_3.set_ylim(-130, 5)
    ax9_3.grid(axis='both')
    ax9_3.legend(loc='upper left')

    fig9.savefig('../results/graph/{}.png'.format(target.replace("-", "(").capitalize() + ")"), bbox_inches='tight')

    fig10.suptitle("{}(digital)".format(target_sort).capitalize(), fontsize=20, fontweight='black')

    ax10_1 = fig10.add_subplot(1, 3, 1)
    ax10_1.plot(x2, y2)
    ax10_1.set_xlabel("t [s]")
    ax10_1.set_ylabel("x(t) [arc-sec]")
    ax10_1.set_xlim(-100, 2125)
    if "long" in target.lower():
        ax10_1.set_ylim(-100, 120)
    elif "cross" in target.lower():
        ax10_1.set_ylim(-120, 120)
    ax10_1.grid(axis='both')

    ax10_2 = fig10.add_subplot(1, 3, 2)
    ax10_2.plot(x2, y9_2, color='gray', alpha=0.5, label='g_obs(t)')
    ax10_2.plot(x2, y6, alpha=0.7, label='g_cal(t)')
    ax10_2.plot(x2, E + x2 * F, color='red', alpha=0.5, label='a: {} ± {}, \nb: {} ± {}'.format(E, dE, F, dF))
    ax10_2.set_xlabel("t [s]")
    ax10_2.set_ylabel("g(t) [μGal]")
    if "long" in target.lower():
        ax10_2.set_xlim(-100, 2400)
        ax10_2.set_ylim(6230, 6400)
    elif "cross" in target.lower():
        ax10_2.set_xlim(-100, 2100)
        ax10_2.set_ylim(6340, 6500)
    ax10_2.grid(axis='both')
    ax10_2.legend(loc='upper left')

    ax10_3 = fig10.add_subplot(1, 3, 3)
    ax10_3.plot(y2, G * y2 + H * (y2 ** 2), label='c: {} ± {}, \nd: {} ± {}'.format(G, dG, H, dH))
    ax10_3.set_xlabel("x(t) [arc-sec]")
    ax10_3.set_ylabel("g(t) [μGal]")
    ax10_3.set_xlim(-120, 120)
    ax10_3.set_ylim(-130, 5)
    ax10_3.grid(axis='both')
    ax10_3.legend(loc='upper left')
    
    fig10.savefig('../results/graph/{}.png'.format("{}(digital)".format(target_sort).capitalize()), bbox_inches='tight')

    # plt.show()


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

def __calculate_error(m, dm):
    m_list = []
    dm_list = []

    for val, err in zip(m, dm):
        getcontext().prec = 1
        getcontext().rounding = ROUND_HALF_UP
        dec_err = Decimal(err) * Decimal(1) #誤差を有効数字1桁に

        getcontext().prec = 28
        dec_val = Decimal(val).quantize(dec_err)

        if float(dec_err).is_integer():
            m_list.append(int(dec_val))
            dm_list.append(int(dec_err))
        else:
            m_list.append(float(dec_val))
            dm_list.append(float(dec_err))
    
    return m_list, dm_list