import numpy as np

def main(master_data, csv_data, coefficient_data):
    # master_data = ["年", "月", "日", "時", "分", "秒(小数点以下切り捨て)", "実験開始からの経過秒", "傾斜値[arc-sec]", "重力値[mGal]"]
    # csv_data = ["年", "月", "日", "時", "分", "秒(小数点以下切り捨て)", "実験開始からの経過秒", "傾斜値[div]"]
    # coefficient_data = [a, b, sa, sb] (単位は全て[arc-sec/div])

    [a, b, _, _] = coefficient_data

    # アナログな傾斜値に関して行列d, Gを求める
    d = []
    G = []
    for csv_list in csv_data:
        tmp_tilt = a * csv_list[7] + b
        tmp_G = [1, csv_list[6], tmp_tilt, tmp_tilt ** 2]
        G.append(tmp_G)

        tmp_gravity_list = []
        for master_list in master_data:
            if csv_list[6] == master_list[6]:
                tmp_gravity_list.append(master_list[8] * 1000) # mGal -> μGal
            if all(master_list == master_data[-1]): # ループの最後
                tmp_average_gravity = np.mean(tmp_gravity_list)
                d.append(tmp_average_gravity)
                tmp_gravity_list = [] # 初期化
    
    m, dm = __inverse(np.array(d), np.array(G))

    # 電子データに対して傾斜量を求める
    dig_d = master_data[:,8] * 1000 # mGal -> μGal
    dig_G = [[1, master_list[6], master_list[7], master_list[7] ** 2] for master_list in master_data]

    dig_m, ddig_m = __inverse(np.array(dig_d), np.array(dig_G))

    return {'analog_m': m, 'digital_m': dig_m, 'd_analog_m': dm, 'd_digital_m': ddig_m, 'gravity_per_minute': G}

def __inverse(d, G):
    G_square = G.transpose() @ G
    G_square_inv = np.linalg.inv(G_square)
    m = G_square_inv @ G.transpose() @ d

    N = G.shape[0]
    M = G.shape[1]
    Cm = (1 / (N - M)) * ((d - G @ m) @ (d - G @ m)) * G_square_inv
    error_list = np.sqrt(np.diag(Cm))

    return m, error_list