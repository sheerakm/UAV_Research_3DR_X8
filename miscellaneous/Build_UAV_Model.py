
from collections import deque
import numpy as np
import math as Math

def Build_UAV_Model():
    constant_vals= {
        "X_u": -0.6112,
        "M_u": 1.912,
        "X_dlon": -17.97,
        "M_dlon": 148.7,
        "Y_v": -0.4277,
        "L_v": -1.644,
        "Y_lat": 18.49,
        "L_lat": 157.3,
        "N_r": -0.5392,
        "N_ped": 18.51,
        "Z_col": -82.28,
        "tau_lon": 0.01718,
        "tau_lat": 0.01747,
        "tau_col": 0.01920,
        "tau_ped": 0.02915,
        "lag": -17.50,
        "lead": -3.169
    }
    g = 9.81

    freq = 100
    time_step = 1 / freq

    A = np.zeros((13, 13))

    A[0][0] = constant_vals["X_u"]
    A[0][6] = -g
    A[0][10] = constant_vals["X_dlon"]
    A[1][1] = constant_vals["Y_v"]
    A[1][7] = g
    A[1][9] = constant_vals["Y_lat"]
    A[2][11] = constant_vals["Z_col"]
    A[3][1] = constant_vals["L_v"]
    A[3][9] = constant_vals["L_lat"]
    A[4][0] = constant_vals["M_u"]
    A[4][10] = constant_vals["M_dlon"]
    A[5][5] = constant_vals["N_r"]
    A[5][12] = constant_vals["N_ped"] - constant_vals["lag"] * constant_vals["lead"]
    A[6][4] = 1.0
    A[7][3] = 1.0
    A[8][5] = 1.0
    A[9][9] = -constant_vals["lag"]
    A[10][10] = -constant_vals["lag"]
    A[11][11] = -constant_vals["lag"]
    A[12][12] = -constant_vals["lag"]

    B = np.zeros((13, 4))

    B[5][3] = constant_vals["lag"] * constant_vals["lead"]
    B[9][0] = constant_vals["lag"]
    B[10][1] = constant_vals["lag"]
    B[11][2] = constant_vals["lag"]
    B[12][3] = constant_vals["lag"]

    buffer_length_lon = Math.ceil(constant_vals["tau_lon"] / time_step)
    buffer_length_lat = Math.ceil(constant_vals["tau_lat"] / time_step)
    buffer_length_col = Math.ceil(constant_vals["tau_col"] / time_step)
    buffer_length_ped = Math.ceil(constant_vals["tau_ped"] / time_step)

    buffer_lon = deque([0] * buffer_length_lon, maxlen=buffer_length_lon)
    buffer_lat = deque([0] * buffer_length_lat, maxlen=buffer_length_lat)
    buffer_col = deque([0] * buffer_length_col, maxlen=buffer_length_col)
    buffer_ped = deque([0] * buffer_length_ped, maxlen=buffer_length_ped)

    return A, B, buffer_lon, buffer_lat, buffer_col, buffer_ped, constant_vals


