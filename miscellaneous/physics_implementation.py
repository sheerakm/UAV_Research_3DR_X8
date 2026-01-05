import numpy as np
import math as Math
import matplotlib.pyplot as plt

from collections import deque

"""
naming convention :

u̇
v̇
ẇ
ṗ
q̇
ṙ
θ
ϕ
ψ
δlat_lag
δlon_lag
δcol_lag
δpeḋ_lag
[du, dv, dw, dp, dq, dr, dtheta, dphi, dpsi, delta_lat_lag, delta_lon_lag, delta_col_lag, delta_ped_lag]
"""

du, dv, dw, dp, dq, dr, dtheta, dphi, dpsi, delta_lat_lag, delta_lon_lag, delta_col_lag, delta_ped_lag = np.zeros(13)

x_next = [du, dv, dw, dp, dq, dr, dtheta, dphi, dpsi, delta_lat_lag, delta_lon_lag, delta_col_lag, delta_ped_lag]

values = {
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

A[0][0]  = values["X_u"]
A[0][6]  = -g
A[0][10] = values["X_dlon"]
A[1][1]  = values["Y_v"]
A[1][7]  = g
A[1][9]  = values["Y_lat"]
A[2][11] = values["Z_col"]
A[3][1]  = values["L_v"]
A[3][9]  = values["L_lat"]
A[4][0]  = values["M_u"]
A[4][10] = values["M_dlon"]
A[5][5]  = values["N_r"]
A[5][12] = values["N_ped"] - values["lag"] * values["lead"]
A[6][4]  = 1.0
A[7][3]  = 1.0
A[8][5]  = 1.0
A[9][9]  = -values["lag"]
A[10][10] = -values["lag"]
A[11][11] = -values["lag"]
A[12][12] = -values["lag"]

B = np.zeros((13, 4))

B[5][3] = values["lag"] * values["lead"]
B[9][0]  = values["lag"]
B[10][1] = values["lag"]
B[11][2] = values["lag"]
B[12][3] = values["lag"]

buffer_length_lon = Math.ceil(values["tau_lon"] / time_step)
buffer_length_lat = Math.ceil(values["tau_lat"] / time_step)
buffer_length_col = Math.ceil(values["tau_col"] / time_step)
buffer_length_ped = Math.ceil(values["tau_ped"] / time_step)


buffer_lon = deque([0]* buffer_length_lon, maxlen=buffer_length_lon)
buffer_lat = deque([0]* buffer_length_lat, maxlen=buffer_length_lat)
buffer_col = deque([0]* buffer_length_col, maxlen=buffer_length_col)
buffer_ped = deque([0]* buffer_length_ped, maxlen=buffer_length_ped)

print(buffer_length_lon)
print(buffer_length_lat)
print(buffer_length_col)
print(buffer_length_ped)


#dx = Ax + By  . A and B are defined now given x and y we can calculate dx


def step(x, y):

  buffer_lon.append(y[1])
  buffer_lat.append(y[0])
  buffer_col.append(y[2])
  buffer_ped.append(y[3])

  delayed_input = np.array([buffer_lat[0], buffer_lon[0], buffer_col[0], buffer_ped[0]])

  dt = A @ x + B @ delayed_input

  return x + dt * time_step

dt = time_step

T = 5.0
steps = int(T / dt)

x = np.zeros(13)
history = []

for k in range(steps):
    if k * dt > 1.0:
        action = np.array([0.5, 0.0, 0.0, 0.0])  # lateral step
    else:
        action = np.zeros(4)

    x = step(x, action)
    history.append(x.copy())

history = np.array(history)



t = np.arange(steps) * dt

# plt.figure()
# plt.plot(t, history[:, 3], label="p (roll rate)")
# plt.plot(t, history[:, 1], label="v (lateral velocity)")
# plt.legend()
# plt.show()

plt.plot(t, history[:,1], label="v")
plt.plot(t, history[:,3], label="p")
plt.plot(t, history[:,7], label="phi")
plt.legend()
plt.show()
