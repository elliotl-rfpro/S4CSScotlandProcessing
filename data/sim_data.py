"""File for global simulation data/settings."""

import numpy as np

times = []
num_times = 19
for i in range(num_times):
    if i <= 11:
        sim_time = f"2024-03-01T08:{5 * i:02d}:00"
    elif 11 <= i <= 23:
        sim_time = f"2024-03-01T09:{5 * (i - 12):02d}:00"
    times.append(sim_time)

fog_levels = [
    0.01500,
    0.01250,
    0.00750,
    0.01000,
    0.01500,
    0.00500,
    0.00750,
    0.00500,
    0.00250,
    0.00750,
    0.01500,
    0.01250,
    0.00500,
    0.00250,
    0.00350,
    0.00250,
    0.00150,
    0.00250,
    0.00350
]

# Reference values
dg_ref = [0.0] * 19
gv_ref = [1.1] * 19

# Exposure
dg = [
    -8.0,
    -8.0,
    -8.0,
    -8.0,
    -8.0,
    -9.0,
    -9.0,
    -10.5,
    -10.0,
    -10.0,
    -9.0,
    -8.0,
    -8.0,
    -10.0,
    -12.0,
    -12.0,
    -12.0,
    -10.0,
    -10.0
]

# Gamma
gv = [
    1.1,
    1.1,
    1.1,
    1.1,
    1.1,
    1.2,
    1.2,
    1.2,
    1.3,
    1.4,
    1.4,
    1.5,
    1.5,
    1.6,
    1.6,
    1.6,
    1.8,
    2.2,
    2.2,
]
gv = np.array(gv)
gv -= 0.3

# --- For manual individual runs --- #
# times = [
#     f"2024-03-01T08:25:00"
# ]

# fog_levels = [
#     0.00500
# ]
