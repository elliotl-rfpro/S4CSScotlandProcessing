"""
Master info
Lat/Long (2kflat/Cardiff): 51.5 N, 3.2W, Solstice zenith: 2008-07-21::14:14, Peak solar luminance: 1.6e9 - 1.9e9
"""
import re
import clr
import time
from typing import List
from matplotlib import pyplot as plt

# Load the rFpro.Controller DLL/Assembly
clr.AddReference("C:/rFpro/2023b/API/rFproControllerExamples/Controller/rFpro.Controller")

# Import rFpro.controller and some other helpful .NET objects
from rFpro import Controller
from System import DateTime, Decimal


def find_and_replace(data: List[str], values: dict) -> None:
    # For each value in the dict, go line by line through the config file until there's a string match that isn't a
    # comment. Then, replace the float on this line with the value in the dict. Janky but fast and works well.
    non_float = re.compile(r'[^\d.]+')
    for key in values:
        name = key.replace('_', '-')
        i = 0
        for line in data:
            if line.__contains__(name) and not line.startswith('#'):
                value = non_float.sub('', line)
                data[i] = data[i].replace(value, str(values[key]))
            i += 1


# Load in the DATA_PATH
DATA_PATH = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data'
CONFIGS_PATH = f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Fog_Comparison/configs'

# Set up simulation settings variables
comments = ''
times = []
num_times = 19
for i in range(num_times):
    if i <= 11:
        sim_time = f"2024-03-01T08:{5 * i:02d}:00"
    elif 11 <= i <= 23:
        sim_time = f"2024-03-01T09:{5 * (i - 12):02d}:00"
    times.append(sim_time)

fog_levels = [
    0.00055,
    0.00100,
    0.00125,
    0.00150,
    0.00175,
    0.00250,
    0.00400,
    0.00600,
    0.00800,
    0.0100,
    0.0125,
    0.0150,
    0.0075,
    0.0025,
    0.001,
    0.0005,
    0.00025,
    0.0001,
    0.00001
]

# Check fog graph looks OK
plt.plot(fog_levels)
plt.show()

# Create an instance of the rFpro.Controller
rFpro = Controller.DeserializeFromFile(f'{CONFIGS_PATH}/WinterTestSite.json')

# Static settings
rFpro.DynamicWeatherEnabled = True
rFpro.ParkedTrafficDensity = Decimal(0.5)
rFpro.Vehicle = 'Hatchback_AWD_Red'
rFpro.Location = 'S4C_WinterTestSite'
rFpro.VehiclePlugin = 'RemoteModelPlugin'

# Printing before main loop
save_loc = f'{DATA_PATH}/Fog_Comparison'
print(f'Saving at: {save_loc}')

# Main loop, each fog has an associated time.
for fog_level, t in zip(fog_levels, times):
    print(f'Fog level: {fog_level}')
    # Load the raytracer.toml file, insert the correct fog params, and save it.
    with open(r"C:/rFpro/2023b/rFpro/Plugins/RaytracePlugin.toml", 'r') as f:
        rt_data = f.readlines()
    for line in range(len(rt_data)):
        if 'fog_density' in rt_data[line]:
            rt_data[line] = f'fog_density = {fog_level}\n'
    with open(r"C:/rFpro/2023b/rFpro/Plugins/RaytracePlugin.toml", 'w') as f:
        f.writelines(rt_data)

    # Configure saving location
    with open(r"C:/rFpro/2023b/rFpro/Plugins/RaytracePlugin.toml", 'r') as f:
        rt_data = f.readlines()
    adj_sim = str(t[:16] + f'_fog{str(fog_level)[2:]}').replace(':', '-')
    for line in range(len(rt_data)):
        if '[output.hdr]' in rt_data[line]:
            rt_data[line + 4] = f'path = "{DATA_PATH}/Fog_Comparison/{adj_sim}.exr"\n'
        if '[output.ldr]' in rt_data[line]:
            rt_data[line + 4] = f'path = "{DATA_PATH}/Fog_Comparison/{adj_sim}.png"\n'
            break
    with open(r"C:/rFpro/2023b/rFpro/Plugins/RaytracePlugin.toml", 'w') as f:
        f.writelines(rt_data)

    # Open the TrainingData.ini file and correctly adjust the saving folder (with date/time)
    with open(r"C:/rFpro/2023b/rFpro/Plugins/WarpBlend/TrainingData.ini", 'r') as f:
        training_data = f.readlines()
    training_data[1] = 'OutputDir=' + save_loc + '\n'
    with open(r"C:/rFpro/2023b/rFpro/Plugins/WarpBlend/TrainingData.ini", 'w') as f:
        f.writelines(training_data)

    # Connect
    rFpro.StartTime = DateTime.Parse(t)
    while rFpro.NodeStatus.NumAlive < rFpro.NodeStatus.NumListeners:
        print(f'{rFpro.NodeStatus.NumAlive} of {rFpro.NodeStatus.NumListeners} Listeners connected.')
        time.sleep(1)
    print(f'{rFpro.NodeStatus.NumAlive} of {rFpro.NodeStatus.NumListeners} Listeners connected.')

    rFpro.StartSession()
    time.sleep(10)
    t1 = time.time()
    while True:
        t2 = time.time()
        if (t2 - t1) >= 250.0:
            rFpro.StopSession()
            break

print("Simulation campaign complete!\n")
print("\nScript executed successfully. Exiting...")
