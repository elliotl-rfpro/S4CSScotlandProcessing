"""Script for trying to work out the mapping between an HDR image and an image with adaptive exposure."""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from PIL import Image

from improc.improc import read_exr
from data.sim_data import times

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


def normalise2rgb(arr, new_min, new_max):
    """
    Normalize an input array between new min and max
    """
    old_min = np.min(arr)
    old_max = np.max(arr)
    normalized_arr = (new_max - new_min) * (arr - old_min) / (old_max - old_min) + new_min
    return normalized_arr


if __name__ == '__main__':
    # Define folders
    base_folder = 'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Fog_Comparison'
    sim_path = f'{base_folder}/Sim_Trees_Removed'
    meas_path = f'{base_folder}/Meas'

    # Define reference ROIs
    sky_reg_meas = [1325, 20, 1375, 70]
    wall_reg_meas = [65, 1320, 115, 1370]
    hill_reg_meas = [2583, 939, 2633, 989]
    fog_reg_meas = [1900, 850, 1950, 900]
    sky_reg_sim = [1325, 20, 1375, 70]
    wall_reg_sim = [45, 1520, 95, 1570]
    hill_reg_sim = [2763, 901, 2813, 951]
    fog_reg_sim = [1935, 850, 1985, 900]

    # Load sets of images
    print(f'Loading images...')
    # Load all .png images into an image array
    sim_images = []
    filenames = []
    for filename in os.listdir(sim_path):
        if filename.endswith('.exr'):
            filenames.append(filename)
            file_path = os.path.join(sim_path, filename)
            image = read_exr(file_path)
            sim_images.append(image)
    sim_images = np.array(sim_images)

    # Load measured images to correct against
    meas_images = []
    meas_filenames = []
    for filename in os.listdir(meas_path):
        if filename.endswith('.png'):
            meas_filenames.append(filename)
            file_path = os.path.join(meas_path, filename)
            image = np.array(Image.open(file_path))
            meas_images.append(image)
    meas_images = np.array(meas_images)
    assert len(sim_images) == len(meas_images), "Different number of simulated and measured images"

    # Step through each image and record the average values of each ROI
    reg1_meas = []
    reg2_meas = []
    reg3_meas = []
    reg4_meas = []
    total_meas = []
    reg1_sim = []
    reg2_sim = []
    reg3_sim = []
    reg4_sim = []
    total_sim = []
    for i in range(len(sim_images)):
        # Measured
        reg1_meas.append(np.mean(meas_images[i][sky_reg_meas[1]:sky_reg_meas[3], sky_reg_meas[0]:sky_reg_meas[2]], axis=(1, 0)))
        reg2_meas.append(np.mean(meas_images[i][wall_reg_meas[1]:wall_reg_meas[3], wall_reg_meas[0]:wall_reg_meas[2]], axis=(1, 0)))
        reg3_meas.append(np.mean(meas_images[i][hill_reg_meas[1]:hill_reg_meas[3], hill_reg_meas[0]:hill_reg_meas[2]], axis=(1, 0)))
        reg4_meas.append(np.mean(meas_images[i][fog_reg_meas[1]:fog_reg_meas[3], fog_reg_meas[0]:fog_reg_meas[2]], axis=(1, 0)))
        total_meas.append(meas_images[i])
        # Simulated
        reg1_sim.append(np.mean(sim_images[i][sky_reg_sim[1]:sky_reg_sim[3], sky_reg_sim[0]:sky_reg_sim[2]], axis=(1, 0)))
        reg2_sim.append(np.mean(sim_images[i][wall_reg_sim[1]:wall_reg_sim[3], wall_reg_sim[0]:wall_reg_sim[2]], axis=(1, 0)))
        reg3_sim.append(np.mean(sim_images[i][hill_reg_sim[1]:hill_reg_sim[3], hill_reg_sim[0]:hill_reg_sim[2]], axis=(1, 0)))
        reg4_sim.append(np.mean(sim_images[i][fog_reg_sim[1]:fog_reg_sim[3], fog_reg_sim[0]:fog_reg_sim[2]], axis=(1, 0)))
        total_sim.append(sim_images[i])
    reg1_meas = np.array(reg1_meas)
    reg2_meas = np.array(reg2_meas)
    reg3_meas = np.array(reg3_meas)
    reg4_meas = np.array(reg4_meas)
    reg1_sim = np.array(reg1_sim)
    reg2_sim = np.array(reg2_sim)
    reg3_sim = np.array(reg3_sim)
    reg4_sim = np.array(reg4_sim)
    total_meas = np.array(total_meas)
    total_sim = np.array(total_sim)

    adj_times = []
    for time_ in times:
        adj_times.append(time_[-8:-3])

    # Plot these averages over time individually
    fig, ax = plt.subplots()
    plt.plot(reg1_meas[:, 0], 'r-.', label='Measured')
    plt.plot(reg1_sim[:, 0], 'r', label='Simulated')
    plt.plot(reg1_meas[:, 1], 'g-.')
    plt.plot(reg1_sim[:, 1], 'g')
    plt.plot(reg1_meas[:, 2], 'b-.')
    plt.plot(reg1_sim[:, 2], 'b')
    plt.title('Average luminance: sky')
    # plt.xticks(adj_times)
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(reg2_meas[:, 0], 'r-.', label='Measured')
    plt.plot(reg2_sim[:, 0], 'r', label='Simulated')
    plt.plot(reg2_meas[:, 1], 'g-.')
    plt.plot(reg2_sim[:, 1], 'g')
    plt.plot(reg2_meas[:, 2], 'b-.')
    plt.plot(reg2_sim[:, 2], 'b')
    plt.title('Average luminance: wall corner')
    # plt.xticks(adj_times)
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(reg3_meas[:, 0], 'r-.', label='Measured')
    plt.plot(reg3_sim[:, 0], 'r', label='Simulated')
    plt.plot(reg3_meas[:, 1], 'g-.')
    plt.plot(reg3_sim[:, 1], 'g')
    plt.plot(reg3_meas[:, 2], 'b-.')
    plt.plot(reg3_sim[:, 2], 'b')
    plt.title('Average luminance: hillside grass')
    # plt.xticks(adj_times)
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(reg4_meas[:, 0], 'r-.', label='Measured')
    plt.plot(reg4_sim[:, 0], 'r', label='Simulated')
    plt.plot(reg4_meas[:, 1], 'g-.')
    plt.plot(reg4_sim[:, 1], 'g')
    plt.plot(reg4_meas[:, 2], 'b-.')
    plt.plot(reg4_sim[:, 2], 'b')
    plt.title('Average luminance: hill in cloud')
    # plt.xticks(adj_times)
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()

    fig, ax = plt.subplots()
    plt.plot(np.mean(total_meas[:, 0], axis=1), 'r-.', label='Measured')
    plt.plot(np.mean(total_sim[:, 0], axis=1), 'r', label='Simulated')
    plt.plot(np.mean(total_meas[:, 1], axis=1), 'g-.')
    plt.plot(np.mean(total_sim[:, 1], axis=1), 'g')
    plt.plot(np.mean(total_meas[:, 2], axis=1), 'b-.')
    plt.plot(np.mean(total_sim[:, 2], axis=1), 'b')
    plt.title('Average luminance: entire scene')
    # plt.xticks(adj_times)
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()

    # Plot the essential statistics for shaping image colouring
    fig, ax = plt.subplots()
    # plt.plot(np.mean(total_meas[:, 0], axis=1), 'r-.', label='Mean R')
    # plt.plot(np.mean(total_meas[:, 1], axis=1), 'g-.', label='Mean G')
    # plt.plot(np.mean(total_meas[:, 2], axis=1), 'b-.', label='Mean B')
    max_r = []
    max_g = []
    max_b = []
    min_r = []
    min_g = []
    min_b = []
    for i in range(total_meas.shape[0]):
        max_r.append(np.max(total_meas[i, :, :, 0]))
        max_g.append(np.max(total_meas[i, :, :, 1]))
        max_b.append(np.max(total_meas[i, :, :, 2]))
        min_r.append(np.min(total_meas[i, :, :, 0]))
        min_g.append(np.min(total_meas[i, :, :, 1]))
        min_b.append(np.min(total_meas[i, :, :, 2]))
    plt.plot(max_r, 'r', label='Max R')
    plt.plot(max_g, 'g', label='Max G')
    plt.plot(max_b, 'b', label='Max B')
    plt.plot(min_r, 'r', label='Min R')
    plt.plot(min_g, 'g', label='Min G')
    plt.plot(min_b, 'b', label='Min B')

    # Ranges
    # plt.plot(np.max(total_meas[:, :, 0], axis=1) - np.min(total_meas[:, :, 0], axis=1), 'r.-', label='Meas Range R')
    # plt.plot(np.max(total_meas[:, :, 1], axis=1) - np.min(total_meas[:, :, 1], axis=1), 'g.-', label='Meas Range G')
    # plt.plot(np.max(total_meas[:, :, 2], axis=1) - np.min(total_meas[:, :, 2], axis=1), 'b.-', label='Meas Range B')
    # range1 = normalise2rgb(np.max(total_sim[:, :, 0], axis=1) - np.min(total_sim[:, :, 0], axis=1), 40, 150)
    # range2 = normalise2rgb(np.max(total_sim[:, :, 1], axis=1) - np.min(total_sim[:, :, 1], axis=1), 40, 150)
    # range3 = normalise2rgb(np.max(total_sim[:, :, 2], axis=1) - np.min(total_sim[:, :, 2], axis=1), 40, 150)
    # plt.plot(range1, 'r-.', label='Sim Range R')
    # plt.plot(range2, 'g-.', label='Sim Range G')
    # plt.plot(range3, 'b-.', label='Sim Range B')

    # Footer
    plt.title('Luminance: entire scene')
    plt.xlabel('Time of day')
    plt.ylabel('Luminance (cd/m^2)')
    plt.legend()
    plt.show()

    print("Script finished successfully.")


