"""Script for performing colour analysis with use of a Macbeth chart"""
import cv2
import numpy as np
import seaborn as sns
from PIL import Image
from improc.utils.macbeth_utils import (apply_colour_filters, plot_differences, calculate_mses, display_colours,
                                        process_macbeth_colours, eval_filters, apply_global_filter)
from improc.improc import simulate_mosaicing_rggb
from camera.camera import auto_white_balance_ref, auto_white_balance, gamma_correction
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from core.object_locations import hdr_regions, wt_regions, rw_regions

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})


def constraint_white_level(filters, image1, image2, white_index: int):
    # Ensures that the level of the darkest colour is approximately equal
    filtered_image = apply_colour_filters(image1, filters)
    return np.sum(filtered_image[white_index]) - np.sum(image2[white_index])


def constraint_black_level(filters, image1, image2, black_index):
    # Ensures that the level of the darkest colour is approximately equal
    filtered_image = apply_colour_filters(image1, filters)
    return np.sum(filtered_image[black_index]) - np.sum(image2[black_index])


def get_reference_macbeth(fpath: str) -> list:
    """
    Load and analyze a reference Macbeth chart.

    Parameters:
    fpath (str): The file path to the Macbeth chart image.

    Returns:
    list: A list of lists where each sublist contains a color name and its corresponding average RGB value.
    """

    # Load the Macbeth chart image
    image = Image.open(fpath)

    # Define the number of rows and columns in the Macbeth chart
    rows = 6
    columns = 4

    # Define the width, height, and separation in x and y of each color square
    square_width = 52
    square_height = 52
    sep_x = 12
    sep_y = 12

    # Initialize a list to store the average color values
    colour_values = []

    # Loop through each color square to calculate average color values
    for row in range(rows):
        for column in range(columns):
            # Calculate coordinates of the current square's top-left corner
            left = column * (square_width + sep_x)
            top = row * (square_height + sep_y)

            # Crop the current square from the image using the calculated coordinates
            current_square = image.crop((left, top, left + square_width, top + square_height))

            # Calculate mean RGB values, then refill original array
            pixels_array = np.array(current_square)[:, :, :3]
            mean_color = np.mean(pixels_array.reshape(-1, pixels_array.shape[-1]), axis=0)
            pixels_array[:, :] = mean_color

            # Store the average RGB values in the list
            colour_values.append(pixels_array)

    # Create a list of lists where each sublist contains a color name and its corresponding average RGB value
    ret_arr = colour_values

    return ret_arr


if __name__ == '__main__':
    # --- REFERENCE --- #
    # # Get reference macbeth chart
    # ref_macbeth = get_reference_macbeth('C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/References/Macbeth_crop.png')

    # --- SIMULATED --- #
    # Load Macbeth chart from simulated .exr HDR buffer
    fname = '20240301_1030_image1'
    sim_image_orig = Image.open(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/References/sim_{fname}_f25_sh05.png')
    sim_image = np.array(sim_image_orig)

    # Adjust hdr regions to account for image shift (not always perfectly aligned)
    tmp_arr = []
    for i in range(len(hdr_regions)):
        hdr_regions[i][0] += 5
        hdr_regions[i][1] += 0
        hdr_regions[i][2] += 5
        hdr_regions[i][3] += 0

    sim_colours_raw = process_macbeth_colours(sim_image, rows=4, columns=6, regions=hdr_regions, ao='yx', avg=False)
    sim_colours = process_macbeth_colours(sim_image, rows=4, columns=6, regions=hdr_regions, ao='yx', avg=True)

    # --- MEASURED --- #
    # Load Macbeth chart from .RAW file
    meas_image = Image.open(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/References/{fname}.png')
    meas_image = np.array(meas_image)

    # Change regions so that they match the initial point of the measured macbeth chart
    for i in range(len(rw_regions)):
        rw_regions[i][0] += 1595
        rw_regions[i][1] += 1063
        rw_regions[i][2] += 1595
        rw_regions[i][3] += 1063

    meas_colours_raw = process_macbeth_colours(meas_image, rows=4, columns=6, regions=rw_regions, ao='yx', avg=False)
    meas_colours = process_macbeth_colours(meas_image, rows=4, columns=6, regions=rw_regions, ao='yx', avg=True)

    # display_colours(meas_colours_raw, title='Measured colours', orientation='portrait')
    # plt.show()

    # --- PROCESSING --- #
    # Calculate colours from filtered image
    sim_colours = process_macbeth_colours(sim_image, rows=4, columns=6, regions=hdr_regions, ao='yx', avg=True)

    # Calculate initial MSE
    init_mse = calculate_mses(np.array(sim_colours), np.array(meas_colours))
    print(f'Init: MSE: {np.sum(init_mse, axis=0)}')

    # Create initial guesses for correction
    # R1, G1, B1, R2, G2, B2, R3, G3, B3, globalR, globalG, globalB
    initial_guess = np.array([0.40, 0.07, 0.05, 0.10, 0.60, 0.05, 0.05, 0.10, 0.60, 0.0, 0.0, 0.0])

    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # x[0] > x[1]
        {'type': 'ineq', 'fun': lambda x: x[1] - x[2]},  # x[1] > x[2]
        {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},  # x[4] > x[3]
        {'type': 'ineq', 'fun': lambda x: x[4] - x[5]},  # x[4] > x[5]
        {'type': 'ineq', 'fun': lambda x: x[7] - x[6]},  # x[7] > x[6]
        {'type': 'ineq', 'fun': lambda x: x[8] - x[7]},  # x[8] > x[7]
        {'type': 'ineq', 'fun': lambda x: x[:9]},  # x[:9] >= 0
        {'type': 'ineq', 'fun': lambda x: 1 - x[:9]},  # x[:9] <= 1
        {'type': 'ineq', 'fun': lambda x: x[9:] + 255},  # x[9:] >= -255
        {'type': 'ineq', 'fun': lambda x: 255 - x[9:]}  # x[9:] <= 255
    ]

    # Minimize the objective function
    result = minimize(eval_filters, initial_guess, args=(sim_colours, meas_colours), method='SLSQP',
                      constraints=constraints)

    # Optimal filters
    optimal_filters = result.x
    print("Optimal filters:", optimal_filters)

    # Apply the optimal filters
    filt_colours = sim_colours.copy()
    filt_colours = apply_colour_filters(filt_colours, optimal_filters)

    # Expand arrays to make them plotable
    size = 100
    filt_colours_arr = np.zeros([24, size, size, 3])
    for i in range(len(meas_colours)):
        filt_colours_arr[i] = np.tile(filt_colours[i], (size, size, 1)).astype(np.uint8)

    # Normalise for plotting
    filt_colours_arr = cv2.normalize(filt_colours_arr, None, np.min(filt_colours_arr), np.max(filt_colours_arr),
                                     cv2.NORM_MINMAX).astype(np.uint8)

    # Display the reference, measured, and corrected images
    display_colours(sim_colours_raw, "Simulated image", orientation='portrait')
    display_colours(meas_colours_raw, "Measured image", orientation='portrait')
    display_colours(filt_colours_arr, "Simulated image with filter model applied", orientation='portrait')
    plt.show()

    # Plot difference metrics
    diff = plot_differences(np.array(sim_colours), np.array(meas_colours), title='Difference for each colour')
    plt.ylim([-255, 255])

    diff2 = plot_differences(np.array(filt_colours), np.array(meas_colours), title='Difference for each colour, corrected')
    plt.ylim([-255, 255])
    plt.show()

    # Plot corrected image against measurement
    meas_image = Image.open(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/References/{fname}.png')
    meas_image = np.array(meas_image)
    filt_image = apply_global_filter(sim_image, optimal_filters)

    # --- MOSAICING --- #
    filt_image = simulate_mosaicing_rggb(filt_image, blend=0.0)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Simulated image')
    ax1.imshow(sim_image_orig)
    plt.grid(False)

    fig2, ax2 = plt.subplots()
    ax2.set_title('Simulated image (corrected)')
    ax2.imshow(filt_image)
    plt.grid(False)

    fig3, ax3 = plt.subplots()
    ax3.set_title('Measured image')
    ax3.imshow(meas_image)
    plt.grid(False)
    plt.show()

    # Save the final file
    cv2.imwrite(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/References/sim_{fname}_f25_sh05_isp.png',
                cv2.cvtColor(filt_image, cv2.COLOR_BGR2RGB))

    print('Script finished successfully.')
