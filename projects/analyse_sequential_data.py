"""Script for analysing sequential data that is output from simulation_campaign.py"""

import moviepy as mp
import numpy as np
from numpy.typing import NDArray
import os
from matplotlib import pyplot as plt
import seaborn as sns
import glob
from PIL import Image
import ffmpeg
import subprocess

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from improc.improc import merge_pngs_to_mp4, simulate_mosaicing_rggb, read_exr

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

# Set the default grid setting to False
plt.rcParams['axes.grid'] = False


def gamma_from_time(time_of_day):
    """
    Finds the correct gamma value depending on time of day, based upon a Gaussian,
    where the peak is at 2.2 at 12pm, and the minimums are at 1.0 at 8am and 5pm.

    Parameters:
    time_of_day (float): Time of day in hours (0-24)

    Parameters:
    time_of_day (float): Time of day in hours (e.g., 8.55 for 8:55am)

    Returns:
    float: Gamma value
    """
    # Convert time_of_day to hours and minutes
    hours = int(time_of_day)
    minutes = (time_of_day - hours) * 100
    time_in_hours = hours + minutes / 60

    # Gaussian parameters
    peak_gamma = 2.0
    min_gamma = 0.01
    peak_time = 12.27
    sunrise = 6.28
    sunset = 18.27

    # Estimate the luminance of the sun using the solar zenith function
    if time_in_hours < sunrise or time_in_hours > sunset:
        luminance = 0.0
    else:
        # Normalize time to range [0, 1] between sunrise and sunset
        normalized_time = (time_in_hours - sunrise) / (sunset - sunrise)
        # Use a symmetric function based on the solar zenith function
        zenith_angle = np.abs(normalized_time - 0.5) * 2
        luminance = np.cos(np.pi * zenith_angle / 2)

    # Calculate gamma based on luminance
    gamma = min_gamma + (peak_gamma - min_gamma) * max(luminance, 0)
    gamma = np.clip(gamma, 0.0, peak_gamma)

    return gamma


def adjust_gamma(image, gamma):
    img_normalized = image / 255.0
    img_gamma_corrected = np.power(img_normalized, 1.0 / gamma)
    img_rescaled = img_gamma_corrected * 255
    img_rescaled = img_rescaled.astype(np.uint8)
    return img_rescaled


def batch_adjust_gamma_exr(folder_path: str, gamma_values: NDArray):
    """
    Adjusts the gamma of all .exr images within a folder using a sequence of gamma values.
    """
    # Get the list of .exr files in the folder
    exr_files = [f for f in os.listdir(folder_path) if f.endswith('.exr')]

    # Assert that the length of gamma values is equal to the number of .exr files
    assert len(gamma_values) == len(
        exr_files), (f"The number of gamma values must be equal to the number of .exr files in the folder "
                     f"(got [{len(gamma_values)}], needs [{len(exr_files)}]).")

    # Process each .exr file in the folder
    for idx, filename in enumerate(exr_files):
        file_path = os.path.join(folder_path, filename)

        # Load the .exr file
        exr_image = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if exr_image is None:
            print(f"Error: Could not load the .exr file {filename}.")
            continue

        # Get the corresponding gamma value from the list
        gamma = float(gamma_values[idx])

        tonemap = cv2.createTonemap(gamma=gamma)
        mapped_image = tonemap.process(exr_image)

        # Convert to 8-bit image for saving
        ldr_image = np.clip(mapped_image * 255, 0, 255).astype('uint8')

        # Save the processed image
        output_filename = os.path.splitext(filename)[0] + f'_gamma_{gamma:.2f}'.replace('.', 'd') + '.png'
        output_path = os.path.join(folder_path, output_filename)
        cv2.imwrite(output_path, ldr_image)

        print(f"Processed and saved: {output_filename}")


def correct_measured_gamma(folder: str):
    """Corrects the gamma of all images within the specified folder (.png) by accounting for the change in brightness
    that results from an image at a different time of day. Uses other helper functions found within this script."""
    # Clean all gamma-corrected images for future passes
    remove_gamma_pngs(folder)

    # Batch adjust gamma of all simulated images
    gamma_values = []
    time_range = np.concatenate((np.arange(8.00, 8.55, 0.05), np.arange(9.00, 9.30, 0.05)))
    for i in time_range:
        gamma_values.append(gamma_from_time(i))
    batch_adjust_gamma_exr(sim_path, np.array(gamma_values))


def remove_gamma_pngs(folder: str):
    """
    Removes all .png files which have "gamma" in their name within a folder.

    Args:
        folder (str): Input folder. Should be absolute path.
    """
    print('Clearing .png files with "gamma" within their names.')
    # Use glob to find all .png files in the folder
    png_files = glob.glob(f'{folder}/*.png')

    # Filter files with 'gamma' in the title
    gamma_files = [f for f in png_files if 'gamma' in os.path.basename(f)]

    # Remove each file
    for filename in gamma_files:
        os.remove(filename)
        print(f'Removed: {filename}')


def pngs_to_mp4(sim_folder: str, meas_folder: str, base_folder: str, include: str = '', vname=''):
    """
    Converts all files with the .png extension to an .mp4 within a given folder. Wrapper for the parent function
    within the improc library.
    """
    # Process .pngs to .mp4 (as required)
    sim_name = f"{sim_folder}/{vname}_sim.mp4"
    meas_name = f"{meas_folder}/{vname}_meas.mp4"
    merge_pngs_to_mp4(sim_folder, 2880, 1860, 2, include=include, fname=sim_name)
    merge_pngs_to_mp4(meas_folder, 2880, 1860, 2, fname=meas_name)

    command = f'ffmpeg -y -i {meas_name} -i {sim_name} -filter_complex hstack {base_folder}/{vname}_comparison.mp4'
    subprocess.call(f'{command}', shell=True)

    # # Load and stack .mp4 videos
    # sim_video = mp.VideoFileClip(sim_name)
    # meas_video = mp.VideoFileClip(meas_name)
    #
    # # Stack the videos vertically
    # final_video = mp.concatenate_videoclips([[meas_video], [sim_video]])
    #
    # # Write the result to a file
    # final_video.write_videofile(f"{base_folder}/{vname}_comparison.mp4")

    print('')


def adjust_exposure_exr(image, ev):
    """
    Changes the exposure of a .exr image according to the supplied EV value
    """
    img_adjusted = image * (2 ** ev)
    return img_adjusted


if __name__ == '__main__':
    print(f'Script starting...')

    # Define folders
    base_folder = 'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Fog_Comparison'
    sim_path = f'{base_folder}/Sim_Accurate_Fog'
    meas_path = f'{base_folder}/Meas'

    # --- IMPROC --- #
    # Load all .png images into an image array
    images = []
    filenames = []
    for filename in os.listdir(sim_path):
        if filename.endswith('.exr'):
            filenames.append(filename)
            file_path = os.path.join(sim_path, filename)
            image = read_exr(file_path)
            images.append(image)
    images = np.array(images)

    # Lists for exposure and gamma changes for each file
    ev = [
        -10.5,
        -10.5,
        -10.5,
        -10.5,
        -10.5,
        -10.5,
        -10.0,
        -10.5,
        -10.0,
        -10.0,
        -9.0,
        -8.5,
        -8.5,
        -10.0,
        -12.0,
        -12.0,
        -12.0,
        -12.0,
        -12.0
    ]
    gv = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.4,
        1.4,
        1.4,
        1.6,
        1.6,
        1.6,
        1.8,
        2.2,
        1.8,
    ]
    gv = np.array(gv)
    gv -= 0.3

    for i in range(len(images)):
        print(f'Processing image: {filenames[i]}...')
        images[i] = adjust_exposure_exr(images[i], ev[i])
        img_norm = cv2.normalize(images[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_corr = adjust_gamma(img_norm, gv[i])
        img_8bit = img_corr.astype(np.uint8)
        sav_image = Image.fromarray(img_8bit)
        sav_image.save(f'{sim_path}/{filenames[i][:-4]}_corr.png')
        # plt.imshow(img_8bit)
        # plt.title(f'{filenames[i]}, ev={ev[i]}')
        # plt.show()
        filt_image = simulate_mosaicing_rggb(img_8bit, blend=0.0)
        sav_image = Image.fromarray(filt_image)
        sav_image.save(f'{sim_path}/{filenames[i][:-4]}_mos.png')

    # Convert .pngs to .mp4
    pngs_to_mp4(sim_path, meas_path, base_folder, include='corr', vname='manual_gamma_corr')
    # pngs_to_mp4(sim_path, meas_path, base_folder, include='mos', vname='manual_gamma_demos')

    # # Get the regions for each set of images
    # sim_regions =

    print('Script finished successfully.')
