"""Script for analysing sequential data that is output from simulation_campaign.py"""

import moviepy as mp
import numpy as np
from numpy.typing import NDArray
import os
from matplotlib import pyplot as plt
import seaborn as sns
import glob
from PIL import Image, ImageDraw, ImageFont
import subprocess

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from improc.improc import merge_pngs_to_mp4, simulate_mosaicing_rggb, read_exr, white_balance_adjust
from data.sim_data import fog_levels, dg, gv, dg_ref, gv_ref

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
    return np.power(image, 1.0 / gamma)


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

    # Stack vertically
    command = f'ffmpeg -y -i {meas_name} -i {sim_name} -filter_complex hstack {base_folder}/{vname}_comparison.mp4'
    subprocess.call(f'{command}', shell=True)

    # # Stack horizontally
    # sim_video = mp.VideoFileClip(sim_name)
    # meas_video = mp.VideoFileClip(meas_name)
    #
    # # Stack the videos vertically
    # final_video = mp.concatenate_videoclips([[meas_video], [sim_video]])
    #
    # # Write the result to a file
    # final_video.write_videofile(f"{base_folder}/{vname}_comparison.mp4")


def add_text_to_image(image_path, output_path, text):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype("arial.ttf", 40)
        text_position = (10, 10)
        draw.text(text_position, text, font=font, fill="white")

        img.save(output_path)


def apply_clipping(image, min_val, max_val):
    """
    Applies clipping to the input image according to the supplied max/min values
    """
    return np.clip(image, min_val, max_val)


def normalise_image(image, new_max, new_min):
    """
    Normalize an image according to new max and new min values
    """
    old_max = np.max(image)
    old_min = np.min(image)
    norm_img = (new_max - new_min) * (image - old_min) / (old_max - old_min) + new_min

    return norm_img


def analyse_rgb_distribution(image, cutoff=255, view=False):
    """
    Analyses the RGB distribution of an input image, returning the image thresholds.
    """
    # Find min, max and avg luminances
    min_lum = np.min(image)
    max_lum = np.max(image)
    avg_lum = np.mean(image)

    # Plot threshold: first instance where value is <=/>= cutoff beyond 0.3/99.7 percentile
    hist, bin_edges = np.histogram(curr_image, bins=256)
    index_997 = np.searchsorted(bin_edges, np.percentile(curr_image, 99.7), side='right') - 1
    t_high = index_997 + np.argmax(hist[index_997:] <= cutoff)
    t_high = bin_edges[t_high]
    t_low = np.argmin(hist <= cutoff)
    t_low = bin_edges[t_low]

    if view:
        # Plot luminance distribution and these variables
        plt.hist(image.ravel(), bins=256, color='black', label='Abs luminance values')
        # Add lines to show macro variables
        plt.axvline(min_lum, color='red', linestyle='dotted', linewidth=1)
        plt.axvline(max_lum, color='red', linestyle='dotted', linewidth=1)
        plt.axvline(avg_lum, color='purple', linestyle='dotted', linewidth=1, label='Mean')
        plt.axvline(np.percentile(curr_image, 99.7), color='orange', linestyle='dotted', linewidth=1, label='+/- 3 stdev')
        plt.axvline(np.percentile(curr_image, 0.3), color='orange', linestyle='dotted', linewidth=1)
        plt.axvline(t_high, color='green', linestyle='dotted', linewidth=1, label='Threshold')
        plt.axvline(t_low, color='green', linestyle='dotted', linewidth=1)
        plt.legend()
        plt.show()

    return t_low, t_high


if __name__ == '__main__':
    print(f'Script starting...')

    # Define folders
    base_folder = 'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Fog_Comparison'
    sim_path = f'{base_folder}/Sim_Trees_Removed'
    meas_path = f'{base_folder}/Meas'
    sim_path_la = f'{base_folder}/Sim_Trees_Removed_Labelled'
    meas_path_la = f'{base_folder}/Meas_Labelled'

    # Define gain, gamma, max/min
    dg = dg_ref
    gv = gv_ref
    min_value = 0
    max_value = 2 ** 12

    # --- IMPROC --- #
    print(f'Loading images...')
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

    # Process images (dg, gamma, mosaic)
    proc_images = []
    awb_images = []
    rgb_ranges = np.linspace(40, 160, 19)
    max_rs = np.linspace(45, 150, 19)
    max_gs = np.linspace(60, 160, 19)
    max_bs = np.linspace(60, 160, 19)
    for i in range(len(images)):
        print(f'Processing image: {filenames[i]}...')
        curr_image = images[i]

        # Analyse image to find thresholds for adjustment
        t_low, t_high = analyse_rgb_distribution(curr_image, cutoff=255, view=False)

        # Clip above/below t_high/t_low
        curr_image = np.clip(curr_image, t_low, t_high)

        # Adjust gamma values
        curr_image = adjust_gamma(curr_image, gv[i])

        # Normalise into RGB space (using info about mins from best guess)
        curr_image[:, :, 0] = normalise_image(curr_image[:, :, 0], max_rs[i], 0)
        curr_image[:, :, 1] = normalise_image(curr_image[:, :, 1], max_gs[i], 5)
        curr_image[:, :, 2] = normalise_image(curr_image[:, :, 2], max_bs[i], 0)

        # # Apply gamma for each channel?
        # curr_image[:, :, 0] = np.power(curr_image[:, :, 0], 1 / 1.1)
        # curr_image[:, :, 1] = np.power(curr_image[:, :, 1], 1 / 1.0)
        # curr_image[:, :, 2] = np.power(curr_image[:, :, 2], 1 / 1.0)

        # # Clip pixel values according to defined min/max
        # curr_image[:, :, 0] = apply_clipping(curr_image[:, :, 0], min_lums[i], max_lums[i] + 10)
        # curr_image[:, :, 1] = apply_clipping(curr_image[:, :, 1], min_lums[i], max_lums[i] + 10)
        # curr_image[:, :, 2] = apply_clipping(curr_image[:, :, 2], min_lums[i], max_lums[i] + 10)

        curr_image = curr_image.astype(np.uint8)

        # Save with corrections
        sname_scaled = 'scaled'
        save_image = Image.fromarray(curr_image.astype(np.uint8))
        save_image.save(f'{sim_path}/{filenames[i][:-4]}_{sname_scaled}.png')

        # # Apply AWB
        # awb_image = white_balance_adjust(curr_image.astype(np.uint8), meas_images[i])
        #
        # # Save AWB images
        # sav_image = Image.fromarray(awb_image)
        # sav_image.save(f'{sim_path}/{filenames[i][:-4]}_gain_awb.png')
        # awb_images.append(awb_image)

        # Demosaic
        curr_image = simulate_mosaicing_rggb(curr_image.astype(np.uint8), blend=0.0)

        # Save with demosaicing
        sname_mos = 'scaled_mos'
        sav_image = Image.fromarray(curr_image)
        sav_image.save(f'{sim_path}/{filenames[i][:-4]}_{sname_mos}.png')
        proc_images.append(curr_image)
    proc_images = np.array(proc_images)
    awb_images = np.array(awb_images)
    print(f'Images processed successfully.')

    # Label the simulated and measured data
    print(f'Labelling simulated images...')
    for i in range(len(images)):
        # Sim
        fp_in_sim = f'{sim_path}/{filenames[i]}'
        fp_out_sim = f'{sim_path_la}/{filenames[i]}'
        in_sim = f'{fp_in_sim[:-4]}_{sname_mos}.png'
        out_sim = f'{fp_out_sim[:-4]}_{sname_mos}_la.png'

        # Define times for text
        if i <= 11:
            curr_time = f"08:{5 * i:02d}"
        elif 11 <= i <= 23:
            curr_time = f"09:{5 * (i - 12):02d}"

        # Define text section
        text = f"{curr_time}\nfog: {fog_levels[i]:.1e}\nDG: 2^{dg[i]:.1f}\ngamma: {gv[i]:.1f}"

        # Annotate sim
        add_text_to_image(in_sim, out_sim, text)

    # Meas
    print(f'Labelling measured images...')
    i = 0
    for filename in os.listdir(meas_path):
        if filename.endswith('.png'):
            in_meas = f'{meas_path}/{filename}'
            fp_out_meas = f'{filename[:-4]}_la.png'
            out_meas = f'{meas_path_la}/{fp_out_meas}'

            # Define times for text
            if i <= 11:
                curr_time = f"08:{5 * i:02d}"
            elif 11 <= i <= 23:
                curr_time = f"09:{5 * (i - 12):02d}"

            # Define text section
            text = f"{curr_time}\n"

            # Annotate meas
            add_text_to_image(in_meas, out_meas, text)
            i += 1
    print(f'Image labelling finished.')

    # Convert .pngs to .mp4
    # pngs_to_mp4(sim_path, meas_path, base_folder, include='corr', vname='manual_gamma_corr')
    # pngs_to_mp4(sim_path, meas_path, base_folder, include='mos', vname='manual_gamma_demos')

    # # Get the regions for each set of images
    # sim_regions =

    print('Script finished successfully.')
