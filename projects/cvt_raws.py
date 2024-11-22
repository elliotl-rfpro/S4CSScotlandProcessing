"""Script for converting .RAW files to a manageable image format within Python"""

import numpy as np
from PIL import Image
import os


def read_rccb_raw(file_path):
    """
    Reads a raw RCCB image file and returns the data as a numpy array.

    Parameters:
    file_path (str): The path to the raw RCCB image file.

    Returns:
    numpy.ndarray: The raw image data as a 1D numpy array of uint8.
    """
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint8)
    return raw_data


def rccb_to_rgb(raw_data, width=2880, height=1860):
    """
    Converts raw RCCB image data to an RGB image.

    Parameters:
    raw_data (numpy.ndarray): The raw RCCB image data.
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    numpy.ndarray: The converted RGB image as a 3D numpy array of uint8.
    """
    # Calculate the total number of pixels
    total_pixels = raw_data.size

    # Ensure the dimensions are correct
    if width * height != total_pixels:
        raise ValueError("Unable to determine correct dimensions from the raw data size.")

    # Initialize the RGB image array with uint16 to prevent overflow
    rgb_image = np.zeros((height, width, 3), dtype=np.uint16)

    # Convert RCCB to RGB
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            r = raw_data[y * width + x]
            c1 = raw_data[y * width + x + 1]
            c2 = raw_data[(y + 1) * width + x]
            b = raw_data[(y + 1) * width + x + 1]

            # Calculate the green channel by averaging the clear channels
            g = (int(c1) + int(c2)) // 2

            # Assign the RGB values to the corresponding pixels
            rgb_image[y, x] = [r, g, b]
            rgb_image[y, x + 1] = [r, g, b]
            rgb_image[y + 1, x] = [r, g, b]
            rgb_image[y + 1, x + 1] = [r, g, b]

    # Convert the image back to uint8
    return rgb_image.astype(np.uint8)


def save_rgb_image(rgb_image, output_path):
    """
    Saves an RGB image to a file if the file does not already exist.

    Parameters:
    rgb_image (numpy.ndarray): The RGB image data.
    output_path (str): The path to save the image file.

    Returns:
    None
    """
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Image not saved.")
    else:
        image = Image.fromarray(rgb_image)
        image.save(output_path)
        print(f"RGB image saved to {output_path}")


if __name__ == '__main__':
    # By default, process all files within the RAW images folder upon the Seagate drive.
    drive_path = 'D:/WMG_LucidCamera'
    save_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Measured/RAW Camera (Processed)'
    folders = [
        '2024-02-11',
        '2024-02-21',
        '2024-02-29',
        '2024-03-01',
        '2024-03-04'
    ]

    # Create array of times
    times = []
    for hour in range(24):
        for minute in range(0, 60, 5):
            times.append(f"{hour:02d}{minute:02d}")

    dates = [
        '0211',
        '0221',
        '0229',
        '0301',
        '0304'
    ]

    for folder, date in zip(folders, dates):
        print(f'Processing folder: {folder}')
        for time in times:
            fname = f'2024{date}_{time}_image1'
            rpath = f'{drive_path}/{folder}/{fname}.raw'
            raw_data = read_rccb_raw(rpath)
            raw_image = rccb_to_rgb(raw_data, width=2880, height=1860)
            save_folder = folder.replace('-', '_')
            save_rgb_image(raw_image, f'{save_path}/{save_folder}/{fname}.png')
