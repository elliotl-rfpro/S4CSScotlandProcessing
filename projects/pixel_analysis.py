import os
from PIL import Image
from PIL.ImageStat import Stat as ims
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

nan = np.nan


def analyse_bmp(folder):
    # Analyse a bitmap image from the data folder. Get all images, see how pixel values change according to time of day
    times = np.arange(datetime(2024, 2, 11), datetime(2024, 2, 12), timedelta(minutes=5)).astype(str)
    for i in range(len(times)):
        times[i] = times[i][-15:-10]

    images = []
    for file in os.listdir(folder):
        if file.endswith('.bmp'):
            images.append(Image.open(folder + '/' + file))
    # Pixels of targets: white center, black reference square, top (blue), mid (red), bottom (green)
    targets = [
        [[1600, 1250], [1660, 1235], [1624, 1186], [1624, 1204], [1624, 1220]],
        [[1790, 1250], [1886, 1250], [1832, 1181], [1832, 1206], [1832, 1231]],
        [[1990, 1170], [nan, nan, nan], [2002, 1145], [2002, 1151], [2002, 1157]],
        [[2080, 1193], [2041, 1193], [2096, 1164], [2096, 1173], [2096, 1183]],
        [[2500, 1300], [2232, 1286], [2571, 1172], [2570, 1219], [2567, 1266]],
    ]
    pixel_1 = []
    pixel_2 = []
    pixel_3 = []
    pixel_4 = []
    pixel_5 = []
    for image in images:
        pixel_1.append(image.getpixel(targets[0][0]))
        pixel_2.append(image.getpixel(targets[0][1]))
        pixel_3.append(image.getpixel(targets[0][2]))
        pixel_4.append(image.getpixel(targets[0][3]))
        pixel_5.append(image.getpixel(targets[0][4]))

    # Plot pixel values
    n = 1
    for pixel in [pixel_1, pixel_2, pixel_3, pixel_4, pixel_5]:
        # Troublesome time array length fix
        if len(times) != len(pixel_1):
            times = times[:-1]
        plt.figure()
        plt.title(folder[-27:])
        plt.plot(np.array(times), np.array(pixel)[:, 0], 'r', label='R')
        plt.plot(np.array(times), np.array(pixel)[:, 1], 'g', label='G')
        plt.plot(np.array(times), np.array(pixel)[:, 2], 'b', label='B')
        plt.ylabel('Pixel intensity')
        plt.xlabel('Time')
        plt.xticks(range(0, len(times), 12), times[::12], rotation=45)
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/results/Claytex_LICamera/pixel_{n}_{folder[-10:]}.png')
        # plt.show()
        plt.close()
        n += 1


def analyse_png(folder):
    # Analyse a bitmap image from the data folder. Get all images, see how pixel values change according to time of day
    times = np.arange(datetime(2024, 2, 11), datetime(2024, 2, 12), timedelta(minutes=5)).astype(str)
    for i in range(len(times)):
        times[i] = times[i][-15:-10]

    images = []
    for file in os.listdir(folder):
        if file.endswith('.png'):
            images.append(Image.open(folder + '/' + file))
    # Pixels of targets: white center, black reference square, top (blue), mid (red), bottom (green)
    # Create bounding boxes for each region, in order to average over pixel values.
    t1_bounds = [[192, 1221, 386, 1354], [477, 1186, 523, 1222],
                 [329, 1034, 404, 1066], [333, 1090, 408, 1120], [333, 1144, 1090, 1176]]
    t2_bounds = [[807, 1265, 1166, 1484], [1270, 1227, 1347, 1293],
                 [1064, 995, 1175, 1039], [1059, 1077, 1173, 1125], [1061, 1157, 1175, 1205]]
    t3_bounds = [[1573, 947, 1660, 1000], [0, 1, 1, 2],
                 [1636, 881, 1664, 902], [1635, 902, 1664, 913], [1638, 926, 1661, 932]]
    t4_bounds = [[1856, 1043, 2000, 1134], [1782, 1038, 1811, 1064],
                 [1960, 935, 2004, 956], [1960, 971, 2004, 991], [1960, 1002, 2004, 1022]]

    target_1 = []
    target_2 = []
    target_3 = []
    target_4 = []
    for image in images:
        # For each image, append all target regions: create clone, crop region, find average value, append
        target_1.append([
            ims(image.copy().crop((t1_bounds[0][0], t1_bounds[0][1], t1_bounds[0][2], t1_bounds[0][3]))).mean,
            ims(image.copy().crop((t1_bounds[1][0], t1_bounds[1][1], t1_bounds[1][2], t1_bounds[1][3]))).mean,
            ims(image.copy().crop((t1_bounds[2][0], t1_bounds[2][1], t1_bounds[2][2], t1_bounds[2][3]))).mean,
            ims(image.copy().crop((t1_bounds[3][0], t1_bounds[3][1], t1_bounds[3][2], t1_bounds[3][3]))).mean,
            ims(image.copy().crop((t1_bounds[4][0], t1_bounds[4][1], t1_bounds[4][2], t1_bounds[4][3]))).mean,
        ])
        target_2.append([
            ims(image.copy().crop((t2_bounds[0][0], t2_bounds[0][1], t2_bounds[0][2], t2_bounds[0][3]))).mean,
            ims(image.copy().crop((t2_bounds[1][0], t2_bounds[1][1], t2_bounds[1][2], t2_bounds[1][3]))).mean,
            ims(image.copy().crop((t2_bounds[2][0], t2_bounds[2][1], t2_bounds[2][2], t2_bounds[2][3]))).mean,
            ims(image.copy().crop((t2_bounds[3][0], t2_bounds[3][1], t2_bounds[3][2], t2_bounds[3][3]))).mean,
            ims(image.copy().crop((t2_bounds[4][0], t2_bounds[4][1], t2_bounds[4][2], t2_bounds[4][3]))).mean,
        ])
        target_3.append([
            ims(image.copy().crop((t3_bounds[0][0], t3_bounds[0][1], t3_bounds[0][2], t3_bounds[0][3]))).mean,
            ims(image.copy().crop((t3_bounds[1][0], t3_bounds[1][1], t3_bounds[1][2], t3_bounds[1][3]))).mean,
            ims(image.copy().crop((t3_bounds[2][0], t3_bounds[2][1], t3_bounds[2][2], t3_bounds[2][3]))).mean,
            ims(image.copy().crop((t3_bounds[3][0], t3_bounds[3][1], t3_bounds[3][2], t3_bounds[3][3]))).mean,
            ims(image.copy().crop((t3_bounds[4][0], t3_bounds[4][1], t3_bounds[4][2], t3_bounds[4][3]))).mean,
        ])
        target_4.append([
            ims(image.copy().crop((t4_bounds[0][0], t4_bounds[0][1], t4_bounds[0][2], t4_bounds[0][3]))).mean,
            ims(image.copy().crop((t4_bounds[1][0], t4_bounds[1][1], t4_bounds[1][2], t4_bounds[1][3]))).mean,
            ims(image.copy().crop((t4_bounds[2][0], t4_bounds[2][1], t4_bounds[2][2], t4_bounds[2][3]))).mean,
            ims(image.copy().crop((t4_bounds[3][0], t4_bounds[3][1], t4_bounds[3][2], t4_bounds[3][3]))).mean,
            ims(image.copy().crop((t4_bounds[4][0], t4_bounds[4][1], t4_bounds[4][2], t4_bounds[4][3]))).mean,
        ])

    # Plot region values for each target
    n = 1
    region_names = ['White surface', 'Black surface', 'Green surface', 'Yellow surface', 'Blue surface']
    for target in [target_1, target_2, target_3, target_4]:
        # Each target has 5 regions
        for i in range(4):
            # Troublesome time array length fix
            if len(times) != len(target_1):
                times = times[:-1]
            plt.figure()
            plt.title(f'{folder[-27:]}, region: {i + 1}')
            plt.plot(np.array(times), np.array(target)[:, i, 0], 'r', label='R')
            plt.plot(np.array(times), np.array(target)[:, i, 1], 'g', label='G')
            plt.plot(np.array(times), np.array(target)[:, i, 2], 'b', label='B')
            plt.ylabel('Pixel intensity')
            plt.xlabel('Time')
            plt.xticks(range(0, len(times), 12), times[::12], rotation=45)
            plt.minorticks_on()
            plt.tight_layout()
            plt.savefig(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/results/NPL_LucidCamera/target_{n}_{region_names[i]}_{folder[-9:]}.png')
            # plt.show()
            plt.close()
        n += 1


if __name__ == '__main__':
    folders = [
        '2024_02_11',
        '2024_02_21',
        '2024_02_29',
        '2024_03_01',
        '2024_03_04'
    ]
    for folder in folders:
        analyse_bmp(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/{folder}')
        analyse_png(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/{folder}')

    print('Done!')
