import os
import shutil
from tqdm import tqdm


def copy_bmps(src: str, dst: str):
    # Copies all .bmp files from the source to the destination. Interval of 5s
    size = 0
    print(f'Moving from {src}...')
    for file in tqdm(os.listdir(src)):
        if file.endswith('_1.bmp') or file.endswith('5-00_Camera14_#0907.png') or file.endswith('0-00_Camera14_#0907.png'):
            pathname = os.path.join(src, file)
            if os.path.isfile(pathname):
                if not os.path.isfile(dst + '/' + pathname):
                    shutil.copy2(pathname, dst)

                    size += os.stat(pathname).st_size / 8e+6
    print(f'Total size: {size} MB')
    print(f'Moved to {dst}...')


def copy_txts(src: str, dst: str):
    # Copies all .txt files from the source to the destination. Interval of 5s
    size = 0
    print(f'Moving from {src}...')
    for file in tqdm(os.listdir(src)):
        if file.endswith('5_IMX728_RCCB_1_RGB_Mean') or file.endswith('0_IMX728_RCCB_1_RGB_Mean'):
            pathname = os.path.join(src, file)
            if os.path.isfile(pathname):
                if not os.path.isfile(dst + '/' + pathname):
                    shutil.copy2(pathname, dst)

                    size += os.stat(pathname).st_size / 8e+6
    print(f'Total size: {size} MB')
    print(f'Moved to {dst}...')
