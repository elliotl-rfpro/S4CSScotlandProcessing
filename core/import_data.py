from improc.utils.dir_tools import copy_txts, raw2exr, remove_exrs, copy_exrs, exr2png

# Claytex camera
sources = [
    'D:/Claytex_LICamera_RCCB/2024-02-11',
    'D:/Claytex_LICamera_RCCB/2024-02-21',
    'D:/Claytex_LICamera_RCCB/2024-02-29',
    'D:/Claytex_LICamera_RCCB/2024-03-01',
    'D:/Claytex_LICamera_RCCB/2024-03-04'
]
destinations = [
    'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/2024_02_11',
    'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/2024_02_21',
    'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/2024_02_29',
    'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/2024_03_01',
    'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/Claytex_LICamera/2024_03_04'
]

# # NPL camera
# sources = [
#     'D:/NPL_LucidCamera/2024-02-11/Camera14_#0907/PNG',
#     'D:/NPL_LucidCamera/2024-02-21/Camera14_#0907/PNG',
#     'D:/NPL_LucidCamera/2024-02-29/Camera14_#0907/PNG',
#     'D:/NPL_LucidCamera/2024-03-01/Camera14_#0907/PNG',
#     'D:/NPL_LucidCamera/2024-03-04/Camera14_#0907/PNG'
# ]
# destinations = [
#     'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/2024_02_11',
#     'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/2024_02_21',
#     'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/2024_02_29',
#     'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/2024_03_01',
#     'C:/Users/ElliotLondon/Documents/PythonLocal/S4CSScotlandProcessing/data/NPL_LucidCamera/2024_03_04',
# ]

if __name__ == '__main__':
    # Move images
    # for source, destination in zip(sources, destinations):
    #     # copy_bmps(source, destination)
    #     copy_txts(source, destination)

    # Clean folder of .exrs
    sources = [
        'D:/WMG_LucidCamera/2024-02-11',
        'D:/WMG_LucidCamera/2024-02-21',
        'D:/WMG_LucidCamera/2024-02-29',
        'D:/WMG_LucidCamera/2024-03-01',
        'D:/WMG_LucidCamera/2024-03-04'
    ]
    for folder in sources:
        # Convert .raw to .exr
        raw2exr(folder, "C:/Users/ElliotLondon/Documents/PythonLocal/S4CSCardington/data/RawToExr.LUCIDTRITON.exe")

        local_fname = folder.replace('-', '_')

        # Copy .exrs to required folder
        copy_exrs(folder, f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Measured/RAW Camera/{local_fname[-10:]}')

        # Turn .exrs into .png files
        exr2png(f'C:/Users/ElliotLondon/Documents/PythonLocal/S4CWinterTesting/data/Measured/RAW Camera/{local_fname[-10:]}')

    # remove_exrs('D:/WMG_LucidCamera/2024-03-01')
