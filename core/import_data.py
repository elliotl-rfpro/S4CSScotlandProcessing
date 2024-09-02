from utils.dir_tools import copy_txts

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
    for source, destination in zip(sources, destinations):
        # copy_bmps(source, destination)
        copy_txts(source, destination)
