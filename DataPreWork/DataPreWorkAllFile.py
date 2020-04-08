from DataPreWork import PreWork

# Input Image Path
inputDirectory_92 = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\92images'
inputDirectory_118 = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\RawImages\\118images'

# PreWorked Output Image Path
outputDirectory_92 = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\PreWorkedImages\\92images'
outputDirectory_118 = 'C:\\Users\\NagyMiklosZoltan\\PycharmProjects\\Szakdolgozat2020\\PreWorkedImages\\118images'

# Gauss Blur kernel size
dim = 3
k_size = (dim, dim)

# Edge Detecition Thresholds
thresholds = 100, 150

# Expected Size to resize
size = (175, 175)

preWork = PreWork.PreWork(g_kernel=k_size, ex_size=size, thresholds=thresholds)
# 92 image set prework
preWork.ImagePreProcessing(input_path=inputDirectory_92,
                           output_path=outputDirectory_92)
# 118 image set prework
preWork.ImagePreProcessing(input_path=inputDirectory_118,
                           output_path=outputDirectory_118)
