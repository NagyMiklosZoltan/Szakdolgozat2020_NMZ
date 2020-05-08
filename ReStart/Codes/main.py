
# from ReStart.Codes.Classification.FullTransferLearning import TrainModelClassification
from ReStart.Codes.Classification.LoadAndUseModelToPredict import GetPrediction
from ReStart.Codes.RDM.GetRDMAverage import calculateClassAverages
from ReStart.Codes.Dataset.MatFilesDataset import getAverageEVC_RDM

root = r'C:\Users\NagyMiklosZoltan\PycharmProjects\Szakdolgozat2020'


rdm_path = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\target_fmri.mat'

p = root + r'\algonautsChallenge2019\Training_Data\92_Image_Set\92images'
m = root + r'\ReStart\Weights\weights-improvement-02-0.05.hdf5'

rdm = getAverageEVC_RDM(rdm_path)

preds = GetPrediction(p, m)
print(preds)

class_rdm = calculateClassAverages(rdm, preds)

print(class_rdm)





