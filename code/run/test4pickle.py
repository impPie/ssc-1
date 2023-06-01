import sys
sys.path.insert(1, '..')
from _5test.evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat
import pickle
import numpy as np

classifierID = "3QSN9X"
with open("../../results/predictlabels/{}_predict.pickle".format(classifierID), 'rb') as out_path:
    y_pred_L=pickle.load(out_path)
# out_path.close()
with open("../../data/pickled/eegAndStage._10per_1.pkl","rb") as dataFileHandler:
    (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)

orig_stageLabels = ['S', 'L', 'R', 'H', 'RW', 'M', 'P', 'F2', '?', '-']
stageLabels4evaluation = orig_stageLabels[:4]

y_pred = [x.upper() if x!= "n" else "S" for x in y_pred_L]
y_pred = np.array(y_pred[11:])
y_test = np.array(stageSeq[11:])

(stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
(stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred,stageLabels4evaluation)
cm = printConfusionMat(stageLabels4confusionMat, confusionMat)
print('sensitivity =', sensitivity)
print('specificity =', specificity)
print('accuracy =', accuracy)
print('precision =', precision)
# rec ConfMat
with open("../../results/confMatrix", 'a') as mt:
        mt.write("\n# {}\n".format(classifierID))
        mt.writelines(cm)
        mt.write("\n")
        mt.writelines(['sensitivity=', str(sensitivity),'\n'])
        mt.writelines(['specificity=', str(specificity),'\n'])
        mt.writelines(['accuracy =  ', str(accuracy),'\n'])
        mt.writelines(['precision = ', str(precision),'\n'])
mt.close()