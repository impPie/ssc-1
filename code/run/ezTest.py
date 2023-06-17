#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
import pickle,numpy as np
from _3train.deepClassifier import DeepClassifier
from os import listdir, makedirs
from utils.fileManagement import selectClassifierID
# from eegFileReaderServer import EEGFileReaderServer
# from classifierClient import ClassifierClient
from utils.parameterSetup import ParameterSetup
from os.path import splitext, isfile, exists
from _4predict.stagePredictor import StagePredictor
from _2featureEx.featureExtractorRawDataWithSTFT import FeatureExtractorRawDataWithSTFT 
from _5test.evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat
import time


class EzT:

    def __init__(self, args):
        self.args = args
        self.classifier_type = 'UTSN-L'
        pass

    def start(self):
        # channelOpt = 1
        params = ParameterSetup()
        self.recordWaves = params.writeWholeWaves
        self.extractorType = params.extractorType
        self.classifierType = params.classifierType
        self.postDir = params.postDir
        self.classLabels = params.sampleClassLabels[:params.maximumStageNum]
        self.predictionState=1

        finalClassifierDir = params.finalClassifierDir
        
        classifierTypeFileName = 'classifierTypes.csv'
        with open(finalClassifierDir + '/' + classifierTypeFileName) as f:
            line = f.readline()
            classifierID, classifierType, samplingFreq, epochTime = [elem.strip() for elem in line.split(',')]
            print(classifierID, ',', classifierType, ',', samplingFreq, ',', epochTime)

        paramFileName = 'params.' + str(classifierID) + '.json'
        paramsForNetworkStructure = ParameterSetup(paramDir=finalClassifierDir, paramFileName=paramFileName)
        self.extractor = FeatureExtractorRawDataWithSTFT()       
        self.y_pred_L=[]
        
        try: 
            classifier = DeepClassifier(self.classLabels, classifierID=classifierID, paramsForDirectorySetup=params, paramsForNetworkStructure=paramsForNetworkStructure)
            model_path = finalClassifierDir + '/weights.' + str(classifierID) + '.pkl'

            print('model_path = ', model_path)
            classifier.load_weights(model_path)

            self.stagePredictor = StagePredictor(paramsForNetworkStructure, self.extractor, classifier, finalClassifierDir, classifierID, params.markovOrderForPrediction)
            
            timeStampSegment=[]
            #------------------------------------------------
            #------------------------------------------------
           
            f_path = finalClassifierDir + '/files_used_for_training.' + str(classifierID) + '.csv'
            trFL=[]
            beeg=[]
            bstg=[]
            with open(f_path) as testF:
                for l in testF:
                    ef,ff,fId = [elem.strip() for elem in l.split(',')]
                    trFL.append(ef)
            testDataP="../../data/pickled"
            for fn in listdir(testDataP):
                if fn.startswith("eegAndStage") and fn not in trFL:
                    with open(testDataP+'/'+fn,"rb") as dataFileHandler:
                        (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
                        beeg.append(eeg)
                        bstg.append(stageSeq)
            beeg=np.reshape(beeg,(-1,512))
            bstg=np.reshape(bstg,(-1))

         
            for eegSegment in beeg:
                if self.predictionState:

                    stagePrediction = self.stagePredictor.predict(
                        eegSegment, timeStampSegment, params.stageLabels4evaluation, params.stageLabel2stageID)

                else:
                    stagePrediction = '?'


                if self.predictionState:
                    # ----
                    # if the prediction is P, then use the previous one
                    if stagePrediction == 'P':
                        # print('stagePrediction == P for wID = ' + str(wID))
                        if len(self.y_pred_L) > 0:
                            finalClassifierDirPrediction = self.y_pred_L[len(
                                self.y_pred_L)-1]
                            # print('stagePrediction replaced to ' + stagePrediction + ' at ' + str(segmentID))
                        else:
                            stagePrediction = 'M'

                    self.y_pred_L.append(stagePrediction)
            # print("Finish predicting--- %s minutes ---" % (int(time.time() - start_time)/60))

            #Record
            with open("../../results/{}_predict.pickle".format(classifierID), 'wb') as out_path:
            # with open("../../results/predictlabels/{}_predict.pickle".format(classifierID), 'wb') as out_path:
                 pickle.dump(self.y_pred_L,out_path)
            out_path.close()
            
            y_pred = [x.upper() if x!= "n" else "S" for x in self.y_pred_L]
            y_pred = np.array(y_pred[11:])
            y_test = np.array(bstg[11:])

            (stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
            (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred, params.stageLabels4evaluation)
       
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
            
        except Exception as e:
            print(str(e))
            raise e
        


if __name__ == '__main__':
    
    start_time = time.time()

    args = sys.argv
    mainapp = EzT(args)
    mainapp.start()
    # print(datetime.datetime.now())
    print("--- %s minutes ---" % (int(time.time() - start_time)/60))
 
