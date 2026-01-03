

%reset -f

import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

def create_sequences(X, y, int_delta, time_steps=7):
    X_out = []; y_out = []
    for i in range(1, len(X)+1):
        if(i < time_steps):
            X_out.append(0*np.ones(time_steps))
            y_out.append(-1)
        else:
            X_out.append(X.iloc[i-time_steps:i].values)
            y_out.append(y.iloc[i-2]-y.iloc[i-3-int_delta])            
            
    return np.array(X_out), np.array(y_out)

def get_data_train(dfTmp, strIndTmp, IndOutTmp, int_delta):
    time_steps = 15    
    
    # print('get_data_train               int_delta:', int_delta, 'strIndTmp:',strIndTmp, 'IndOutTmp:',IndOutTmp)
    
    strIndY = IndOutTmp
    dfTmp['IND'] = dfTmp[strIndTmp]
    
    scaler1 = StandardScaler()
    scaler1 = scaler1.fit(np.array(dfTmp['IND']).reshape(-1,1))
    dfTmp['IND'] = scaler1.transform(np.array(dfTmp['IND']).reshape(-1,1))
    dfTmp['IND'] = round(100*dfTmp['IND'])/100
    
    X_trainTmp, y_trainTmp  = create_sequences(dfTmp['IND'], dfTmp[strIndY], int_delta, time_steps)    
    
    X_train = X_trainTmp
    y_train = np.where(y_trainTmp > 0, 1, 0)
    
    return X_train, y_train

extensionFile = "_inf24_15m.pkl"
# extensionFile = "_24_15m.pkl"

# tabSymbol = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT', 'LTC', 'LINK', 'XLM']
tabSymbol = ['BTC']

tabFil = [15, 20]
# tabFil = [20]
tabDelta = [0, 1, 2, 3]
# tabDelta = [0]
tabIndi = ['RSI5', 'RSI9', 'RSI14', 'CCI9', 'CCI13', 'CCI20', 'MACD13', 'MACD26', 'MACD40']
tabIndiOut = ['MACD13', 'MACD26', 'MACD40']


for aa in range(0, len(tabSymbol)):  
    symbol = tabSymbol[aa] + 'USDT'
    strFile = "C:\\ml/CLOSE_FIL/data" + symbol + "Train" + extensionFile
    df = pickle.load(open(strFile,'rb'))
    print('###############################################################################     Crypto  :', tabSymbol[aa])
    result_dict = {}; int_dict = 0
    for ff in range(0, len(tabFil)):
        int_fil = tabFil[ff]
        for ii in range(0, len(tabDelta)):
            int_delta = tabDelta[ii]
            for oo in range(0, len(tabIndiOut)):
                IndOut = 'FL_' + tabIndiOut[oo] + '_' + str(int_fil)
                for jj in range(0, len(tabIndi)):            
                    Indicateur = tabIndi[jj]                    
                    # print('int_fil:', int_fil, 'int_delta:', int_delta, 'Indicateur:', Indicateur, 'IndOut:', IndOut)
                    
                    X_trainAll, y_trainAll = get_data_train(df, Indicateur, IndOut, int_delta)
                    
                    file_name = "NnetIndi_" + tabIndiOut[oo] + "_"  + Indicateur + "_" + str(int_delta) + "_"+ str(int_fil)
                    Name_File = "C:\\ml/CLOSE_FIL/modelTrain" + file_name + ".pkl"
                    print(Name_File)
                    model_nnet = pickle.load(open(Name_File,'rb'))
                    
                    y_pred_proba = model_nnet.predict_proba(X_trainAll)[:, 1]  # Probabilité d'achat (classe 1)
                    
                    # y_train_proba = stacking_model2.predict_proba(X_train_all)[:, 1]  # Probabilité d'achat (classe 1)
                    auc_train = roc_auc_score(y_trainAll, y_pred_proba)
                    # print(f"AUC Score pour le modèle Stacking Entraînement avec predict_proba (Train) : {auc_train}")
                    
                    # Conversion en classes avec un seuil de 0.5
                    threshold = 0.5
                    y_train_pred = (y_pred_proba > threshold).astype(int)  # Achat si p > 0.5, sinon Vente
                    train_report = classification_report(y_trainAll, y_train_pred, output_dict=True)
                    accuracy_train = train_report['accuracy']
                    # print(f"{Indicateur} : Accuracy pour le modèle (Train) : {accuracy_train:.2%}")
                    
                    Pind = "P0_" + tabIndiOut[oo] + "_" + Indicateur + "_" + str(int_delta) + '_' + str(int_fil)
                    Yind = "Y_" + tabIndiOut[oo] + "_" + str(int_delta) + "_"+ str(int_fil)
                    df[Yind] = y_trainAll
                    df[Pind] = y_pred_proba
                    
                    accuracy_train = round(accuracy_train, 3)
                    result_dict[int_dict] = {'int_fil': int_fil,
                                  'int_delta': int_delta,
                                  'IndOut': tabIndiOut[oo],
                                  'Indicateur': Indicateur,
                                  'accuracy_train': accuracy_train}
                    print(result_dict[int_dict])
                    int_dict = int_dict  + 1  
    
    df = df.drop(columns=['IND', 'FL_MACD13_15', 'FL_MACD13_20', 'FL_MACD26_15', 'FL_MACD26_20', 'FL_MACD40_15', 'FL_MACD40_20', 
                          'MACD13', 'MACD26', 'MACD40', 'RSI5', 'RSI9', 'RSI14', 'CCI9', 'CCI13', 'CCI20'])
    df = df[100:len(df)-100].reset_index(drop=True)
    
    strFile = "C:\\ml/CLOSE_FIL/data" + symbol + "BrutIndNnet" + extensionFile
    pickle.dump(df, open(strFile, "wb"))
    

df_result_dict = pd.DataFrame(result_dict).transpose()





# from tensorflow.keras.models import load_model
# from tensorflow.python.saved_model.load import LoadOptions

# # options = LoadOptions(experimental_io_device="/job:localhost")
# # model = load_model(r"C:\ml/RCF_FIL/modelTrainmodel00.h5")

# import tensorflow.compat.v1 as tf1
# from tensorflow.keras.models import load_model

# # Charger le modèle avec compatibilité rétroactive
# # tf1.disable_eager_execution()
# model = load_model("C:/ml/RCF_FIL/modelTrainmodel00.h5")

# # model.predict([X_trainIndi0, X_trainIndi1, X_trainIndi2]).flatten() 

# import tensorflow as tf
# print("Version de TensorFlow :", tf.__version__)
# print("GPU disponible :", tf.config.list_physical_devices('GPU'))


# model_nnet = pickle.load(open('C:\\ml/RCF_FIL/modelTrainNnetIndi_MACD9_1_20.pkl','rb'))

