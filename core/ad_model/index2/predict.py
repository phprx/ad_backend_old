import dill as pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.formula.api as sm
from core.ad_model.index2.core_utilities import patient_staging
from core.ad_model.index2.gaussian_mixture_model import calculate_prob_mm


def predict_AD_index(DataTest):
    f = open("core/ad_model/index2/ModelOutput_all.pkl", "rb+")
    ModelOutput = pickle.load(f)
    BiomarkerList = ModelOutput[0]
    pi0 = ModelOutput[6]
    event_centers = ModelOutput[5]

    # /计算阈值
    DTrain = pd.read_csv('core/ad_model/index2/Data_7.csv')
    Ytrain = DTrain['Diagnosis']
    idx = Ytrain != 'MCI';
    Ytrain = Ytrain[idx];
    Ytrain[Ytrain == 'CN'] = 0;
    Ytrain[Ytrain == 'AD'] = 1;
    Strain = ModelOutput[7][0]['Stages'];
    Strain = Strain.values[idx];
    fpr, tpr, thresholds = metrics.roc_curve(pd.to_numeric(Ytrain.values), Strain)
    max_Strain = max(Strain)
    bacc = []
    for j in thresholds:
        Ypred1 = np.zeros(Strain.shape)
        Ypred1[Strain >= j] = 1
        cm1 = metrics.confusion_matrix(pd.to_numeric(Ytrain.values), Ypred1)
        sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        bacc.append((sensitivity1 + specificity1) / 2)
    # 阈值
    thr_opt = thresholds[np.argmax(bacc)]

    # 获取预测样本
    UniqueSubjIDs = pd.Series.unique(DataTest['PTID'])
    Factors = ['Age', 'Sex', 'ICV']
    sex = np.zeros(len(DataTest['PTID']), dtype=int)
    idx_male = DataTest['Sex'] == 'Male';
    idx_male = np.where(idx_male);
    idx_male = idx_male[0];
    idx_female = DataTest['Sex'] == 'Female';
    idx_female = np.where(idx_female);
    idx_female = idx_female[0];
    sex[idx_male] = 1;
    sex[idx_female] = 0;
    DataTest = DataTest.drop('Sex', axis=1)
    DataTest = DataTest.assign(Sex=pd.Series(sex, DataTest.index))
    mean_confval = np.zeros(len(Factors))

    sex = np.zeros(len(DTrain['Diagnosis']), dtype=int)
    idx_male = DTrain['Sex'] == 'Male';
    idx_male = np.where(idx_male);
    idx_male = idx_male[0];
    idx_female = DTrain['Sex'] == 'Female';
    idx_female = np.where(idx_female);
    idx_female = idx_female[0];
    sex[idx_male] = 1;
    sex[idx_female] = 0;
    DTrain = DTrain.drop('Sex', axis=1)
    DTrain = DTrain.assign(Sex=pd.Series(sex, DTrain.index))

    for j in range(len(Factors)):
        mean_confval[j] = np.nanmean(DTrain[Factors[j]].values)

    # 对获得的预测样本数据处理
    f = open("core/ad_model/index2/ols_Output.pkl", "rb+")
    betalist = pickle.load(f)

    droplist = ['PTID', 'Diagnosis', 'EXAMDATE']
    DataBiomarkers = DataTest
    DataBiomarkers = DataBiomarkers.drop(Factors, axis=1)
    H = list(DataBiomarkers)
    for j in droplist:
        if any(j in f for f in H):
            DataBiomarkers = DataBiomarkers.drop(j, axis=1)
    BiomarkersList = list(DataBiomarkers)
    BiomarkersListnew = []
    for i in range(len(BiomarkersList)):
        BiomarkersListnew.append(BiomarkersList[i].replace(' ', '_'))
        BiomarkersListnew[i] = BiomarkersListnew[i].replace('-', '_')
    for i in range(len(BiomarkersList)):
        DataTest = DataTest.rename(columns={BiomarkersList[i]: BiomarkersListnew[i]})
    Deviation = (DataTest[Factors].astype('float64') - mean_confval)
    Deviation[np.isnan(Deviation)] = 0

    for i in range(len(BiomarkersList)):
        betai = betalist[i].values
        betai_slopes = betai[1:]
        CorrectionFactor = np.dot(Deviation.values, betai_slopes)  # 偏差*线性回归参数得到修正参数
        DataTest[BiomarkersListnew[i]] = DataTest[BiomarkersListnew[i]].astype('float64') - CorrectionFactor
    DataTest = DataTest.drop(Factors, axis=1)
    for i in range(len(BiomarkersList)):
        DataTest = DataTest.rename(columns={BiomarkersListnew[i]: BiomarkersList[i]})

    str_confounders = ''
    mean_confval = np.zeros(len(Factors))
    for j in range(len(Factors)):
        str_confounders = str_confounders + '+' + Factors[j]
    str_confounders = str_confounders[1:]

    ## Multiple linear regression
    betalist = []
    for i in range(len(BiomarkersList)):
        str_formula = BiomarkersListnew[i] + '~' + str_confounders
        # sm.ols是用最小二乘法来拟合每个属性与性别年龄和ICV的关系
        result = sm.ols(formula=str_formula, data=DTrain[idx]).fit()
        betalist.append(result.params)
    num_events = len(BiomarkersList);
    num_subjects = DataTest.shape[0]
    matData = np.zeros((num_subjects, num_events, 1))
    for i in range(7):
        matData[:, i, 0] = DataTest[BiomarkersList[i]].values
    DataTest = matData

    Nfeats = DataTest.shape[2]
    if Nfeats == 1:
        params = np.zeros((DataTest.shape[1], 5))
        params[:, :2] = ModelOutput[1]
        params[:, 2:4] = ModelOutput[2]
    params[:, 4] = ModelOutput[3]

    p_yes, p_no, likeli_pre, likeli_post = calculate_prob_mm(DataTest[:, :, 0], params, val_invalid=np.nan);
    subj_stages = patient_staging(pi0, event_centers, p_yes, p_no, ['exp', 'p']);
    print(thr_opt)  # 分期阈值
    print(subj_stages)  # 病人病情分期大于阈值为AD小于阈值为CN
    if (subj_stages[0] > thr_opt):
        return "该患者有痴呆风险，建议到医院进行治疗"
    else:
        return "该患者目前正常，建议继续观察"

