import numpy as np
import time as tm
import pandas as pd
import random
from random import randint
import math
import xlsxwriter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display

from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, precision_recall_curve, recall_score, roc_curve, roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV

def Abs_Distance(md, dfS, target):
    prob1 = dfS['P_def'][md]
    prob2 = target

    a=np.where(prob1<prob2, prob1, prob2)
    b=np.where(prob1>prob2, prob1, prob2)

    k   = 1-sum(a)/sum(b)

    return k 

def Add_Distance(dfS, target):
    t_start = tm.time()
    d = []
    for i in Progressbar(range(dfS.shape[0]), every=1):
        d_ = Abs_Distance(i, dfS, target)
        d.append(d_)
    dfS['Distance']=d
    t_end = tm.time()
    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))

def Calculate_Metrics(transformed_dataset, target, metric, path, transformation):
    t_start = tm.time()
    total_pi = []
    
    if metric == 'all':
        metrics = ['roc', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'distance']
    else:
        metrics = metric
        
    var_names = ['Ratio', 'Combination'] + metrics
        
    global_pi = np.zeros((len(var_names),transformed_dataset.shape[1]))
        
    for var in Progressbar(range(transformed_dataset.shape[1]), every=1):
            
        scrs = transformed_dataset[:,var]
        total_pi = []         
        for metric in metrics:
            ppred = PowerIndex(target, scrs, metric) 
            total_pi.append([ppred])
                 
        ppreds=np.array(total_pi)
        global_pi[0,var]=var+1
        global_pi[1,var]=var+1
        global_pi[2:(len(ppreds)+2),var]=ppreds.ravel()
            
    var_names=np.array(var_names)
    global_pi1 = np.c_[var_names, global_pi]
    global_pi2 = global_pi1[:,1:].astype(float)
    global_pi2[2:,:] = np.around(global_pi2[2:,:]*100,3)
    global_pi2 = np.c_[global_pi1[:,0], global_pi2] 
    t_end = tm.time()
    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))
    return global_pi1

def Calibrate_Multivariate(data, defs, metric):
    lr   = LogisticRegression(C=1e9, fit_intercept=False).fit(data, defs)
    bt   = lr.coef_
    scrs = np.ravel(np.dot(data,np.transpose(bt)))

    pi   = PowerIndex(defs, scrs, metric)

    return pi, scrs, bt

def Graph(target, data):
    scrs = data
    defs = target
    df1=np.extract(defs==1,scrs)
    df0=np.extract(defs==0,scrs)

    sorter = np.argsort(scrs[:])
    scrs = scrs[sorter]
    defs = defs[sorter]
    ys = 1 / (1 + np.exp( -scrs) )
    
    fig, ax1 = plt.subplots()
    ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.grid(False)
    plt.title('PD Best Model')
    ax1.set_ylabel('Frequency', color='k')
    ax1.hist(df1, bins=20, normed=1, facecolor='r', edgecolor='k', alpha=0.3)
    ax1.hist(df0, bins=20, normed=1, facecolor='b', edgecolor='k', alpha=0.3)
    ax2.set_ylabel('PD', color='r')
    ax2.plot(scrs, ys,"r-", markersize=2) 
    plt.margins(0,0)
    plt.show()
    pred = np.where(ys >.5, 1, 0)
    cnf_matrix = confusion_matrix(defs, pred)
    plt.show()
       
    fig, ax1 = plt.subplots()
    font = FontProperties()
    df_cm = pd.DataFrame(np.array(cnf_matrix), range(2),
                  range(2))
    plt.clf()
    plt.imshow(df_cm, interpolation='nearest', cmap=plt.cm.Dark2)
    plt.grid(False)
    classNames = ['Negative','Positive']
    plt.title('Default or Not Default Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FN'], ['FP', 'TP']]
    font.set_weight('bold')
    for i in range(2):
        for j in range(2):
            plt.text(i,j, str(s[i][j])+" = "+str(df_cm[i][j]),fontproperties=font,color='white', fontsize=11, horizontalalignment='center')
    plt.show()
    tn, fp, fn, tp = cnf_matrix[0,0], cnf_matrix[0,1], cnf_matrix[1,0], cnf_matrix[1,1]

    accuracy  = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    F         = 2 * (precision*recall) / (precision+recall)
    print('===========================================')
    print('                  METRICS                  ')
    print('===========================================')
    print('AUC          ', PowerIndex(defs, scrs,'auc'))
    print('Accuracy     ', accuracy)
    print('Precision:   ', precision)
    print('Recall:      ', recall)
    print('F:           ', F)
    print('===========================================')
    return [tn, fp, fn, tp]

def Metrics(defs, scrs, prediction):
    co = confusion_matrix(defs, prediction)
    tn, fp, fn, tp = co[0,0], co[0,1], co[1,0], co[1,1]
    accuracy  = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    F         = 2 * (precision*recall) / (precision+recall)
    print(co)
    print('ROC          ', PowerIndex(defs, scrs,'roc'))
    print('AUC          ', PowerIndex(defs, scrs,'auc'))
    print('Accuracy     ', accuracy)
    print('Precision:   ', precision)
    print('Recall:      ', recall)
    print('F:           ', F)

def Modified_Jaccard_Distance(target,p_def):
    prob1 = p_def
    prob2 = target

    a=np.where(prob1<prob2, prob1, prob2)
    b=np.where(prob1>prob2, prob1, prob2)

    k   = 1-sum(a)/sum(b)

    return k 

def Multivariate(n_t, n_i, n_f, data, target, path):
    t_start = tm.time()
    nr = data.shape[1]
    ml = list(range(1,nr))
    metrics = ['roc','accuracy','precision','recall','f1','auc']
    p_i = []
    linea = []
    pp = []
    col0=[];col1 = [];col2=[];col3=[];col4=[];col5=[];col6=[];col7=[];col8=[];col9=[];col10=[]
    random.seed(91)
    for i in Progressbar(range(n_t), every=n_t/100):
        n = random.randint(n_i,n_f)
        random.shuffle(ml)
        md = []
        p_i = []
        pp = []
        for j in range(n):
            md.append(ml[j])        
        try:
            nDt = len(target)
            scr = np.array([1]*nDt)
            for i in md:
                scr_i = data[:,i-1]
                scr = np.c_[scr, scr_i]
            for j in md:
                p_i.append(j)
            for m in metrics:
                pi, VScr, bt = Calibrate_Multivariate(scr, target, m)
                ys = 1 / (1 + np.exp( -VScr) )
                pred = np.where(ys >.5, 1, 0)
                p_i.append(pi)
                pp.append(pi)
            linea.append(p_i)
            col0.append(md);col1.append(pp[0]*100);col2.append(pp[1]*100);col3.append(pp[2]*100)
            col4.append(pp[3]*100);col5.append(pp[4]*100);col6.append(pp[5]*100);col7.append(VScr)
            col8.append(pred);col9.append(ys);col10.append(bt)
        except:
            e='error'

    t_end = tm.time()
    
    names = ['Models','Roc','Accuracy','Precision','Recall','F1','Auc','Score','Prediction','P_def','Betas']
    columns_ = {'Models':col0,'Roc':col1,'Accuracy':col2,'Precision':col3,'Recall':col4,'F1':col5,'Auc':col6,'Score':col7,'Prediction':col8,'P_def':col9,'Betas':col10}
    
    df = pd.DataFrame(columns_,columns=names)

    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))
   
    return df
     
def Multivariate_Best_Model(number_iterations, X_train, y_train, X_test, y_test, metric, path, n_i, n_f):
    t_start = tm.time()
        
    df = Multivariate(number_iterations, n_i, n_f, X_train, y_train, path)
            
    if metric == 'Distance' or metric == 'distance':
        if metric == 'distance':
            metric = 'Distance'
        Add_Distance(df, y_train)
        asc = True
    else:
        asc = False
    df_sort = df.sort_values([metric], ascending=asc)
    dfS = df_sort.reset_index(drop=True)

    t_end = tm.time()
    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))
    return dfS

def PowerIndex(defs, scrs, metric='roc'):
    metric     = metric.lower()
    p_def      = 1/(1+np.exp(-scrs))
    prediction = np.where(p_def >.5, 1, 0)
    if metric == 'roc':
        pi = ROC(defs, scrs)
    elif metric == 'accuracy':
        pi = accuracy_score(defs, prediction)
    elif metric == 'precision':
        pi = precision_score(defs, prediction)
    elif metric == 'recall':
        pi = recall_score(defs, prediction)
    elif metric == 'f1':
        pi = f1_score(defs, prediction)
    elif metric == 'auc':
        pi = roc_auc_score(defs, prediction)
    elif metric == 'distance':
        pi = Modified_Jaccard_Distance(defs,p_def)
    return pi

def Products_Analysis(data, transformed_dataset, target, global_pi, metrics, m_prod, transformation, path, threshold):
    t_start = tm.time()
    list_metrics = ['roc', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'distance']
    i_m_p = list_metrics.index(m_prod)
    pi_indiv = global_pi[i_m_p + 2,1:].astype(float) 
         
    combinations = [] 
    new_dataset = transformed_dataset
    
    n = transformed_dataset.shape[1] 
         
    for i in Progressbar(range(n), every=1): 
            for k in range(n): 
                scrs = []
                if k>=i:
                    scrs = data[:,i]*data[:,k]
                    if transformation == 'logit':
                        pi_betas, s = Univariate(target, scrs, metrics)
                        pi_betas = pi_betas[0:-1]
                    else:
                        if metrics == 'all':
                            metrics = list_metrics
                        else:
                            metrics = metrics
                        defs = target
                        scrs = np.array(scrs)
                        defs = np.array(defs)  
                        
                        scr, dfl = Uniform_Regularization(scrs, defs)
                        m = np.mean(scr)
                        s = np.std(scr)
                        scrs = (scrs-m)/s
                        
                        pi_betas = []         
                        for metric in metrics:
                            ppred = PowerIndex(defs, scrs, metric) 
                            pi_betas.append([ppred]) 

                    if pi_betas[0] > pi_indiv[i] and pi_betas[0] > pi_indiv[k] and np.array(pi_betas[0]) > threshold:
                        i_k_pi = np.array([i,k])
                        all_pi = np.array(pi_betas[0:len(pi_betas)]).ravel()
                        i_k_pi = np.concatenate((i_k_pi, all_pi))
                        combinations.append(i_k_pi)
                        new_dataset = np.c_[new_dataset,scrs]
   
    if len(combinations) > 0:
       
        combinations = np.array(combinations)

        num_variable = [n]*combinations.shape[0]
        num_variable = np.add(num_variable, list(range(1,combinations.shape[0]+1)))

        combinations1 = np.c_[num_variable,combinations]

        df = pd.DataFrame(combinations1)
        n = len(combinations1)
        r = []

        for i in range(n):
            r.append(str(int(combinations1[i,1])) + 'x' + str(int(combinations1[i,2])))

        df[1] = r
        df.drop(df.columns[[2]], axis=1, inplace=True)

        df1 = pd.DataFrame(global_pi)
        df1 = pd.DataFrame.transpose(df1)
        df = df.T.reset_index(drop=True).T
        new_dataset_df = pd.concat([df1,df],0)
        
    else:
        print('There are not new combinations')
        df1 = pd.DataFrame(global_pi)
        df1 = pd.DataFrame.transpose(df1)
        new_dataset_df = df1
    t_end = tm.time()
    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))
    return new_dataset, new_dataset_df

def Progressbar(sequence, every=None, size=None, name='Items'):
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
        
def ROC(defs, scrs):
    nDt   = len(defs)
    n_def = defs.sum()   
    
    acum_a = 0
    acum_b = 0

    sum_acum = 0
    defs     = defs[np.argsort(-scrs)]
    for i in defs:
        acum_a   = acum_a + i
        sum_acum = sum_acum + 0.5 * (acum_a + acum_b)
        acum_b   = acum_a
    ROC = (2 * (sum_acum / n_def) - nDt) / (nDt - n_def)
    return ROC

def Transformation(data, target, model):
    t_start   = tm.time()
    ini       = 1
    fin       = data.shape[1] + 1
    newdata   = []
    
    if model == 'logit':
        for r_i in Progressbar(range(ini,fin), every=1):
            defs = target
            scrs = data[:,r_i-1]
            p, scrs = Univariate(target, scrs, metric='all')
     
            newdata.append(scrs)
        aux = np.asarray(newdata)
        newdata = aux.transpose()
    else:
        for r_i in Progressbar(range(ini,fin), every=1):
            defs = target
            scrs = data[:,r_i-1]
            scrs = np.array(scrs)
            defs = np.array(defs)  
            scr, dfl = Uniform_Regularization(scrs, defs)
            
            m = np.mean(scr)
            s = np.std(scr)
            scrs = (scrs-m)/s
            
            inds = scrs.argsort()
            sortedTarget = defs[inds]
            
            half_1 = sum(sortedTarget[0:int(len(sortedTarget)/2)])
            half_2 = sum(sortedTarget[int(len(sortedTarget)/2)+1:])
            
            if half_1 > half_2:
                scrs = -scrs
                       
            newdata.append(scrs)
        aux = np.asarray(newdata)
        newdata = aux.transpose()

    t_end = tm.time()
    print ('Runtime: {:6.0f} seg'.format(t_end-t_start))
    return newdata

def Uniform_Regularization(scrs, defs):
    def1    = np.compress(defs==1,scrs)
    def0    = np.compress(defs==0,scrs)
    n_def   = len(def1)
    n_nodef = len(def0)
    if n_def < n_nodef:
        deci    = int(n_nodef / n_def)
        nmax    = deci * n_def

        dec     = np.array(def1) 
        dfl     = np.array([1] * n_def)

        for i in range(0, nmax, deci):
            dec = np.append(dec,[def0[i+int(deci/2)]])
            dfl = np.append(dfl,[0])
    else: 
        dec = scrs
        dfl = defs

    return  dec, dfl

def Univariate(target, scrs, metric='all'):
    np.seterr(all='print')
    pi_betas = []
    defs = target
    scrs = np.array(scrs) 
    defs = np.array(defs) 
    
    scr, dfl = Uniform_Regularization(scrs, defs) 
    nDt = len(scr) 
    
    scr0 = np.array( [1] * nDt )  
    scr = np.c_[scr0, scr] 
    
    lr = LogisticRegression(C=1e9, fit_intercept = False).fit(scr, dfl) 
    bt = np.ravel(lr.coef_) 
    
    
    scrs = 1 / (1 + np.exp( -(bt[0] + bt[1] * scrs)) )
    m = np.mean(scrs)
    s = np.std(scrs)
    scrs = (scrs-m)/s
    
    if metric == 'all':
        metrics = ['roc', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'distance']
    else: 
        metrics = metric  
    for metric in metrics:
        pred = PowerIndex(defs, scrs, metric) 
        pi_betas.append([pred])
                
    pi_betas.append([bt])
    
    return pi_betas, scrs 