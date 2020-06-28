import pandas as pd
import numpy as np
import pickle
import os


x_test = pd.read_csv('/data/input_data/base_data_test_X.csv')

#  区分类别变量
cate_var = x_test.iloc[:, 1:].select_dtypes(include=['object']).columns.values

x_test = x_test.drop(list(cate_var), axis=1)

test_cust = x_test[['cust_id']]
x_test = x_test.drop(['cust_id'], axis=1)
x_test.fillna((-999), inplace=True)

# load model
clfs = pickle.load(open(r'model/lr.pkl', 'rb'))

result = clfs.predict_proba(x_test)

y_pred_test = pd.DataFrame({'cust_id': test_cust.cust_id,
                            'y_pred_pro': result[:,1]}, columns=['cust_id', 'y_pred_pro'])

result = np.array(y_pred_test['y_pred_pro'])

pect_n20 = np.percentile(result, 20)
pect_n40 = np.percentile(result, 40)
pect_n60 = np.percentile(result, 60)
pect_n80 = np.percentile(result, 80)

def target_qf(result, pect_n20, pect_n40, pect_n60, pect_n80):
    if result < pect_n20:
        return 1
    elif pect_n20 <= result < pect_n40:
        return 2
    elif pect_n40 <= result < pect_n60:
        return 3
    elif pect_n60 <= result < pect_n80:
        return 4
    elif pect_n80 <= result:
        return 5

y_pred_test['y_pred'] = y_pred_test.apply(lambda x:target_qf(x.y_pred_pro, pect_n20, pect_n40, pect_n60, pect_n80),axis=1)
y_pred_test = y_pred_test.drop(['y_pred_pro'], axis=1)


y_pred_test.to_csv(r'/home/cdsw/result.csv', index=False)
