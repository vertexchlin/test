#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#1 load model and load data
#numpy大量矩陣維度與矩陣運算, log
import numpy as np
#Paython上Excel所有操作, 欄位的加總、分群、樞紐分析表、小計、畫折線圖、圓餅圖
import pandas as pd
#畫圖範圍框架
import matplotlib.pyplot as plt
#seaborn直方圖, heatmap
import seaborn as sns




#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#2 data processing
din.info()      type(int64, float64, object, ...)
din.describe()  max, min, mean, mode, 75%, 25%, 
din.head(10)    
din.tail()
din.shape       (dimension)
din.columns     feature name


#2-1將training set與testing set concat起來處理
#保證這兩個sets在同feature做同樣的處理

#2-2 training set Nan補值/testing set Nan補值
din.isnull().sum()  查看null個數
din.drop('Cabin', axis=1, inplace=True)
din.drop(['Cabin'], axis = 1, inplace = True)
din.Embarked.fillna(din.Embarked.max(), inplace = True)
din.Age.fillna(din.Age.mean, inplace = True)

#2-3 Object to Int
#非numeric要全部轉成numeric才可以做training
#converting categorical feature to numeric
din.Sex = din.Sex.map({0:'female', 1:'male'}).astype(object)
din['Sex'] = din.Sex.map({'female':0, 'male':1}).astype(int)
#2-4多連續值切分範圍
train_df['AgeBand'] = pd.cut(train_df['Age'], 5) #cut into same width
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
#2-5查看X(AgeBand)與y(Survived)的關係
train_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean().
  sort_values(by = 'AgeBand', ascending = True) #groupby
#2-6把連續X(Fare)分小類成X(Fare), 並把datatype轉成int
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
#2-7normalization
#2-8heatmap
corrmat = train_df.corr()
sns.heatmap(corrmat, vmin=-0.3, vmax=0.8, square=True)

for column in corrmat[corrmat.SalePrice > 0.6].index:
    plt.subplot(2,2,2)
    plt.scatter(train_df[column], train_df['SalePrice'])
    plt.show()
#2-9把前10名的feature自動列出來
#2-10先使用 Neighborhood 做區域分類，再用區域內 LotFrontage 的中位數進行補值
X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#2-11取代值
X[X['GarageYrBlt'] == 2207].index
X.loc[X[X['GarageYrBlt'] == 2207].index, 'GarageYrBlt'] = 2007
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-    
#3 split training set & validation set
#3-1機器學習模塊
from sklearn import model_selection
#3-2把training set分成training set & validation set
x_ = din.drop(['Cabin','Survived'], axis=1)
y_ = din.Survived
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x_, y_, random_state=0, test_size=0.33)
#3-3k-fold

#3-4label encoding
final_X = pd.DataFrame()
for columns in object_columns:
    final_X = pd.concat([final_X, rank_label_encoding(X,columns)], axis=1)
for columns in not_object_columns:
    final_X = pd.concat([final_X, X[columns]], axis=1)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#4 Model
#4-1 predict, accuracy
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2)
clf.fit(x_train,y_train)
y_preds = clf.predict(x_valid)
from sklearn.metrics import accuracy_score
accuracy_score(y_valid, y_preds) 

#4-3 XGBRegressor fit
from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate = 0.01, n_estimators = 3300,
                        objective = "reg:linear",
                                     max_depth= 3, min_child_weight=2,
                                     gamma = 0, subsample=0.6,
                                     colsample_bytree=0.7,
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0, reg_lambda= 1)
xgb_model.fit(x_train, y_train)

#4-4 LGBMRegressor
from lightgbm import LGBMRegressor
lgbm_model = LGBMRegressor(learning_rate = 0.01, n_estimators = 2900,
                        objective='regression',
                                     max_depth= 3,min_child_weight=0,
                                     gamma = 0, 
                                     subsample=0.6, colsample_bytree=0.6, 
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0.1, reg_lambda= 0)
lgbm_model.fit(x_train, y_train)

#4-5SVR
from sklearn.svm import SVR
SVR_model  = SVR(C = 10, epsilon = 0.1, gamma = 1e-06)
SVR_model.fit(x_train, y_train)

#4-6ElasticNetCV
from sklearn.linear_model import ElasticNetCV
alphas = [0.0001, 0.0002, 0.0003]
l1ratio = [0.5, 0.6, 0.7, 0.8, 0.7]
elastic_model = ElasticNetCV(max_iter=1e7, alphas = alphas, cv = kfolds, l1_ratio = l1ratio)
elastic_model.fit(x_train, y_train)

print(elastic_model.alpha_) #印出最佳解之alpha
print(elastic_model.l1_ratio_)#印出最佳解之l1_ratio


#4-7cross validation 
from sklearn.model_selection import  KFold
kfolds = KFold(n_splits=6)
from sklearn.model_selection import cross_val_score
def cv_rmse(model, X, y):
    return np.sqrt(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv = kfolds))
cv_error = {"xgb": cv_rmse(xgb_model, x_train, y_train),
            "lgbm":cv_rmse(lgbm_model, x_train, y_train),
            "SVR": cv_rmse(SVR_model, x_train, y_train),
            "elastic":cv_rmse(elastic_model, x_train, y_train)}

cv_error

#4-8參數調整
from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
def XGBRegressor_cv(x,y):
    cv_params = {'learning_rate': [0.005,0.01, 0.05, 0.07]}

    other_params = dict(learning_rate = 0.01, n_estimators = 3300,
                        objective = "reg:linear",
                                     max_depth= 3, min_child_weight=2,
                                     gamma = 0, subsample=0.6,
                                     colsample_bytree=0.7,
                                     scale_pos_weight=1,seed=0, 
                                     reg_alpha= 0, reg_lambda= 1)

    model = XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="neg_mean_squared_log_error", cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(x, y)
    evalute_result = optimized_GBM.grid_scores_
    print('每輪迭代運行結果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return model
XGBRegressor_cv(train_X,train_y)

#4-9 validation error
valid_data = {"xgb":xgb_model.predict(x_valid),
            "lgbm":lgbm_model.predict(x_valid),
            "elastic": elastic_model.predict(x_valid),
            "SVR":SVR_model.predict(x_valid)}

valid_error = dict()
for model,v in valid_data.items():
    valid_error[model] = np.power((v - y_valid),2).mean()
print(valid_error)

for train_df in combine: #為什麼一定要加這一行
    train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)


