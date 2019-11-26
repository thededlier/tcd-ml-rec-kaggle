import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

#Input of training and test sheet
X1=pd.read_csv(r'tcdml1920-rec-click-pred--training.csv')
X2=pd.read_csv(r'tcdml1920-rec-click-pred--test.csv')
X2.isnull().sum()
X2=X2.drop(X2.index[9145:10791])
X2=X2.drop(X2.index[9145:10791])
#Combining both the datasets
X3=X1.append(X2)
X3.isnull().sum()

#Dropping fields
X3=X3.drop(["clicks","ctr","rec_processing_time","app_version","response_delivered",
            "abstract_detected_language","query_document_id", "abstract_char_count","abstract_word_count",
            "timezone_by_ip","local_hour_of_request","local_time_of_request","number_of_authors",
            "num_pubs_by_first_author","first_author_id","user_id","session_id","user_os","user_os_version",
            "user_java_version","user_timezone","document_language_provided", "year_published","time_recs_recieved",
            "time_recs_displayed","time_recs_viewed","number_of_recs_in_set", "query_identifier"],axis=1)

#Filling NA values
X3['app_lang']=X3['app_lang'].fillna("\\N")
X3['country_by_ip']=X3['country_by_ip'].fillna("\\N")
nan_fields = ['query_detected_language', 'country_by_ip', 'cbf_parser', 'item_type', 'app_lang']
X3[nan_fields]=X3[nan_fields].replace('\\N','NAN')
X3['query_word_count']=X3['query_word_count'].replace('\\N',0)
X3['query_char_count']=X3['query_char_count'].replace('\\N',0)
X3['recommendation_algorithm_id_used']=X3['recommendation_algorithm_id_used'].replace('\\N',0)

#Encoding of column values
labelencoder_x=LabelEncoder()
X3['query_detected_language']=labelencoder_x.fit_transform(X3['query_detected_language'])
X3['item_type']=labelencoder_x.fit_transform(X3['item_type'])
X3['app_lang']=labelencoder_x.fit_transform(X3['app_lang'])
X3['country_by_ip']=labelencoder_x.fit_transform(X3['country_by_ip'])
X3['algorithm_class']=labelencoder_x.fit_transform(X3['algorithm_class'])
X3['cbf_parser']=labelencoder_x.fit_transform(X3['cbf_parser'])
X3['search_title']=labelencoder_x.fit_transform(X3['search_title'])
X3['search_keywords']=labelencoder_x.fit_transform(X3['search_keywords'])
X3['search_abstract']=labelencoder_x.fit_transform(X3['search_abstract'])
X3['query_detected_language']=labelencoder_x.fit_transform(X3['query_detected_language'])
X3['application_type']=X3['application_type'].replace('0','digital_library')
X3['application_type'] = X3['application_type'].map({'blog': 0, 'digital_library': 1, 'e-commerce': 2}).astype(int)
X3.isin(['\N']).sum(axis=0)
X3.dtypes
X3['query_word_count']=X3['query_word_count'].astype(int)
from datetime import datetime
X3['request_received']=pd.to_datetime(X3['request_received'])
X3['request_received']=X3['request_received'].dt.dayofyear 
#X4=X3.corr(method ='pearson')
df=pd.DataFrame(X3)
df.isin(['\N']).sum(axis=0)
df.dtypes

#Splitting again both the datasets
X4=df.iloc[0:385686,:]
X4.isnull().sum()
num_nans = X4.size - X4.count().sum()
X4.dtypes
X5=df.iloc[385687:394832,:]

#Splitting of the two datasets, training and test dataset into three classes according to organization_id
df1=X4[X4.organization_id==1]
df2=X4[X4.organization_id==4]
df3=X4[X4.organization_id==8]
df4=X5[X5.organization_id==1]
df5=X5[X5.organization_id==4]
df6=X5[X5.organization_id==8]

#Training of data with organization id=1
X=df1.loc[:, df1.columns!='set_clicked']
y=df1['set_clicked']
from sklearn.model_selection import KFold#, train_test_split
kfold = KFold(n_splits=7, shuffle=True, random_state=50)
X=np.array(X.copy())
y=np.array(y.copy())
for train_index, test_index in kfold.split(X):  
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
#from sklearn.ensemble import RandomForestClassifier
#lm=RandomForestClassifier(n_estimators = 100, random_state = 400, max_features=16)
from sklearn.neighbors import KNeighborsClassifier
lm = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
lm.fit(X_train, y_train)
predictions1 = lm.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions1))
z_test=df4.loc[:, df4.columns!='set_clicked']
df4['set_clicked'] = lm.predict(z_test)

#Training of data with organization id=4
X=df2.loc[:, df2.columns!='set_clicked']
y=df2['set_clicked']
from sklearn.model_selection import KFold#, train_test_split
kfold = KFold(n_splits=7, shuffle=True, random_state=50)
X=np.array(X.copy())
y=np.array(y.copy())
for train_index, test_index in kfold.split(X):  
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
#from sklearn.ensemble import RandomForestClassifier
#lm=RandomForestClassifier(n_estimators = 100, random_state = 400, max_features=16)
from sklearn.neighbors import KNeighborsClassifier
lm = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
lm.fit(X_train, y_train)
predictions2 = lm.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions2))
z_test=df5.loc[:, df5.columns!='set_clicked']
df5['set_clicked'] = lm.predict(z_test)

#Training of data with organization id=8
X=df3.loc[:, df3.columns!='set_clicked']
y=df3['set_clicked']
from sklearn.model_selection import KFold#, train_test_split
kfold = KFold(n_splits=7, shuffle=True, random_state=50)
X=np.array(X.copy())
y=np.array(y.copy())
for train_index, test_index in kfold.split(X):  
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
#from sklearn.ensemble import RandomForestClassifier
#lm=RandomForestClassifier(n_estimators = 100, random_state = 400, max_features=16)
from sklearn.neighbors import KNeighborsClassifier
lm = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform')
lm.fit(X_train, y_train)
predictions3 = lm.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions3))
z_test=df6.loc[:, df6.columns!='set_clicked']
df6['set_clicked'] = lm.predict(z_test)

#Predicting the final output
final=[df4,df5,df6]
result = pd.concat(final)
result.to_csv(r'tcdml1920-rec-click-pred--submission file.csv')

