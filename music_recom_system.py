# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:17:17 2020

@author: Zi MO
"""

#import numpy as np
import pandas as pd
import warnings
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

train = pd.read_csv('C:/Users/Lenovo/Desktop/kkbox/train.csv',dtype={'msno' : 'category',
                                                                     'source_system_tab' : 'category',
                                                                     'source_screen_name' : 'category',
                                                                     'source_type' : 'category',
                                                                     'target' : np.uint8,
                                                                     'song_id' : 'category'})
test = pd.read_csv('C:/Users/Lenovo/Desktop/kkbox/test.csv',dtype={'msno' : 'category',
                                                                   'source_system_tab' : 'category',
                                                                   'source_screen_name' : 'category',
                                                                   'source_type' : 'category',
                                                                   'song_id' : 'category'})

songs = pd.read_csv('C:/Users/Lenovo/Desktop/kkbox/songs.csv',dtype={'genre_ids': 'category',
                                                                     'language' : 'category',
                                                                     'artist_name' : 'category',
                                                                     'composer' : 'category',
                                                                     'lyricist' : 'category',
                                                                     'song_id' : 'category'})

train = pd.merge(train, songs, on='song_id', how='left')
test = pd.merge(test, songs, on='song_id', how='left')

members = pd.read_csv('C:/Users/Lenovo/Desktop/kkbox/members.csv',dtype={'city' : 'category',
                                                                         'bd' : np.uint8,
                                                                         'gender' : 'category',
                                                                         'registered_via' : 'category'})


members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
#members['registration_day'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
#members['expiration_day'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members = members.drop(['registration_init_time','expiration_date'], axis=1)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')


songs_extra = pd.read_csv('C:/Users/Lenovo/Desktop/kkbox/song_extra_info.csv')


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)


train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')

for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


train.info()

# Create a Cross Validation with 4 splits
kf = KFold(n_splits=4)

predictions = np.zeros(shape=[len(test)])

from sklearn.metrics import accuracy_score
# For each KFold
for train_indices ,validate_indices in kf.split(train) : 
    lgb_train = lgb.Dataset(train.drop(['target'],axis=1).loc[train_indices,:],label=train.loc[train_indices,'target'])
    lgb_val = lgb.Dataset(train.drop(['target'],axis=1).loc[validate_indices,:],label=train.loc[validate_indices,'target'])

    params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.2 ,
            'verbose': 0,
            'num_leaves': 2**8,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            'num_rounds': 100,
            'metric' : 'auc'
        }
    
    results = {}
    # Train the model    
    lgbm_model = lgb.train(params, 
                           train_set = lgb_train, 
                           valid_sets = (lgb_val, lgb_train),
                           valid_names=('validate','train'),
                           verbose_eval=5,
                           evals_result= results)
    predictions += lgbm_model.predict(test.drop(['id'],axis=1))
    
    pred_train = lgbm_model.predict(train.drop(['target'],axis=1).loc[train_indices,:], num_iteration= 100)
    pred_val = lgbm_model.predict(train.drop(['target'],axis=1).loc[validate_indices,:], num_iteration = 100)
    print('train accuracy: {:.5} '.format(accuracy_score(train.loc[train_indices,'target'],pred_train > 0.5)))
    print('valid accuracy: {:.5} \n'.format(accuracy_score(train.loc[validate_indices,'target'],pred_val > 0.5)))
    lgb.plot_metric(results)
    lgb.plot_importance(lgbm_model,importance_type = "gain")
    
    del lgbm_model
    # We get the ammount of predictions from the prediction list, by dividing the predictions by the number of Kfolds.



predictions = predictions/4

subm = pd.DataFrame()
ids = test['id'].values
subm['id'] = ids
subm['target'] = predictions
subm.to_csv('C:/Users/Lenovo/Desktop/kkbox/4fold_lgbm_submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')


