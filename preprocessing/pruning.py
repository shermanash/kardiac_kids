'''
Regularization by Dataset; i.e., Pruning
----------------------------------------------
A way to preprocess UCI Heart Disease datasets


Goal: 
Pare down features for each dataset to arrive at only those above a given level of informativeness.

Method:
Regularize via logistic regression to identify key features in each dataset.

Strategy:
1. get_cv_scores for logistic regression and tweak C to optimize score
2. Get .coefs_ for optimized model (first have to train_test_split)
3. Look at coefs and see which have abs vals over a given threshold; these are relative strongest features. Associate these with indices to be able to tie back to dataset DataFrames.
4. Finally, pare down cleaned DataFrames for each dataset to only top features.


Run `from pruning import prune` and you'll be ready to go (just feed in params, as described in the `prune` function).


Main Function:
prune

Helper Functions:
get_datasets
clean_datasets
optimize_score (the regularization function)
get_coefs
get_top_features

Thoughts:
1. Play around with c_range and threshold values! The ones I used are more to test this process quickly rather than to get iron-clad "best" features.
2. Beware, large c_range ranges will take longer to process. However, one reason I went with logistic regression is that it's quicker than SVM. tqdm may also help speed things?
3. What would feature reduction look like using a different approach (random forest, SVM, something else)?


Ben Ellentuck
2/11/16
'''

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
import tqdm
from operator import itemgetter


def prune(cnx, c_range, threshold):
    '''
    Pares down Cleveland Clinic, Hungarian Institute of Cardiology, Swiss University Hospitals,
    and Long Beach V.A. Medical Center datasets to most relevant features (according to a 
    given threshold) for binary classification of heart disease. 
    
    Heart disease binary classification refers to the "num" feature in the original datasets,
    where "num" is defined as "diagnosis of heart disease (angiographic disease status)," a
    value of 0 indicates "< 50% diameter narrowing," and a value of 1 indicates "> 50% 
    diameter narrowing." I.e., 0 = no heart disease and 1 = heart disease.
    
    Using logistic regression to identify key features in each dataset, `prune` returns a
    dictionary containing pared-down datasets and another dictionary mapping the relevant
    features to their coefficient values for each dataset.
    
    Note the variance as well as overlap between the different sets of relevant features. 
    `prune` does not control for consistency among features across datasets, nor does it 
    control for a certain number of features in a given pared-down set. Instead, the 
    `threshold` variable demarcates between informative and uninformative feature coefficients.
    
    
    Parameters
    ----------
    cnx : sqlalchemy.engine.base.Engine 
        Allows access to datasets via postgres/Postico. Using the `sqlalchemy.create_engine`
        module, feed in psql username, password, ip, and network arguments.
        E.g.: `create_engine('postgresql://shermanash:[password]@54.236.113.118:5432/jaysips')`
        
    c_range : numpy.ndarray or list
        Range of values for logistic regression C input parameter to test, in order to 
        optimize/"tune" the model. Optimized model = highest cross_val_score, given all 
        features. `tqdm` lets you know how far along in a particular round of testing you are
        (there are 4 rounds, one for each dataset).
        Recommended syntax: np.arange(start, stop, step), where start, stop, and step are
        all floats.
        
    threshold : float
        Dividing line between "informative" and "uninformative" feature coefficients. Features
        whose coefficients are informative are to be kept in pruned dataset, while those whose
        coefficients are uninformative are to be discarded.
        
    
    Returns
    -------
    pared_down_datasets, dict of hospital names (keys) and DataFrames pared-down to most 
                         relevant features (values).
    
    pared_down_features, dict of hospital names (keys) and dicts (values) of relevant feature
                         names (keys; unicode, but can still access via pd.columns method) and
                         corresponding linear regression thetas-i.e.-coefficients (values).    
    '''
    cleaned = clean_datasets(get_datasets(cnx))
    top_features = get_top_features(get_coefs(
            cleaned, optimize_score(cleaned, c_range)), threshold)
    
    pared_down_datasets = {}
    pared_down_features = {}

    for name, features in top_features.iteritems():
        pared_down_datasets[name] = pd.concat([cleaned[name].iloc[:, feature] 
                                               for feature, _ in features.iteritems()], 
                                              axis=1)
        pared_down_features[name] = {cleaned[name].columns[feature]: importance
                                     for feature, importance in features.iteritems()}
    
    return pared_down_datasets, pared_down_features

def get_datasets(cnx):
    '''Retrieves datasets from psql.
    '''
    cleveland = pd.read_sql_query('''SELECT * FROM cleveland''', cnx)
    hungary = pd.read_sql_query('''SELECT * FROM hungary''', cnx)
    longbeach = pd.read_sql_query('''SELECT * FROM longbeach''', cnx)
    swiss = pd.read_sql_query('''SELECT * FROM swiss''', cnx)
    datasets = {'Cleveland Clinic': cleveland, 'Hungarian Institute of Cardiology': hungary, 
            'Swiss University Hospitals': swiss, 'Long Beach V.A. Medical Center': longbeach}
    return datasets

def clean_datasets(datasets):
    '''Narrows datasets down to 46 non-extraneous features.
    '''
    cleaned = {}
    for name, hosp in datasets.iteritems():
        hosp = hosp.iloc[:, :58]
        hosp['num'] = hosp['num'].replace(2, 1).replace(3, 1).replace(4, 1)
        hosp = hosp.replace(-9, np.nan)
        hosp = hosp.drop(hosp.iloc[:, 44:50], axis=1).drop(hosp.iloc[:, 51:54], axis=1)
        hosp = hosp.drop('pncaden', axis=1).drop('dm', axis=1)
        hosp = hosp.replace(np.nan, 0)
        cleaned[name] = hosp
    return cleaned

def optimize_score(cleaned, c_range):
    '''Regularizes logistic regression models to come up with optimal C values, given all 
    features, for each dataset.
    '''
    scores = {}
    for name, hosp in cleaned.iteritems():
        score = (0, 0)
        for c in tqdm.tqdm(c_range):
            model = LogisticRegression(C=c)
            new_score = np.mean(cross_val_score(model, hosp.iloc[:, :-1], hosp['num']))
            if new_score > score[0]:
                score = (new_score, c)
        scores[name] = score
    print 'Regularization completed.'
    return scores

def get_coefs(cleaned, scores):
    '''Gets coefficients for every feature from each dataset for regularized logistic 
    regression models.
    '''
    coefs = {}
    for name, hosp in cleaned.iteritems():
        X_train, _, y_train, _ = train_test_split(hosp.iloc[:, :-1], hosp['num'], 
                                                        test_size = 0.25)
        model = LogisticRegression(C=scores[name][1])
        model.fit(X_train, y_train)
        coefs[name] = model.coef_
    return coefs

def get_top_features(coefs, threshold):
    '''Distinguishes informative from uninformative features for a given model according to a
    set threshold.
    '''
    top_features = {}
    for name, features in coefs.iteritems():
        top_features[name] = {i: features[0][i] 
                              for i in range(0, len(features[0]-1))
                              if np.absolute(features[0][i]) >= threshold}
    return top_features