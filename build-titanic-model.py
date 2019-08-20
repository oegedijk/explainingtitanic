
# coding: utf-8

# # Imports

# In[17]:

get_ipython().system('pip install -U --ignore-installed -r requirements.txt')


# In[1]:

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '0')


# In[2]:

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[4]:

import numpy as np
import pandas as pd

import gc
import pickle

from pathlib import Path


# ### load some project specific modules:

# In[5]:

from sklearn.metrics import roc_auc_score


# In[6]:

from data_transform_methods import *
from explainer_methods import *
from optimizer_methods import *
from explainer import *


# # load data

# In[7]:

d = pd.read_csv("train.csv")


# ### feature engineering:

# In[8]:

d['Familysize'] = d.SibSp + d.Parch
d['Cabin'] = d.Cabin.str[0]


# In[9]:

d.shape
d.head().T


# In[10]:

show_cardinality_of_cats(d)


# # Transform data:

# In[11]:

TARGET='Survived'


# ### drop name, ticket and id columns:

# In[12]:

drop_columns=['Name', 'Ticket', 'PassengerId']

substring_drop_list = []

d = clean_data(d, drop_columns, substring_drop_list, drop_dates=True)


# ## Generate training and test set:
# - In this case we generate test set of 200 so that we'll have enoughd data points to calculate shap values, etc

# In[13]:

test_idxs = d.sample(200).index
d_train = d[~d.index.isin(test_idxs)]
d_test = d[d.index.isin(test_idxs)]
d_train.shape, d_test.shape


# ## Generate tree and linear datasets:

# #### Cols to generate isolation forest outlier score for:

# In[14]:

tree_transformer =  fit_transformer(d_train, target=TARGET, numfill='ExtremeValue')


# In[15]:

tree_data = (*get_transformed_X_y(d_train, tree_transformer, TARGET, add_random=False),
             *get_transformed_X_y(d_test, tree_transformer, TARGET, add_random=False))


# In[16]:

(X_train, y_train, X_test, y_test) = tree_data


# In[17]:

X_train.head()


# In[18]:

len(X_train), y_train.sum(), y_train.mean()
len(X_test), y_test.sum(), y_test.mean()


# ## Run the garbage collector:

# In[19]:

gc.collect()


# # Optimize hyperparameters

# ### optimize model:

# In[20]:

#models = ['RandomForestClassifier', 'BalancedRandomForestClassifier', 
#            'XGBClassifier', 'LogisticRegression']
models = ['RandomForestClassifier']
best_model, trials = classifier_optimize(tree_data, None, models, 
                                         roc_auc_score, needs_proba=True, n_evals=200, cv=5)


# In[21]:

best_model


# In[22]:

model, (X_train, y_train, X_test, y_test), transformer = get_best_model_and_data(best_model, tree_data, 
                                            None, None, tree_transformer, None)
model.fit(X_train, y_train)


# In[23]:

pred_probas = model.predict_proba(X_test)
print('roc auc score:', roc_auc_score(y_test, pred_probas[:,1]))
print(classification_report(y_test, np.where(pred_probas[:,1]>0.5, 1, 0)))


# In[24]:

model_bunch = ModelBunch(model, transformer, TARGET, use_columns=X_train.columns)
pickle.dump(model_bunch, open(Path.cwd() / 'titanic_model_bunch.pkl','wb'))


# In[25]:

model_bunch.predict_proba(d_test)


# In[26]:

explainer = TreeClassifierExplainer(model_bunch, d_test, 
                                    metric=roc_auc_score, labels=['Not Survived', 'Survived'])


# In[27]:

explainer.calculate_properties()


# In[28]:

explainer.contrib_df(index=0)


# In[29]:

explainer.plot_confusion_matrix()


# In[30]:

explainer.plot_precision()


# In[31]:

pickle.dump(explainer, open(Path.cwd() / 'titanic_explainer.pkl', 'wb'))


# In[33]:

get_ipython().system(' python app.py')


# In[39]:

get_ipython().system('git add .')
get_ipython().system('git commit -m "changed title"')
get_ipython().system('git push -u origin master')


# In[ ]:



