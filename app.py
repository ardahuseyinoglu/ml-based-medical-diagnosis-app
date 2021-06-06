import streamlit as st
import pandas as pd
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import SessionState
session = SessionState.get(run_id=0)

#-----------------------------------------------------------------------------------


#st.set_page_config(page_title='ML based MDS')

st.title('Machine Learning Based Medical Diagnosis System')
st.write("\n")
st.write("\n")
st.write("You can enter the data of the patient to get the prediction for diagnosis and probability scores for each disease.")
st.write("""
### How to use
- We built the best model for you. So, the default features are recommended.
- However, you can select or remove the features you want from the left sidebar. Whenever you add a new feature, it will be displayed at the end on the left sidebar to enter the data.
- Please press the **Predict** button, after you enter data for all displayed features on the left sidebar.
- You will see the diseases that the patient may have with probability values and the most possible disease as prediction.
""")



#-------------------------------------------------------------------------------------

X_train = pd.read_csv('features.csv')
y_train = pd.read_csv('target.csv')

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)


def scale_value(feature_name, value):
    value = np.array([value]).reshape(1, -1)
    sc = MinMaxScaler()
    sc.fit(X_train[[feature_name]])
    scaled_value = float(sc.transform(value))
    return scaled_value

#-------------------------------------------------------------------------------------


feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18', 'Feature_19', 'Feature_20', 'Feature_21', 'Feature_22', 'Feature_23', 'Feature_24', 'Feature_25', 'Feature_26', 'Feature_27', 'Feature_28', 'Feature_29', 'Feature_30', 'Feature_31', 'Feature_32', 'Feature_33', 'Feature_34', 'Feature_35', 'Feature_36', 'Feature_37', 'Feature_38', 'Feature_39', 'Feature_40', 'Feature_41', 'Feature_42', 'Feature_43', 'Feature_44', 'Feature_45', 'Feature_46', 'Feature_47', 'Feature_48', 'Feature_49', 'Feature_50']


with open('best_model_info.json') as json_file:
    default_features = json.load(json_file)

if st.sidebar.button("Clear data and reset to the default features"):
    session.run_id += 1

st.sidebar.write("\n")    
st.sidebar.write("**Select features to be used in prediction**")
selected_features = st.sidebar.multiselect('Select one or more features', feature_names, default=default_features['best_model_feature_names'], key=session.run_id)

st.sidebar.write("\n")
st.sidebar.write("**Provide data for all displayed features**")
selected_features_dict = {}
for feature in selected_features:
    if feature == 'Feature_1':
        selected_features_dict[feature] = st.sidebar.radio(feature, ['Male', 'Female'], key=session.run_id)
    elif feature in ['Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']:
        selected_features_dict[feature] = st.sidebar.number_input(feature, key=session.run_id)
    elif feature == 'Feature_28':
        selected_features_dict[feature] = st.sidebar.selectbox(feature, ['Every Day', '3-4 Days a Week', '1-2 Days a Week', '1-2 Days a Month'], key=session.run_id)
    elif feature == 'Feature_29':
        selected_features_dict[feature] = st.sidebar.selectbox(feature, ['Mornings', 'Evenings', 'No Difference'], key=session.run_id)
    elif feature == 'Feature_44':
        selected_features_dict[feature] = st.sidebar.number_input(feature, key=session.run_id)
    else:
        selected_features_dict[feature] = st.sidebar.radio(feature, ['Yes', 'No'], key=session.run_id)


st.sidebar.write("\n")       
predict_button = st.sidebar.button('Predict')
           
#-------------------------------------------------------------------------------------

if predict_button:
    selected_features_dict_copy = selected_features_dict.copy()

    for feature_name, value in selected_features_dict_copy.items():
        if feature_name == 'Feature_1':
            if value == 'Male':
                selected_features_dict_copy[feature_name] = 1.0
            else:
                selected_features_dict_copy[feature_name] = 0.0
        elif feature_name in ['Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']:
            selected_features_dict_copy[feature_name] = scale_value(feature_name, np.array(value))
        elif feature_name == 'Feature_28':
            if value == '1-2 Days a Month':
                selected_features_dict_copy[feature_name] = 1.0
            else:
                selected_features_dict_copy[feature_name] = 0.0
        elif feature_name == 'Feature_29':
            if value == 'Mornings':
                selected_features_dict_copy[feature_name] = 1.0
            else:
                selected_features_dict_copy[feature_name] = 0.0
        elif feature_name == 'Feature_44':
            selected_features_dict_copy[feature_name] = scale_value(feature_name, value)
        else:
            if value == 'Yes':
                selected_features_dict_copy[feature_name] = 1.0
            else:
                selected_features_dict_copy[feature_name] = 0.0

    if 'Feature_28' in selected_features_dict_copy.keys():
        selected_features_dict_copy['Feature_28_1-2 Days a Month'] = selected_features_dict_copy['Feature_28']
        selected_features_dict_copy.pop('Feature_28')

    if 'Feature_29' in selected_features_dict_copy.keys():
        selected_features_dict_copy['Feature_29_Mornings'] = selected_features_dict_copy['Feature_29']
        selected_features_dict_copy.pop('Feature_29')
    
    
    best_features = list(selected_features_dict_copy.keys())
    X_train = X_train[best_features]    
    test_sample = pd.DataFrame(selected_features_dict_copy, index=['0',])    

    rfc = RandomForestClassifier(criterion= 'entropy', max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 15)
    rfc.fit(X_train, y_train)
    pred = rfc.predict(test_sample)
    probs = rfc.predict_proba(test_sample)
    

    st.write("""### Prediction""")
    st.markdown(f"The model predicts the disease as **{le.classes_[pred][0]}** in according to the provided data for selected features.")
        
    
    prob_df = pd.DataFrame({'Disease': le.classes_, 'Probability value': probs[0]*100}).sort_values('Probability value', ascending=False).reset_index(drop=True)
    prob_df.index += 1
    st.table(prob_df)

    