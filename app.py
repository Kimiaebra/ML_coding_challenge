import streamlit as st
import pandas as pd
import time
from predictor import prepare_data, train_model, split_data, predict, org2pid, load_model_and_transformers, evaluate_small_classes
from negative_predictor import prepare_data as neg_prepare_data
from negative_predictor import train_model as neg_train_model
from negative_predictor import split_data as neg_split_data
from negative_predictor import add_negative_samples
from negative_predictor import org2pid as neg_org2pid
from negative_predictor import predict as neg_predict
from negative_predictor import load_model_and_transformers as neg_load_model_and_transformers

import numpy as np
import sklearn 



def run_prediction_with_filtering_feature(file, use_pretrained):
    data = pd.read_csv(file)
    target= np.array(data['part_id'])
    if use_pretrained:
        # Load pre-trained model and transformers
        model, vectorizer, le_pid, le_org, org_part_map = load_model_and_transformers('model_and_transformers.pkl')
        # Prepare data using loaded transformers
        X_test, y_test, _, _, _ = prepare_data(data, le_org, le_pid, vectorizer)
    else:
        # Prepare data and train new model
        X, y, le_pid, le_org, vectorizer = prepare_data(data)
        org_part_map = org2pid(data)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        model = train_model(X_train, y_train, le_pid, le_org, vectorizer,org_part_map, save_model=False)
    
    num_test_items = X_test.shape[0]
    start_time = time.time()
    predictions, report = predict(model, X_test, y_test, org_part_map, le_pid)
    acc=sklearn.metrics.accuracy_score(target,predictions ,normalize=True, sample_weight=None)
    total_time = (time.time() - start_time)/ num_test_items
    

    return predictions, total_time,report,acc


def run_prediction_with_negative_samples(file, use_pretrained):
    data = pd.read_csv(file)
    target= np.array(data['part_id'])
    if use_pretrained:
        # Load pre-trained model and transformers
        model, vectorizer, le_pid, le_org = neg_load_model_and_transformers('model_and_transformers_negative_sample.pkl')
        # Prepare data using loaded transformers
        X_test, y_test, _, _, _ = neg_prepare_data(data, le_org, le_pid, vectorizer)
    else:
        # Prepare data and train new model
        X, y, le_pid, le_org, vectorizer = neg_prepare_data(data)
        X_train, X_test, y_train, y_test = neg_split_data(X, y, test_size=0.2)
        org_part_map = neg_org2pid(X_train,y_train)
        X_train,y_train = add_negative_samples( X_train,y_train,org_part_map)
        model = neg_train_model(X_train, y_train, le_pid, le_org, vectorizer, save_model=False)
    
    num_test_items = X_test.shape[0]
    start_time = time.time()
    predictions, report = neg_predict(model, X_test, y_test, le_pid)
    acc=sklearn.metrics.accuracy_score(target,predictions ,normalize=True, sample_weight=None)
    total_time = (time.time() - start_time)/ num_test_items
    

    return predictions, total_time,report,acc

st.title("Part ID Prediction Tool")

st.markdown("""
### Choose Training Method:
- There are 2 Models to embed this information in the training process,Since not all organizations create/can create all 476 of these parts_ids. 
- **NegativeSampling Mode**: Augments the provided datasets by adding negative samples to help the model learn that not all part IDs 
    get produced by each organization. This is done by introducing samples for part IDs that an organization 
    does not produce, while positive samples are given weights.
- **FeatureFiltering Mode**: Prediction across all categories and post-process the predictions to adjust based on the organization-part_id mapping. 
""")
st.markdown("""
### Choose Prediction Mode:
- **Use Pretrained Model**: This mode utilizes a model that was previously trained on the entire provided original dataset. This is for your unseen dataset of 5000 entries mentioned in the task description. after uploading your dataset click the "Predict" button.
- **Train Model On the Fly**: This mode dynamically trains a new model using 80% of the provided original dataset that has to be uploaded and then uses the remaining 20% as test dataset to be predicted upon clicking on "Predict" button. This version does not take a new test dataset to be predicted.
- **Note**: a random subset of the provided dataset is located in github https://github.com/Kimiaebra/ML_coding_challenge/blob/56e2b49dfc91a1534d34258897270b3693904146/saved_raw_test_data.csv merely to test the functionality of the pretrained version, ofcourse it's biased and trained model has seen the whole data
""")

method = st.radio(
    "Choose Training Method:",
    ["Feature Filtering", "Negative Samples"],
)

mode = st.radio("Choose Prediction Mode:", ["Use Pretrained Model", "Train Model On the Fly"])

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    if st.button('Predict'):
        if method == "Feature Filtering":
            predictions,total_time, report,acc = run_prediction_with_filtering_feature(uploaded_file, mode == "Use Pretrained Model")
        else:
            predictions,total_time, report,acc = run_prediction_with_negative_samples(uploaded_file, mode == "Use Pretrained Model")

        st.write(f"Total time taken to predict for each row in average: {total_time:.4f} seconds")  
        st.write(f"Accuracy of prediction: {acc:.4f}")
        results_df = pd.DataFrame(predictions, columns=['Predictions'])
        st.write(results_df)
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Prediction Results",
                        data=csv,
                        file_name='results.csv',
                        mime='text/csv')
        
            
