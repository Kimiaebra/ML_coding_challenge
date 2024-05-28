#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:53:30 2024

@author: kimiaebrahimi
"""


'''
About the dataset: 
    
    - unique values: 57 organization, 13163 descriptions, 476 part_ids 
    - imbalanced regarding part_id : id_11 751 entry , id_7 2 entry 
    - there are part_ids that are produced in only a subset of organization not all of them. 
        - Edge cases : 
            - There are 445 part_ids that only get produced in one oraganization 
            - There are no part IDs that are produced by all organizations in the dataset.
questions : 
    - labelencore or one-hot encoding 
    
            
scores: 
    f1= 0.65 without oversampling 
    f1= 0.81 with oversampling 
    f1= 0.79  without filtering
    
'''


import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import scipy.sparse as sp
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from scipy.sparse import vstack
import time
import random 
from sklearn.utils.class_weight import compute_class_weight
import joblib


# Load the data
DATA= pd.read_csv('dataset.csv')

def prepare_data(data, le_org=None, le_pid=None, vectorizer=None):
    """
    Prepares the data by encoding and vectorizing, optionally using existing transformers.

    Args:
        data: DataFrame with raw data.
        le_org: Pre-fitted LabelEncoder for organizations, or None to fit a new one.
        le_pid: Pre-fitted LabelEncoder for part IDs, or None to fit a new one.
        vectorizer: Pre-fitted CountVectorizer for descriptions, or None to fit a new one.

    Returns:
        tuple: A tuple containing the resampled feature matrix, target vector, and label encoder for part IDs,
        label encoder for organization ,vectorizer for description
    """

    # Encode the 'organization' and 'part_id' columns 
    if le_org is None:
        le_org = LabelEncoder()
        data['organization_encoded'] = le_org.fit_transform(data['organization'])
    else:
        data['organization_encoded'] = le_org.transform(data['organization'])

    if le_pid is None:
        le_pid = LabelEncoder()
        data['part_id_encoded'] = le_pid.fit_transform(data['part_id'])
    else:
        data['part_id_encoded'] = le_pid.transform(data['part_id'])

    # Vectorize the descriptions
    if vectorizer is None:
        vectorizer = CountVectorizer()
        X_descriptions = vectorizer.fit_transform(data['description'])
    else:
        X_descriptions = vectorizer.transform(data['description'])

    X_full = sp.hstack((X_descriptions, data['organization_encoded'].values[:, None]), format='csr')
    y_full = data['part_id_encoded']

    return X_full, y_full, le_pid, le_org, vectorizer

def save_raw_test_data(data, test_size=0.2, file_name='raw_test_data.csv'):
    """
    Splits the raw data into training and test sets and saves the test set to a CSV file.
    This function does not apply any transformations or encoding to the data.

    Args:
        data (DataFrame): The raw data.
        test_size (float): Proportion of the dataset to include in the test split.
        file_name (str): Filename for the saved test data CSV.

    Returns:
        Saves test data to a CSV file and returns file path.
    """
    # Split data into training and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Save the test data to a CSV file
    test_data.to_csv(file_name, index=False)
    print(f"Test data saved to {file_name}")

    return file_name

def split_data(X,y,test_size=0.2):

    if test_size == 0:
        # If test_size is 0, use all data for training
        X_train, y_train = X, y
        X_test, y_test = None, None  # No test data
    else:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = custom_stratified_split(X, y, test_size)

    # oversample minority class of training data
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
   
    return   X_train_resampled,X_test,y_train_resampled,y_test
    
def train_model(X_train, y_train, le_pid, le_org, vectorizer,org_part_map, n_estimators=100, save_model=False):
    """
    Trains a RandomForest classifier and saves the model with its associated transformers.

    Args:
        X_train (sparse matrix): Feature matrix for training.
        y_train (array): Target vector for training.
        le_pid, le_org, vectorizer: Transformers used for preprocessing.
        n_estimators (int): Number of trees in the forest.
        save_model (bool): Whether to save the model.

    Returns:
        Trained RandomForestClassifier.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    if save_model:
        model_components = {
            'model': clf,
            'label_encoder_part_id': le_pid,
            'label_encoder_org': le_org,
            'vectorizer': vectorizer,
            'org_part_map': org_part_map,
        }
        joblib.dump(model_components, 'model_and_transformers.pkl',compress=3)

    return clf

def load_model_and_transformers(filepath='model_and_transformers.pkl'):
    components = joblib.load(filepath)
    model = components['model']
    le_pid = components['label_encoder_part_id']
    le_org = components['label_encoder_org']
    vectorizer = components['vectorizer']
    org_part_map = components.get('org_part_map', {})  # Default to empty dict if not found
    return model, vectorizer, le_pid, le_org, org_part_map


def custom_stratified_split(X, y, test_size):
    """
    Perform a stratified split of a dataset with special handling for classes with exactly two or three instances,
    making sure each of them gets at least one intance

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)   The input samples
    y : array-like, shape (n_samples,)   The target labels associated with the samples
    test_size : float The proportion of the dataset to include in the test split.

    """
    
    # Find unique classes and their counts
    unique, counts = np.unique(y, return_counts=True)
    
    # Find classes with exactly two and three instances
    class_two_instances = unique[counts == 2]
    class_three_instances = unique[counts == 3]

    # Initializing lists for train and test indices
    train_idx = []
    test_idx = []

    # Distribute two-instance classes into training and testing sets,making sure each of them gets one intance
    for c in class_two_instances:
        indices = np.where(y == c)[0]
        np.random.shuffle(indices)
        train_idx.extend([indices[0]])
        test_idx.extend([indices[1]])

    # Distribute three-instance classes into training and testing sets,making sure each of them gets at least one intance
    for c in class_three_instances:
        indices = np.where(y == c)[0]
        np.random.shuffle(indices)
        train_idx.extend(indices[:2])   # two go to train
        test_idx.extend(indices[2:])    # one goes to test

    # Convert lists to arrays for proper indexing and concatenation
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # Filter out already assigned indices
    remaining_indices = np.setdiff1d(np.arange(len(y)), np.concatenate([train_idx, test_idx]))

    # Split remaining data
    X_remain, y_remain = X[remaining_indices], y[remaining_indices]
    X_train, X_test, y_train_part, y_test_part = train_test_split(
        X_remain, y_remain, test_size=test_size, stratify=y_remain, random_state=42)

    # Combine manually split and stratified split
    X_train = vstack([X_train, X[train_idx]])
    X_test = vstack([X_test, X[test_idx]])
    y_train = pd.concat([y_train_part, y.iloc[train_idx]], ignore_index=True)
    y_test = pd.concat([y_test_part, y.iloc[test_idx]], ignore_index=True)

    return X_train, X_test, y_train, y_test


def predict(model,X_test,y_test,org_part_map,le_pid):
    """
    Predict the constrained part IDs for the test dataset based on the provided model and 
    organizational constraints.
"""
    # Predict probabilities or log probabilities depending on your model setup
    y_prob = model.predict_proba(X_test)
    y_pred_constrained = []
    # max_indices = np.argmax(y_prob, axis=1)
    
    # Ensure we use a consistent set of labels
    all_labels = np.arange(len(le_pid.classes_))

    # Get organization indices 
    org_indices = X_test[:, -1].toarray().flatten().astype(int)
    for idx, probs in enumerate(y_prob):
        org_id = org_indices[idx]
        valid_parts = org_part_map[org_id]  
        # Filter probabilities to only include valid parts, set others to a very low value
        filtered_probs = np.full(probs.shape, -np.inf)  # Use a very low value for impossible classes
        filtered_probs[list(valid_parts)] = probs[list(valid_parts)]
        
        # Select the part ID with the highest probability among valid ones
        constrained_part_id = np.argmax(filtered_probs)
        y_pred_constrained.append(constrained_part_id)
        
    y_pred_constrained = np.array(y_pred_constrained)

    class_report_dict = classification_report(y_test, y_pred_constrained,  labels=all_labels,
                                          target_names=le_pid.inverse_transform(all_labels),output_dict=True)
    
    # class_report_dict = classification_report(y_test, y_pred_constrained,  labels=all_labels,
    #                                       target_names=le_pid.inverse_transform(all_labels))
    # print("Classification Report:\n", class_report_dict)
    # return y_pred_constrained,class_report

    return y_pred_constrained,class_report_dict

def evaluate_small_classes(class_report,class_instances):
    # Convert the dictionary to a DataFrame. Exclude the summary rows if they exist.
    report_df = pd.DataFrame.from_dict({(i): class_report[i]
                                        for i in class_report.keys()
                                        if i not in ['accuracy', 'macro avg', 'weighted avg']}, orient='index')

    # Filter to get classes with exactly 1 instance in the test set
    filtered_report = report_df[report_df['support'] == class_instances]
    mean_f1_score = filtered_report['f1-score'].mean()
    print(f"Mean F1-Score for classes with {class_instances} instances:", mean_f1_score)
    return mean_f1_score

def org2pid(data):
    
    # Create a mapping from organizations to parts they create
    org_part_map = defaultdict(set)
    for _, row in data.iterrows():
        org_part_map[row['organization_encoded']].add(row['part_id_encoded'])
    
    return org_part_map

# Define a simple search function
def search_part_id(organization, description):
    for record in data_list:
        if record['organization'] == organization and record['description'] == description:
            return record['part_id']
    return None

if __name__=='__main__':
    
    # reads the data, performs feature embeddings, extracts feature and target vectors 
    X,y,le_pid,le_org,vectorizer = prepare_data(DATA)
    org_part_map=org2pid(DATA)
    # performs data split
    X_train,X_test,y_train,y_test= split_data(X,y,test_size=0)
    
    # trains the model 
    model= train_model(X_train, y_train,  le_pid, le_org, vectorizer,org_part_map, save_model = True)

    # Start timing for inference
    start_time = time.time()
    num_test_items = X_test.shape[0]
    prediction,report = predict(model,X_test,y_test,org_part_map,le_pid)

    # End timing for inference
    end_time = time.time()

    ml_inference_time = (end_time - start_time)/ num_test_items
    print(f"Machine Learning Inference Time: {ml_inference_time:.4f} seconds")
    
    
    
    
    # Convert data for easier search
    data_list = DATA[['organization', 'description', 'part_id']].to_dict('records')

    # Ensure the CountVectorizer is handling the feature names correctly.
    # Verify vocabulary bounds directly.
    features = vectorizer.get_feature_names_out()
    
    # Generate test descriptions safely
    test_descriptions = []
    for i in range(X_test.shape[0]):
        feature_indices = X_test[i, :-1].nonzero()[1]  # Exclude the last column (organization_encoded)
        words = [features[idx] for idx in feature_indices if idx < len(features)]
        test_descriptions.append(' '.join(words))

    test_data = pd.DataFrame({
        'organization': le_org.inverse_transform(X_test[:, -1].toarray().ravel()),
        'description': test_descriptions
    })

    # Start timing the search algorithm
    start_time = time.time()

    # Applying the search algorithm to the prepared test data
    search_results = [search_part_id(row['organization'], row['description']) for _, row in test_data.iterrows()]

    # Stop timing after the search completes
    end_time = time.time()
    search_time = (end_time - start_time)/ num_test_items
    print(f"Simple Search Time: {search_time:.4f} seconds")
    