##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:53:30 2024

@author: kimiaebrahimi
"""


'''
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
import time
from scipy.sparse import hstack
import joblib


# Load the data
DATA= pd.read_csv('dataset.csv')


def prepare_data(data, le_org=None, le_pid=None, vectorizer=None):
    """
    This function Prepares the input data for training the model:
    STEPS: 
        1) encoding categorical variables "organization" and "part_id" and vectorizing descriptions 
        2) Prepares feature matrix and target vector
            2.1) new column 'label' is added. label 1 is set for positive class and 0 for negative class. 
                positive class are datapoints were organization have only part_ids that they produce (original data)
                negative class are datapoints were organization have  part_ids that they DONT produce (augmented data, 
                                                                                                       essencial for training!!)
    RETURNS: 
        X_full_df (DataFrame): feature matrix
        y_full (DataFrame): target vector
        le_pid: label encoder for 'part_id'
        le_org:label encoder for 'organization'
        vectorizer: Vectorizer for 'description'
    
    
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
    

    # Prepare the full feature matrix by combining description vectors with encoded organization and part_id
    X_full = hstack([
        X_descriptions,
        data['organization_encoded'].values[:, None],
        data['part_id_encoded'].values[:, None]
    ], format='csr')
    
    # Convert the sparse matrix to a DataFrame
    feature_names = vectorizer.get_feature_names_out()
    feature_names = list(feature_names) + ['organization_encoded', 'part_id_encoded']
    X_full_df = pd.DataFrame(X_full.toarray(), columns=feature_names)
    
    # label 1 to denote Positive/Original samples
    X_full_df=label_positive_data(X_full_df) 
    
    # drop traget vector 
    X_full_df=X_full_df.drop(columns=['part_id_encoded'])
    
    # target vector 
    y_full = data['part_id_encoded']
    
    return X_full_df, y_full,le_pid,le_org,vectorizer

def add_negative_samples(feature,target, org_part_map):
    
    """
    Augments the provided datasets by adding negative samples to help the model learn that not all part IDs 
    get produced by each organization. This is done by introducing samples for part IDs that an organization 
    does not produce, while positive samples are given weights. The presence of both positive and negative samples 
    allows for meaningful weighting and more effective model training.

    Parameters:
    feature (DataFrame)
    target (DataFrame)
    org_part_map (dict): A mapping of each organization to the set of part IDs they produce.

    Returns:
    tuple: A tuple containing two DataFrames:
        - augmented_X (DataFrame): The features DataFrame augmented with negative samples.
        - augmented_y (DataFrame): The targets DataFrame augmented with  negative sample labels.
    """
    
    # all unique part_ids
    all_parts = set.union(*org_part_map.values())
    
    new_Xs = []
    new_ys=[]
    
    # iterate over each organization 
    for org, parts in org_part_map.items():
        # non_produced parts by org 
        non_parts = list(all_parts - parts)
        # to sythesis description for negative samples,gather descriptions from actual produced parts of this organization
        org_descriptions = feature[feature['organization_encoded'] == org].filter(regex='^ano')
        
        # for each non_part randomly select a description from 'org_descriptions'
        for non_part in non_parts:
            if len(org_descriptions)>0:  # Ensure there are descriptions to choose from
                
                # Randomly pick a description from the same organization
                chosen_description = org_descriptions.sample(n=1)
                # Create a new row with the chosen description and append it
                new_x = chosen_description.squeeze().to_dict()  # Convert the single-row DataFrame to a dictionary
                new_x.update({
                    'organization_encoded': org,
                    'label': 0  # label 0 to denote negative samples
                })
                new_ys.append(non_part)
                new_Xs.append(new_x)
    
    # concatinate negative samples with existing postive samples from input
    negative_samples_X = pd.DataFrame(new_Xs)
    negative_samples_y = pd.DataFrame(new_ys)
    augmented_X= pd.concat([feature, negative_samples_X],axis=0)
    augmented_y = pd.concat([target, negative_samples_y.squeeze()], axis=0)
    
    return augmented_X,augmented_y

def label_positive_data(data):
    """
    Adds a label column to the existing data, setting it to 1 for all existing entries
    
    """
    data['label'] = 1
    return data

    
def train_model(X_train,y_train, le_pid, le_org, vectorizer,n_estimators=100,save_model=False):
    
    """
    Trains a RandomForest classifier using the provided data 

    Args:
        X (sparse matrix): The feature matrix
        y (array): The target vector
        test_size (float): Fraction of the data to be used as test set
        n_estimators (int): Number of trees in the forest

    Returns:
        tuple: A tuple containing the trained model
        
    """
    
    weight_positive=5
    weight_negative=1
    sample_weights= np.where(X_train.loc[:,'label'] == 1, weight_positive, weight_negative)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train,sample_weight=sample_weights)
    if save_model:
        model_components = {
            'model': clf,
            'label_encoder_part_id': le_pid,
            'label_encoder_org': le_org,
            'vectorizer': vectorizer,
        }
        joblib.dump(model_components, 'model_and_transformers.pkl',compress=3)

    return clf


def split_data(X, y, test_size):
    """
    Perform a custom stratified split of a dataset with special handling for classes with exactly two or three instances

    Parameters
    ----------
    X : The input samples
    y : The target labels associated with the samples.
    test_size : float,  The proportion of the dataset to include in the test split.

    Returns
    -------
    X_train : training samples
    X_test : testing samples
    y_train : training labels
    y_test : he testing labels
    
    """
    
    # Find unique classes and their counts
    unique, counts = np.unique(y, return_counts=True)
    
    # Find classes with exactly two and three instances
    class_two_instances = unique[counts == 2]
    class_three_instances = unique[counts == 3]

    # Initializing lists for train and test indices
    train_idx = []
    test_idx = []

    # Distribute two-instance classes into training and testing sets
    for c in class_two_instances:
        indices = np.where(y == c)[0]
        np.random.shuffle(indices)
        train_idx.extend([indices[0]])
        test_idx.extend([indices[1]])

    # Distribute three-instance classes into training and testing sets
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
    X_remain, y_remain = X.iloc[remaining_indices,:], y[remaining_indices]
    X_train, X_test, y_train_part, y_test_part = train_test_split(
        X_remain, y_remain, test_size=test_size, stratify=y_remain, random_state=42)

    # Combine manually split and stratified split
    X_train = pd.concat([X_train, X.iloc[train_idx]],axis=0)
    X_test = pd.concat([X_test, X.iloc[test_idx]],axis=0)
    y_train = pd.concat([y_train_part, y.iloc[train_idx]],axis=0)
    y_test = pd.concat([y_test_part, y.iloc[test_idx]],axis=0)

    return X_train, X_test, y_train, y_test


def predict(model,X_test,y_test,le_pid):
    """
    Predict the constrained part IDs for the test dataset based on the provided model and 
    organizational constraints.
"""
    # Predict probabilities or log probabilities depending on your model setup
    y_prob = model.predict_proba(X_test)
    max_indices = np.argmax(y_prob, axis=1)

    # class_report = classification_report(y_test, max_indices, 
    #                                     target_names=le_pid.inverse_transform(range(len(le_pid.classes_))))
    
    # print("Classification Report:\n", class_report)
    # class_report = classification_report(y_test, max_indices, 
    #                                       target_names=le_pid.inverse_transform(range(len(le_pid.classes_))),output_dict=True)
    
    return le_pid.inverse_transform(max_indices),None
    
def load_model_and_transformers(filepath='model_and_transformers_binary.pkl'):
    components = joblib.load(filepath)
    model = components['model']
    le_pid = components['label_encoder_part_id']
    le_org = components['label_encoder_org']
    vectorizer = components['vectorizer']
    return model, vectorizer, le_pid, le_org


def evaluate_small_classes(class_report,class_instances):
    """
    Evaluates the performance of small classes within a classification report by calculating the mean F1-score
    for classes that have a specific number of instances in the test set.
    
    Parameters:
    class_report (dict): A dictionary containing classification metrics for each class, generated 
                         by sklearn's classification_report function.
                         
    class_instances (int): number of instances in a monority class  
    
    Returns:
    float: The mean F1-score based on only the classes that have exactly the number of instances specified by 'class_instances'.
    """

    # Convert the dictionary to a DataFrame. Exclude the summary rows if they exist.
    report_df = pd.DataFrame.from_dict({(i): class_report[i]
                                        for i in class_report.keys()
                                        if i not in ['accuracy', 'macro avg', 'weighted avg']}, orient='index')

    # Filter to get classes with exactly 1 instance in the test set
    filtered_report = report_df[report_df['support'] == class_instances]
    mean_f1_score = filtered_report['f1-score'].mean()
    print(f"Mean F1-Score for classes with {class_instances} instances:", mean_f1_score)
    return mean_f1_score


def org2pid(X_data,y_data):
    """
    Generates a mapping of organizations to the parts they create based on provided datasets.
    
    """
    y_data_df=pd.DataFrame(y_data)
    # Create a mapping from organizations to parts they create
    org_part_map = defaultdict(set)
   
    # Check if the length of both DataFrames is the same
    if len(X_data) != len(y_data_df):
        raise ValueError("X_data and y_data must be of the same length.")
    
    # Iterate over rows of both X_data and y_data simultaneously
    for (idx_x, row_x), (idx_y, row_y) in zip(X_data.iterrows(), y_data_df.iterrows()):
        # Update the set for the organization with the part_id from y_data
        org_part_map[row_x['organization_encoded']].add(row_y['part_id_encoded'])
    
    return org_part_map


if __name__=='__main__':
    
    # reads the data, performs feature embeddings, extracts feature and target vectors 
    X,y,le_pid,le_org,vectorizer = prepare_data(DATA)

    # performs data split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2) 
    
    # maps orgranization to part_ids which get produced in this organization 
    mapping=org2pid(X_train,y_train)
    
    # add negative class to trainig data
    X_train,y_train= add_negative_samples(X_train,y_train,mapping)
    
    # trains the model 
    model= train_model(X_train, y_train, le_pid,le_org,vectorizer, save_model=False)

    # Start timing for inference
    start_time = time.time()
    num_test_items = X_test.shape[0]
    prediction,report = predict(model,X_test,y_test,le_pid)
    f1= evaluate_small_classes(report,2)
    # End timing for inference
    end_time = time.time()
    ml_inference_time = (end_time - start_time)/ num_test_items
    print(f"Machine Learning Inference Time: {ml_inference_time:.4f} seconds")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    