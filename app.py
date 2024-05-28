import streamlit as st
import pandas as pd
import time
from predictor import prepare_data, train_model, split_data, predict, org2pid, load_model_and_transformers, evaluate_small_classes

def run_prediction(file, use_pretrained):
    data = pd.read_csv(file)
    
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
    total_time = (time.time() - start_time)/ num_test_items
    predictions = ['id_' + str(pred) for pred in predictions] if predictions is not None else []

    return predictions, total_time,report

st.title("Part ID Prediction Tool")

# Option to select prediction mode
mode = st.radio("Choose Prediction Mode:", ["Use Pretrained Model", "Train Model On the Fly"])

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    if st.button('Predict'):
        predictions,total_time, report = run_prediction(uploaded_file, mode == "Use Pretrained Model")
        f1= evaluate_small_classes(report,2)
        print(f1)
        st.write(f"Total time taken to predict for each row in average: {total_time:.4f} seconds")
        results_df = pd.DataFrame(predictions, columns=['Predictions'])
        st.write(results_df)
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Prediction Results",
                           data=csv,
                           file_name='results.csv',
                           mime='text/csv')
