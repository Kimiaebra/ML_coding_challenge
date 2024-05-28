# ML_coding_challenge

## Environment Setup and Installation

### Requirements
- Python 3.9

# Clone the repository
git clone https://github.com/Kimiaebra/ML_coding_challenge.git
cd ML_coding_challenge

# Create a Conda envirement
conda create -n [env_name] python=3.9
conda activate [env_name]

# Install the required packages
pip install -r requirements.txt

# Running the Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py

# Choose Training Model

There are 2 Models to embed this information in the training process, since not all organizations create/can create all 476 of these parts_ids:

### NegativeSampling Mode:
- Augments the provided datasets by adding negative samples to help the model learn that not all part IDs get produced by each organization. This is done by introducing samples for part IDs that an organization does not produce (negative samples), while positive samples are given weights.

### FeatureFiltering Mode:
- Prediction across all categories and post-process the predictions to adjust based on the organization-part_id mapping.

# Choose Prediction Mode

### Use Pretrained Model:
- This mode utilizes a model that was previously trained on the entire provided original dataset. This is for your unseen dataset of 5000 entries mentioned in the task description. After uploading your dataset, click the "Predict" button.

### Train Model On the Fly:
- This mode dynamically trains a new model using 80% of the provided original dataset that has to be uploaded and then uses the remaining 20% as test dataset to be predicted upon clicking on "Predict" button. This version does not take a new test dataset to be predicted.

Note: A random subset of the provided dataset is located in GitHub at [this link](https://github.com/Kimiaebra/ML_coding_challenge/blob/56e2b49dfc91a1534d34258897270b3693904146/saved_raw_test_data.csv), merely for testing the functionality of the pretrained version, of course, it's biased as the trained model has seen the whole data.
