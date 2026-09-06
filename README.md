Customer Personality App
This repository contains the code and resources for the Customer Personality App. It is a machine learning-based application designed to analyze customer data, engineer features, and segment customers using a clustering pipeline.  
ZIP
+ 1
📁 Project Structure
The project is organized into several key directories and files:
app.py: The main application file used to run the user interface.  
ZIP
requirements.txt: Contains the list of Python dependencies required to run the project.  
ZIP
rebuild_model.py: A standalone script used to retrain and rebuild the machine learning models.  
ZIP
src/: Contains the core source code, including clustering_pipeline.py which handles the machine learning clustering logic.  
ZIP
config/: Houses configuration files, notably recommendations.py, which likely manages the logic for providing insights based on customer segments.  
ZIP
models/: Stores serialized, pre-trained machine learning models, specifically clustering_model.joblib and clustering_model_org.joblib.  
ZIP
data/: The directory containing the datasets used in this project:  
ZIP
customer_personality_feature_engineered.csv: The processed dataset with engineered features.  
ZIP
customer_segments.csv: The output data containing the finalized customer cluster assignments.  
ZIP
.vscode/: Contains workspace settings (settings.json) for Visual Studio Code.  
ZIP
🚀 Getting Started
Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.
Installation
Clone the repository to your local machine.
Install the required dependencies using the requirements.txt file:  
ZIP
Bash
pip install -r requirements.txt
Usage
To launch the application, run the main app file:  
ZIP
Bash
python app.py
(Note: If app.py is built with a framework like Streamlit, you may need to use streamlit run app.py instead).
Model Retraining
If you update the dataset or wish to tweak the clustering parameters, you can regenerate the .joblib model files by running the rebuild script:  
ZIP
Bash
python rebuild_model.py
