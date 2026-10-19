A Streamlit web app that automates the core machine learning workflow: upload a dataset, explore it with auto-generated plots, train multiple models in one click, and compare their accuracy on a leaderboard.

Features
Data upload: CSV or Excel, with a quick preview and summary statistics
Automated EDA: histograms, box plots, count plots, correlation heatmap, scatter plots
Preprocessing: missing value handling, categorical encoding, feature scaling, train-test split
Model building: choose the target column and problem type (classification or regression), then train several models at once
Model comparison: accuracy, precision, recall, F1 (classification) or R², MAE, RMSE (regression), shown in a ranked table and chart
Export: download the results table and the best model as a .pkl file

Tech Stack

Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn

Workflow

Upload Data → EDA → Preprocess → Select Target & Models → Train → Compare → Export

Author

Anuj
