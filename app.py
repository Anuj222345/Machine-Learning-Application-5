import streamlit as st 




import pandas as pd 
import numpy as np 


import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

import seaborn as sns



from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def main():
    """Semi Auto App with Streamlit """
    
    st.title("Semi Auto ML App")
    st.text("Using Streamlit ==1.56.0+")
    
    activities = ["EDA", "Plot", "Model Building", "About"]
    
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Show Shape"):
                st.write(df.shape)
    
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
                
            if st.checkbox("Show Summary"):
                st.write(df.describe())
                
            if st.checkbox("Select Columns To Show"):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)
                        
                
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-2].value_counts())
    
    
    
    
    
    
    
    elif choice == 'Plot':
        st.subheader("Data Visualization")
        
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            if st.checkbox("Correlation with Seaborn"):
                corr = df.select_dtypes(include='number').corr()
                st.write("Columns in heatmap:", corr.columns.tolist())
            
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(corr, annot=True, cmap="rocket", ax=ax)
                st.pyplot(fig)
                
            if st.checkbox("Pie Chart"):
                pie_columns = [c for c in df.columns if df[c].nunique() <= 10]
                column_to_plot = st.selectbox("Select 1 Column", pie_columns)

                fig, ax = plt.subplots()
                df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_label("Class")
                st.pyplot(fig)
                
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
            
            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
                
                if type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)
                    
                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)

                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_columns_names].plot(kind=type_of_plot, ax=ax)
                    st.pyplot(fig)
        

        
    
    
    
    elif choice == 'Model Building':
        st.subheader("Building ML Model")
        
        data = st.file_uploader("Upload Dataset",type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            
            
            X = df.iloc[:, 2:].dropna(axis=1, how="all")
            Y = df.iloc[:, 1]
            seed = 7
            
            
            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))
            
            
            
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            
            for name,model in models:
                Kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state=seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=Kfold,scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(), "standard_deviation":cv_results.std()}
                all_models.append(accuracy_results)
                
            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name", "Model Accuracy", "Standard Deviation"]))
                
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)
        
    
    
    
    elif choice == 'About':
        st.subheader("About")
        
    
if __name__ == '__main__':
    main()
