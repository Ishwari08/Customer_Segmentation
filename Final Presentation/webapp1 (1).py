import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
st.title('Predict Customer Segmentation')
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    st.write(df1.head())

    # Select features and target variable
    x = df1.loc[:, ['Education','Income','Recency','NumWebVisitsMonth','Complain','Response','age','Living_With','Years_Since_Registration','Customer_Spent','Total_purcheses','Num_Children','Total_Accept_cmp']]
    y = df1['Clusters']

    # Standardize the features
    SS = StandardScaler()
    SS_X = SS.fit_transform(x)
    X = pd.DataFrame(SS_X, columns=x.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    # Train the model
    model = RandomForestClassifier(max_depth=4, n_estimators=200, max_samples=None, max_features=1.0, random_state=None)
    model.fit(X_train, y_train)

    # Save the model
    with open('my_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    st.write("Model trained and saved successfully as 'my_model.pkl'")

    # Load the pre-trained model
    model = pickle.load(open("my_model.pkl", "rb"))

    # Make predictions
    predictions = model.predict(X_test)
    st.write("Predictions on test set:")
    st.write(predictions.round().astype(int))

    # Allow user to input new data for prediction
    st.write("Input new data for prediction:")

    # Input fields for all features
    Education = st.number_input("Education", value=0)
    Income = st.number_input("Income", value=0.0)
    Recency = st.number_input("Recency", value=0.0)
    NumWebVisitsMonth = st.number_input("Number of Web Visits in a Month", value=0)
    Complain = st.number_input("Complain (0 or 1)", value=0)
    Response = st.number_input("Response (0 or 1)", value=0)
    Age = st.number_input("Age", value=0)
    Living_With = st.number_input("Living With (0, 1, 2)", value=0)
    Years_Since_Registration = st.number_input("Years Since Registration", value=0.0)
    Customer_Spent = st.number_input("Customer Spent", value=0.0)
    Total_purcheses = st.number_input("Total Purchases", value=0)
    Num_Children = st.number_input("Number of Children", value=0)
    Total_Accept_cmp = st.number_input("Total Accepted Campaigns", value=0)

    if st.button("Predict"):
        new_data = pd.DataFrame({
            'Education': [Education],
            'Income': [Income],
            'Recency': [Recency],
            'NumWebVisitsMonth': [NumWebVisitsMonth],
            'Complain': [Complain],
            'Response': [Response],
            'age': [Age],
            'Living_With': [Living_With],
            'Years_Since_Registration': [Years_Since_Registration],
            'Customer_Spent': [Customer_Spent],
            'Total_purcheses': [Total_purcheses],
            'Num_Children': [Num_Children],
            'Total_Accept_cmp': [Total_Accept_cmp]
        })
        
        new_data_scaled = SS.transform(new_data)
        new_prediction = model.predict(new_data_scaled)
        new_prediction_int = int(round(new_prediction[0]))

        # Map cluster numbers to descriptive names
        cluster_names = {
            0: "High income, low spend",
            1: "Highest spent and highest earning",
            2: "Low income, low spend",
            3: "High income, high spending",
            4: "Lowest spent, low income"
        }

        predicted_cluster_name = cluster_names.get(new_prediction_int, "Unknown cluster")
        st.write(f"The predicted cluster is: {new_prediction_int} - {predicted_cluster_name}")
