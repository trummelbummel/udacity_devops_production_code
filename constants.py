'''
This file defines all variables for the churn main experiment workflow.
'''

folders=['./models/', './logs/', './images/'] # Directories necessary for execution.
bestmodel = './model/rf_model.pkl'
datapath = "./data/bank_data.csv"
imagepath = "./images/"
model_pth='./models/'

# Model training variables
modeltraining=True
modelname='rf' # or lrc

n_estimators=[100, 200, 1000]
minleafs=[1, 3, 5]
penalty=['elastic', 'l1']
logistic_c=[1, 10, 100, 1000]

# Definitions for plotting
plot_univar = ['Churn', 'Customer_Age']
plot_univar_cat = ['Income_Category']
dependent='Churn'
negative_class="Existing Customer"
dependent_inputcol='Attrition_Flag'
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]