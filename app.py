import streamlit as st
import datetime
import pandas as pd
import joblib
import lightgbm

#to get prediction of Fraud or not Fraud
def prediction(merchantName, transactionAmount, currentBalance, availableMoney, transactionDate,accountOpenDate,
 currentExpDate):

    # #dates
    # txn_date = transactionDate
    # acc_open_date = datetime.strptime(accountOpenDate,'%Y-%m-%d').date()
    # curr_exp_date = datetime.strptime(currentExpDate,'%Y-%m-%d').date()
 
    # Pre-processing user input
    card_age = abs(currentExpDate - accountOpenDate).days
    account_age = abs(transactionDate - accountOpenDate).days

    data = {'merchantName':[merchantName],'account_age':[account_age],'transactionAmount':[transactionAmount],
            'card_age':[card_age],'currentBalance':[currentBalance],'availableMoney':[availableMoney]}

    sample = pd.DataFrame(data)

    sample['merchantName'] = pd.Categorical(sample['merchantName'])

    #loading the trained model
    classifier = joblib.load("./lgbm_top6_features.pkl")

    # Making predictions 
    prediction = classifier.predict(sample)[0]
     
    if prediction == True:
        pred = 'fraudulent transaction '
    else:
        pred = 'non-fraudulent transaction '
    return pred

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page
    st.title('Credit Card Fraud Detection!')

    #read merchant names
    merchants = pd.read_csv('merchant.csv')
    mech_names = merchants['Merchant'].to_list()
      
    # following lines create boxes in which user can enter data required to make prediction 
    merchantName = st.selectbox('Merchant Name',mech_names)
    transactionAmount = st.number_input("Transaction Amount (USD)") 
    currentBalance = st.number_input("Current Account Balance (USD)")
    availableMoney = st.number_input("Available Money (USD)")
    txn_date = st.date_input("Transaction Date")
    acc_open_date = st.date_input("Account Open Date")
    card_exp_date = st.date_input("Card Expiry Date")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(merchantName, transactionAmount, currentBalance, availableMoney, txn_date, acc_open_date,card_exp_date) 
        st.success('This is a {} transaction'.format(result))
     
if __name__=='__main__': 
    main()