import pandas as pd
import joblib
import streamlit as st
# from sklearn.tree import DecisionTreeClassifier

st.title("Mobile Money Fraud Detection")

with st.sidebar:
    st.header("Data Requirements")
    st.caption("To use the model, you need to upload a csv file with nine columns (features). Column names are not important though won't affect the model.\nNote that the model is still in production, it may or may not be perfect with the predictions.")
    with st.expander("Data format"):
        st.markdown("• step")
        st.markdown("• initiator_id")
        st.markdown("• recipient_id")
        st.markdown("• transaction_type")
        st.markdown("• amount")
        st.markdown("• oldBalInitiator")
        st.markdown("• newBalInitiator")
        st.markdown("• oldBalRecipient")
        st.markdown("• newBalRecipient")
    st.divider()
    st.caption("<p style = 'text-align:center'>Developed by Kinataama Lauren and Sukwe Benjamin</p>", unsafe_allow_html = True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Trust our model to predict for you", on_click = clicked, args = [1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, low_memory = True)
        st.header('Uploaded data sample')
        st.write(df.head())
        model = joblib.load('best_student_model.joblib')
        pred = model.predict(df)
        pred = pd.DataFrame(pred, columns=['Prediction'])
        pred['Prediction'] = pred['Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})
        st.header('Predicted values')
        st.write(pred.head())

        pred = pred.to_csv(index=False).encode('utf-8')
        st.download_button('Download prediction',
                        pred,
                        'prediction.csv',
                        'text/csv',
                        key='download-csv')