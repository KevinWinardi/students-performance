import streamlit as st
import pandas as pd
from data_preprocessing import data_preprocessing, encoder_Gender, encoder_Application_mode, encoder_Course, encoder_Daytime_evening_attendance, encoder_Debtor, encoder_Displaced, encoder_Scholarship_holder, encoder_Tuition_fees_up_to_date
from prediction import prediction

st.sidebar.image('img/logo.png', width=100)
st.sidebar.header("Students Performance App (Prototype)")
st.sidebar.markdown("This app predicts the performance of students based on 15 affect features.")

with st.form('student_form'):
    st.subheader('Profil')
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox(label='Gender', options=encoder_Gender.classes_, index=0, key='Gender')
    with col2:
        Age_at_enrollment = int(st.number_input(label='Age At Enrollment', min_value=17, max_value=100, help='Input integer number', key='Age_at_enrollment'))

    st.subheader('Admission Information')
    col1, col2, col3 = st.columns(3)
    with col1:
        Application_mode = st.selectbox(label='Application_mode', options=encoder_Application_mode.classes_, index=0, key='Application_mode')
    with col2:
        Course = st.selectbox(label='Course', options=encoder_Course.classes_, index=0, key='Course')
    with col3:
        Daytime_evening_attendance = st.selectbox(label='Daytime / Evening Attendance', options=encoder_Daytime_evening_attendance.classes_, index=0, key='Daytime_evening_attendance')

    st.subheader('Student Condition')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Debtor = st.selectbox(label='Debtor', options=encoder_Debtor.classes_, index=0, key='Debtor')
    with col2:
        Displaced = st.selectbox(label='Displaced', options=encoder_Displaced.classes_, index=0, key='Displaced')
    with col3:
        Scholarship_holder = st.selectbox(label='Scholarship Holder', options=encoder_Scholarship_holder.classes_, index=0, key='Scholarship_holder')
    with col4:
        Tuition_fees_up_to_date = st.selectbox(label='Tuition Fees Up To Date', options=encoder_Tuition_fees_up_to_date.classes_, index=0, key='Tuition_fees_up_to_date')

    st.subheader('Curricular Units 1st Sem')
    col1, col2, col3 = st.columns(3)
    with col1:
        Curricular_units_1st_sem_approved = int(st.number_input(label='Approved', min_value=0, help='Input integer number', key='Curricular_units_1st_sem_approved'))
    with col2:
        Curricular_units_1st_sem_grade = float(st.number_input(label='Grade', min_value=0.0, help='Input float number', key='Curricular_units_1st_sem_grade'))
    with col3:
        Curricular_units_1st_sem_without_evaluations = int(st.number_input(label='Without Evaluations', min_value=0, help='Input integer number', key='Curricular_units_1st_sem_without_evaluations'))

    st.subheader('Curricular Units 2nd Sem')
    col1, col2, col3 = st.columns(3)
    with col1:
        Curricular_units_2nd_sem_approved = int(st.number_input(label='Approved', min_value=0, help='Input integer number', key='Curricular_units_2nd_sem_approved'))
    with col2:
        Curricular_units_2nd_sem_grade = float(st.number_input(label='Grade', min_value=0.0, help='Input float number', key='Curricular_units_2nd_sem_grade'))
    with col3:
        Curricular_units_2nd_sem_without_evaluations = int(st.number_input(label='Without Evaluations', min_value=0, help='Input integer number', key='Curricular_units_2nd_sem_without_evaluations'))

    submit_button = st.form_submit_button(label='Generate Prediction')

if submit_button:
    data = pd.DataFrame({
        'Gender': [Gender],
        'Age_at_enrollment': [Age_at_enrollment],
        'Application_mode': [Application_mode],
        'Course': [Course],
        'Daytime_evening_attendance': [Daytime_evening_attendance],
        'Debtor': [Debtor],
        'Displaced': [Displaced],
        'Scholarship_holder': [Scholarship_holder],
        'Tuition_fees_up_to_date': [Tuition_fees_up_to_date],
        'Curricular_units_1st_sem_approved': [Curricular_units_1st_sem_approved],
        'Curricular_units_1st_sem_grade': [Curricular_units_1st_sem_grade],
        'Curricular_units_1st_sem_without_evaluations': [Curricular_units_1st_sem_without_evaluations],
        'Curricular_units_2nd_sem_approved': [Curricular_units_2nd_sem_approved],
        'Curricular_units_2nd_sem_grade': [Curricular_units_2nd_sem_grade],
        'Curricular_units_2nd_sem_without_evaluations': [Curricular_units_2nd_sem_without_evaluations]
    })
    
    new_data = data_preprocessing(data)

    with st.expander("View the Data"):
        st.dataframe(data=data, width=800, height=10)
    
    with st.spinner('Making the prediction...'):
        result = prediction(new_data)
        if result=='Dropout':
            st.error(f"Prediction Result: {result}")
        else:
            st.success(f"Prediction Result: {result}")