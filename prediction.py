import joblib

model = joblib.load('model/rf_model.joblib')
result_Status = joblib.load('model/encoder_Status.joblib')

def prediction(data):
    result = model.predict(data)

    # Hasil result berupa float 0.0 atau 1.0, maka perlu konversi int agar encoder bekerja
    result = result.astype(int) 
    final_result = result_Status.inverse_transform(result)[0]
    return final_result