import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib  # Para guardar y cargar el modelo

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.binary_cols = [
            'mainroad', 'guestroom', 'basement', 
            'hotwaterheating', 'airconditioning', 'prefarea'
        ]
        
    def train_and_save_model(self, data_path='Housing.csv', model_path='house_price_model.pkl'):
        """Entrena y guarda el modelo"""
        data = pd.read_csv(data_path)
        
        # Preprocesamiento
        data[self.binary_cols] = data[self.binary_cols].apply(
            lambda x: x.map({'yes': 1, 'no': 0}))
        
        self.encoder = LabelEncoder()
        data['furnishingstatus'] = self.encoder.fit_transform(data['furnishingstatus'])
        
        X = data.drop('price', axis=1)
        y = data['price']
        
        # Dividir datos
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_train)
        
        # Guardar componentes
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.encoder,
            'binary_cols': self.binary_cols
        }, model_path)
        
    def load_model(self, model_path='house_price_model.pkl'):
        """Carga el modelo entrenado"""
        components = joblib.load(model_path)
        self.model = components['model']
        self.scaler = components['scaler']
        self.encoder = components['encoder']
        self.binary_cols = components['binary_cols']
        return self
        
    def predict(self, input_data):
        """Realiza una predicción con nuevos datos"""
        # Convertir a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocesamiento
        input_df[self.binary_cols] = input_df[self.binary_cols].apply(
            lambda x: x.map({'yes': 1, 'no': 0}))
        
        input_df['furnishingstatus'] = self.encoder.transform(
            input_df['furnishingstatus'])
        
        # Escalar características
        scaled_input = self.scaler.transform(input_df)
        
        # Predecir
        return self.model.predict(scaled_input)[0]

# Ejemplo de uso (para entrenar el modelo)
if __name__ == '__main__':
    predictor = HousePriceModel()
    predictor.train_and_save_model()
    print("Modelo entrenado y guardado exitosamente!")

# Al final del archivo modelo_prediccion.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

def get_statistics(path='Housing.csv'):
    return pd.read_csv(path)

def get_model_performance(path='Housing.csv'):
    data = pd.read_csv(path)
    binary_cols = ['mainroad', 'guestroom', 'basement', 
                   'hotwaterheating', 'airconditioning', 'prefarea']
    data[binary_cols] = data[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))

    encoder = LabelEncoder()
    data['furnishingstatus'] = encoder.fit_transform(data['furnishingstatus'])

    X = data.drop('price', axis=1)
    y = data['price']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    return y_test, y_pred
