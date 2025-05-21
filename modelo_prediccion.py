import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.binary_cols = [
            'mainroad', 'guestroom', 'basement', 
            'hotwaterheating', 'airconditioning', 'prefarea'
        ]
        
    def train_and_save_model(self, data_path='Housing.csv', model_path='house_price_model.pkl'):
        """Entrena y guarda el modelo mejorado"""
        # Cargar y limpiar datos
        data = pd.read_csv(data_path)
        
        # Eliminar duplicados
        data = data.drop_duplicates()
        
        # Manejo de outliers
        data = self.handle_outliers(data)
        
        # Ingeniería de características
        data = self.feature_engineering(data)
        
        # Preprocesamiento
        X = data.drop('price', axis=1)
        y = np.log(data['price'])  # Transformación logarítmica
        
        # Definir transformaciones
        numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
                            'price_per_sqft', 'total_rooms']
        categorical_features = ['furnishingstatus']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features),
                ('binary', 'passthrough', self.binary_cols)
            ])
        
        # Crear pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
        
        # Optimización de hiperparámetros
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5]
        }
        
        # Búsqueda de mejores parámetros
        grid_search = GridSearchCV(self.model, param_grid, cv=5, 
                                  scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        # Evaluación final
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Métricas de rendimiento
        mae = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
        rmse = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((np.exp(y_test) - np.exp(y_pred)) / np.exp(y_test))) * 100
        
        print("\nResultados Finales:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f} ({r2*100:.1f}%)")
        print(f"MAPE: {mape:.2f}%")
        
        # Guardar modelo
        joblib.dump(self.model, model_path)
        
    def handle_outliers(self, data):
        """Manejo de outliers usando el método IQR"""
        numeric_cols = ['area', 'price']
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[(data[col] >= (Q1 - 1.5 * IQR)) & (data[col] <= (Q3 + 1.5 * IQR))]
        return data
        
    def feature_engineering(self, data):
        """Creación de nuevas características"""
        # Convertir variables binarias
        data[self.binary_cols] = data[self.binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
        
        # Nuevas características
        data['price_per_sqft'] = data['price'] / data['area']
        data['total_rooms'] = data['bedrooms'] + data['bathrooms']
        data['luxury_score'] = data['airconditioning'] + data['prefarea'] + data['hotwaterheating']
        
        # Transformación logarítmica para características sesgadas
        data['area_log'] = np.log(data['area'])
        
        return data
        
    def load_model(self, model_path='house_price_model.pkl'):
        """Carga el modelo entrenado"""
        self.model = joblib.load(model_path)
        return self
        
    def predict(self, input_data):
        """Realiza una predicción con nuevos datos"""
        input_df = pd.DataFrame([input_data])
        input_df = self.feature_engineering(input_df)
        log_pred = self.model.predict(input_df)
        return np.exp(log_pred)[0]

# Ejecutar entrenamiento
if __name__ == '__main__':
    predictor = HousePriceModel()
    predictor.train_and_save_model()
    print("modelo")
    
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