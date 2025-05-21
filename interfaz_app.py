# interfaz.py
import tkinter as tk
from tkinter import ttk, messagebox
from modelo_prediccion import HousePriceModel, get_statistics, get_model_performance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicación de Predicción de Precios de Viviendas")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f4f7")

        self.predictor = HousePriceModel().load_model()

        style = ttk.Style()
        style.configure('TButton', font=('Segoe UI', 11), padding=10)
        style.configure('TLabel', font=('Segoe UI', 12, 'bold'))

        ttk.Label(self.root, text="Menú Principal", font=('Segoe UI', 14, 'bold')).pack(pady=20)

        ttk.Button(self.root, text="Predecir Precio", command=self.open_prediction_window).pack(pady=10)
        ttk.Button(self.root, text="Estadísticas del Dataset", command=self.show_statistics).pack(pady=10)
        ttk.Button(self.root, text="Rendimiento del Modelo", command=self.show_model_performance).pack(pady=10)

    def open_prediction_window(self):
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title("Predecir Precio")
        prediction_window.geometry("500x600")
        PredictPriceInterface(prediction_window, self.predictor)

    def show_statistics(self):
        data = get_statistics()
        numeric_data = data.select_dtypes(include=['number'])

        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlación entre Variables Numéricas')
        plt.show()

        if 'price' in numeric_data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(numeric_data['price'], kde=True)
            plt.title('Distribución de Precios de Viviendas')
            plt.xlabel('Precio')
            plt.ylabel('Frecuencia')
            plt.show()

    def show_model_performance(self):
        y_test, y_pred = get_model_performance()
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        messagebox.showinfo("Rendimiento del Modelo",
                           f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\nMAPE: {mape:.2f}%")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Precio Real')
        plt.ylabel('Precio Predicho')
        plt.title('Regresión Lineal: Real vs Predicho')
        plt.show()

class PredictPriceInterface:
    def __init__(self, root, predictor):
        self.root = root
        self.predictor = predictor
        self.entries = {}
        self.bool_map = {'Sí': 'yes', 'No': 'no'}
        self.furnishing_map = {
            'No Amoblado': 'unfurnished',
            'Semi-Amoblado': 'semi-furnished',
            'Amoblado': 'furnished'
        }
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Ingrese características de la propiedad", font=('Segoe UI', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)

        fields = [
            ('area', 'Área (pies cuadrados)', 'entry'),
            ('bedrooms', 'Número de habitaciones', 'entry'),
            ('bathrooms', 'Número de baños', 'entry'),
            ('stories', 'Número de pisos', 'entry'),
            ('mainroad', 'Conexión a vía principal', 'combobox', ['No', 'Sí']),
            ('guestroom', 'Cuarto de huéspedes', 'combobox', ['No', 'Sí']),
            ('basement', 'Sótano', 'combobox', ['No', 'Sí']),
            ('hotwaterheating', 'Agua caliente', 'combobox', ['No', 'Sí']),
            ('airconditioning', 'Aire acondicionado', 'combobox', ['No', 'Sí']),
            ('parking', 'Espacios de estacionamiento', 'entry'),
            ('prefarea', 'Área preferencial', 'combobox', ['No', 'Sí']),
            ('furnishingstatus', 'Amoblado', 'combobox', ['No Amoblado', 'Semi-Amoblado', 'Amoblado'])
        ]

        for i, (field, label, field_type, *options) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            if field_type == 'entry':
                entry = ttk.Entry(frame)
                entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
                self.entries[field] = entry
            elif field_type == 'combobox':
                combo = ttk.Combobox(frame, values=options[0], state='readonly')
                combo.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
                combo.current(0)
                self.entries[field] = combo

        ttk.Button(frame, text="Predecir Precio", command=self.predict_price).grid(
            row=len(fields)+1, column=0, columnspan=2, pady=20)

        self.result_label = ttk.Label(frame, text="", font=('Segoe UI', 11, 'bold'))
        self.result_label.grid(row=len(fields)+2, column=0, columnspan=2)

        frame.columnconfigure(1, weight=1)

    def predict_price(self):
        try:
            area = float(self.entries['area'].get())
            bedrooms = int(self.entries['bedrooms'].get())
            bathrooms = int(self.entries['bathrooms'].get())
            stories = int(self.entries['stories'].get())
            parking = int(self.entries['parking'].get())

            input_data = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'stories': stories,
                'mainroad': self.bool_map[self.entries['mainroad'].get()],
                'guestroom': self.bool_map[self.entries['guestroom'].get()],
                'basement': self.bool_map[self.entries['basement'].get()],
                'hotwaterheating': self.bool_map[self.entries['hotwaterheating'].get()],
                'airconditioning': self.bool_map[self.entries['airconditioning'].get()],
                'parking': parking,
                'prefarea': self.bool_map[self.entries['prefarea'].get()],
                'furnishingstatus': self.furnishing_map[self.entries['furnishingstatus'].get()]
            }

            predicted_price = self.predictor.predict(input_data)
            data = get_statistics()
            mean_price = data['price'].mean()
            percentage_diff = ((predicted_price - mean_price) / mean_price) * 100

            self.result_label.config(
                text=f"Precio estimado: ${predicted_price:,.2f}\n"
                     f"({percentage_diff:+.1f}% vs precio promedio)",
                foreground="green"
            )

            # Gráfico de distribución
            plt.figure(figsize=(10, 6))
            sns.histplot(data['price'], kde=True)
            plt.axvline(predicted_price, color='red', linestyle='--', label='Predicción')
            plt.title(f'Distribución de Precios (Predicción: ${predicted_price:,.2f})')
            plt.xlabel('Precio')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.show()

            # Gráfico comparativo
            stats = data['price'].describe()
            plt.figure(figsize=(10, 6))
            bars = plt.bar(['Mínimo', 'Promedio', 'Máximo', 'Tu Predicción'],
                          [stats['min'], stats['mean'], stats['max'], predicted_price])
            bars[-1].set_color('red')
            plt.title('Comparación con Estadísticas del Dataset')
            plt.ylabel('Precio')
            plt.xticks(rotation=45)
            plt.show()

        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = HousePriceApp(root)
    root.mainloop()