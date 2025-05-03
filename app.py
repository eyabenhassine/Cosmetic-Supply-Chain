import pandas as pd
import joblib
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import warnings
import traceback

warnings.filterwarnings("ignore")  # Ignorer les avertissements pour simplifier

print("Starting app.py...")

# Initialisation de l'application Flask
app = Flask(__name__)
print("Flask app initialized.")

# Informations de connexion à la base de données
host = "localhost"
port = "5432"
database = "DW_supplyChain"
user = "postgres"
password = "1234"  # Remplace par ton vrai mot de passe
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
print("Connecting to database...")
try:
    engine = create_engine(connection_string)
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    engine = None

# Fonction pour créer la matrice de recommandation (pour kNN)
def create_recommendation_matrix(df):
    print("Creating recommendation matrix...")
    pivot_table = df.pivot_table(index='shop_id', columns='productname', values='total_quantity', fill_value=0)
    print("Recommendation matrix created.")
    return pivot_table

# Fonction pour entraîner le modèle kNN
def train_knn_model(pivot_table):
    print("Training kNN model...")
    knn_model = NearestNeighbors(metric='cosine')
    knn_model.fit(pivot_table)
    print("kNN model trained.")
    return knn_model

# Fonction pour obtenir les recommandations (kNN)
def recommend_products_knn(shop_id, pivot_table, knn_model, top_n=5, n_neighbors=3):
    if shop_id not in pivot_table.index:
        return None, f"Shop ID {shop_id} non trouvé dans les données."
    
    shop_index = pivot_table.index.tolist().index(shop_id)
    distances, indices = knn_model.kneighbors([pivot_table.iloc[shop_index]], n_neighbors=n_neighbors+1)
    
    similar_shops_indices = indices.flatten()[1:]
    similar_shops = pivot_table.iloc[similar_shops_indices]
    scores = similar_shops.sum(axis=0)
    
    purchased_products = pivot_table.loc[shop_id][pivot_table.loc[shop_id] > 0].index
    scores = scores.drop(purchased_products, errors='ignore')
    
    recommended_products = scores.sort_values(ascending=False).head(top_n)
    return recommended_products, None

# Fonction pour générer un graphique des recommandations (kNN)
def plot_recommendations(recommendations, shop_id):
    plt.figure(figsize=(10, 6))
    recommendations.plot(kind='bar', color='dodgerblue')
    plt.title(f"Produits Recommandés pour Shop ID {shop_id}")
    plt.xlabel("Nom du produit")
    plt.ylabel("Score agrégé (par voisins)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

# Fonction pour entraîner et sauvegarder le modèle SARIMA
def train_sarima_model():
    print("Training SARIMA model...")
    # Charger le JSON
    json_path = os.path.join(os.path.dirname(__file__), 'orders_products_2022_2024.json')
    print(f"Looking for JSON file at: {json_path}")
    if not os.path.exists(json_path):
        return None, None, "Erreur : Le fichier 'orders_products_2022_2024.json' n'a pas été trouvé."
    
    data = pd.read_json(json_path)
    print("JSON file loaded.")
    
    # Étape 1 : Extraire les produits et quantités
    products = []
    for order in data.to_dict('records'):
        order_date = pd.to_datetime(order['OrderDate'])
        for product in order['Products']:
            product['order_date'] = order_date
            products.append(product)
    
    products_df = pd.DataFrame(products)
    print("Products DataFrame created.")
    
    # Étape 2 : Agréger par produit et par mois
    products_df['order_date'] = pd.to_datetime(products_df['order_date'])
    products_df['month'] = products_df['order_date'].dt.to_period('M')
    monthly_demand = products_df.groupby(['productid', 'month'])['quantity'].sum().reset_index()
    print("Monthly demand aggregated.")
    print(monthly_demand.head().to_string())
    print(f"Data types: {monthly_demand.dtypes}")
    print(f"Missing values: {monthly_demand.isnull().sum()}")
    
    monthly_demand['month'] = monthly_demand['month'].dt.to_timestamp()
    monthly_demand['month_of_year'] = monthly_demand['month'].dt.month
    
    # Étape 3 : Préparer les prédictions pour chaque produit (limiter aux 100 premiers produits)
    all_predictions = []
    historical_data = []
    product_ids = monthly_demand['productid'].unique()[:100]  # Prendre seulement les 100 premiers produits
    print(f"Number of unique product IDs to process: {len(product_ids)}")
    for i, product_id in enumerate(product_ids):
        print(f"Processing product ID {product_id} ({i+1}/{len(product_ids)})")
        product_data = monthly_demand[monthly_demand['productid'] == product_id].copy()
        print(f"Data for product ID {product_id}:\n{product_data[['month', 'quantity']].to_string()}")
        
        # Vérifier les doublons dans 'month' et agréger si nécessaire
        if product_data['month'].duplicated().any():
            print(f"Duplicate months found for product ID {product_id}, aggregating data...")
            product_data = product_data.groupby('month')['quantity'].sum().reset_index()
        
        if len(product_data) < 12:  # Réduire à 12 mois pour inclure plus de produits
            print(f"Skipping product ID {product_id} due to insufficient data (<12 months)")
            continue
        
        product_data['type'] = 'historical'
        product_data['productid'] = product_id
        historical_data.append(product_data)
        print(f"Added historical data for product ID {product_id}")
        
        product_data_ts = product_data.set_index('month')['quantity']
        print(f"Shape of product_data_ts for product ID {product_id}: {product_data_ts.shape}")
        if product_data_ts.shape[0] < 2 or not product_data_ts.index.is_monotonic_increasing:
            print(f"Skipping product ID {product_id} due to insufficient data (<2 points) or non-monotonic index")
            continue
        
        # Vérifier si toutes les valeurs sont identiques (aucune variation)
        if product_data_ts.nunique() == 1:
            print(f"Skipping product ID {product_id} due to no variation in data (all values are {product_data_ts.iloc[0]})")
            continue
        
        print(f"Fitting SARIMA model for product_id {product_id}...")
        
        try:
            # Essayer d'abord avec le modèle saisonnier
            try:
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
                model_fit = model.fit(disp=False)
            except Exception as e:
                print(f"Seasonal SARIMA failed for product ID {product_id}: {e}, trying non-seasonal ARIMA...")
                # Revenir à un modèle non saisonnier si le modèle saisonnier échoue
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                model_fit = model.fit(disp=False)
            
            print(f"SARIMA model fitted for product ID {product_id}")
            
            forecast_steps = 20  # Mai 2025 à Décembre 2026
            forecast = model_fit.forecast(steps=forecast_steps)
            print(f"Forecast generated for product ID {product_id}")
            
            future_dates = pd.date_range(start='2025-05-01', periods=forecast_steps, freq='M')
            forecast_df = pd.DataFrame({
                'month': future_dates,
                'predicted_quantity': forecast.round().astype(int),
                'productid': product_id,
                'type': 'predicted'
            })
            all_predictions.append(forecast_df)
            print(f"Prediction DataFrame created for product ID {product_id}")
        except Exception as e:
            print(f"Error fitting SARIMA model for product ID {product_id}: {e}")
            traceback.print_exc()
            continue
    
    if not all_predictions:
        return None, None, "Erreur : Aucune prédiction n'a pu être générée (pas assez de données pour les produits)."
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    historical_df = pd.concat(historical_data, ignore_index=True)
    historical_df = historical_df.rename(columns={'quantity': 'quantity'})
    
    # Sauvegarder les données
    joblib.dump(predictions_df, 'sarima_forecast_sales.pkl')
    joblib.dump(historical_df, 'sarima_historical_sales.pkl')
    print("SARIMA model trained and saved.")
    
    return predictions_df, historical_df, None

# Fonction pour générer les graphiques SARIMA
def generate_sarima_plots(predictions_df, historical_df):
    print("Generating SARIMA plots...")
    # Courbe globale
    historical_global = historical_df.groupby('month')['quantity'].sum().reset_index()
    historical_global['type'] = 'historical'
    predicted_global = predictions_df.groupby('month')['predicted_quantity'].sum().reset_index()
    predicted_global['type'] = 'predicted'
    
    plt.figure(figsize=(12, 6))
    plt.plot(historical_global['month'], historical_global['quantity'], label='Historique (tous produits)', marker='o', color='blue')
    plt.plot(predicted_global['month'], predicted_global['predicted_quantity'], label='Prédit (tous produits)', marker='x', linestyle='--', color='orange')
    plt.title('Quantités historiques et prédites pour tous les produits (SARIMA, mai 2025 - déc 2026)')
    plt.xlabel('Mois')
    plt.ylabel('Quantité totale')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    curve_plot = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    buffer1.close()
    plt.close()
    
    # Histogramme
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions_df['predicted_quantity'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution des quantités prédites pour tous les produits (SARIMA, mai 2025 - déc 2026)')
    plt.xlabel('Quantité prédite')
    plt.ylabel('Fréquence')
    plt.grid(True)
    mean_pred = predictions_df['predicted_quantity'].mean()
    plt.axvline(mean_pred, color='red', linestyle='--', label=f'Moyenne: {mean_pred:.2f}')
    plt.legend()
    
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    hist_plot = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    buffer2.close()
    plt.close()
    
    # Courbe saisonnière
    seasonal_data = historical_df.groupby('month_of_year')['quantity'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(seasonal_data['month_of_year'], seasonal_data['quantity'], label='Moyenne (tous produits)', marker='o', color='green')
    plt.title('Variation saisonnière moyenne des quantités pour tous les produits (SARIMA)')
    plt.xlabel('Mois de l’année (1 = Janvier, 12 = Décembre)')
    plt.ylabel('Quantité moyenne')
    plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    buffer3 = io.BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    seasonal_plot = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    buffer3.close()
    plt.close()
    
    print("SARIMA plots generated.")
    return curve_plot, hist_plot, seasonal_plot

# Fonction pour entraîner et sauvegarder tous les modèles
def train_and_save_models():
    print("Training all models...")
    # --- Modèle Clustering_Sales (K-Means) ---
    print("Training Clustering Sales model...")
    query_kmeans = """
    SELECT quantity, unit_price, total
    FROM public."Fact_sales"
    WHERE quantity IS NOT NULL AND unit_price IS NOT NULL AND total IS NOT NULL
    """
    df_kmeans = pd.read_sql(query_kmeans, engine)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_kmeans[['quantity', 'unit_price', 'total']])
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    kmeans.fit(X_scaled)
    joblib.dump(scaler, 'scaler_sales_clustering_sales.pkl')
    joblib.dump(kmeans, 'clustering_sales.pkl')
    print("Clustering Sales model trained.")

    # --- Modèle Demand_Prediction_Sales (Random Forest) ---
    print("Training Demand Prediction Sales model...")
    query_rf = """
    SELECT "FK_shop", "FK_product", "quantity", "unit_price", "total", "FK_date"
    FROM public."Fact_sales"
    WHERE "quantity" IS NOT NULL AND "unit_price" IS NOT NULL AND total IS NOT NULL
    """
    df_rf = pd.read_sql(query_rf, engine)
    df_rf['demand_category'] = pd.cut(df_rf['quantity'], bins=[0, 10, 15, float('inf')], labels=['basse', 'moyenne', 'élevée'])
    df_rf['month'] = pd.to_datetime(df_rf['FK_date'], format='%Y%m%d').dt.month
    X_rf = df_rf[['unit_price', 'total', 'FK_shop', 'FK_product', 'month']]
    y_rf = df_rf['demand_category']
    X_train_rf, _, y_train_rf, _ = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_rf, y_train_rf)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_smote, y_train_smote)
    joblib.dump(rf, 'rf_sales_demand.pkl')
    print("Demand Prediction Sales model trained.")

    # --- Modèle Product_Recommendation_Sales (kNN) ---
    print("Training Product Recommendation Sales model...")
    query_knn = """
    SELECT 
        s."PK_shops" AS shop_id, 
        p.productname, 
        SUM(fs.quantity) AS total_quantity
    FROM public."Fact_sales" fs
    JOIN public."Dim_Shops" s ON fs."FK_shop" = s."PK_shops"
    JOIN public."Dim_Cosmetic_Products" p ON fs."FK_product" = p."PK_Products"
    WHERE 
        fs."FK_shop" IS NOT NULL 
        AND fs."FK_product" IS NOT NULL 
        AND fs.quantity IS NOT NULL
    GROUP BY 
        s."PK_shops", 
        p.productname
    """
    df_knn = pd.read_sql(query_knn, engine)
    pivot_table = create_recommendation_matrix(df_knn)
    knn_model = train_knn_model(pivot_table)
    joblib.dump(knn_model, 'knn_sales_recommendation.pkl')
    joblib.dump(pivot_table, 'pivot_table_sales_recommendation.pkl')
    print("Product Recommendation Sales model trained.")

    # --- Modèle Time_Series_Forecast_Sales (SARIMA) ---
    print("Training Time Series Forecast Sales model...")
    predictions_df, historical_df, sarima_error = train_sarima_model()
    if sarima_error:
        print(sarima_error)
        predictions_df, historical_df = None, None
    print("Time Series Forecast Sales model training completed.")

    return scaler, kmeans, rf, knn_model, pivot_table, predictions_df, historical_df

# Vérifier si tous les fichiers .pkl existent, sinon entraîner et sauvegarder
pkl_files = [
    'scaler_sales_clustering_sales.pkl',
    'clustering_sales.pkl',
    'rf_sales_demand.pkl',
    'knn_sales_recommendation.pkl',
    'pivot_table_sales_recommendation.pkl',
    'sarima_forecast_sales.pkl',
    'sarima_historical_sales.pkl'
]
print("Checking for .pkl files...")
if not all(os.path.exists(pkl) for pkl in pkl_files):
    print("Some .pkl files are missing. Training models...")
    scaler, kmeans, rf, knn_model, pivot_table, predictions_df, historical_df = train_and_save_models()
else:
    print("Loading .pkl files...")
    scaler = joblib.load('scaler_sales_clustering_sales.pkl')
    kmeans = joblib.load('clustering_sales.pkl')
    rf = joblib.load('rf_sales_demand.pkl')
    knn_model = joblib.load('knn_sales_recommendation.pkl')
    pivot_table = joblib.load('pivot_table_sales_recommendation.pkl')
    predictions_df = joblib.load('sarima_forecast_sales.pkl')
    historical_df = joblib.load('sarima_historical_sales.pkl')
print(".pkl files processed.")

# Liste des shops pour le menu déroulant
SHOPS = ['41', '61', '81']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/axis/<axis_name>')
def axis(axis_name):
    if axis_name.lower() == 'sales':
        models = [
            {'id': 'clustering_sales', 'name': 'Clustering Sales'},
            {'id': 'demand_prediction_sales', 'name': 'Demand Prediction Sales'},
            {'id': 'product_recommendation_sales', 'name': 'Product Recommendation Sales'},
            {'id': 'time_series_forecast_sales', 'name': 'Time Series Forecast Sales'}
        ]
        return render_template('models.html', axis=axis_name, models=models)
    elif axis_name.lower() in ['production', 'stock']:
        return render_template('axis.html', axis=axis_name)
    else:
        return render_template('index.html', error="Axe non valide")

@app.route('/test_model/sales/clustering_sales', methods=['GET', 'POST'])
def test_clustering_sales():
    if request.method == 'POST':
        try:
            quantity = float(request.form['quantity'])
            unit_price = float(request.form['unit_price'])
            total = quantity * unit_price
            input_data = pd.DataFrame([[quantity, unit_price, total]], 
                                     columns=['quantity', 'unit_price', 'total'])
            input_scaled = scaler.transform(input_data)
            cluster = kmeans.predict(input_scaled)[0]
            return render_template('test_model.html', 
                                 axis='sales', 
                                 model_name='Clustering Sales', 
                                 model_id='clustering_sales', 
                                 fields=['quantity', 'unit_price'], 
                                 prediction=f'Le point (total calculé: {total}) appartient au cluster {cluster}',
                                 quantity=quantity,
                                 unit_price=unit_price,
                                 total=total)
        except Exception as e:
            return render_template('test_model.html', 
                                 axis='sales', 
                                 model_name='Clustering Sales', 
                                 model_id='clustering_sales', 
                                 fields=['quantity', 'unit_price'], 
                                 prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html', 
                         axis='sales', 
                         model_name='Clustering Sales', 
                         model_id='clustering_sales', 
                         fields=['quantity', 'unit_price'])

@app.route('/test_model/sales/demand_prediction_sales', methods=['GET', 'POST'])
def test_demand_prediction_sales():
    if request.method == 'POST':
        try:
            unit_price = float(request.form['unit_price'])
            total = float(request.form['total'])
            fk_shop = int(request.form['FK_shop'])
            fk_product = int(request.form['FK_product'])
            month = int(request.form['month'])
            input_data = pd.DataFrame([[unit_price, total, fk_shop, fk_product, month]], 
                                     columns=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'])
            prediction = rf.predict(input_data)[0]
            return render_template('test_model.html', 
                                 axis='sales', 
                                 model_name='Demand Prediction Sales', 
                                 model_id='demand_prediction_sales', 
                                 fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'], 
                                 prediction=f'Catégorie de demande prédite: {prediction}',
                                 unit_price=unit_price,
                                 total=total,
                                 FK_shop=fk_shop,
                                 FK_product=fk_product,
                                 month=month)
        except Exception as e:
            return render_template('test_model.html', 
                                 axis='sales', 
                                 model_name='Demand Prediction Sales', 
                                 model_id='demand_prediction_sales', 
                                 fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'], 
                                 prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html', 
                         axis='sales', 
                         model_name='Demand Prediction Sales', 
                         model_id='demand_prediction_sales', 
                         fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'])

@app.route('/test_model/sales/product_recommendation_sales', methods=['GET', 'POST'])
def test_product_recommendation_sales():
    if request.method == 'POST':
        try:
            shop_id = int(request.form['shop_id'])
            n_neighbors = int(request.form['n_neighbors'])
            top_n = int(request.form['top_n'])

            recommendations, error = recommend_products_knn(
                shop_id=shop_id,
                pivot_table=pivot_table,
                knn_model=knn_model,
                top_n=top_n,
                n_neighbors=n_neighbors
            )

            if error:
                return render_template('test_model.html',
                                     axis='sales',
                                     model_name='Product Recommendation Sales',
                                     model_id='product_recommendation_sales',
                                     fields=['shop_id', 'n_neighbors', 'top_n'],
                                     shops=SHOPS,
                                     prediction=f'Erreur: {error}',
                                     shop_id=shop_id,
                                     n_neighbors=n_neighbors,
                                     top_n=top_n)

            graphic = plot_recommendations(recommendations, shop_id)

            table_html = '<table class="recommendation-table">'
            table_html += '<thead><tr><th>Produit</th><th>Score</th></tr></thead>'
            table_html += '<tbody>'
            for product, score in recommendations.items():
                table_html += f'<tr><td>{product}</td><td><strong style="color: #28a745;">{score:.1f}</strong></td></tr>'
            table_html += '</tbody></table>'

            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Product Recommendation Sales',
                                 model_id='product_recommendation_sales',
                                 fields=['shop_id', 'n_neighbors', 'top_n'],
                                 shops=SHOPS,
                                 prediction=table_html,
                                 graphic=graphic,
                                 shop_id=shop_id,
                                 n_neighbors=n_neighbors,
                                 top_n=top_n)
        except Exception as e:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Product Recommendation Sales',
                                 model_id='product_recommendation_sales',
                                 fields=['shop_id', 'n_neighbors', 'top_n'],
                                 shops=SHOPS,
                                 prediction=f'Erreur: {str(e)}',
                                 shop_id=request.form.get('shop_id', ''),
                                 n_neighbors=request.form.get('n_neighbors', ''),
                                 top_n=request.form.get('top_n', ''))
    return render_template('test_model.html',
                         axis='sales',
                         model_name='Product Recommendation Sales',
                         model_id='product_recommendation_sales',
                         fields=['shop_id', 'n_neighbors', 'top_n'],
                         shops=SHOPS)

@app.route('/test_model/sales/time_series_forecast_sales', methods=['GET', 'POST'])
def test_time_series_forecast_sales():
    product_id = None
    date_str = None
    
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        date_str = request.form.get('date')
    elif request.method == 'GET':
        product_id = request.args.get('product_id')
        date_str = request.args.get('date')
    
    if not product_id or not date_str:
        return render_template('test_model.html',
                             axis='sales',
                             model_name='Time Series Forecast Sales',
                             model_id='time_series_forecast_sales',
                             fields=['product_id', 'date'],
                             prediction="Erreur : Veuillez fournir un product_id et une date (ex. ?product_id=PR-1103&date=2025-06-01)")

    try:
        date = pd.to_datetime(date_str)
        if historical_df is None:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=['product_id', 'date'],
                                 prediction="Erreur : Les données historiques ne sont pas disponibles.")

        # Filtrer les données historiques pour le product_id spécifié
        product_data = historical_df[historical_df['productid'] == product_id].copy()
        if len(product_data) < 12:  # Réduire à 12 mois pour inclure plus de produits
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=['product_id', 'date'],
                                 prediction=f"Erreur : Insufficient data (<12 months) for product ID {product_id}")

        product_data_ts = product_data.set_index('month')['quantity']
        if product_data_ts.shape[0] < 2 or not product_data_ts.index.is_monotonic_increasing:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=['product_id', 'date'],
                                 prediction=f"Erreur : Insufficient data (<2 points) or non-monotonic index for product ID {product_id}")

        # Vérifier si toutes les valeurs sont identiques
        if product_data_ts.nunique() == 1:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=['product_id', 'date'],
                                 prediction=f"Erreur : No variation in data for product ID {product_id} (all values are {product_data_ts.iloc[0]})")

        # Ajuster le modèle SARIMA
        try:
            model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
            model_fit = model.fit(disp=False)
        except Exception as e:
            print(f"Seasonal SARIMA failed for product ID {product_id}: {e}, trying non-seasonal ARIMA...")
            model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)

        # Prédire pour la date spécifiée
        forecast_steps = (date - pd.to_datetime('2025-05-01')).days // 30 + 1  # Approximation en mois
        if forecast_steps < 1:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=['product_id', 'date'],
                                 prediction=f"Erreur : La date {date_str} est antérieure à mai 2025")

        forecast = model_fit.forecast(steps=forecast_steps)[-1]  # Prendre la dernière valeur
        predicted_quantity = round(forecast)

        return render_template('test_model.html',
                             axis='sales',
                             model_name='Time Series Forecast Sales',
                             model_id='time_series_forecast_sales',
                             fields=['product_id', 'date'],
                             prediction=f"Quantité prédite pour {product_id} le {date_str}: {predicted_quantity}",
                             product_id=product_id,
                             date=date_str)

    except Exception as e:
        return render_template('test_model.html',
                             axis='sales',
                             model_name='Time Series Forecast Sales',
                             model_id='time_series_forecast_sales',
                             fields=['product_id', 'date'],
                             prediction=f"Erreur : {str(e)}")

if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        traceback.print_exc()