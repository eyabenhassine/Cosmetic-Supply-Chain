import pandas as pd
import joblib
from flask import Flask, request, render_template  # type: ignore
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend Agg pour éviter les erreurs Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import warnings
import traceback
import ast

warnings.filterwarnings("ignore")

print("Starting app.py...")

app = Flask(__name__)
print("Flask app initialized.")

host = "localhost"
port = "5432"
database = "DW_supplyChain"
user = "postgres"
password = "1234"
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
print("Connecting to database...")
try:
    engine = create_engine(connection_string)
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    engine = None

# Récupérer la liste des catégories, marques et entrepôts pour l'axe Stock
try:
    query_categories = """
    SELECT DISTINCT category
    FROM public."Dim_Cosmetic_Products"
    WHERE category IS NOT NULL
    """
    categories_df = pd.read_sql(query_categories, engine)
    CATEGORIES = categories_df['category'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des catégories: {e}")
    CATEGORIES = ['Hair Care', 'Body Care', 'Foot Care']

try:
    query_brands = """
    SELECT DISTINCT brandname
    FROM public."Dim_Cosmetic_Products"
    WHERE brandname IS NOT NULL
    """
    brands_df = pd.read_sql(query_brands, engine)
    BRANDS = brands_df['brandname'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des marques: {e}")
    BRANDS = ['BrandA', 'BrandB', 'BrandC']

try:
    query_warehouses = """
    SELECT DISTINCT warehousename
    FROM public."Dim_Warehouse"
    WHERE warehousename IS NOT NULL
    """
    warehouses_df = pd.read_sql(query_warehouses, engine)
    WAREHOUSES = warehouses_df['warehousename'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des entrepôts: {e}")
    WAREHOUSES = ['WH1', 'WH2', 'WH3', 'WH4', 'WH5']

# Récupérer la liste des pays des fournisseurs et des entrepôts pour l'axe Stock
try:
    query_supplier_countries = """
    SELECT DISTINCT l.country
    FROM "Dim_Suppliers" s
    JOIN "Dim_Location" l ON s."FK_location" = l."PK_location"
    WHERE l.country IS NOT NULL
    """
    supplier_countries_df = pd.read_sql(query_supplier_countries, engine)
    SUPPLIER_COUNTRIES = supplier_countries_df['country'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des pays des fournisseurs: {e}")
    SUPPLIER_COUNTRIES = ['France', 'Germany', 'USA']

try:
    query_warehouse_countries = """
    SELECT DISTINCT l.country
    FROM "Dim_Warehouse" w
    JOIN "Dim_Location" l ON w."FK_location" = l."PK_location"
    WHERE l.country IS NOT NULL
    """
    warehouse_countries_df = pd.read_sql(query_warehouse_countries, engine)
    WAREHOUSE_COUNTRIES = warehouse_countries_df['country'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des pays des entrepôts: {e}")
    WAREHOUSE_COUNTRIES = ['France', 'Germany', 'USA']

# Récupérer la liste des ProductName
try:
    query_products = """
    SELECT DISTINCT productname
    FROM public."Dim_Cosmetic_Products"
    WHERE productname IS NOT NULL
    """
    products_df = pd.read_sql(query_products, engine)
    PRODUCT_NAMES = products_df['productname'].tolist()
except Exception as e:
    print(f"Erreur lors de la récupération des ProductName: {e}")
    PRODUCT_NAMES = ['Hair Serum 1', 'Body Wash 2', 'Foot Cream 3', 'Conditioner 4', 'Eau de Toilette 5']

SHOPS = ['41', '61', '81']

# Liste des fichiers .pkl pour référence
pkl_files = [
    'scaler_sales_clustering_sales.pkl',
    'clustering_sales.pkl',
    'rf_sales_demand.pkl',
    'knn_sales_recommendation.pkl',
    'pivot_table_sales_recommendation.pkl',
    'sarima_forecast_sales.pkl',
    'sarima_historical_sales.pkl',
    'scaler_production_clustering.pkl',
    'clustering_production.pkl',
    'cluster_labels_production.pkl',
    'product_consumption.pkl',
    'rf_material_consumption.pkl',
    'scaler_material_consumption.pkl',
    'label_encoder_category.pkl',
    'label_encoder_material_category.pkl',
    'material_consumption_dataset.pkl',
    'stock_prediction_model.pkl',
    'le_category.pkl',
    'le_brand.pkl',
    'le_warehouse.pkl',
    'stock_classification_pipeline.pkl'
]
print("Skipping initial .pkl loading. Models will be loaded or trained on demand.")

# Fonctions utilitaires
def create_recommendation_matrix(df):
    print("Creating recommendation matrix...")
    pivot_table = df.pivot_table(index='shop_id', columns='productname', values='total_quantity', fill_value=0)
    print("Recommendation matrix created.")
    return pivot_table

def train_knn_model(pivot_table):
    print("Training kNN model...")
    knn_model = NearestNeighbors(metric='cosine')
    knn_model.fit(pivot_table)
    print("kNN model trained.")
    return knn_model

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

def train_sarima_model():
    print("Training SARIMA model...")
    json_path = os.path.join(os.path.dirname(__file__), 'orders_products_2022_2024.json')
    print(f"Looking for JSON file at: {json_path}")
    if not os.path.exists(json_path):
        return None, None, "Erreur : Le fichier 'orders_products_2022_2024.json' n'a pas été trouvé."
    
    data = pd.read_json(json_path)
    print("JSON file loaded.")
    
    products = []
    for order in data.to_dict('records'):
        order_date = pd.to_datetime(order['OrderDate'])
        for product in order['Products']:
            product['order_date'] = order_date
            products.append(product)
    
    products_df = pd.DataFrame(products)
    print("Products DataFrame created.")
    
    products_df['order_date'] = pd.to_datetime(products_df['order_date'])
    products_df['month'] = products_df['order_date'].dt.to_period('M')
    monthly_demand = products_df.groupby(['productid', 'month'])['quantity'].sum().reset_index()
    print("Monthly demand aggregated.")
    
    monthly_demand['month'] = monthly_demand['month'].dt.to_timestamp()
    monthly_demand['month_of_year'] = monthly_demand['month'].dt.month
    
    all_predictions = []
    historical_data = []
    product_ids = monthly_demand['productid'].unique()[:100]
    print(f"Number of unique product IDs to process: {len(product_ids)}")
    for i, product_id in enumerate(product_ids):
        print(f"Processing product ID {product_id} ({i+1}/{len(product_ids)})")
        product_data = monthly_demand[monthly_demand['productid'] == product_id].copy()
        
        if product_data['month'].duplicated().any():
            product_data = product_data.groupby('month')['quantity'].sum().reset_index()
        
        if len(product_data) < 12:
            continue
        
        product_data['type'] = 'historical'
        product_data['productid'] = product_id
        historical_data.append(product_data)
        
        product_data_ts = product_data.set_index('month')['quantity']
        if product_data_ts.shape[0] < 2 or not product_data_ts.index.is_monotonic_increasing:
            continue
        
        if product_data_ts.nunique() == 1:
            continue
        
        try:
            try:
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
                model_fit = model.fit(disp=False)
            except:
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                model_fit = model.fit(disp=False)
            
            forecast_steps = 20
            forecast = model_fit.forecast(steps=forecast_steps)
            
            future_dates = pd.date_range(start='2025-05-01', periods=forecast_steps, freq='M')
            forecast_df = pd.DataFrame({
                'month': future_dates,
                'predicted_quantity': forecast.round().astype(int),
                'productid': product_id,
                'type': 'predicted'
            })
            all_predictions.append(forecast_df)
        except Exception as e:
            print(f"Error fitting SARIMA model for product_id {product_id}: {e}")
            continue
    
    if not all_predictions:
        return None, None, "Erreur : Aucune prédiction n'a pu être générée."
    
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    historical_df = pd.concat(historical_data, ignore_index=True)
    historical_df = historical_df.rename(columns={'quantity': 'quantity'})
    
    joblib.dump(predictions_df, 'sarima_forecast_sales.pkl')
    joblib.dump(historical_df, 'sarima_historical_sales.pkl')
    print("SARIMA model trained and saved.")
    
    return predictions_df, historical_df, None

def generate_sarima_plots(predictions_df, historical_df):
    print("Generating SARIMA plots...")
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

def convert_to_float(value):
    try:
        if isinstance(value, str):
            value = value.strip("[]")
            value_list = value.split(",")
            floats = [float(x.strip()) for x in value_list if x.strip() != '']
            return sum(floats) / len(floats) if floats else None
        return float(value)
    except Exception as e:
        print(f"Erreur: {value} -> {e}")
        return None

def train_and_save_models():
    print("Training all models...")
    # Sales Models
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

    print("Training Time Series Forecast Sales model...")
    predictions_df, historical_df, sarima_error = train_sarima_model()
    if sarima_error:
        print(sarima_error)
        predictions_df, historical_df = None, None
    print("Time Series Forecast Sales model training completed.")

    # Production Model: K-Means Clustering
    print("Training Clustering Production model...")
    query_kmeans_prod = """
    SELECT 
        f."FK_product",
        f."FK_base_material",
        f."QuantityUsed",
        p."PK_Products",
        p."productname",
        m."PK_Base_Materials",
        m."Material_Name"
    FROM "Fact_Production" f
    JOIN "Dim_Cosmetic_Products" p ON f."FK_product" = p."PK_Products"
    JOIN "Dim_Base_Materials" m ON f."FK_base_material" = m."PK_Base_Materials"
    """
    merged = pd.read_sql(query_kmeans_prod, engine) if engine else pd.DataFrame({
        'FK_product': [1, 2, 1, 3, 2],
        'FK_base_material': [101, 102, 101, 103, 102],
        'QuantityUsed': ['[10.0]', '[20.0]', '[15.0]', '[25.0]', '[30.0]'],
        'PK_Products': [1, 2, 1, 3, 2],
        'productname': ['ProductA', 'ProductB', 'ProductA', 'ProductC', 'ProductB'],
        'PK_Base_Materials': [101, 102, 101, 103, 102],
        'Material_Name': ['Material1', 'Material2', 'Material1', 'Material3', 'Material2']
    })

    merged['QuantityUsed'] = merged['QuantityUsed'].apply(convert_to_float)
    merged['QuantityUsed'] = merged['QuantityUsed'].fillna(merged['QuantityUsed'].median())
    product_consumption = merged.groupby(['FK_product', 'productname'])['QuantityUsed'].sum().reset_index()
    product_consumption.columns = ['ProductID', 'ProductName', 'TotalConsumption']
    
    scaler_prod = MinMaxScaler()
    product_consumption['TotalConsumption_scaled'] = scaler_prod.fit_transform(product_consumption[['TotalConsumption']])
    kmeans_prod = KMeans(n_clusters=2, random_state=42)
    kmeans_prod.fit(product_consumption[['TotalConsumption_scaled']])
    
    product_consumption['Cluster'] = kmeans_prod.labels_
    cluster_means = product_consumption.groupby('Cluster')['TotalConsumption'].mean()
    low_cluster = cluster_means.idxmin()
    cluster_labels = {low_cluster: "Faible consommation", 1-low_cluster: "Forte consommation"}
    product_consumption['Consommation'] = product_consumption['Cluster'].map(cluster_labels)
    
    joblib.dump(scaler_prod, 'scaler_production_clustering.pkl')
    joblib.dump(kmeans_prod, 'clustering_production.pkl')
    joblib.dump(cluster_labels, 'cluster_labels_production.pkl')
    joblib.dump(product_consumption, 'product_consumption.pkl')
    print("Clustering Production model trained.")

    # Production Model: Material Consumption Classification
    print("Training Material Consumption Classification model...")
    query_rf_prod = """
    SELECT 
        f."FK_product",
        f."FK_base_material",
        f."QuantityUsed",
        f."Dosage",
        p."PK_Products",
        p."productname",
        p."category",
        m."PK_Base_Materials",
        m."Material_Name",
        m."Material_Category"
    FROM "Fact_Production" f
    JOIN "Dim_Cosmetic_Products" p ON f."FK_product" = p."PK_Products"
    JOIN "Dim_Base_Materials" m ON f."FK_base_material" = m."PK_Base_Materials"
    """
    merged_rf = pd.read_sql(query_rf_prod, engine) if engine else pd.DataFrame({
        'FK_product': [1, 2, 1, 3, 2],
        'FK_base_material': [101, 102, 101, 103, 102],
        'QuantityUsed': ['[10.0]', '[20.0]', '[15.0]', '[25.0]', '[30.0]'],
        'Dosage': [0.1, 0.2, 0.15, 0.25, 0.3],
        'PK_Products': [1, 2, 1, 3, 2],
        'productname': ['ProductA', 'ProductB', 'ProductA', 'ProductC', 'ProductB'],
        'category': ['Hair Care', 'Body Care', 'Hair Care', 'Foot Care', 'Body Care'],
        'PK_Base_Materials': [101, 102, 101, 103, 102],
        'Material_Name': ['Material1', 'Material2', 'Material1', 'Material3', 'Material2'],
        'Material_Category': ['Oil', 'Cream', 'Oil', 'Powder', 'Cream']
    })

    merged_rf['QuantityUsed'] = merged_rf['QuantityUsed'].apply(convert_to_float)
    merged_rf['QuantityUsed'] = merged_rf['QuantityUsed'].fillna(merged_rf['QuantityUsed'].median())
    
    product_consumption_rf = merged_rf.groupby('FK_product')['QuantityUsed'].sum().reset_index()
    product_consumption_rf.columns = ['FK_product', 'TotalConsumption']  # Correction ici
    
    threshold = product_consumption_rf['TotalConsumption'].median()
    product_consumption_rf['Label'] = product_consumption_rf['TotalConsumption'].apply(
        lambda x: 1 if x >= threshold else 0
    )
    
    dataset = product_consumption_rf.merge(merged_rf[['FK_product', 'Dosage', 'QuantityUsed', 'productname', 'category', 'Material_Category']], 
                                          on='FK_product', how='left')
    
    label_encoder_cat = LabelEncoder()
    dataset['category'] = label_encoder_cat.fit_transform(dataset['category'])
    
    label_encoder_mat = LabelEncoder()
    dataset['Material_Category'] = label_encoder_mat.fit_transform(dataset['Material_Category'])
    
    X = dataset[['Dosage', 'QuantityUsed', 'category', 'Material_Category']]
    y = dataset['Label']
    
    X = X.fillna(X.median())
    
    scaler_rf_prod = StandardScaler()
    X_scaled = scaler_rf_prod.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    rf_prod = RandomForestClassifier(random_state=42)
    rf_prod.fit(X_train, y_train)
    
    dataset['Consommation'] = dataset['Label'].map({1: "Forte consommation", 0: "Faible consommation"})
    joblib.dump(rf_prod, 'rf_material_consumption.pkl')
    joblib.dump(scaler_rf_prod, 'scaler_material_consumption.pkl')
    joblib.dump(label_encoder_cat, 'label_encoder_category.pkl')
    joblib.dump(label_encoder_mat, 'label_encoder_material_category.pkl')
    joblib.dump(dataset, 'material_consumption_dataset.pkl')
    print("Material Consumption Classification model trained.")

    return (scaler, kmeans, rf, knn_model, pivot_table, predictions_df, historical_df, 
            scaler_prod, kmeans_prod, cluster_labels, product_consumption, 
            scaler_rf_prod, rf_prod, label_encoder_cat, label_encoder_mat, dataset)

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
    elif axis_name.lower() == 'production':
        models = [
            {'id': 'clustering_production', 'name': 'Clustering Production'},
            {'id': 'material_consumption_classification', 'name': 'Material Consumption Classification'}
        ]
        return render_template('models.html', axis=axis_name, models=models)
    elif axis_name.lower() == 'stock':
        models = [
            {'id': 'stock_prediction', 'name': 'Stock Prediction'},
            {'id': 'stock_classification', 'name': 'Stock Classification'}
        ]
        return render_template('models.html', axis=axis_name, models=models)
    else:
        return render_template('index.html', error="Axe non valide")

@app.route('/test_model/sales/clustering_sales', methods=['GET', 'POST'])
def test_clustering_sales():
    if request.method == 'POST':
        try:
            # Charger ou entraîner le modèle
            if not os.path.exists('scaler_sales_clustering_sales.pkl') or not os.path.exists('clustering_sales.pkl'):
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
            else:
                print("Loading Clustering Sales model...")
                scaler = joblib.load('scaler_sales_clustering_sales.pkl')
                kmeans = joblib.load('clustering_sales.pkl')

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
            # Charger ou entraîner le modèle
            if not os.path.exists('rf_sales_demand.pkl'):
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
            else:
                print("Loading Demand Prediction Sales model...")
                rf = joblib.load('rf_sales_demand.pkl')

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
                                 month=month,
                                 shops=SHOPS)
        except Exception as e:
            return render_template('test_model.html', 
                                 axis='sales', 
                                 model_name='Demand Prediction Sales', 
                                 model_id='demand_prediction_sales', 
                                 fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'], 
                                 prediction=f'Erreur: {str(e)}',
                                 shops=SHOPS)
    return render_template('test_model.html', 
                         axis='sales', 
                         model_name='Demand Prediction Sales', 
                         model_id='demand_prediction_sales', 
                         fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'],
                         shops=SHOPS)

@app.route('/test_model/sales/product_recommendation_sales', methods=['GET', 'POST'])
def test_product_recommendation_sales():
    if request.method == 'POST':
        try:
            # Charger ou entraîner le modèle
            if not os.path.exists('knn_sales_recommendation.pkl') or not os.path.exists('pivot_table_sales_recommendation.pkl'):
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
            else:
                print("Loading Product Recommendation Sales model...")
                knn_model = joblib.load('knn_sales_recommendation.pkl')
                pivot_table = joblib.load('pivot_table_sales_recommendation.pkl')

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

@app.route('/test_model/sales/time_series_forecast_sales', methods=['GET'])
def test_time_series_forecast_sales():
    try:
        # Charger ou entraîner le modèle
        if not os.path.exists('sarima_forecast_sales.pkl') or not os.path.exists('sarima_historical_sales.pkl'):
            print("Training Time Series Forecast Sales model...")
            predictions_df, historical_df, sarima_error = train_sarima_model()
            if sarima_error:
                return render_template('test_model.html',
                                     axis='sales',
                                     model_name='Time Series Forecast Sales',
                                     model_id='time_series_forecast_sales',
                                     fields=[],
                                     prediction=f"Erreur : {sarima_error}")
        else:
            print("Loading Time Series Forecast Sales model...")
            predictions_df = joblib.load('sarima_forecast_sales.pkl')
            historical_df = joblib.load('sarima_historical_sales.pkl')

        if predictions_df is None or historical_df is None:
            return render_template('test_model.html',
                                 axis='sales',
                                 model_name='Time Series Forecast Sales',
                                 model_id='time_series_forecast_sales',
                                 fields=[],
                                 prediction="Erreur : Les données de prévision ou historiques ne sont pas disponibles.")
        
        curve_plot, hist_plot, seasonal_plot = generate_sarima_plots(predictions_df, historical_df)
        
        return render_template('test_model.html',
                             axis='sales',
                             model_name='Time Series Forecast Sales',
                             model_id='time_series_forecast_sales',
                             fields=[],
                             curve_plot=curve_plot,
                             hist_plot=hist_plot,
                             seasonal_plot=seasonal_plot)
    except Exception as e:
        return render_template('test_model.html',
                             axis='sales',
                             model_name='Time Series Forecast Sales',
                             model_id='time_series_forecast_sales',
                             fields=[],
                             prediction=f"Erreur : {str(e)}")

@app.route('/test_model/production/clustering_production', methods=['GET', 'POST'])
def test_clustering_production():
    if request.method == 'POST':
        try:
            # Charger ou entraîner le modèle
            if not all(os.path.exists(f) for f in ['scaler_production_clustering.pkl', 'clustering_production.pkl', 'cluster_labels_production.pkl', 'product_consumption.pkl']):
                print("Training Clustering Production model...")
                query_kmeans_prod = """
                SELECT 
                    f."FK_product",
                    f."FK_base_material",
                    f."QuantityUsed",
                    p."PK_Products",
                    p."productname",
                    m."PK_Base_Materials",
                    m."Material_Name"
                FROM "Fact_Production" f
                JOIN "Dim_Cosmetic_Products" p ON f."FK_product" = p."PK_Products"
                JOIN "Dim_Base_Materials" m ON f."FK_base_material" = m."PK_Base_Materials"
                """
                merged = pd.read_sql(query_kmeans_prod, engine) if engine else pd.DataFrame({
                    'FK_product': [1, 2, 1, 3, 2],
                    'FK_base_material': [101, 102, 101, 103, 102],
                    'QuantityUsed': ['[10.0]', '[20.0]', '[15.0]', '[25.0]', '[30.0]'],
                    'PK_Products': [1, 2, 1, 3, 2],
                    'productname': ['ProductA', 'ProductB', 'ProductA', 'ProductC', 'ProductB'],
                    'PK_Base_Materials': [101, 102, 101, 103, 102],
                    'Material_Name': ['Material1', 'Material2', 'Material1', 'Material3', 'Material2']
                })

                merged['QuantityUsed'] = merged['QuantityUsed'].apply(convert_to_float)
                merged['QuantityUsed'] = merged['QuantityUsed'].fillna(merged['QuantityUsed'].median())
                product_consumption = merged.groupby(['FK_product', 'productname'])['QuantityUsed'].sum().reset_index()
                product_consumption.columns = ['ProductID', 'ProductName', 'TotalConsumption']
                
                scaler_prod = MinMaxScaler()
                product_consumption['TotalConsumption_scaled'] = scaler_prod.fit_transform(product_consumption[['TotalConsumption']])
                kmeans_prod = KMeans(n_clusters=2, random_state=42)
                kmeans_prod.fit(product_consumption[['TotalConsumption_scaled']])
                
                product_consumption['Cluster'] = kmeans_prod.labels_
                cluster_means = product_consumption.groupby('Cluster')['TotalConsumption'].mean()
                low_cluster = cluster_means.idxmin()
                cluster_labels = {low_cluster: "Faible consommation", 1-low_cluster: "Forte consommation"}
                product_consumption['Consommation'] = product_consumption['Cluster'].map(cluster_labels)
                
                joblib.dump(scaler_prod, 'scaler_production_clustering.pkl')
                joblib.dump(kmeans_prod, 'clustering_production.pkl')
                joblib.dump(cluster_labels, 'cluster_labels_production.pkl')
                joblib.dump(product_consumption, 'product_consumption.pkl')
                print("Clustering Production model trained.")
            else:
                print("Loading Clustering Production model...")
                scaler_prod = joblib.load('scaler_production_clustering.pkl')
                kmeans_prod = joblib.load('clustering_production.pkl')
                cluster_labels = joblib.load('cluster_labels_production.pkl')
                product_consumption = joblib.load('product_consumption.pkl')

            # Générer les visualisations
            cluster_counts = product_consumption['Consommation'].value_counts().to_dict()
            cluster_counts_html = '<table class="recommendation-table">'
            cluster_counts_html += '<thead><tr><th>Consommation</th><th>Nombre de produits</th></tr></thead>'
            cluster_counts_html += '<tbody>'
            for cons, count in cluster_counts.items():
                cluster_counts_html += f'<tr><td>{cons}</td><td>{count}</td></tr>'
            cluster_counts_html += '</tbody></table>'

            preview_data = product_consumption[['ProductName', 'TotalConsumption', 'Consommation']].head(5)
            preview_html = '<table class="recommendation-table">'
            preview_html += '<thead><tr><th>ProductName</th><th>TotalConsumption</th><th>Consommation</th></tr></thead>'
            preview_html += '<tbody>'
            for _, row in preview_data.iterrows():
                preview_html += f'<tr><td>{row["ProductName"]}</td><td>{row["TotalConsumption"]:.1f}</td><td>{row["Consommation"]}</td></tr>'
            preview_html += '</tbody></table>'

            product_name = request.form['product_name']
            quantity_used = float(request.form['quantity_used'])
            input_data = pd.DataFrame([[quantity_used]], columns=['TotalConsumption'])
            input_scaled = scaler_prod.transform(input_data)
            cluster = kmeans_prod.predict(input_scaled)[0]
            cluster_label = cluster_labels.get(cluster, f"Cluster {cluster}")
            prediction = f'Produit: {product_name}, Consommation: {cluster_label} pour QuantityUsed = {quantity_used}'
            return render_template('test_model.html',
                                 axis='production',
                                 model_name='Clustering Production',
                                 model_id='clustering_production',
                                 fields=['product_name', 'quantity_used'],
                                 product_names=PRODUCT_NAMES,
                                 cluster_counts=cluster_counts_html,
                                 preview=preview_html,
                                 prediction=prediction,
                                 product_name=product_name,
                                 quantity_used=quantity_used)
        except Exception as e:
            return render_template('test_model.html',
                                 axis='production',
                                 model_name='Clustering Production',
                                 model_id='clustering_production',
                                 fields=['product_name', 'quantity_used'],
                                 product_names=PRODUCT_NAMES,
                                 prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html',
                         axis='production',
                         model_name='Clustering Production',
                         model_id='clustering_production',
                         fields=['product_name', 'quantity_used'],
                         product_names=PRODUCT_NAMES)

@app.route('/test_model/production/material_consumption_classification', methods=['GET', 'POST'])
def test_material_consumption_classification():
    try:
        # Charger ou entraîner le modèle
        if not all(os.path.exists(f) for f in ['rf_material_consumption.pkl', 'scaler_material_consumption.pkl', 'label_encoder_category.pkl', 'label_encoder_material_category.pkl', 'material_consumption_dataset.pkl']):
            print("Training Material Consumption Classification model...")
            query_rf_prod = """
            SELECT 
                f."FK_product",
                f."FK_base_material",
                f."QuantityUsed",
                f."Dosage",
                p."PK_Products",
                p."productname",
                p."category",
                m."PK_Base_Materials",
                m."Material_Name",
                m."Material_Category"
            FROM "Fact_Production" f
            JOIN "Dim_Cosmetic_Products" p ON f."FK_product" = p."PK_Products"
            JOIN "Dim_Base_Materials" m ON f."FK_base_material" = m."PK_Base_Materials"
            """
            merged_rf = pd.read_sql(query_rf_prod, engine) if engine else pd.DataFrame({
                'FK_product': [1, 2, 1, 3, 2],
                'FK_base_material': [101, 102, 101, 103, 102],
                'QuantityUsed': ['[10.0]', '[20.0]', '[15.0]', '[25.0]', '[30.0]'],
                'Dosage': [0.1, 0.2, 0.15, 0.25, 0.3],
                'PK_Products': [1, 2, 1, 3, 2],
                'productname': ['ProductA', 'ProductB', 'ProductA', 'ProductC', 'ProductB'],
                'category': ['Hair Care', 'Body Care', 'Hair Care', 'Foot Care', 'Body Care'],
                'PK_Base_Materials': [101, 102, 101, 103, 102],
                'Material_Name': ['Material1', 'Material2', 'Material1', 'Materialsob', 'Material2'],
                'Material_Category': ['Oil', 'Cream', 'Oil', 'Powder', 'Cream']
            })

            merged_rf['QuantityUsed'] = merged_rf['QuantityUsed'].apply(convert_to_float)
            merged_rf['QuantityUsed'] = merged_rf['QuantityUsed'].fillna(merged_rf['QuantityUsed'].median())
            
            product_consumption_rf = merged_rf.groupby('FK_product')['QuantityUsed'].sum().reset_index()
            product_consumption_rf.columns = ['FK_product', 'TotalConsumption']
            
            threshold = product_consumption_rf['TotalConsumption'].median()
            product_consumption_rf['Label'] = product_consumption_rf['TotalConsumption'].apply(
                lambda x: 1 if x >= threshold else 0
            )
            
            dataset = product_consumption_rf.merge(merged_rf[['FK_product', 'Dosage', 'QuantityUsed', 'productname', 'category', 'Material_Category']], 
                                                  on='FK_product', how='left')
            
            label_encoder_cat = LabelEncoder()
            dataset['category'] = label_encoder_cat.fit_transform(dataset['category'])
            
            label_encoder_mat = LabelEncoder()
            dataset['Material_Category'] = label_encoder_mat.fit_transform(dataset['Material_Category'])
            
            X = dataset[['Dosage', 'QuantityUsed', 'category', 'Material_Category']]
            y = dataset['Label']
            
            X = X.fillna(X.median())
            
            scaler_rf_prod = StandardScaler()
            X_scaled = scaler_rf_prod.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            
            rf_prod = RandomForestClassifier(random_state=42)
            rf_prod.fit(X_train, y_train)
            
            dataset['Consommation'] = dataset['Label'].map({1: "Forte consommation", 0: "Faible consommation"})
            joblib.dump(rf_prod, 'rf_material_consumption.pkl')
            joblib.dump(scaler_rf_prod, 'scaler_material_consumption.pkl')
            joblib.dump(label_encoder_cat, 'label_encoder_category.pkl')
            joblib.dump(label_encoder_mat, 'label_encoder_material_category.pkl')
            joblib.dump(dataset, 'material_consumption_dataset.pkl')
            print("Material Consumption Classification model trained.")
        else:
            print("Loading Material Consumption Classification model...")
            rf_prod = joblib.load('rf_material_consumption.pkl')
            scaler_rf_prod = joblib.load('scaler_material_consumption.pkl')
            label_encoder_cat = joblib.load('label_encoder_category.pkl')
            label_encoder_mat = joblib.load('label_encoder_material_category.pkl')
            dataset = joblib.load('material_consumption_dataset.pkl')

        # Préparer un tableau des prédictions pour tous les produits
        dataset_with_predictions = dataset.copy()
        X = dataset_with_predictions[['Dosage', 'QuantityUsed', 'category', 'Material_Category']]
        X_scaled = scaler_rf_prod.transform(X)
        dataset_with_predictions['Predicted_Label'] = rf_prod.predict(X_scaled)
        dataset_with_predictions['Consommation'] = dataset_with_predictions['Predicted_Label'].map({1: "Forte consommation", 0: "Faible consommation"})
        
        preview_columns = ['TotalConsumption', 'Consommation']
        if 'productname' in dataset_with_predictions.columns:
            preview_columns.insert(0, 'productname')
        else:
            preview_columns.insert(0, 'FK_product')
        
        preview_data = dataset_with_predictions[preview_columns].head(5)
        preview_html = '<table class="recommendation-table">'
        preview_html += '<thead><tr>'
        for col in preview_columns:
            preview_html += f'<th>{col.replace("_", " ").title()}</th>'
        preview_html += '</tr></thead>'
        preview_html += '<tbody>'
        for _, row in preview_data.iterrows():
            preview_html += '<tr>'
            for col in preview_columns:
                value = row[col]
                if col == 'TotalConsumption':
                    value = f"{value:.1f}"
                preview_html += f'<td>{value}</td>'
            preview_html += '</tr>'
        preview_html += '</tbody></table>'

        if request.method == 'POST':
            product_name = request.form['product_name']
            product_id = product_name.split()[-1] if ' ' in product_name else product_name
            
            if 'productname' in dataset.columns:
                product_data = dataset[dataset['productname'] == product_name]
            else:
                product_data = dataset[dataset['FK_product'].astype(str) == str(product_id)]
            
            if product_data.empty:
                return render_template('test_model.html',
                                     axis='production',
                                     model_name='Material Consumption Classification',
                                     model_id='material_consumption_classification',
                                     fields=['product_name'],
                                     product_names=PRODUCT_NAMES,
                                     preview=preview_html,
                                     prediction=f'Erreur: Produit {product_name} non trouvé dans les données.')
            
            input_data = product_data[['Dosage', 'QuantityUsed', 'category', 'Material_Category']].iloc[0:1]
            input_scaled = scaler_rf_prod.transform(input_data)
            
            prediction = rf_prod.predict(input_scaled)[0]
            prediction_label = "Forte consommation" if prediction == 1 else "Faible consommation"
            
            return render_template('test_model.html',
                                 axis='production',
                                 model_name='Material Consumption Classification',
                                 model_id='material_consumption_classification',
                                 fields=['product_name'],
                                 product_names=PRODUCT_NAMES,
                                 preview=preview_html,
                                 prediction=f'Produit: {product_name}, Consommation: {prediction_label}',
                                 product_name=product_name)
    except Exception as e:
        return render_template('test_model.html',
                             axis='production',
                             model_name='Material Consumption Classification',
                             model_id='material_consumption_classification',
                             fields=['product_name'],
                             product_names=PRODUCT_NAMES,
                             preview=preview_html if 'preview_html' in locals() else '',
                             prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html',
                         axis='production',
                         model_name='Material Consumption Classification',
                         model_id='material_consumption_classification',
                         fields=['product_name'],
                         product_names=PRODUCT_NAMES,
                         preview=preview_html if 'preview_html' in locals() else '')

@app.route('/test_model/stock/stock_prediction', methods=['GET', 'POST'])
def test_stock_prediction():
    if request.method == 'POST':
        try:
            model = None
            le_category = None
            le_brand = None
            le_warehouse = None
            df = None
            if not os.path.exists('stock_prediction_model.pkl'):
                print("Training stock prediction model...")
                query = """
                SELECT 
                    fs."FK_product",
                    fs."FK_warehouse",
                    AVG(fs."Quantity") AS stock_quantity,
                    AVG(fs."Capacity") AS stock_capacity,
                    SUM(s.quantity) AS sales_quantity,
                    AVG(s.unit_price) AS unit_price,
                    SUM(s.total) AS sales_total,
                    cp."category" AS product_category,
                    cp."brandname" AS product_brand,
                    w."warehousename",
                    d.year,
                    d.month
                FROM fact_stock fs
                LEFT JOIN "Fact_sales" s ON fs."FK_product" = s."FK_product"
                LEFT JOIN "Dim_Cosmetic_Products" cp ON fs."FK_product" = cp."PK_Products"
                LEFT JOIN "Dim_Warehouse" w ON fs."FK_warehouse" = w."PK_Warehouse"
                LEFT JOIN dim_date d ON s."FK_date" = d.pk_dates
                GROUP BY fs."FK_product", fs."FK_warehouse", cp."category", cp."brandname", w."warehousename", d.year, d.month
                """
                df = pd.read_sql(query, engine)
                le_category = LabelEncoder()
                le_brand = LabelEncoder()
                le_warehouse = LabelEncoder()

                df['product_category_encoded'] = le_category.fit_transform(df['product_category'])
                df['product_brand_encoded'] = le_brand.fit_transform(df['product_brand'])
                df['warehousename_encoded'] = le_warehouse.fit_transform(df['warehousename'])

                features = ['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                            'product_category_encoded', 'product_brand_encoded', 'warehousename_encoded']
                target = 'stock_quantity'

                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)

                joblib.dump(model, 'stock_prediction_model.pkl')
                joblib.dump(le_category, 'le_category.pkl')
                joblib.dump(le_brand, 'le_brand.pkl')
                joblib.dump(le_warehouse, 'le_warehouse.pkl')
                print("Model and LabelEncoders saved.")
            else:
                print("Loading existing stock prediction model...")
                model = joblib.load('stock_prediction_model.pkl')
                le_category = joblib.load('le_category.pkl')
                le_brand = joblib.load('le_brand.pkl')
                le_warehouse = joblib.load('le_warehouse.pkl')

            stock_capacity = float(request.form['stock_capacity'])
            sales_quantity = float(request.form['sales_quantity'])
            unit_price = float(request.form['unit_price'])
            sales_total = float(request.form['sales_total'])
            product_category = request.form['product_category']
            product_brand = request.form['product_brand']
            warehousename = request.form['warehousename']

            product_category_encoded = le_category.transform([product_category])[0]
            product_brand_encoded = le_brand.transform([product_brand])[0]
            warehousename_encoded = le_warehouse.transform([warehousename])[0]

            input_data = pd.DataFrame([[stock_capacity, sales_quantity, unit_price, sales_total,
                                      product_category_encoded, product_brand_encoded, warehousename_encoded]],
                                    columns=['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                                             'product_category_encoded', 'product_brand_encoded', 'warehousename_encoded'])

            prediction = model.predict(input_data)[0]
            return render_template('test_model.html',
                                 axis='stock',
                                 model_name='Stock Prediction',
                                 model_id='stock_prediction',
                                 fields=['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                                         'product_category', 'product_brand', 'warehousename'],
                                 prediction=f'Stock quantity predicted: {prediction:.2f}',
                                 stock_capacity=stock_capacity,
                                 sales_quantity=sales_quantity,
                                 unit_price=unit_price,
                                 sales_total=sales_total,
                                 product_category=product_category,
                                 product_brand=product_brand,
                                 warehousename=warehousename,
                                 categories=CATEGORIES,
                                 brands=BRANDS,
                                 warehouses=WAREHOUSES)
        except Exception as e:
            return render_template('test_model.html',
                                 axis='stock',
                                 model_name='Stock Prediction',
                                 model_id='stock_prediction',
                                 fields=['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                                         'product_category', 'product_brand', 'warehousename'],
                                 prediction=f'Erreur: {str(e)}',
                                 categories=CATEGORIES,
                                 brands=BRANDS,
                                 warehouses=WAREHOUSES)
    return render_template('test_model.html',
                         axis='stock',
                         model_name='Stock Prediction',
                         model_id='stock_prediction',
                         fields=['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                                 'product_category', 'product_brand', 'warehousename'],
                         categories=CATEGORIES,
                         brands=BRANDS,
                         warehouses=WAREHOUSES)

@app.route('/test_model/stock/stock_classification', methods=['GET', 'POST'])
def test_stock_classification():
    if request.method == 'POST':
        try:
            model_pipeline = None
            data = None
            if not os.path.exists('stock_classification_pipeline.pkl'):
                print("Training stock classification model...")
                query = """
                WITH sales_stats AS (
                    SELECT 
                        "FK_product",
                        AVG("quantity"::FLOAT) as avg_daily_sales,
                        COALESCE(STDDEV("quantity"::FLOAT), 0) as sales_volatility,
                        COUNT(*) as transaction_count,
                        MAX(s.suppliername) as suppliername,
                        MAX(l.country) as supplier_country
                    FROM "Fact_sales" fs
                    LEFT JOIN "Dim_Suppliers" s ON fs."FK_supplier" = s.supplierid
                    LEFT JOIN "Dim_Location" l ON s."FK_location" = l."PK_location"
                    GROUP BY "FK_product"
                ),
                product_info AS (
                    SELECT 
                        p."PK_Products",
                        p.productid,
                        p.productname,
                        p.category,
                        p.brandname
                    FROM "Dim_Cosmetic_Products" p
                ),
                warehouse_info AS (
                    SELECT 
                        w."PK_Warehouse",
                        w.warehousename,
                        l.city as warehouse_city,
                        l.country as warehouse_country
                    FROM "Dim_Warehouse" w
                    JOIN "Dim_Location" l ON w."FK_location" = l."PK_location"
                )
                SELECT 
                    st."FK_product",
                    st."FK_warehouse",
                    pi.productname,
                    pi.category,
                    pi.brandname,
                    ss.suppliername,
                    ss.supplier_country,
                    wi.warehousename,
                    wi.warehouse_city,
                    wi.warehouse_country,
                    st."Quantity" as current_stock,
                    st."Capacity" as max_capacity,
                    (st."Capacity" - st."Quantity") as remaining_space,
                    COALESCE(ss.avg_daily_sales, 0) as avg_daily_sales,
                    COALESCE(ss.sales_volatility, 0) as sales_volatility,
                    COALESCE(ss.transaction_count, 0) as transaction_count,
                    CASE 
                        WHEN st."Quantity" / NULLIF(GREATEST(ss.avg_daily_sales, 0.001), 0) < 7 THEN 1 
                        ELSE 0 
                    END as rupture_risk
                FROM "fact_stock" st
                JOIN product_info pi ON st."FK_product" = pi."PK_Products"
                JOIN warehouse_info wi ON st."FK_warehouse" = wi."PK_Warehouse"
                LEFT JOIN sales_stats ss ON st."FK_product" = ss."FK_product"
                """
                data = pd.read_sql(query, engine)
                print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")

                X = data.drop(['rupture_risk', 'FK_product', 'FK_warehouse', 'productname', 
                              'warehousename', 'suppliername'], axis=1)
                y = data['rupture_risk']

                numeric_features = [
                    'current_stock', 'max_capacity', 'remaining_space', 
                    'avg_daily_sales', 'sales_volatility', 'transaction_count'
                ]
                categorical_features = [
                    'category', 'brandname', 'supplier_country', 
                    'warehouse_city', 'warehouse_country'
                ]

                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])

                model_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
                ])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model_pipeline.fit(X_train, y_train)

                joblib.dump(model_pipeline, 'stock_classification_pipeline.pkl')
                print("Stock classification model trained and saved.")
            else:
                print("Loading existing stock classification model...")
                model_pipeline = joblib.load('stock_classification_pipeline.pkl')
                query = """
                WITH sales_stats AS (
                    SELECT 
                        "FK_product",
                        AVG("quantity"::FLOAT) as avg_daily_sales,
                        COALESCE(STDDEV("quantity"::FLOAT), 0) as sales_volatility,
                        COUNT(*) as transaction_count,
                        MAX(s.suppliername) as suppliername,
                        MAX(l.country) as supplier_country
                    FROM "Fact_sales" fs
                    LEFT JOIN "Dim_Suppliers" s ON fs."FK_supplier" = s.supplierid
                    LEFT JOIN "Dim_Location" l ON s."FK_location" = l."PK_location"
                    GROUP BY "FK_product"
                ),
                product_info AS (
                    SELECT 
                        p."PK_Products",
                        p.productid,
                        p.productname,
                        p.category,
                        p.brandname
                    FROM "Dim_Cosmetic_Products" p
                ),
                warehouse_info AS (
                    SELECT 
                        w."PK_Warehouse",
                        w.warehousename,
                        l.city as warehouse_city,
                        l.country as warehouse_country
                    FROM "Dim_Warehouse" w
                    JOIN "Dim_Location" l ON w."FK_location" = l."PK_location"
                )
                SELECT 
                    st."FK_product",
                    st."FK_warehouse",
                    pi.productname,
                    pi.category,
                    pi.brandname,
                    ss.suppliername,
                    ss.supplier_country,
                    wi.warehousename,
                    wi.warehouse_city,
                    wi.warehouse_country,
                    st."Quantity" as current_stock,
                    st."Capacity" as max_capacity,
                    (st."Capacity" - st."Quantity") as remaining_space,
                    COALESCE(ss.avg_daily_sales, 0) as avg_daily_sales,
                    COALESCE(ss.sales_volatility, 0) as sales_volatility,
                    COALESCE(ss.transaction_count, 0) as transaction_count
                FROM "fact_stock" st
                JOIN product_info pi ON st."FK_product" = pi."PK_Products"
                JOIN warehouse_info wi ON st."FK_warehouse" = wi."PK_Warehouse"
                LEFT JOIN sales_stats ss ON st."FK_product" = ss."FK_product"
                """
                data = pd.read_sql(query, engine)

            current_stock = float(request.form['current_stock'])
            max_capacity = float(request.form['max_capacity'])
            avg_daily_sales = float(request.form['avg_daily_sales'])
            sales_volatility = float(request.form['sales_volatility'])
            transaction_count = float(request.form['transaction_count'])
            category = request.form['category']
            brandname = request.form['brandname']
            supplier_country = request.form['supplier_country']
            warehouse_city = request.form['warehouse_city']
            warehouse_country = request.form['warehouse_country']

            input_data = pd.DataFrame([{
                'current_stock': current_stock,
                'max_capacity': max_capacity,
                'remaining_space': max_capacity - current_stock,
                'avg_daily_sales': avg_daily_sales,
                'sales_volatility': sales_volatility,
                'transaction_count': transaction_count,
                'category': category,
                'brandname': brandname,
                'supplier_country': supplier_country,
                'warehouse_city': warehouse_city,
                'warehouse_country': warehouse_country
            }])

            prediction = model_pipeline.predict(input_data)[0]
            prediction_proba = model_pipeline.predict_proba(input_data)[0][1]
            prediction_label = "Risque de rupture" if prediction == 1 else "Pas de risque"

            X = data.drop(['FK_product', 'FK_warehouse', 'productname', 'warehousename', 'suppliername'], axis=1)
            y_pred = model_pipeline.predict(X)
            data['predicted_risk'] = y_pred
            data['risk_probability'] = model_pipeline.predict_proba(X)[:, 1]

            data['actual_risk'] = data.apply(lambda row: 1 if row['current_stock'] / max(row['avg_daily_sales'], 0.001) < 7 else 0, axis=1)
            cm = confusion_matrix(data['actual_risk'], data['predicted_risk'])
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            buffer1 = io.BytesIO()
            plt.savefig(buffer1, format='png')
            buffer1.seek(0)
            cm_plot = base64.b64encode(buffer1.getvalue()).decode('utf-8')
            buffer1.close()
            plt.close()

            high_risk = data.sort_values('risk_probability', ascending=False).head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=high_risk, x='risk_probability', y='productname')
            plt.title('Top 10 Produits à Risque de Rupture')
            plt.xlabel('Probabilité de Rupture')
            plt.ylabel('Nom du Produit')
            buffer2 = io.BytesIO()
            plt.savefig(buffer2, format='png')
            buffer2.seek(0)
            high_risk_plot = base64.b64encode(buffer2.getvalue()).decode('utf-8')
            buffer2.close()
            plt.close()

            return render_template('test_model.html',
                                 axis='stock',
                                 model_name='Stock Classification',
                                 model_id='stock_classification',
                                 fields=['current_stock', 'max_capacity', 'avg_daily_sales', 'sales_volatility',
                                         'transaction_count', 'category', 'brandname', 'supplier_country',
                                         'warehouse_city', 'warehouse_country'],
                                 prediction=f'Prédiction: {prediction_label} (Probabilité: {prediction_proba:.2f})',
                                 current_stock=current_stock,
                                 max_capacity=max_capacity,
                                 avg_daily_sales=avg_daily_sales,
                                 sales_volatility=sales_volatility,
                                 transaction_count=transaction_count,
                                 category=category,
                                 brandname=brandname,
                                 supplier_country=supplier_country,
                                 warehouse_city=warehouse_city,
                                 warehouse_country=warehouse_country,
                                 categories=CATEGORIES,
                                 brands=BRANDS,
                                 supplier_countries=SUPPLIER_COUNTRIES,
                                 warehouse_cities=WAREHOUSES,
                                 warehouse_countries=WAREHOUSE_COUNTRIES,
                                 cm_plot=cm_plot,
                                 high_risk_plot=high_risk_plot)
        except Exception as e:
            return render_template('test_model.html',
                                 axis='stock',
                                 model_name='Stock Classification',
                                 model_id='stock_classification',
                                 fields=['current_stock', 'max_capacity', 'avg_daily_sales', 'sales_volatility',
                                         'transaction_count', 'category', 'brandname', 'supplier_country',
                                         'warehouse_city', 'warehouse_country'],
                                 prediction=f'Erreur: {str(e)}',
                                 categories=CATEGORIES,
                                 brands=BRANDS,
                                 supplier_countries=SUPPLIER_COUNTRIES,
                                 warehouse_cities=WAREHOUSES,
                                 warehouse_countries=WAREHOUSE_COUNTRIES)
    return render_template('test_model.html',
                         axis='stock',
                         model_name='Stock Classification',
                         model_id='stock_classification',
                         fields=['current_stock', 'max_capacity', 'avg_daily_sales', 'sales_volatility',
                                 'transaction_count', 'category', 'brandname', 'supplier_country',
                                 'warehouse_city', 'warehouse_country'],
                         categories=CATEGORIES,
                         brands=BRANDS,
                         supplier_countries=SUPPLIER_COUNTRIES,
                         warehouse_cities=WAREHOUSES,
                         warehouse_countries=WAREHOUSE_COUNTRIES)

if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")