import pandas as pd
import joblib
from flask import Flask, request, render_template # type: ignore
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

# Sales-related functions (unchanged)
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
    print(monthly_demand.head().to_string())
    print(f"Data types: {monthly_demand.dtypes}")
    print(f"Missing values: {monthly_demand.isnull().sum()}")
    
    monthly_demand['month'] = monthly_demand['month'].dt.to_timestamp()
    monthly_demand['month_of_year'] = monthly_demand['month'].dt.month
    
    all_predictions = []
    historical_data = []
    product_ids = monthly_demand['productid'].unique()[:100]
    print(f"Number of unique product IDs to process: {len(product_ids)}")
    for i, product_id in enumerate(product_ids):
        print(f"Processing product ID {product_id} ({i+1}/{len(product_ids)})")
        product_data = monthly_demand[monthly_demand['productid'] == product_id].copy()
        print(f"Data for product ID {product_id}:\n{product_data[['month', 'quantity']].to_string()}")
        
        if product_data['month'].duplicated().any():
            print(f"Duplicate months found for product ID {product_id}, aggregating data...")
            product_data = product_data.groupby('month')['quantity'].sum().reset_index()
        
        if len(product_data) < 12:
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
        
        if product_data_ts.nunique() == 1:
            print(f"Skipping product ID {product_id} due to no variation in data (all values are {product_data_ts.iloc[0]})")
            continue
        
        print(f"Fitting SARIMA model for product_id {product_id}...")
        
        try:
            try:
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(1, 1, 0, 12))
                model_fit = model.fit(disp=False)
            except Exception as e:
                print(f"Seasonal SARIMA failed for product ID {product_id}: {e}, trying non-seasonal ARIMA...")
                model = SARIMAX(product_data_ts, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
                model_fit = model.fit(disp=False)
            
            print(f"SARIMA model fitted for product ID {product_id}")
            
            forecast_steps = 20
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
            print(f"Error fitting SARIMA model for product_id {product_id}: {e}")
            traceback.print_exc()
            continue
    
    if not all_predictions:
        return None, None, "Erreur : Aucune prédiction n'a pu être générée (pas assez de données pour les produits)."
    
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

    merged['QuantityUsed'] = merged['QuantityUsed'].apply(convert_to_float)
    merged['QuantityUsed'] = merged['QuantityUsed'].fillna(merged['QuantityUsed'].median())
    product_consumption = merged.groupby(['FK_product', 'productname'])['QuantityUsed'].sum().reset_index()
    product_consumption.columns = ['ProductID', 'ProductName', 'TotalConsumption']
    
    scaler_prod = MinMaxScaler()
    product_consumption['TotalConsumption_scaled'] = scaler_prod.fit_transform(product_consumption[['TotalConsumption']])
    kmeans_prod = KMeans(n_clusters=2, random_state=42)
    kmeans_prod.fit(product_consumption[['TotalConsumption_scaled']])
    
    # Déterminer les étiquettes des clusters
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
    product_consumption_rf.columns = ['ProductID', 'TotalConsumption']
    
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
    'material_consumption_dataset.pkl'
]
print("Checking for .pkl files...")
if not all(os.path.exists(pkl) for pkl in pkl_files):
    print("Some .pkl files are missing. Training models...")
    (scaler, kmeans, rf, knn_model, pivot_table, predictions_df, historical_df, 
     scaler_prod, kmeans_prod, cluster_labels, product_consumption, 
     scaler_rf_prod, rf_prod, label_encoder_cat, label_encoder_mat, material_consumption_dataset) = train_and_save_models()
else:
    print("Loading .pkl files...")
    scaler = joblib.load('scaler_sales_clustering_sales.pkl')
    kmeans = joblib.load('clustering_sales.pkl')
    rf = joblib.load('rf_sales_demand.pkl')
    knn_model = joblib.load('knn_sales_recommendation.pkl')
    pivot_table = joblib.load('pivot_table_sales_recommendation.pkl')
    predictions_df = joblib.load('sarima_forecast_sales.pkl')
    historical_df = joblib.load('sarima_historical_sales.pkl')
    scaler_prod = joblib.load('scaler_production_clustering.pkl')
    kmeans_prod = joblib.load('clustering_production.pkl')
    cluster_labels = joblib.load('cluster_labels_production.pkl')
    product_consumption = joblib.load('product_consumption.pkl')
    scaler_rf_prod = joblib.load('scaler_material_consumption.pkl')
    rf_prod = joblib.load('rf_material_consumption.pkl')
    label_encoder_cat = joblib.load('label_encoder_category.pkl')
    label_encoder_mat = joblib.load('label_encoder_material_category.pkl')
    material_consumption_dataset = joblib.load('material_consumption_dataset.pkl')
print(".pkl files processed.")

SHOPS = ['41', '61', '81']

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
            {'id': 'stock_prediction', 'name': 'Stock Prediction'}
        ]
        return render_template('models.html', axis=axis_name, models=models)
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
                         fields=['unit_price', 'total', 'FK_shop', 'FK_product', 'month'],
                         shops=SHOPS)

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

@app.route('/test_model/sales/time_series_forecast_sales', methods=['GET'])
def test_time_series_forecast_sales():
    try:
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

    if request.method == 'POST':
        try:
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
                                 cluster_counts=cluster_counts_html,
                                 preview=preview_html,
                                 prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html',
                         axis='production',
                         model_name='Clustering Production',
                         model_id='clustering_production',
                         fields=['product_name', 'quantity_used'],
                         product_names=PRODUCT_NAMES,
                         cluster_counts=cluster_counts_html,
                         preview=preview_html)

<<<<<<< HEAD
=======
@app.route('/predict/production/clustering_production', methods=['POST'])
def predict_clustering_production():
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

    try:
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
                             cluster_counts=cluster_counts_html,
                             preview=preview_html,
                             prediction=f'Erreur: {str(e)}')

>>>>>>> ace9e64209c78a540539c30511d5e56ec11afeb7
@app.route('/test_model/production/material_consumption_classification', methods=['GET', 'POST'])
def test_material_consumption_classification():
    # Préparer un tableau des prédictions pour tous les produits
    dataset_with_predictions = material_consumption_dataset.copy()
    X = dataset_with_predictions[['Dosage', 'QuantityUsed', 'category', 'Material_Category']]
    X_scaled = scaler_rf_prod.transform(X)
    dataset_with_predictions['Predicted_Label'] = rf_prod.predict(X_scaled)
    dataset_with_predictions['Consommation'] = dataset_with_predictions['Predicted_Label'].map({1: "Forte consommation", 0: "Faible consommation"})
    
    # Vérifier si 'productname' existe dans le DataFrame, sinon utiliser 'FK_product' comme fallback
    preview_columns = ['TotalConsumption', 'Consommation']
    if 'productname' in dataset_with_predictions.columns:
        preview_columns.insert(0, 'productname')
    else:
        preview_columns.insert(0, 'FK_product')
        print("Warning: 'productname' not found in dataset_with_predictions, using 'FK_product' instead.")
    
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
        try:
            product_name = request.form['product_name']
            # Extraire l'ID du produit si le format est "Nom ID" ou utiliser directement comme ID
            product_id = product_name.split()[-1] if ' ' in product_name else product_name
            
            # Filtrer les données en utilisant 'FK_product' ou 'productname' selon disponibilité
            if 'productname' in material_consumption_dataset.columns:
                product_data = material_consumption_dataset[material_consumption_dataset['productname'] == product_name]
            else:
                product_data = material_consumption_dataset[material_consumption_dataset['FK_product'].astype(str) == str(product_id)]
            
            if product_data.empty:
                return render_template('test_model.html',
                                     axis='production',
                                     model_name='Material Consumption Classification',
                                     model_id='material_consumption_classification',
                                     fields=['product_name'],
                                     product_names=PRODUCT_NAMES,
                                     preview=preview_html,
                                     prediction=f'Erreur: Produit {product_name} non trouvé dans les données.')
            
            # Extraire les features pour la prédiction
            input_data = product_data[['Dosage', 'QuantityUsed', 'category', 'Material_Category']].iloc[0:1]
            input_scaled = scaler_rf_prod.transform(input_data)
            
            # Faire la prédiction
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
                                 preview=preview_html,
                                 prediction=f'Erreur: {str(e)}')
    return render_template('test_model.html',
                         axis='production',
                         model_name='Material Consumption Classification',
                         model_id='material_consumption_classification',
                         fields=['product_name'],
                         product_names=PRODUCT_NAMES,
                         preview=preview_html)

<<<<<<< HEAD
@app.route('/test_model/stock/stock_prediction', methods=['GET', 'POST'])
def test_stock_prediction():
    model = None
    le_category = None
    le_brand = None
    le_warehouse = None
    df = None

    if request.method == 'POST':
        try:
            # Charger ou entraîner le modèle uniquement lors de la requête POST
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
                print("Data loaded successfully. First few rows:")
                print(df.head())

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

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Mean Squared Error: {mse:.2f}")
                print(f"R² Score: {r2:.2f}")

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
                    w."warehousename"
                FROM fact_stock fs
                LEFT JOIN "Fact_sales" s ON fs."FK_product" = s."FK_product"
                LEFT JOIN "Dim_Cosmetic_Products" cp ON fs."FK_product" = cp."PK_Products"
                LEFT JOIN "Dim_Warehouse" w ON fs."FK_warehouse" = w."PK_Warehouse"
                GROUP BY fs."FK_product", fs."FK_warehouse", cp."category", cp."brandname", w."warehousename"
                """
                df = pd.read_sql(query, engine)

            # Préparer les données d'entrée à partir du formulaire
            stock_capacity = float(request.form['stock_capacity'])
            sales_quantity = float(request.form['sales_quantity'])
            unit_price = float(request.form['unit_price'])
            sales_total = float(request.form['sales_total'])
            product_category = request.form['product_category']
            product_brand = request.form['product_brand']
            warehousename = request.form['warehousename']

            # Encoder les variables catégoriques
            product_category_encoded = le_category.transform([product_category])[0]
            product_brand_encoded = le_brand.transform([product_brand])[0]
            warehousename_encoded = le_warehouse.transform([warehousename])[0]

            input_data = pd.DataFrame([[stock_capacity, sales_quantity, unit_price, sales_total,
                                      product_category_encoded, product_brand_encoded, warehousename_encoded]],
                                    columns=['stock_capacity', 'sales_quantity', 'unit_price', 'sales_total',
                                             'product_category_encoded', 'product_brand_encoded', 'warehousename_encoded'])

            # Faire la prédiction
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

=======
>>>>>>> ace9e64209c78a540539c30511d5e56ec11afeb7
if __name__ == '__main__':
    print("Starting Flask server...")
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")