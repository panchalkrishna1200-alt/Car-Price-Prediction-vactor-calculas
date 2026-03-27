import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ========================================
# PAGE SETUP
# ========================================
st.set_page_config(page_title="Vector Calculus Project", layout="wide")
st.title("🚗 Car Price Prediction with Vector Calculus & AI")

# ========================================
# LOAD DATA
# ========================================
st.sidebar.header("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("car.csv")

df.columns = df.columns.str.strip()
st.success("Dataset Loaded")
st.dataframe(df.head())

# ========================================
# DATA CLEANING
# ========================================
st.header("🧹 Data Cleaning")

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

df = df[df['Price'].str.lower() != 'ask for price']
df['Price'] = df['Price'].str.replace(',', '').astype(int)

df['kms_driven'] = df['kms_driven'].str.replace('kms', '').str.replace(',', '').str.strip()
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
df['kms_driven'] = df['kms_driven'].fillna(df['kms_driven'].median()).astype(int)

df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['name'] = df['name'].apply(lambda x: ' '.join(str(x).split()[:3]))

st.write("Cleaned Data")
st.dataframe(df.head())

# ========================================
# VISUALIZATION
# ========================================
st.header("📊 Data Visualization")

fig, ax = plt.subplots()
sns.histplot(df['Price'], kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, ax=ax)
st.pyplot(fig)

# ========================================
# MODEL TRAINING
# ========================================
st.header("🤖 Model Training")

features = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
target = 'Price'

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

cat = ['name', 'company', 'fuel_type']
num = ['year', 'kms_driven']

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)
])

model_rf = Pipeline([
    ('prep', preprocessor),
    ('reg', RandomForestRegressor())
])

model_lr = Pipeline([
    ('prep', preprocessor),
    ('reg', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

pred_rf = model_rf.predict(X_test)
pred_lr = model_lr.predict(X_test)

st.write("Random Forest R²:", r2_score(y_test, pred_rf))
st.write("Linear Regression R²:", r2_score(y_test, pred_lr))

# ========================================
# GRADIENT DESCENT
# ========================================
st.header("📐 Gradient Descent")

def f(x, y):
    return x**2 + y**2

def grad(x, y):
    return np.array([2*x, 2*y])

x, y = 8, 8
lr = 0.1

px, py = [x], [y]

for i in range(20):
    g = grad(x, y)
    x -= lr * g[0]
    y -= lr * g[1]
    px.append(x)
    py.append(y)

xv = np.linspace(-10, 10, 100)
yv = np.linspace(-10, 10, 100)
Xg, Yg = np.meshgrid(xv, yv)
Zg = f(Xg, Yg)

fig, ax = plt.subplots()
ax.contour(Xg, Yg, Zg)
ax.plot(px, py, 'ro-')
st.pyplot(fig)

# ========================================
# VECTOR FIELD
# ========================================
st.header("🌊 Vector Field")

Xv, Yv = np.meshgrid(np.linspace(-5,5,10), np.linspace(-5,5,10))
U = -2*Xv
V = -2*Yv

fig, ax = plt.subplots()
ax.quiver(Xv, Yv, U, V)
st.pyplot(fig)

# ========================================
# DIVERGENCE
# ========================================
st.header("🔵 Divergence")

div = 2*Xv + 2*Yv

fig, ax = plt.subplots()
cont = ax.contourf(Xv, Yv, div)
plt.colorbar(cont)
st.pyplot(fig)

# ========================================
# CURL
# ========================================
st.header("🌀 Curl")

Uc = -Yv
Vc = Xv

fig, ax = plt.subplots()
ax.quiver(Xv, Yv, Uc, Vc)
st.pyplot(fig)

# ========================================
# LINE INTEGRAL
# ========================================
st.header("📏 Line Integral")

t = np.linspace(0, 2*np.pi, 100)
x_path = np.cos(t)
y_path = np.sin(t)

Fx = -y_path
Fy = x_path

dx = -np.sin(t)
dy = np.cos(t)

line_integral = np.sum(Fx*dx + Fy*dy)
st.write("Line Integral Value:", round(line_integral,2))

# ========================================
# MULTIPLE INTEGRAL
# ========================================
st.header("📦 Double Integral")

x = np.linspace(0, 2, 50)
y = np.linspace(0, 2, 50)

X, Y = np.meshgrid(x, y)
Z = X + Y

dx = dy = 0.04
integral = np.sum(Z)*dx*dy

st.write("Double Integral:", round(integral,2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
st.pyplot(fig)

# ========================================
# PREDICTION
# ========================================
st.header("🔮 Car Price Prediction")

company = st.selectbox("Company", df['company'].unique())
name = st.selectbox("Model", df[df['company']==company]['name'].unique())
year = st.slider("Year", 2005, 2024, 2019)
kms = st.number_input("KMs", 0, 500000, 50000)
fuel = st.selectbox("Fuel", df['fuel_type'].unique())

if st.button("Predict"):
    input_df = pd.DataFrame({
        'name':[name],
        'company':[company],
        'year':[year],
        'kms_driven':[kms],
        'fuel_type':[fuel]
    })
    
    price = model_rf.predict(input_df)[0]
    st.success(f"Predicted Price: ₹{price:,.2f}")

# ========================================
# END
# ========================================
st.markdown("---")
st.success("✅ Project Completed: All Vector Calculus Concepts Applied")
