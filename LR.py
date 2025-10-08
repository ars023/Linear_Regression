import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="Linear Regression Model", layout="wide")
st.title("Linear Regression Analysis")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

# Initialize variables
X = None
y = None
data_loaded = False

# Check if file is uploaded and process data
if uploaded_file is not None:
    try:
        data = np.genfromtxt(uploaded_file, delimiter=",", skip_header=1)
        
        # Check if data has at least 2 columns
        if data.ndim == 1 or data.shape[1] < 2:
            st.error("File must have at least 2 columns (X and y)")
        else:
            X = data[:, 0].reshape(-1, 1)
            y = data[:, 1]
            
            # Check for NaN values
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                st.error("Data contains NaN values. Please check your file format and ensure all values are numeric.")
            else:
                data_loaded = True
                st.success(f"Data loaded successfully! Shape: {X.shape[0]} samples, 2 features")
                
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.info("Make sure your file is a CSV with numeric values separated by commas.")
else:
    # Generate example synthetic data if no file is uploaded
    st.info("No file uploaded. Using synthetic data for demonstration.")
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 2 * X.flatten() + 1 + np.random.randn(100) * 2
    data_loaded = True

# Custom Linear Regression Class
class MyLinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        # Initialize parameters
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y
        
        # Gradient descent
        for i in range(self.n_iters):
            self.update_weights()
    
    def update_weights(self):
        y_pred = self.predict(self.X)
        
        # Calculate gradients
        dw = (2/self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (2/self.m) * np.sum(y_pred - self.y)
        
        # Update weights
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
    
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.w) + self.b

# Only proceed if we have valid data
if data_loaded and X is not None and y is not None:
    
    # Sidebar parameters
    st.sidebar.header("Model Parameters")
    lr = st.sidebar.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
    iters = st.sidebar.number_input("Number of Iterations", min_value=100, max_value=10000, value=1000, step=100)
    
    # Initialize and train models
    custom_model = MyLinearRegression(learning_rate=lr, n_iters=iters)
    custom_model.fit(X, y)
    y_pred_custom = custom_model.predict(X)
    
    # Calculate metrics for custom model
    custom_mse = mean_squared_error(y, y_pred_custom)
    custom_r2 = r2_score(y, y_pred_custom)
    
    # Split data for sklearn model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train sklearn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    y_pred_sklearn_full = sklearn_model.predict(X)
    
    # Calculate metrics for sklearn model
    sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
    sklearn_r2 = r2_score(y_test, y_pred_sklearn)
    
    # Retrain button
    if st.sidebar.button("Re-train Model"):
        custom_model = MyLinearRegression(learning_rate=lr, n_iters=iters)
        custom_model.fit(X, y)
        y_pred_custom = custom_model.predict(X)
        custom_mse = mean_squared_error(y, y_pred_custom)
        custom_r2 = r2_score(y, y_pred_custom)
        st.sidebar.success("Model re-trained!")
    
    # Display metrics in sidebar
    st.sidebar.subheader("Model Performance")
    st.sidebar.markdown(f"**Custom Model MSE:** {custom_mse:.4f}")
    st.sidebar.markdown(f"**Custom Model R²:** {custom_r2:.4f}")
    st.sidebar.markdown(f"**Sklearn MSE:** {sklearn_mse:.4f}")
    st.sidebar.markdown(f"**Sklearn R²:** {sklearn_r2:.4f}")
    
    # Prediction input
    st.sidebar.subheader("Make Predictions")
    pred_value = st.sidebar.number_input("Enter X value for prediction:", value=0.0, step=0.1)
    
    custom_prediction = custom_model.predict(np.array([[pred_value]]))[0]
    sklearn_prediction = sklearn_model.predict(np.array([[pred_value]]))[0]
    
    st.sidebar.markdown(f"**Custom Model Prediction:** {custom_prediction:.4f}")
    st.sidebar.markdown(f"**Sklearn Prediction:** {sklearn_prediction:.4f}")
    
    # Create visualizations
    fig_custom = px.scatter(x=X[:,0], y=y, labels={"x":"X", "y":"y"}, 
                           title="Custom Linear Regression Model")
    fig_custom.add_scatter(x=X[:,0], y=y_pred_custom, mode="lines", 
                          name="Custom Model", line=dict(color="red"))
    
    fig_sklearn = px.scatter(x=X[:,0], y=y, labels={"x":"X", "y":"y"}, 
                            title="Scikit-learn Linear Regression")
    fig_sklearn.add_scatter(x=X[:,0], y=y_pred_sklearn_full, mode="lines", 
                           name="Sklearn Model", line=dict(color="blue"))
    
    # Display plots and model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_custom, use_container_width=True)
        st.markdown("### Custom Model Parameters")
        st.markdown(f"**Weight (slope):** {custom_model.w[0]:.6f}")
        st.markdown(f"**Bias (intercept):** {custom_model.b:.6f}")
        st.markdown(f"**Equation:** y = {custom_model.w[0]:.4f}x + {custom_model.b:.4f}")
    
    with col2:
        st.plotly_chart(fig_sklearn, use_container_width=True)
        st.markdown("### Scikit-learn Model Parameters")
        st.markdown(f"**Weight (slope):** {sklearn_model.coef_[0]:.6f}")
        st.markdown(f"**Bias (intercept):** {sklearn_model.intercept_:.6f}")
        st.markdown(f"**Equation:** y = {sklearn_model.coef_[0]:.4f}x + {sklearn_model.intercept_:.4f}")
    
    # Display data statistics
    st.subheader("Data Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Samples", len(X))
    with col2:
        st.metric("X Mean", f"{np.mean(X):.2f}")
    with col3:
        st.metric("Y Mean", f"{np.mean(y):.2f}")
    with col4:
        st.metric("Correlation", f"{np.corrcoef(X.flatten(), y)[0,1]:.3f}")

else:
    st.warning("Please upload a valid CSV file with numeric data to get started!")
    st.markdown("""
    ### Expected File Format:
    - CSV file with at least 2 columns
    - First column: X values (independent variable)
    - Second column: Y values (dependent variable)
    - All values should be numeric
    - Example:
    ```
    X,Y
    1,2.1
    2,4.2
    3,6.1
    4,8.0
    ```
    """)
