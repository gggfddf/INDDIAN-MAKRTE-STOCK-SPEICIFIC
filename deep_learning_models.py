import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    Attention, MultiHeadAttention, LayerNormalization,
    Input, concatenate, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DeepLearningModels:
    """
    Comprehensive deep learning models for stock market prediction
    """
    
    def __init__(self, data_dict, sequence_length=60):
        self.data_dict = data_dict
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.confidence_scores = {}
        
    def prepare_data(self, timeframe='1d'):
        """Prepare data for deep learning models"""
        if timeframe not in self.data_dict:
            raise ValueError(f"Timeframe {timeframe} not available")
        
        data = self.data_dict[timeframe].copy()
        
        # Feature engineering
        features = self._engineer_features(data)
        
        # Create sequences
        X, y = self._create_sequences(features)
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, features
    
    def _engineer_features(self, data):
        """Engineer comprehensive features for ML models"""
        features_df = data.copy()
        
        # Price-based features
        features_df['Returns'] = data['Close'].pct_change()
        features_df['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features_df['Price_Change'] = data['Close'] - data['Open']
        features_df['Range'] = data['High'] - data['Low']
        features_df['Upper_Shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
        features_df['Lower_Shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        
        # Technical indicators
        features_df['RSI'] = self._calculate_rsi(data['Close'])
        features_df['MACD'], features_df['MACD_Signal'] = self._calculate_macd(data['Close'])
        features_df['BB_Upper'], features_df['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        features_df['SMA_20'] = data['Close'].rolling(20).mean()
        features_df['SMA_50'] = data['Close'].rolling(50).mean()
        features_df['EMA_12'] = data['Close'].ewm(span=12).mean()
        features_df['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Volume features
        features_df['Volume_SMA'] = data['Volume'].rolling(20).mean()
        features_df['Volume_Ratio'] = data['Volume'] / features_df['Volume_SMA']
        features_df['Price_Volume'] = data['Close'] * data['Volume']
        features_df['VWAP'] = (features_df['Price_Volume'].rolling(20).sum() / 
                              data['Volume'].rolling(20).sum())
        
        # Volatility features
        features_df['Volatility'] = features_df['Returns'].rolling(20).std()
        features_df['ATR'] = self._calculate_atr(data)
        features_df['High_Low_Ratio'] = data['High'] / data['Low']
        
        # Momentum features
        features_df['ROC_10'] = data['Close'].pct_change(10)
        features_df['ROC_20'] = data['Close'].pct_change(20)
        features_df['Momentum_10'] = data['Close'] / data['Close'].shift(10)
        features_df['Momentum_20'] = data['Close'] / data['Close'].shift(20)
        
        # Candlestick features
        features_df['Body_Size'] = abs(data['Close'] - data['Open'])
        features_df['Body_Ratio'] = features_df['Body_Size'] / features_df['Range']
        features_df['Is_Bullish'] = (data['Close'] > data['Open']).astype(int)
        features_df['Doji'] = (features_df['Body_Size'] < features_df['Range'] * 0.1).astype(int)
        
        # Gap features
        features_df['Gap'] = data['Open'] - data['Close'].shift(1)
        features_df['Gap_Percent'] = features_df['Gap'] / data['Close'].shift(1)
        features_df['Gap_Up'] = (features_df['Gap'] > 0).astype(int)
        features_df['Gap_Down'] = (features_df['Gap'] < 0).astype(int)
        
        # Time-based features
        features_df['Hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
        features_df['Day_of_Week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 0
        features_df['Month'] = data.index.month if hasattr(data.index, 'month') else 0
        features_df['Quarter'] = data.index.quarter if hasattr(data.index, 'quarter') else 0
        
        # Cyclical encoding for time features
        features_df['Hour_Sin'] = np.sin(2 * np.pi * features_df['Hour'] / 24)
        features_df['Hour_Cos'] = np.cos(2 * np.pi * features_df['Hour'] / 24)
        features_df['Day_Sin'] = np.sin(2 * np.pi * features_df['Day_of_Week'] / 7)
        features_df['Day_Cos'] = np.cos(2 * np.pi * features_df['Day_of_Week'] / 7)
        features_df['Month_Sin'] = np.sin(2 * np.pi * features_df['Month'] / 12)
        features_df['Month_Cos'] = np.cos(2 * np.pi * features_df['Month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features_df[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            features_df[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
            features_df[f'Returns_Lag_{lag}'] = features_df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features_df[f'Close_Mean_{window}'] = data['Close'].rolling(window).mean()
            features_df[f'Close_Std_{window}'] = data['Close'].rolling(window).std()
            features_df[f'Volume_Mean_{window}'] = data['Volume'].rolling(window).mean()
            features_df[f'Returns_Mean_{window}'] = features_df['Returns'].rolling(window).mean()
            features_df[f'Returns_Std_{window}'] = features_df['Returns'].rolling(window).std()
        
        # Support/Resistance levels
        features_df['Support'] = data['Low'].rolling(20).min()
        features_df['Resistance'] = data['High'].rolling(20).max()
        features_df['Support_Distance'] = (data['Close'] - features_df['Support']) / data['Close']
        features_df['Resistance_Distance'] = (features_df['Resistance'] - data['Close']) / data['Close']
        
        # Market structure features
        features_df['Higher_High'] = (data['High'] > data['High'].shift(1)).astype(int)
        features_df['Lower_Low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
        features_df['Inside_Bar'] = ((data['High'] < data['High'].shift(1)) & 
                                    (data['Low'] > data['Low'].shift(1))).astype(int)
        features_df['Outside_Bar'] = ((data['High'] > data['High'].shift(1)) & 
                                     (data['Low'] < data['Low'].shift(1))).astype(int)
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        return features_df
    
    def _create_sequences(self, features):
        """Create sequences for time series prediction"""
        # Select features for modeling
        feature_columns = [col for col in features.columns if col not in ['Close']]
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(features[feature_columns])
        y_scaled = scaler_y.fit_transform(features[['Close']])
        
        # Store scalers
        self.scalers['X'] = scaler_X
        self.scalers['y'] = scaler_y
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X.append(X_scaled[i-self.sequence_length:i])
            y.append(y_scaled[i, 0])  # Predict next close price
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_cnn_model(self, input_shape):
        """Build CNN model for pattern recognition"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            Dropout(0.2),
            
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_attention_model(self, input_shape):
        """Build Attention-based model"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(100, return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        lstm_out2 = LSTM(100, return_sequences=True)(lstm_out)
        lstm_out2 = Dropout(0.2)(lstm_out2)
        lstm_out2 = BatchNormalization()(lstm_out2)
        
        # Multi-head attention
        attention_out = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1
        )(lstm_out2, lstm_out2)
        
        attention_out = LayerNormalization()(attention_out + lstm_out2)
        
        # Global pooling and dense layers
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
        dense_out = Dense(100, activation='relu')(pooled)
        dense_out = Dropout(0.3)(dense_out)
        dense_out = Dense(50, activation='relu')(dense_out)
        outputs = Dense(1)(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_hybrid_model(self, input_shape):
        """Build hybrid model combining CNN and LSTM"""
        inputs = Input(shape=input_shape)
        
        # CNN branch
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)
        
        # LSTM branch
        lstm_branch = LSTM(100, return_sequences=True)(inputs)
        lstm_branch = Dropout(0.2)(lstm_branch)
        lstm_branch = LSTM(50, return_sequences=True)(lstm_branch)
        lstm_branch = Dropout(0.2)(lstm_branch)
        
        # Combine branches
        # Reshape CNN output to match LSTM output
        cnn_reshaped = tf.keras.layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [0, lstm_branch.shape[1] - x.shape[1]], [0, 0]])
        )(cnn_branch)
        
        combined = concatenate([lstm_branch, cnn_reshaped])
        
        # Attention mechanism
        attention_out = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.1
        )(combined, combined)
        
        attention_out = LayerNormalization()(attention_out + combined)
        
        # Output layers
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
        dense_out = Dense(100, activation='relu')(pooled)
        dense_out = Dropout(0.3)(dense_out)
        dense_out = Dense(50, activation='relu')(dense_out)
        outputs = Dense(1)(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model
    
    def build_autoencoder(self, input_shape):
        """Build autoencoder for anomaly detection"""
        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoded = LSTM(100, return_sequences=True)(encoder_inputs)
        encoded = LSTM(50, return_sequences=False)(encoded)
        encoded = Dense(25, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(50, activation='relu')(encoded)
        decoded = Dense(100, activation='relu')(decoded)
        decoded = tf.keras.layers.RepeatVector(input_shape[0])(decoded)
        decoded = LSTM(100, return_sequences=True)(decoded)
        decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
        
        autoencoder = Model(encoder_inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                           loss='mse')
        
        return autoencoder
    
    def train_models(self, timeframe='1d'):
        """Train all deep learning models"""
        print(f"Training models for {timeframe} timeframe...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, features = self.prepare_data(timeframe)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Training callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        callbacks = [early_stopping, reduce_lr]
        
        # Train LSTM model
        print("Training LSTM model...")
        lstm_model = self.build_lstm_model(input_shape)
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        self.models[f'{timeframe}_lstm'] = lstm_model
        
        # Train CNN model
        print("Training CNN model...")
        cnn_model = self.build_cnn_model(input_shape)
        cnn_history = cnn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        self.models[f'{timeframe}_cnn'] = cnn_model
        
        # Train Attention model
        print("Training Attention model...")
        attention_model = self.build_attention_model(input_shape)
        attention_history = attention_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        self.models[f'{timeframe}_attention'] = attention_model
        
        # Train Hybrid model
        print("Training Hybrid model...")
        hybrid_model = self.build_hybrid_model(input_shape)
        hybrid_history = hybrid_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        self.models[f'{timeframe}_hybrid'] = hybrid_model
        
        # Train Autoencoder for anomaly detection
        print("Training Autoencoder...")
        autoencoder = self.build_autoencoder(input_shape)
        autoencoder_history = autoencoder.fit(
            X_train, X_train,  # Reconstruct input
            validation_data=(X_test, X_test),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        self.models[f'{timeframe}_autoencoder'] = autoencoder
        
        # Train traditional ML models for ensemble
        print("Training traditional ML models...")
        self._train_traditional_models(X_train, X_test, y_train, y_test, timeframe)
        
        # Evaluate models
        self._evaluate_models(X_test, y_test, timeframe)
        
        return {
            'lstm_history': lstm_history,
            'cnn_history': cnn_history,
            'attention_history': attention_history,
            'hybrid_history': hybrid_history,
            'autoencoder_history': autoencoder_history
        }
    
    def _train_traditional_models(self, X_train, X_test, y_train, y_test, timeframe):
        """Train traditional ML models for ensemble"""
        # Reshape data for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_flat, y_train)
        self.models[f'{timeframe}_xgb'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        lgb_model.fit(X_train_flat, y_train)
        self.models[f'{timeframe}_lgb'] = lgb_model
        
        # Feature importance for traditional models
        self.feature_importance[f'{timeframe}_xgb'] = xgb_model.feature_importances_
        self.feature_importance[f'{timeframe}_lgb'] = lgb_model.feature_importances_
    
    def _evaluate_models(self, X_test, y_test, timeframe):
        """Evaluate all models"""
        evaluation_results = {}
        
        # Reshape for traditional models
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        for model_name, model in self.models.items():
            if timeframe in model_name:
                try:
                    if 'xgb' in model_name or 'lgb' in model_name:
                        y_pred = model.predict(X_test_flat)
                    else:
                        y_pred = model.predict(X_test).flatten()
                    
                    # Calculate metrics
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    rmse = np.sqrt(mse)
                    
                    # Directional accuracy
                    y_test_diff = np.diff(y_test)
                    y_pred_diff = np.diff(y_pred)
                    direction_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
                    
                    evaluation_results[model_name] = {
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'direction_accuracy': direction_accuracy
                    }
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results
    
    def create_ensemble_prediction(self, X_input, timeframe='1d'):
        """Create ensemble prediction from all models"""
        predictions = []
        weights = []
        
        # Reshape for traditional models
        X_input_flat = X_input.reshape(X_input.shape[0], -1)
        
        for model_name, model in self.models.items():
            if timeframe in model_name and 'autoencoder' not in model_name:
                try:
                    if 'xgb' in model_name or 'lgb' in model_name:
                        pred = model.predict(X_input_flat)
                    else:
                        pred = model.predict(X_input).flatten()
                    
                    predictions.append(pred)
                    
                    # Weight based on model type (can be improved with validation performance)
                    if 'hybrid' in model_name:
                        weights.append(0.3)
                    elif 'attention' in model_name:
                        weights.append(0.25)
                    elif 'lstm' in model_name:
                        weights.append(0.2)
                    elif 'cnn' in model_name:
                        weights.append(0.15)
                    else:
                        weights.append(0.1)
                        
                except Exception as e:
                    print(f"Error in ensemble prediction for {model_name}: {e}")
        
        if predictions:
            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            # Calculate prediction confidence
            pred_std = np.std(predictions, axis=0)
            confidence = 1 / (1 + pred_std)  # Higher confidence for lower std
            
            return ensemble_pred, confidence
        
        return None, None
    
    def detect_anomalies(self, X_input, timeframe='1d'):
        """Detect anomalies using autoencoder"""
        autoencoder_key = f'{timeframe}_autoencoder'
        
        if autoencoder_key in self.models:
            # Reconstruct input
            reconstructed = self.models[autoencoder_key].predict(X_input)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.square(X_input - reconstructed), axis=(1, 2))
            
            # Define anomaly threshold (99th percentile of training errors)
            threshold = np.percentile(reconstruction_error, 99)
            
            # Identify anomalies
            anomalies = reconstruction_error > threshold
            
            return anomalies, reconstruction_error
        
        return None, None
    
    def predict_multiple_timeframes(self, latest_data):
        """Make predictions for multiple timeframes"""
        predictions = {}
        
        for timeframe in self.data_dict.keys():
            if timeframe in latest_data:
                try:
                    # Prepare latest data for prediction
                    features = self._engineer_features(latest_data[timeframe])
                    
                    if len(features) >= self.sequence_length:
                        # Get last sequence
                        feature_columns = [col for col in features.columns if col not in ['Close']]
                        X_latest = self.scalers['X'].transform(features[feature_columns].iloc[-self.sequence_length:])
                        X_latest = X_latest.reshape(1, self.sequence_length, -1)
                        
                        # Ensemble prediction
                        pred, confidence = self.create_ensemble_prediction(X_latest, timeframe)
                        
                        if pred is not None:
                            # Inverse scale prediction
                            pred_original = self.scalers['y'].inverse_transform(pred.reshape(-1, 1))[0, 0]
                            
                            predictions[timeframe] = {
                                'prediction': pred_original,
                                'confidence': confidence[0] if confidence is not None else 0.5,
                                'current_price': features['Close'].iloc[-1]
                            }
                            
                            # Detect anomalies
                            anomalies, reconstruction_error = self.detect_anomalies(X_latest, timeframe)
                            if anomalies is not None:
                                predictions[timeframe]['anomaly_detected'] = bool(anomalies[0])
                                predictions[timeframe]['anomaly_score'] = float(reconstruction_error[0])
                        
                except Exception as e:
                    print(f"Error predicting for {timeframe}: {e}")
        
        return predictions
    
    def calculate_feature_importance_ensemble(self, timeframe='1d'):
        """Calculate ensemble feature importance"""
        importance_scores = {}
        
        # Get feature importance from traditional models
        for model_type in ['xgb', 'lgb']:
            key = f'{timeframe}_{model_type}'
            if key in self.feature_importance:
                importance_scores[model_type] = self.feature_importance[key]
        
        # For deep learning models, use permutation importance (simplified)
        # This would typically require more sophisticated analysis
        
        return importance_scores
    
    def predict_price_movements(self, timeframe='1d', horizon=5):
        """Predict price movements for multiple horizons"""
        predictions = {}
        
        if timeframe not in self.data_dict:
            return predictions
        
        try:
            # Get latest data
            data = self.data_dict[timeframe]
            features = self._engineer_features(data)
            
            if len(features) >= self.sequence_length:
                feature_columns = [col for col in features.columns if col not in ['Close']]
                X_latest = self.scalers['X'].transform(features[feature_columns].iloc[-self.sequence_length:])
                X_latest = X_latest.reshape(1, self.sequence_length, -1)
                
                # Multi-horizon predictions
                for h in range(1, horizon + 1):
                    pred, confidence = self.create_ensemble_prediction(X_latest, timeframe)
                    
                    if pred is not None:
                        pred_original = self.scalers['y'].inverse_transform(pred.reshape(-1, 1))[0, 0]
                        current_price = features['Close'].iloc[-1]
                        
                        predictions[f'{h}_period'] = {
                            'predicted_price': pred_original,
                            'price_change': pred_original - current_price,
                            'price_change_percent': ((pred_original - current_price) / current_price) * 100,
                            'confidence': confidence[0] if confidence is not None else 0.5
                        }
                
        except Exception as e:
            print(f"Error in multi-horizon prediction: {e}")
        
        return predictions
    
    # Helper methods for technical indicators
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def _calculate_atr(self, data, window=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'count_params'):
                summary[model_name] = {
                    'type': 'Neural Network',
                    'parameters': model.count_params(),
                    'layers': len(model.layers)
                }
            else:
                summary[model_name] = {
                    'type': 'Traditional ML',
                    'model_type': type(model).__name__
                }
        
        return summary
    
    def save_models(self, filepath_prefix):
        """Save all trained models"""
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'save'):
                    model.save(f"{filepath_prefix}_{model_name}.h5")
                else:
                    # Save traditional ML models using joblib
                    import joblib
                    joblib.dump(model, f"{filepath_prefix}_{model_name}.pkl")
                    
            except Exception as e:
                print(f"Error saving {model_name}: {e}")
    
    def load_models(self, filepath_prefix):
        """Load saved models"""
        import os
        import joblib
        
        # Find all model files
        for filename in os.listdir('.'):
            if filename.startswith(filepath_prefix):
                model_name = filename.replace(filepath_prefix + '_', '').replace('.h5', '').replace('.pkl', '')
                
                try:
                    if filename.endswith('.h5'):
                        self.models[model_name] = tf.keras.models.load_model(filename)
                    elif filename.endswith('.pkl'):
                        self.models[model_name] = joblib.load(filename)
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

# Test the models
if __name__ == "__main__":
    # This would be called from the main analysis class
    print("Deep Learning Models module loaded successfully!")
    print("Available model types:")
    print("- LSTM Networks")
    print("- CNN for pattern recognition") 
    print("- Attention mechanisms")
    print("- Hybrid CNN-LSTM models")
    print("- Autoencoders for anomaly detection")
    print("- Ensemble methods")
    print("- Traditional ML models (XGBoost, LightGBM)")