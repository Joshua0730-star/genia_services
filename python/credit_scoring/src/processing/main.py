"""
Data Preprocessing Module for German Credit Risk Dataset
Handles feature engineering, encoding, and scaling following SOLID principles
Su único objetivo es transformar los datos originales del dataset en datos numéricos y normalizados que la red neuronal pueda entender.
"""
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder ## <-- StandardScaler (para normalizar los valores numericos) y OneHotEncoder (para codificar las caracteristicas no numericas como textos)
from sklearn.compose import ColumnTransformer ### <-- para tratar varios tipos de datos de manera adecuada
from sklearn.pipeline import Pipeline ### <-- para concatenar los preprocessores adecuados


class CreditDataPreprocessor:
    """
    Preprocessor for German Credit Risk dataset
    Handles categorical encoding, numerical scaling, and feature engineering
    """
    def __init__(self):
        self.numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
        self.categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        self.target_feature = 'Risk'
        
    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Builds a pipeline for preprocessing the data
        1. Scale numerical features (StandardScaler)
        2. Codify categorical features (OneHotEncoder)

        Su trabajo consiste en construir un objeto capaz de aprender cómo deben transformarse los datos.
        """
        
        # features transformers
        numeric_tf = Pipeline(steps=[("scaler", StandardScaler())]) ## <--- hacemos que los valores numericos tengna escalas similares para un mejor funcionamiento
        categorical_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]) ## <-- las caracteristicas de este tipo las volvemos numericas. (las redes neuronales no pueden trabajar con caracteristicas no numericas) (handle_unknown="ignore" significa que se ignora el valor no reconocido, Es una medida de seguridad para evitar que el modelo falle cuando aparezcan categorías nuevas.)
        
        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_tf, self.numerical_features), ### <-- ( nombre, transformador, columnas a transformar), indicamos que queremos llamar "num" este resultado y que aplique el transformador numeric_tf a las columnas self.numerical_features
                ('cat', categorical_tf, self.categorical_features)
            ],
            remainder='passthrough' ### <-- Este parámetro indica qué hacer con las columnas que no aparecen en ningún transformador.
        )
        
        # adjust
        x_train = df.drop(self.target_feature, axis=1) ## <-- eliminamos la columna que queremos predecir
        preprocessor.fit(x_train) ### <-- preparamos el preprocesador y que aprenda a como realizar las trasnformaciones para mas adelante hacer preprocessor.transform(x)
        return preprocessor
    
    def process_data(self, df: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply processing and separate features
        Args:
            df (pd.DataFrame): DataFrame to process.
            preprocessor (ColumnTransformer): fit preprocessing
        """
        
        df_copy = df.copy() ## <-- sacamos una copia del dataframe original
        df_copy[self.target_feature] = df_copy[self.target_feature].map({'bad': 0, 'good': 1}) ### <-- mapeamos los valores de la columna "Risk" para que tenga valores binarios (0 y 1), donde dice bad = 0 y good = 1
        
        y = df_copy[self.target_feature] ## <-- sacamos la columna que queremos predecir en Y
        x = df_copy.drop(self.target_feature, axis=1) 
        
        x_processed = preprocessor.transform(x) ### <-- aplicamos la transformacion a los datos de inputs
        return x_processed, y ### <-- regresamos los datos procesados y los datos de salida