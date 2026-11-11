import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional,List

# Creamos la clase de nuestra red neuronal

## todas las redes neuronales tienen que heredar de la clase nn.Module. y asi tenemos todas las funciones y herramientas para construir nuestra red neuronal
class CreditScoringModel(nn.Module):
    """
    Multi-layer Perceptron for credit scoring

    Architecture:
    - Input Layer: num_features (depends on preprocessing)
    - Hidden Layers: Dynamic
    - Output Layer: Linear (-> 1) para producir loggits
    """
    def __init__(self, num_features: int, hidden_layers: List[int], dropout_rate: float = 0.1, use_batch_norm: bool = True, activation_fn: str = "ReLU"):

        super(CreditScoringModel, self).__init__() # inicializamos la clase

        self.num_features = num_features
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation_fn_name = activation_fn



        layers = []
        input_size = num_features

        for i, layer_size in enumerate(hidden_layers):

            ## Linear Layer (neural) <-- creamos la primera capa de nuestra red neuronal con los parametros de entrada del numero de caracteristicas y el numero de neuronas que vamos a usar en esa capa
            layers.append(nn.Linear(input_size, layer_size))


            ## batch normalization <-- si estamos usando batch normalization, creamos la capa de batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))

            ## activation function <-- creamos la funcion de activacion que vamos a usar en nuestra red neuronal
            if activation_fn == "ReLU":
                layers.append(nn.ReLU())
            elif activation_fn == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            elif activation_fn == "GELU":
                layers.append(nn.GELU())

            ## dropout <-- si estamos usando dropout, creamos la capa de dropout
            layers.append(nn.Dropout(dropout_rate))

            ## output size --> input size
            input_size = layer_size ## 64

        ## output layer 
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Forward pass trough the network
        args:
            x: input tensor of shape (batch_size, num_features)

        returns:
            ouput tensor of shape (batch_size, 1) with probabilities
        """

        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor: 
        ##"Voy a usar la red únicamente para predecir. No guardes el grafo computacional ni calcules gradientes.
        """
        Get prediction probabilities
        Args:
            x: Input tensor
            
        Returns:
            Probabilities for both classes (good=1, bad=0)
        """
        with torch.no_grad():
            logits = self.forward(x) ## lo que se hace aqui es indicar que hagamos una propagacion hacia adelante en la red neuronal y me retorne el valor de la ultima neurona o capa
            prob_good = torch.sigmoid(logits) ## para clasificar el problema binario y obtener un valor entre 0 y 1.
            prob_bad = 1 - prob_good
            return torch.cat([prob_bad, prob_good], dim=1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0=bad, 1=good)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return (probabilities > threshold).int() ## <-- convierte un true = 1 y un false = 0 

    def get_model_info(self) -> dict: ## <-- devuelve un diccionario con la informacion de la arquitectura de nuestra red neuronal
        """Get model architecture information"""
        return {
            "model_type": "CreditScoringModel",
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation_fn": self.activation_fn_name,
            "architecture": {
                "input_layer": self.num_features,
                "hidden_layers": self.hidden_layers_config,
                "output_layer": 1
            },
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        

## Explicacion de los parametros
# num_features: int: es el numero de caracteristicas que vamos a entrenar en nuestra red neuronal

# hidden_layers: List[int]: es una lista de enteros que indican el numero de neuronas que vamos a usar en cada capa de nuestra red neuronal

## [128, 64] <-- quiere decir que en nuestra red neuronal vamos a tener 2 capas ocultas, la primera con 128 neuronas y la segunda con 64 neuronas

# dropout_rate: float: es un parametro que indica el porcentaje de neuronas que vamos a eliminar de manera aleatoria de nuestra red neuronal. esto ayuda a reducir el error de nuestra red neuronal

# use_batch_norm: bool: es un parametro que indica si vamos a usar batch normalization en nuestra red neuronal. es tomar la salida de la neurona y normalizarla

# activation_fn: str: es un parametro que indica la funcion de activacion que vamos a usar en nuestra red neuronal. por defecto es el sigmoide


# importante es notar que este código no realiza el aprendizaje. En el constructor solo se define la estructura de la red y se inicializan sus pesos.