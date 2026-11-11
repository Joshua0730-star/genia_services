## Perceptron Multicapa

Se organiza jerarquicamente en una secuencia de capas compuestas de neuronas, y esta organizacion jerarquica de capas y neueronas es lo que le permite aprender representaciones complejas de los datos a medida que la informacion va avanzando por nuestra red neuronal

aqui distinguimos 3 tipos de capas que van secuencialmente:

- Input Layer: Su funcion es simplemente recibir informacion del mundo exterior y pasarla a la primera capa oculta
- Hidden Layers: En estas capas es donde ocurre la mayor parte del aprendizaje, cada neurona de la etapa oculta recibe la informacion de la neurona de la capa anterior, realiza un procedimiento matematico y entrega su resultado a la neurona de la siguiente etapa. entre mayor profundidad se tenga en la red neuronal mayor numero de capas ocultas va a haber.
- Output Layer: esta capa final y su salida depende directamente del tipo de problema que queremos solucionar.

### Como funciona nuestra red neuronal y como funciona cada una de las neuronas

En cada una de las neuronas de las capas profundas/ocultas tenemos unos valores de entrada que son los de salida de la neurona de la capa anterior.

tambien tenemos un `peso`: cada conexion entre 2 neuronas tiene un peso asociado, este peso es un parametro numerico que la red `aprende` durante el proceso de entrenamiento. y ese peso va a determinar que tan importante es la conexion entre las neuronas para la red.

`Sesgo`: es otro parametro aprendible que se suma a la combinacion de las entradas ponderadas. permite a la neurona ajustar el umbral de activacion de cada una de las neuronas.

### Pasos

1. `Preactivazion o Logit`: Calcular la suma ponderada de las entradas a la que se le añade el sesgo.

tomamos el vector de entradas `x` y el vector de pesos `w`. los multiplicamos y a ese resultado sumamos el sesgo `b`.

z = (x1 \* w1) + (x2 \* w2) + b

z = x \* w + b

Este valor `z` es el valor de entrada para el siguiente componente crucial de la neurona: La funcion de activacion.

2. `Funcion de activacion`: Es una operacion matematica no lineal que se aplica al resultado de la suma ponderada (z) para generar la salida final de la neurona (a).
   Determina si la neurona es `activa` o `no activa`.

Que tipos de funciones podemos usar para la activacion de una red neuronal?

- Sigmoid: `a = 1 / (1 + e^-z)`
- Tanh: `a = tanh(z)`
- ReLu: `a = max(0, z)`
- Leaky ReLu: `a = max(0.01 \* z, z)`

Una de las mas usadas es la funcion ReLu

![image](images/relu.png)

recordemos que esto es una f(x) que nos da un valor en Y, el valor positivo dado en f(x) sera el mismo valor dado en Y. pero si es negativo, el valor dado en Y sera 0.

Para que usamos la funcion de activacion??.

su trabajo principal es agregar componentes no lineales a la red neuronal.

Esta funcion de activacion permite que la red neuronal no sea completamente lineal. ya que sin una funcion de activacion, la red neuronal no podra aprender. y acoplarse con la funcion de costo que nos interesa. y ademas seria lo mismo tener muchas capas que 1 sola capa si no hay funcion de activacion.

##### funcion logistica (sigmoide):

esta funcion nos da un valor positvo entre 0 y 1. si import que tan grande o pequeño sea su valor de entrada. sirve mucho para clasificacion binaria. en la ultima neurona de la red.

esta funcion sin embargo no es muy eficiente para las redes neuronales hoy en dia. porque su gradiente/derivada no es muy util a la hora de usar para la funcion de costo. ya que llega un punto donde los valores son muy pequeños por lo cual si se usa esta funcion de aactivacion en toda la red aprenderia muy lento porque la actualizacion de los pesos y sesgos seria minima.

`utilidad`

- problemas binarios

`Problemas o desventajas`

- derivada muy pequeña (desvanecimiento de gradiente)
- valores positivos y no maneja negativos

##### Tangente hiperbolica (tahn):

esta comprobado que esta funcion es mucho mejor que la anterior en la mayoria de los casos. ya que nos comprende valores entre -1 y 1. y esta centrada en 0, su versatilidad nos permite manejar valores negativos y positivos. ademas su derivada es mucho mas eficiente y alta. es decir que buscaremos funciones con derivadas altas para que el ajuste de los parametros de la red sea eficiente.

pero esta funcion tiene el mismo problema que la funcion de sigmoide. ya que si los valores son muy grades o pequeños el gradiente es minimo. y tambien porque ambas usan calculos exponenciales en ejecucion el cual genera un gran costo computacional

`utilidad`

- valores mas generales entre -1 y 1.
- es mas versatil y general

`Problemas o desventajas`

- derivada muy pequeña (desvanecimiento de gradiente)
- caras de calcular en ejecucion computacional

##### softsign:

similar a tanh pero con una derivada muy pequeña. y un mejor gradiente.

##### ReLU:

se ha comprobado en 2018 que esta funcion es mejor que la tangente hiperbolica (tanh) en la mayoria de los casos. y que en la forward pass y backpropagation es mas eficiente que tanh hasta 6 veces mas rapida.

esta funcion es extremadamente simple y no tiene tanto costo computacional. simplemente regresa el valor que sea mayor:

`max(0, x)`
esta funcion da una gradiente constante de 1 para todos los positivos. y 0 en todos los negativos.

``ventajas`:

- es mas simple
- es mas rapido
- gradiente constante y valores positivos infinitos

`problemas`

- Neuronas muertas. ya que el valor de retorno es 0 en valores negativos.

Sin embargo esta funcion es una de las mas importantes hoy en dia y la que es por defecto

debido a ese problema de neuronas muertas algunos investigadores se han propuesto mejorar esta funcion con variaciones:

- `Leaky ReLu`: Max (0.01x, x) <- el cual busca que para los negativos devuelva un valor diferente a cero. sin embargo su gradiente se desvanece poco a poco para los negativos

- `Parametric ReLu`: Max (0.01x, x) <- if (x > 0) x, if x <= 0 ax. donde a: es un numero parametrizable

- `Gelu`: esta funcion no es monotonica (quiere decir que las funciones que solo pasan una sola vez por el 0 en incrementan constantemente)

##### softplus

in (1 + e^x)

esta propone algunas mejoras frente a ReLu pero no es muy comun y ademas tiene costos computacionales muy altos.

veamos algunas funciones de activacion mas modernas

##### Swish

x \* sigmoid(beta x)

su grafica es suave y no es monotonica (quiere decir que puede bajar y subir varias veces por el 0)
esta comprobada que es mucho mejor que RElu en la mayoria de los casos pero tiene un costo computacional un poco mas alto.
es un buen candidato para reemplazar a ReLU

##### Mish

Propuesta en 2020 y mejora significativamente superando a ReLU y Swish en muchos casos sobre todo en vision computacionakl .No es monotonica.

la mayoria de mejoras modernas estan proponiendo una base no monotonica de funciones.

formula: x \* tanh(softplus(x))

mantiene un poco de informacion negativa evitando las neuronas muertas. pero tiene un costo computacional mas alto que el de ReLu.

#### Funciones de activacion que se usan a diario

##### funcion de identidad (lineal)

Util para las redes de regresion que deben dar como salida un numero, por ejemplo: una red que calcula el precio de un producto en funcion de sus caracteristicas.

##### Softmax

Usado en redes de clasificacion donde la capa de salida tiene varias neuronas que representan cateegorias y cada una asociada una probabilidad segun el valor de la entrada.

#### Cuando usar x funcion ?

depende de algunos factores pero aqui hay unos casos de uso comunes para cada tipo de funcion:

- sigmoid: en la capa de salida para problemas de clasificacion binaria
- tangente hiperbolica: es superior a la funcion logistica para capas ocultas pero no mejor que ReLU
- ReLu: la que normalmente vamos a usar a menos que quieras experimentar con otras. si hay algun problema con las redes muertas se puede probar disminuyendo la tasa de aprendizaje o probando con funciones como (Leaky ReLu, parametric ReLu) pero ten en cuenta que pueden caer en el desvanecimiento de gradientes.
- GeLu: de manera general tiene mejores resultados que ReLu especialmente parra redes muy grandes o transformers
- Swish: funciona muy bien y en general le gana a Relu solo que su costo computacional es muy alto.
- Softmax: en la ultima capa de redes de clasificacion
- funcion de identidad: redes lineales

#### funciones Oscilatorias

3. `Decision Final`: Las activaciones de la ultima capa oculta se pasan a la capa de salida.
   para nuestro problema de clasificacion binaria, la capa de salida tiene una sola neurona. esta neurona calcula su suma ponderada.

En lugar de `ReLu` Se usa la funcion `Sigmoide`.

![image](images/sigmoide.png)

### ¿ Como aprende ua red neuronal profunda?

#### 1. Aprendizaje

El proceso de aprendizaje consiste en ajustar sistematicamente estos pesos y sesgos para que las predicciones del modelo se acerquenlo mas posibles a los resultados reales

Este proceso se basa en 3 pilares:

- **Funcion de perdida**: Para que la red pueda mejorar, primero necesita una forma de medir que tan equivocadas estan sus predicciones.

`Loss Function` es una formular matematica que calcula un unico valor escalar que representa la diferencia entra la prediccion del modelo y el valor real.

para nuestro ejemplo usaremos la funcion de perdida de entropia cruzada

- **Algoritmo de rertopropagacion**: Una vez se calcula el error total como ajustamos los millones de pesos y sesgos de la red para reducir este error??,

veamos paso a paso como funciona este algoritmo

- 1. Propagacion hacia adelante
- 2. calculamos el valor del error

- **Optimizador**: Se encarga de en base a los parametros y errores dados de nuestra red llevar esos puntos en la direccion opuesta para minimizar la perdida

TenemosUna red aprende modificando los parametros de la red para poder hacer predicciones correctamente.
¿ y como modificamos los parametros de la red? Pues hacemos una propagacion hacia adelante, tomamos ese resultado final para pasarlo a la funcion de perdida, en la funcion de perdida lo que se hace es cuantificar que tanto error tenemos (que tan lejos estamos de la verdad). luego tomamos ese error y con la regla de la cadena lo retropropagamos hacia atras por toda la red, en este paso lo que buscamos identificar es que tan culpables son cada una de las capas en ese error final. una vez se encuentran el error de todas y cada una de las capas, podemos calcular el gradiente de la perdida respecto a cada uno de los parametros para movilizarnos en direccion contraria al gradiente(que es un vector que indica la direccion de hacia donde crece el error).

Una vez tenemos el gradiente sabemos hacia donde tenemos que ajustar los parametros pero no sabemos cuanto.

Y aqui es donde entra en juego `el optimizador`. que nos dice cuanto tenemos que ir avanzando en el ajuste de nuestros parametros para ir encontrando el error minimo.

#### Estrategias de aprendizaje

Cuando entrenamos una red neuronal tenemos 2 casos, `over-fitting (sobreajuste)` y `under-fitting (subajuste)`.

##### Over-fitting

`Cuando un modelo tiene un rendimiento casi perfecto en el conjunto de datos de entrenamiento, pero con datos nuevos es muy malo`

Algunas tecnicas para este caso son:

- Dropout: Apagar y encender neuronas random a la hora de entrenar
- Early Stopping: detener el entrenamiento cuando el error no va mejorando (cuando nuestra red ya no esta aprendiendo)

##### Under-fitting

`Un modelo deficiente con datos de entrenamiento y con datos nuevos, (es como un modelo que no hace nada bien 😂)`

- Mejorar los datos de entrenamiento
- Mejorar la arquitectura de la red

`Generalizacion`
Un Modelo de Deep Learning debe tener la capacidad de hacer predicciones precisas sobre datos nuevos y no vistos. esta capacidad se conoce como generalizacion.

### Diseño y Flujo de la plataforma

![image](images/arquitectura_proyecto.png)

#### Explicacion breve

En nuestro proyecto tenemos un dataset de entrenamiento, en nuestra capa de entrada (input) de nuestra red neuronal vamos a tener que ingresar 12 caracteristicas, probablemente nuestra red neuronal tenga 12 neuronas en la capa de entrada. (queremos solucionar un problema binario) donde x1 = age, x2 = sex, y asi sucesivamente. pero ademas tenemos varios tipos de datos en nuestro dataset (numericas, categoricas como el sexo.)

- Lo primero que hacemos es importar la libreria `torch` porque con ella es que vamos a construir nuestra red neuronal.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

##### ¿Qué es un logit?

Un logit representa la "evidencia" que tiene la red para decidir una clase.

Un logit muy positivo indica que la red está muy convencida de la clase positiva.
Un logit muy negativo indica que está muy convencida de la clase negativa.
Un logit cercano a cero indica incertidumbre.

Por ejemplo:

Logit = 8

La red está muy segura de que el cliente es bueno.

Mientras que:

Logit = -8

La red está muy segura de que el cliente es malo.

#### Preprocesamiento

Es importante entender que la red neuronal nunca trabaja directamente con el CSV original.

Siempre recibe datos previamente procesados.

``map() es un método de Pandas.

Su función consiste en reemplazar valores utilizando un diccionario.``

##### ¿ Porque escoger bien el dispositivo es importante ?

`
Porque determina dónde se ejecutarán millones de operaciones matemáticas`.

Durante el entrenamiento se realizan continuamente:

multiplicaciones de matrices;
sumas;
funciones de activación;
cálculo de gradientes;
actualización de pesos.

Todo eso ocurrirá en el dispositivo que hayas seleccionado.

##### ¿Qué es realmente un optimizador?

El optimizador es el algoritmo encargado de actualizar los pesos.

Piensa en él como el "método" para mover la red hacia un error menor.

Todos reciben el gradiente calculado por backpropagation, pero cada uno decide cómo utilizarlo.

Los más comunes son:

SGD (Stochastic Gradient Descent): hace actualizaciones sencillas siguiendo el gradiente.
Adam: adapta automáticamente el tamaño del paso para cada peso y suele converger más rápido.
RMSprop: ajusta el aprendizaje según el historial reciente de los gradientes.

No cambian la arquitectura de la red; cambian la forma en que los pesos se actualizan.

# Bloque: Configuración del entrenamiento (Training Config)

Este bloque del constructor (`__init__`) es el encargado de leer desde el archivo **YAML** todos los **hiperparámetros de entrenamiento**.

Es importante entender que **este bloque todavía no entrena la red neuronal**.

Su única responsabilidad es leer la configuración y almacenarla dentro del objeto `CreditScoringModelTraining` para que más adelante pueda utilizarse durante el entrenamiento.

El código es el siguiente:

```python
# -------------------------------------------------
# Training config
# -------------------------------------------------
train_cfg = self.params["training_config"]

self.batch_size = train_cfg["batch_size"]
self.learning_rate = train_cfg["learning_rate"]
self.num_epochs = train_cfg["num_epochs"]
self.weight_decay = train_cfg["weight_decay"]

self.optimizer_name = train_cfg["optimizer"]
self.scheduler_name = train_cfg["scheduler"]

self.early_stopping_patience = train_cfg["early_stopping_patience"]
```

---

# Antes de comenzar: Parámetros vs Hiperparámetros

Es fundamental diferenciar tres conceptos distintos dentro de una red neuronal.

## 1. Arquitectura

La arquitectura responde a la pregunta:

> **¿Cómo está construida la red neuronal?**

Por ejemplo:

- Número de capas ocultas.
- Número de neuronas por capa.
- Función de activación.
- Uso de Batch Normalization.
- Dropout.

Estos parámetros describen la estructura del modelo.

---

## 2. Parámetros entrenables

Son los valores que la red neuronal aprende automáticamente durante el entrenamiento.

Por ejemplo:

- Pesos (_Weights_).
- Bias.

Inicialmente estos valores son aleatorios.

Después de cada iteración del entrenamiento cambian mediante Backpropagation y el optimizador.

Ejemplo:

```text
Antes:

Peso = 0.82

↓

Después del entrenamiento

↓

Peso = 0.76
```

Estos son los verdaderos **parámetros del modelo**.

---

## 3. Hiperparámetros

Los hiperparámetros **no son aprendidos por la red**.

Son valores definidos por el desarrollador antes de comenzar el entrenamiento.

Ejemplos:

- Learning Rate.
- Batch Size.
- Número de Epochs.
- Optimizer.
- Scheduler.
- Weight Decay.
- Early Stopping.

Si en el YAML escribimos:

```yaml
learning_rate: 0.001
```

La red nunca modifica ese valor.

Siempre será responsabilidad del desarrollador elegirlo.

---

# 1. Batch Size

```python
self.batch_size = train_cfg["batch_size"]
```

## ¿Qué representa?

El Batch Size indica **cuántas muestras procesa la red neuronal antes de actualizar sus pesos**.

Supongamos un dataset con:

```text
1000 muestras
```

Y un Batch Size de:

```yaml
batch_size: 100
```

Entonces el entrenamiento se divide así:

```text
Dataset

1000 muestras

│

├── Batch 1 → 100 muestras
├── Batch 2 → 100 muestras
├── Batch 3 → 100 muestras
...
└── Batch 10 → 100 muestras
```

Cada batch realiza el siguiente proceso:

```text
Forward Pass

↓

Calcular Loss

↓

Backpropagation

↓

Actualizar pesos
```

Después comienza el siguiente batch.

### Batch Size grande

Ejemplo:

```yaml
batch_size: 512
```

Ventajas:

- Menos actualizaciones.
- Mejor aprovechamiento de la GPU.
- Gradientes más estables.

Desventajas:

- Mayor consumo de memoria.
- Puede generalizar ligeramente peor.

### Batch Size pequeño

Ejemplo:

```yaml
batch_size: 8
```

Ventajas:

- Muchas actualizaciones.
- Puede escapar mejor de mínimos locales.

Desventajas:

- Entrenamiento más lento.
- Gradientes más ruidosos.

---

# 2. Learning Rate

```python
self.learning_rate = train_cfg["learning_rate"]
```

Es probablemente el hiperparámetro más importante de todo el entrenamiento.

## ¿Qué controla?

Controla **qué tan grande será cada actualización de los pesos**.

Durante el entrenamiento ocurre:

```text
Forward

↓

Loss

↓

Backpropagation

↓

Actualizar pesos
```

El Learning Rate determina el tamaño del paso en esa actualización.

Ejemplo:

Supongamos un peso:

```text
0.80
```

Con:

```yaml
learning_rate: 0.1
```

Podría actualizarse a:

```text
0.70
```

Mientras que con:

```yaml
learning_rate: 0.001
```

Podría cambiar únicamente a:

```text
0.799
```

### Learning Rate muy pequeño

```text
0.000001
```

La red aprende muy lentamente.

Puede necesitar miles de épocas.

### Learning Rate muy grande

```text
1
```

Los pesos cambian demasiado en cada actualización.

La red puede nunca converger.

Visualmente:

Learning Rate pequeño:

```text
●
↓
↓
↓
↓
Mínimo
```

Learning Rate grande:

```text
↓↓↓↓↓↓

Se pasa del mínimo

Vuelve

Se vuelve a pasar
```

---

# 3. Número de Epochs

```python
self.num_epochs = train_cfg["num_epochs"]
```

Una **Epoch** significa:

> **Recorrer completamente el dataset una vez.**

Ejemplo:

Dataset:

```text
10.000 muestras
```

Batch Size:

```text
100
```

Entonces:

```text
100 batches

↓

1 Epoch
```

Si:

```yaml
num_epochs: 50
```

La red verá el dataset completo:

```text
50 veces
```

---

# 4. Weight Decay

```python
self.weight_decay = train_cfg["weight_decay"]
```

Su objetivo es evitar que los pesos crezcan demasiado.

Durante el entrenamiento algunos pesos pueden alcanzar valores muy grandes.

Ejemplo:

```text
35

-80

120

-95
```

Esto puede indicar que el modelo está memorizando excesivamente los datos.

Weight Decay añade una penalización a la función de pérdida.

Conceptualmente:

```text
Loss

+

Penalización por pesos grandes
```

Como consecuencia, el optimizador prefiere pesos más pequeños.

Esto ayuda a reducir el **Overfitting**.

---

# 5. Optimizer

```python
self.optimizer_name = train_cfg["optimizer"]
```

Aquí únicamente se guarda el nombre del optimizador.

Todavía no se crea.

Más adelante aparecerá código similar a:

```python
torch.optim.Adam(...)
```

o

```python
torch.optim.SGD(...)
```

## ¿Qué hace un optimizador?

El optimizador es el algoritmo encargado de actualizar los pesos utilizando los gradientes calculados durante Backpropagation.

Todos reciben la misma información (los gradientes), pero cada uno decide de forma distinta cómo modificar los pesos.

Algunos ejemplos:

- **SGD (Stochastic Gradient Descent):** sigue directamente la dirección del gradiente.
- **Adam:** adapta automáticamente el tamaño de las actualizaciones para cada parámetro.
- **RMSprop:** ajusta el aprendizaje utilizando información histórica de los gradientes.

El optimizador no modifica la arquitectura de la red.

Únicamente modifica la forma en que aprende.

---

# 6. Scheduler

```python
self.scheduler_name = train_cfg["scheduler"]
```

El Scheduler es un mecanismo que modifica automáticamente el **Learning Rate** durante el entrenamiento.

Normalmente se utiliza un Learning Rate alto al principio para aprender rápidamente.

Posteriormente se reduce para realizar ajustes más precisos.

Visualmente:

```text
Primeras épocas

↓

Learning Rate alto

↓

Aprendizaje rápido

↓

Últimas épocas

↓

Learning Rate pequeño

↓

Ajustes finos
```

Puede compararse con lijar una pieza de madera:

- Primero se utiliza una lija gruesa para dar forma rápidamente.
- Después una lija fina para obtener un acabado preciso.

---

# 7. Early Stopping

```python
self.early_stopping_patience = train_cfg["early_stopping_patience"]
```

Este hiperparámetro sirve para detener automáticamente el entrenamiento cuando el modelo deja de mejorar.

Supongamos la pérdida de validación:

```text
0.60

0.45

0.31

0.27

0.26

0.26

0.26

0.26
```

La mejora se ha detenido.

Si:

```yaml
early_stopping_patience: 3
```

El entrenamiento esperará tres épocas sin mejora.

Si después de esas tres épocas la pérdida continúa igual o empeora, el entrenamiento finalizará automáticamente.

Esto evita entrenar más tiempo del necesario y ayuda a prevenir el Overfitting.

---

# Resumen

Los hiperparámetros controlan **cómo aprende la red neuronal**, mientras que la arquitectura controla **cómo está construida**.

| Hiperparámetro | Función |
| -------------- | ------- |
| **Batch Size** | Número  |

# Explicación de los hiperparámetros: Loss Function, Scheduler y Early Stopping

Estos hiperparámetros controlan distintos aspectos del entrenamiento de la red neuronal.

```yaml
loss_function:
  use_pos_weight: true

scheduler:
  name: "ReduceLROnPlateau"
  patience: 5
  factor: 0.5

early_stopping:
  patience: 10
  delta: 0.001
```

---

# 1. Loss Function

```yaml
loss_function:
  use_pos_weight: true
```

## ¿Qué es la Loss Function?

La **Loss Function (Función de pérdida)** es la función que mide **qué tan equivocada está la red neuronal**.

Después de realizar un **Forward Pass**, la red genera una predicción.

Por ejemplo:

```text
Valor real:

1 (Cliente bueno)

↓

La red predice:

0.92
```

Como la predicción es bastante buena, la pérdida será pequeña.

Ahora imagina otro caso.

```text
Valor real:

1

↓

La red predice:

0.12
```

Aquí el error es mucho mayor.

La función de pérdida convierte ese error en un número.

```text
Predicción correcta

↓

Loss pequeña

↓

Buen aprendizaje
```

```text
Predicción incorrecta

↓

Loss grande

↓

El optimizador debe corregir los pesos
```

Durante todo el entrenamiento el objetivo del modelo es:

> **Minimizar la Loss Function.**

---

# ¿Qué significa use_pos_weight?

```yaml
use_pos_weight: true
```

Este parámetro solamente tiene sentido cuando existe un **desbalance entre las clases**.

Por ejemplo:

```text
1000 clientes

↓

900 buenos

100 malos
```

Observa el problema.

Si la red siempre responde:

```text
"Bueno"
```

Obtendrá:

```text
900 aciertos

100 errores

↓

90% de precisión
```

Parece un excelente modelo.

Pero realmente nunca aprendió a detectar clientes malos.

---

## ¿Cómo lo soluciona pos_weight?

La idea es darle **más importancia a la clase minoritaria**.

En lugar de tratar ambos errores igual:

```text
Error cliente bueno = 1

Error cliente malo = 1
```

Se cambia por algo como:

```text
Error cliente bueno = 1

Error cliente malo = 9
```

Ahora equivocarse con un cliente malo cuesta mucho más.

La red empezará a prestar mucha más atención a esa clase.

---

## ¿Qué ocurre internamente?

Cuando se utiliza PyTorch normalmente aparece algo parecido a:

```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=...
)
```

Ese `pos_weight` aumenta la penalización cuando la red falla en ejemplos positivos.

Por eso:

```yaml
use_pos_weight: true
```

significa:

> Calcula automáticamente un peso para equilibrar ambas clases.

---

# 2. Scheduler

```yaml
scheduler:
  name: "ReduceLROnPlateau"
```

## ¿Qué es un Scheduler?

El Scheduler es un algoritmo encargado de modificar automáticamente el **Learning Rate** durante el entrenamiento.

Recordemos:

```text
Learning Rate

↓

Tamaño de los pasos
```

Al principio queremos pasos grandes.

Al final queremos pasos pequeños.

---

## ¿Qué hace ReduceLROnPlateau?

Es uno de los schedulers más utilizados.

Su comportamiento es muy sencillo.

Va observando una métrica.

Normalmente:

```text
Validation Loss
```

Mientras la pérdida siga disminuyendo:

```text
0.60

↓

0.45

↓

0.38

↓

0.29
```

No hace absolutamente nada.

Pero llega un momento:

```text
0.29

↓

0.29

↓

0.30

↓

0.29
```

La pérdida deja de mejorar.

Entonces ReduceLROnPlateau dice:

> "Parece que el modelo ya no está aprendiendo con este Learning Rate."

Y decide reducirlo.

---

# patience

```yaml
patience: 5
```

La paciencia indica:

> ¿Cuántas épocas esperar antes de reducir el Learning Rate?

Ejemplo.

```text
Epoch 20

↓

Validation Loss = 0.25
```

Luego:

```text
Epoch 21

0.25
```

```text
Epoch 22

0.25
```

```text
Epoch 23

0.25
```

Todavía no hace nada.

Cuando pasan:

```text
5 épocas
```

sin mejora,

recién ahí reduce el Learning Rate.

---

# factor

```yaml
factor: 0.5
```

Indica cuánto reducir el Learning Rate.

Supongamos:

```text
Learning Rate

0.001
```

Con:

```yaml
factor: 0.5
```

Se convierte en:

```text
0.001

↓

0.0005
```

Si vuelve a estancarse:

```text
0.0005

↓

0.00025
```

Cada vez aprende de forma más fina.

---

# ¿Por qué no bajar siempre el Learning Rate?

Porque al inicio queremos aprender rápido.

Al final queremos afinar.

Es parecido a bajar una montaña.

Primero:

```text
Pasos grandes
```

Después:

```text
Pasos pequeños
```

para no pasarnos del objetivo.

---

# 3. Early Stopping

```yaml
early_stopping:
```

El objetivo del Early Stopping es completamente diferente.

Mientras que el Scheduler intenta:

> **Seguir entrenando pero con un Learning Rate menor.**

Early Stopping dice:

> **El modelo ya no mejora. Es momento de detener el entrenamiento.**

---

# patience

```yaml
patience: 10
```

Funciona igual que en el Scheduler.

Pero aquí no reduce el Learning Rate.

Aquí cuenta:

```text
Épocas sin mejorar
```

Si llegan a:

```text
10
```

El entrenamiento termina.

---

## Ejemplo

```text
Epoch

Validation Loss
```

```text
20

0.231
```

```text
21

0.229
```

```text
22

0.229
```

```text
23

0.229
```

```text
24

0.230
```

...

Después de 10 épocas sin mejora:

```text
STOP
```

El entrenamiento finaliza automáticamente.

---

# delta

```yaml
delta: 0.001
```

Este parámetro suele generar dudas.

No toda mejora es realmente una mejora.

Supongamos:

```text
0.245000

↓

0.244999
```

La diferencia es:

```text
0.000001
```

Eso prácticamente no cambia nada.

Con:

```yaml
delta: 0.001
```

Solo se considerará una mejora cuando la pérdida disminuya al menos:

```text
0.001
```

Ejemplos:

```text
0.250

↓

0.2498
```

Mejora:

```text
0.0002
```

No cuenta.

---

```text
0.250

↓

0.248
```

Mejora:

```text
0.002
```

Sí cuenta.

Esto evita que pequeñas fluctuaciones debidas al ruido hagan creer al algoritmo que el modelo sigue mejorando.

---

# Diferencia entre Scheduler y Early Stopping

Aunque ambos observan normalmente la **Validation Loss**, hacen cosas completamente distintas.

| Scheduler                               | Early Stopping                                   |
| --------------------------------------- | ------------------------------------------------ |
| Reduce el Learning Rate.                | Detiene completamente el entrenamiento.          |
| Intenta que el modelo siga aprendiendo. | Decide que ya no vale la pena seguir entrenando. |
| El entrenamiento continúa.              | El entrenamiento termina.                        |

---

# Resumen general

Estos hiperparámetros ayudan a controlar el aprendizaje del modelo:

- **`use_pos_weight`**: aumenta la importancia de la clase minoritaria cuando los datos están desbalanceados.
- **`ReduceLROnPlateau`**: reduce automáticamente el Learning Rate cuando el modelo deja de mejorar.
- **`scheduler.patience`**: número de épocas que espera antes de reducir el Learning Rate.
- **`scheduler.factor`**: porcentaje con el que se reduce el Learning Rate.
- **`early_stopping.patience`**: número de épocas sin mejora antes de detener el entrenamiento.
- **`early_stopping.delta`**: mejora mínima necesaria para considerar que realmente hubo progreso.

En conjunto, estos hiperparámetros permiten que el entrenamiento sea más eficiente, evitan entrenar durante más tiempo del necesario y ayudan a obtener un modelo que generalice mejor sobre datos nuevos.

# Bloque: Preparación del entorno de entrenamiento

Después de leer toda la configuración del archivo YAML, el constructor comienza a preparar el entorno donde se ejecutará el entrenamiento.

Este bloque no entrena la red neuronal ni crea el modelo.

Su objetivo es dejar listo todo lo necesario antes de comenzar el entrenamiento.

El código es el siguiente:

```python
# -------------------------------------------------
# Reproducibilidad
# -------------------------------------------------
np.random.seed(self.random_state)
torch.manual_seed(self.random_state)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(self.random_state)

# -------------------------------------------------
# Device
# -------------------------------------------------
self.device = self._resolve_device()

# -------------------------------------------------
# Instancias auxiliares
# -------------------------------------------------
self.data_preprocessor = CreditDataPreprocessor()

self.history: Dict[str, List[float]] = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_auc": [],
    "val_auc": [],
}
```

---

# 1. Reproducibilidad

```python
np.random.seed(self.random_state)
```

Esta línea fija la semilla (_seed_) del generador de números aleatorios de **NumPy**.

## ¿Qué es una semilla?

Muchas operaciones de Machine Learning utilizan números aleatorios.

Por ejemplo:

- Inicializar pesos.
- Mezclar el dataset.
- Dividir entrenamiento y validación.
- Aplicar Dropout.

Aunque parezcan completamente aleatorios, realmente son generados por un algoritmo llamado **Pseudo Random Number Generator (PRNG)**.

Ese algoritmo siempre comienza desde un número inicial llamado **seed**.

Visualmente:

```text
Seed = 42

↓

Generador

↓

0.37

↓

0.81

↓

0.12

↓

0.94
```

Si vuelves a utilizar exactamente la misma semilla:

```text
Seed = 42

↓

Generador

↓

0.37

↓

0.81

↓

0.12

↓

0.94
```

Obtendrás exactamente la misma secuencia.

---

## ¿Por qué esto es importante?

Supongamos que entrenas hoy una red neuronal.

Obtienes:

```text
Accuracy = 91.4%
```

Mañana ejecutas exactamente el mismo código.

Sin fijar la semilla podrías obtener:

```text
Accuracy = 90.8%
```

o

```text
Accuracy = 91.9%
```

No porque el código haya cambiado, sino porque algunas operaciones aleatorias fueron diferentes.

Al fijar la semilla:

```python
np.random.seed(42)
```

las operaciones aleatorias de NumPy siempre producirán los mismos resultados, haciendo que los experimentos sean reproducibles.

---

# 2. Semilla de PyTorch

```python
torch.manual_seed(self.random_state)
```

Esta instrucción hace exactamente la misma idea, pero para PyTorch.

PyTorch utiliza números aleatorios en muchas situaciones, por ejemplo:

- Inicialización de los pesos de las capas (`nn.Linear`).
- Aplicación de Dropout.
- Algunas operaciones internas de entrenamiento.

Sin fijar la semilla, cada entrenamiento comenzaría con pesos iniciales distintos.

Ejemplo:

Primer entrenamiento:

```text
Peso inicial

0.42
```

Segundo entrenamiento:

```text
Peso inicial

-0.18
```

Con la misma semilla:

```python
torch.manual_seed(42)
```

los pesos iniciales serán siempre los mismos, permitiendo comparar experimentos de forma justa.

---

# 3. ¿Por qué también existe esta parte?

```python
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(self.random_state)
```

Aquí aparece un detalle importante.

Hasta ahora la semilla solo se había configurado para la CPU.

Pero si el entrenamiento utiliza una GPU NVIDIA mediante CUDA, la GPU tiene su propio generador de números aleatorios.

Por eso primero se comprueba:

```python
torch.cuda.is_available()
```

Esta función devuelve:

```text
True
```

si existe una GPU CUDA disponible.

o

```text
False
```

si únicamente se utilizará la CPU.

Si existe una GPU, entonces:

```python
torch.cuda.manual_seed_all(self.random_state)
```

establece la misma semilla para todos los generadores aleatorios asociados a las GPU disponibles.

Esto garantiza que tanto CPU como GPU produzcan resultados reproducibles en la medida de lo posible.

---

# 4. Selección del dispositivo

```python
self.device = self._resolve_device()
```

Aquí todavía no vemos la implementación de `_resolve_device()`, pero podemos entender perfectamente su objetivo.

Su responsabilidad es decidir dónde se ejecutará el entrenamiento.

Generalmente devolverá alguno de estos dispositivos:

```text
CPU
```

o

```text
CUDA (GPU NVIDIA)
```

En PyTorch esto suele representarse mediante un objeto:

```python
torch.device("cpu")
```

o

```python
torch.device("cuda")
```

Más adelante, cuando se cree el modelo, aparecerá algo similar a:

```python
model.to(self.device)
```

Esto moverá todos los pesos del modelo al dispositivo seleccionado.

También los datos deberán enviarse al mismo dispositivo durante el entrenamiento:

```python
x = x.to(self.device)
y = y.to(self.device)
```

Una regla muy importante en PyTorch es:

> El modelo y los tensores deben estar en el mismo dispositivo.

Si el modelo está en la GPU y los datos permanecen en la CPU, PyTorch generará un error.

---

# 5. Creación del preprocesador

```python
self.data_preprocessor = CreditDataPreprocessor()
```

Aquí el `Trainer` crea una instancia del preprocesador que analizamos anteriormente.

Recordemos que esa clase era responsable de:

- Escalar variables numéricas.
- Codificar variables categóricas.
- Separar características y etiquetas.

El `Trainer` no implementa esa lógica directamente.

Simplemente crea un objeto especializado para delegarle esa responsabilidad.

Este diseño sigue el principio de responsabilidad única (Single Responsibility Principle).

Cada clase tiene una única función claramente definida.

---

# 6. Historial del entrenamiento

Finalmente aparece:

```python
self.history = {
    ...
}
```

Este diccionario servirá para almacenar todas las métricas obtenidas durante el entrenamiento.

Inicialmente todas las listas están vacías.

```python
{
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_auc": [],
    "val_auc": []
}
```

¿Por qué listas?

Porque en cada época se obtendrá un nuevo valor.

Supongamos un entrenamiento de cinco épocas.

La pérdida de entrenamiento podría evolucionar así:

```text
Epoch 1 → 0.68

Epoch 2 → 0.51

Epoch 3 → 0.39

Epoch 4 → 0.30

Epoch 5 → 0.24
```

Entonces la lista terminaría siendo:

```python
train_loss = [
    0.68,
    0.51,
    0.39,
    0.30,
    0.24
]
```

Lo mismo ocurre con el resto de métricas.

Por ejemplo:

```python
val_accuracy = [
    0.74,
    0.79,
    0.84,
    0.87,
    0.89
]
```

---

# ¿Para qué sirve guardar este historial?

Tener todas las métricas almacenadas permite posteriormente:

- Dibujar gráficas del entrenamiento.
- Comparar entrenamiento y validación.
- Detectar Overfitting.
- Analizar la evolución del modelo.
- Registrar resultados en MLflow.

Por ejemplo, una gráfica típica utiliza directamente estos valores:

```text
Loss

│\
│ \
│  \
│   \
│    \_____

└───────────── Epoch
```

Sin este historial sería imposible reconstruir cómo aprendió el modelo.

---

# Resumen del bloque

Este bloque prepara completamente el entorno antes de comenzar el entrenamiento.

Cada parte tiene una responsabilidad específica:

| Instrucción                    | Función                                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------- |
| `np.random.seed()`             | Hace reproducibles las operaciones aleatorias de NumPy.                                         |
| `torch.manual_seed()`          | Hace reproducibles las operaciones aleatorias de PyTorch en CPU.                                |
| `torch.cuda.manual_seed_all()` | Hace reproducibles las operaciones aleatorias en GPU.                                           |
| `_resolve_device()`            | Decide si el entrenamiento se ejecutará en CPU o GPU.                                           |
| `CreditDataPreprocessor()`     | Crea el objeto encargado del preprocesamiento de los datos.                                     |
| `history`                      | Almacena todas las métricas obtenidas durante el entrenamiento para analizarlas posteriormente. |

En conjunto, este bloque deja preparado el entorno de trabajo para que el entrenamiento pueda comenzar de forma reproducible, organizada y con toda la información necesaria para registrar la evolución del modelo.

# Método `_load_and_split_data()`

Este es el primer método funcional del `Trainer`.

```python
def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
```

Su responsabilidad es muy clara:

> **Cargar el dataset desde el disco y dividirlo en dos conjuntos: entrenamiento y validación.**

Observa que este método **no entrena la red**, **no crea el modelo** y **no realiza el preprocesamiento**.

Únicamente prepara los datos para las siguientes etapas.

Visualmente, el flujo es:

```text
Archivo CSV

↓

Leer DataFrame

↓

Limpiar columnas innecesarias

↓

Dividir dataset

↓

DataFrame entrenamiento
+
DataFrame validación
```

---

# Tipo de retorno

```python
-> Tuple[pd.DataFrame, pd.DataFrame]
```

El método devuelve dos objetos.

```text
(
    df_train,
    df_val
)
```

Es decir:

- DataFrame para entrenamiento.
- DataFrame para validación.

---

# Registro de información (Logging)

Las primeras líneas son:

```python
log.info("--- Load data ---")
```

y

```python
log.info(f"✔ Loading data from: {self.dataset_path}")
```

Estas instrucciones únicamente escriben mensajes en el sistema de logs.

No modifican absolutamente nada del entrenamiento.

Su objetivo es que, cuando ejecutes el programa, puedas ver qué está ocurriendo.

Por ejemplo:

```text
--- Load data ---

✔ Loading data from:

data/german_credit.csv
```

Cuando un entrenamiento dura varios minutos o incluso horas, estos mensajes ayudan mucho para saber en qué etapa se encuentra el proceso.

---

# Verificación de existencia del archivo

Después aparece:

```python
if not self.dataset_path.exists():
```

Recordemos que:

```python
self.dataset_path
```

es un objeto `Path`.

El método:

```python
.exists()
```

pregunta al sistema operativo:

> **¿Existe realmente este archivo?**

Devuelve:

```text
True
```

si el archivo existe.

o

```text
False
```

si no existe.

---

## ¿Por qué comprobar esto?

Imagina que el YAML dice:

```yaml
dataset_path: data/german_credit.csv
```

Pero accidentalmente borras el archivo.

Sin esta comprobación, el programa fallaría más adelante con un error poco claro.

En cambio aquí se detecta inmediatamente.

---

# Lanzamiento de una excepción

```python
raise FileNotFoundError(
    f"No se encontró el dataset: {self.dataset_path}"
)
```

La palabra clave:

```python
raise
```

significa:

> **Detener inmediatamente la ejecución y lanzar un error.**

No intenta continuar.

No intenta crear un DataFrame vacío.

Simplemente informa claramente cuál es el problema.

Este tipo de validaciones son una buena práctica porque permiten detectar errores al inicio del proceso.

---

# Lectura del CSV

```python
df = pd.read_csv(self.dataset_path)
```

Aquí sí ocurre la primera operación real sobre los datos.

`pandas.read_csv()`:

- abre el archivo CSV,
- interpreta filas y columnas,
- crea un `DataFrame`.

Por ejemplo:

```text
CSV

Age,Job,Risk

25,2,good

40,1,bad
```

Se convierte en:

```text
DataFrame

Age   Job   Risk

25     2    good

40     1    bad
```

A partir de este momento todas las operaciones se realizarán sobre ese DataFrame.

---

# Eliminación de la columna "Unnamed: 0"

Después aparece:

```python
if "Unnamed: 0" in df.columns:
```

Primero comprobamos si existe una columna llamada:

```text
Unnamed: 0
```

¿Por qué aparece esa columna?

Normalmente ocurre cuando un DataFrame se guarda en CSV incluyendo el índice.

Ejemplo:

```python
df.to_csv(...)
```

produce algo parecido a:

```text
,Age,Job,Risk

0,25,2,good

1,40,1,bad
```

Cuando se vuelve a leer:

```python
pd.read_csv(...)
```

esa primera columna pasa a llamarse:

```text
Unnamed: 0
```

Pero realmente no contiene información útil.

Solo era el índice del DataFrame anterior.

---

## Eliminación

```python
df = df.drop(columns=["Unnamed: 0"])
```

Aquí se elimina completamente esa columna.

Es equivalente a decir:

> "Quiero todas las columnas excepto `Unnamed: 0`."

El resultado es un DataFrame limpio.

---

# Separación entrenamiento-validación

Ahora llegamos a la parte más importante del método.

```python
df_train, df_val = train_test_split(...)
```

Aquí aparece una función muy utilizada de `scikit-learn`.

Su responsabilidad es dividir el dataset en dos conjuntos independientes.

Visualmente:

```text
Dataset completo

↓

train_test_split

↓

Entrenamiento

+

Validación
```

---

# ¿Por qué dividir el dataset?

Imagina un profesor.

Primero te enseña ejercicios.

Después hace un examen con ejercicios nuevos.

Si el examen fuera exactamente el mismo que los ejercicios vistos en clase, no sabría si realmente aprendiste o simplemente los memorizaste.

Con una red neuronal ocurre exactamente lo mismo.

Si entrenamos y evaluamos usando los mismos datos, el modelo podría memorizar las respuestas.

Por eso se utilizan dos conjuntos diferentes:

```text
Entrenamiento

↓

Aprender
```

```text
Validación

↓

Evaluar qué tan bien generaliza
```

---

# `test_size`

```python
test_size=self.test_size
```

Este parámetro indica qué porcentaje del dataset se reservará para validación.

Ejemplo:

```python
test_size = 0.2
```

Entonces:

```text
1000 muestras

↓

800 entrenamiento

200 validación
```

---

# `random_state`

```python
random_state=self.random_state
```

La división del dataset se realiza de forma aleatoria.

Pero gracias a `random_state` esa división siempre será la misma.

Por ejemplo:

Sin semilla:

```text
Entrenamiento

A

C

F

G
```

La siguiente ejecución podría generar:

```text
Entrenamiento

B

D

E

H
```

Con un `random_state` fijo:

Siempre se obtendrá exactamente la misma división.

Esto permite que los experimentos sean reproducibles.

---

# `stratify`

```python
stratify=df[self.data_preprocessor.target_feature]
```

Este es probablemente el parámetro más importante de `train_test_split`.

Su función es mantener la misma proporción de clases en ambos conjuntos.

Supongamos un dataset con:

```text
1000 clientes

↓

700 buenos

300 malos
```

Eso equivale aproximadamente a:

```text
70%

30%
```

Sin `stratify`, una división aleatoria podría producir algo como:

Entrenamiento:

```text
760 buenos

40 malos
```

Validación:

```text
40 buenos

160 malos
```

Las proporciones quedarían completamente desbalanceadas.

En consecuencia, el modelo se entrenaría y evaluaría sobre distribuciones distintas.

Con `stratify`:

```text
Entrenamiento

70%

30%
```

```text
Validación

70%

30%
```

Ambos conjuntos mantienen exactamente la misma distribución de clases.

En este proyecto se utiliza:

```python
self.data_preprocessor.target_feature
```

que corresponde a la columna objetivo:

```text
Risk
```

Por tanto, `train_test_split` garantiza que la proporción de clientes **good** y **bad** sea prácticamente la misma tanto en entrenamiento como en validación.

---

# Retorno del método

Finalmente:

```python
return df_train, df_val
```

El método devuelve los dos DataFrames creados.

Es importante observar que todavía contienen los datos originales.

Por ejemplo:

```text
Sex = male

Housing = own

Risk = good
```

Las variables categóricas aún no han sido codificadas.

Los valores numéricos aún no han sido escalados.

Todo ese trabajo lo realizará posteriormente el `CreditDataPreprocessor`.

---

# Resumen

Este método constituye la primera etapa del flujo de datos del entrenamiento.

Su responsabilidad consiste en:

1. Verificar que el archivo del dataset exista.
2. Leer el CSV utilizando Pandas.
3. Eliminar una posible columna de índice (`Unnamed: 0`).
4. Dividir el dataset en entrenamiento y validación mediante `train_test_split`.
5. Mantener la proporción de clases utilizando `stratify`.
6. Devolver ambos DataFrames para que posteriormente sean preprocesados.

El flujo completo puede resumirse así:

```text
Archivo CSV
      │
      ▼
Leer con Pandas
      │
      ▼
Eliminar columnas innecesarias
      │
      ▼
train_test_split
      │
      ├──────────────► DataFrame entrenamiento
      │
      └──────────────► DataFrame validación
```

La decisión de separar este proceso en un método independiente sigue nuevamente el principio de **responsabilidad única**: este método únicamente se encarga de cargar y dividir los datos, mientras que el preprocesamiento y el entrenamiento quedan delegados a otras partes del sistema.

# Método `_preprocess_data()`

Este método representa la transición entre el mundo de **Pandas** y el mundo de **PyTorch**.

Hasta este punto del entrenamiento únicamente tenemos dos `DataFrame`:

```text
df_train

df_val
```

Sin embargo, una red neuronal implementada con PyTorch **no puede trabajar directamente con DataFrames**.

Las capas como:

```python
nn.Linear(...)
```

esperan recibir como entrada objetos de tipo:

```python
torch.Tensor
```

Por esa razón este método tiene cuatro responsabilidades principales:

1. Ajustar el preprocesador utilizando únicamente el conjunto de entrenamiento.
2. Transformar tanto entrenamiento como validación utilizando ese mismo preprocesador.
3. Convertir los datos procesados en tensores de PyTorch.
4. Guardar el preprocesador entrenado para reutilizarlo posteriormente durante la inferencia.

Visualmente el flujo es:

```text
DataFrame entrenamiento
                │
                ▼
        Ajustar preprocesador
                │
                ▼
Transformar entrenamiento
Transformar validación
                │
                ▼
Convertir a Tensor
                │
                ▼
Guardar preprocesador
                │
                ▼
Retornar tensores
```

---

# Definición del método

```python
def _preprocess_data(
    self,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
```

Recibe:

- El DataFrame de entrenamiento.
- El DataFrame de validación.

Y devuelve cuatro tensores:

```text
x_train_tensor

y_train_tensor

x_val_tensor

y_val_tensor
```

Es decir, todo queda listo para alimentar la red neuronal.

---

# Inicio del proceso

```python
log.info("--- Preprocessing data ---")
```

Como vimos anteriormente, esto únicamente escribe un mensaje en el sistema de logs.

No modifica ningún dato.

Simplemente informa que comienza la etapa de preprocesamiento.

---

# Ajustar el preprocesador

```python
preprocessor = self.data_preprocessor.fit_preprocessor(df_train)
```

Esta línea es una de las más importantes del método.

Recordemos la clase `CreditDataPreprocessor`.

Allí estudiamos el método:

```python
fit_preprocessor(df)
```

Ese método hacía tres cosas:

1. Construía el `ColumnTransformer`.
2. Construía los `Pipeline`.
3. Ejecutaba:

```python
preprocessor.fit(x_train)
```

---

## ¿Qué significa "fit"?

Aquí es donde muchas personas se confunden.

**Fit NO significa transformar.**

Significa:

> **Aprender información del conjunto de entrenamiento.**

Por ejemplo.

El `StandardScaler` necesita conocer:

- media
- desviación estándar

Supongamos la columna:

```text
Edad

20

30

40

50
```

Durante el `fit` calcula:

```text
Media = 35

Desviación = 11.18
```

Pero todavía no modifica ninguna fila.

Simplemente aprende esos valores.

---

El `OneHotEncoder` también aprende información.

Por ejemplo:

```text
Housing

Own

Rent

Free
```

Aprende que existen exactamente tres categorías.

Tampoco transforma nada todavía.

---

En resumen:

```text
fit()

↓

Aprender estadísticas
```

---

# Transformar entrenamiento

Después aparece:

```python
x_train_processed, y_train = (
    self.data_preprocessor.process_data(
        df_train,
        preprocessor
    )
)
```

Aquí ya no estamos utilizando `fit`.

Ahora utilizamos:

```python
transform()
```

Recordemos que `process_data()` hacía:

```python
preprocessor.transform(x)
```

---

## ¿Qué ocurre durante transform?

Ahora sí se aplican las transformaciones.

Ejemplo.

Antes:

```text
Edad

20

40

60
```

Después del `StandardScaler`:

```text
-1.22

0.10

1.13
```

---

Variables categóricas.

Antes:

```text
Housing

Own

Rent

Free
```

Después del `OneHotEncoder`:

```text
Own Free Rent

1    0    0

0    1    0

0    0    1
```

El resultado ya es completamente numérico.

---

# Transformar validación

Después:

```python
x_val_processed, y_val =
    self.data_preprocessor.process_data(
        df_val,
        preprocessor
    )
```

Esta línea parece igual.

Pero existe una diferencia muy importante.

Aquí **NO se vuelve a ejecutar `fit()`**.

Únicamente:

```text
transform()
```

---

## ¿Por qué?

Porque el conjunto de validación representa datos que el modelo nunca ha visto.

Si ejecutáramos:

```python
fit(df_val)
```

el `StandardScaler` calcularía una nueva media.

El `OneHotEncoder` aprendería nuevas categorías.

Eso significaría que estamos utilizando información del conjunto de validación durante el entrenamiento.

Este problema recibe el nombre de:

```text
Data Leakage
```

Es uno de los errores más comunes en Machine Learning.

La regla correcta siempre es:

```text
Train

↓

fit()

↓

Aprender estadísticas
```

Después:

```text
Validation

↓

transform()

↓

Usar las estadísticas del Train
```

Nunca al revés.

---

# Conversión a Tensor

Llegamos ahora al paso donde entramos definitivamente en PyTorch.

Primero:

```python
x_train_tensor =
    torch.tensor(
        x_train_processed,
        dtype=torch.float32
    )
```

Hasta ahora:

```text
x_train_processed
```

era una matriz de NumPy.

Pero PyTorch trabaja con:

```python
torch.Tensor
```

Por eso utilizamos:

```python
torch.tensor(...)
```

Visualmente:

```text
NumPy Array

↓

torch.tensor()

↓

Tensor
```

---

# ¿Por qué float32?

```python
dtype=torch.float32
```

La mayoría de operaciones matemáticas de PyTorch utilizan este tipo de dato.

Además:

- ocupa menos memoria que `float64`;
- es más rápido;
- está optimizado para GPU.

Por eso prácticamente todos los modelos utilizan:

```python
torch.float32
```

---

# Conversión de las etiquetas

Ahora aparece:

```python
y_train_tensor =
    torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).view(-1,1)
```

Aquí sucede algo interesante.

Primero:

```python
y_train.values
```

convierte la Serie de Pandas en un arreglo de NumPy.

Ejemplo:

Antes:

```text
Pandas Series

0

1

0

1
```

Después:

```text
NumPy

[0,1,0,1]
```

Luego:

```python
torch.tensor(...)
```

produce:

```text
Tensor

[0,1,0,1]
```

---

# ¿Qué hace `.view(-1,1)`?

Esta instrucción cambia la forma (_shape_) del tensor.

Inicialmente:

```text
[0,1,0,1]
```

tiene forma:

```text
(4,)
```

Es un vector.

Después:

```python
.view(-1,1)
```

obtendremos:

```text
[[0],

 [1],

 [0],

 [1]]
```

Forma:

```text
(4,1)
```

---

## ¿Qué significa el -1?

PyTorch calcula automáticamente ese tamaño.

Por ejemplo.

Si existen:

```text
1000 ejemplos
```

`.view(-1,1)` significa:

```text
(1000,1)
```

Si existen:

```text
350 ejemplos
```

obtendremos:

```text
(350,1)
```

Nunca necesitamos escribir ese número manualmente.

---

## ¿Por qué necesitamos una columna?

Recuerda la salida del modelo.

La última capa era:

```python
nn.Linear(...,1)
```

Eso produce:

```text
(batch_size,1)
```

Por lo tanto las etiquetas deben tener exactamente la misma forma.

Modelo:

```text
(64,1)
```

Etiquetas:

```text
(64,1)
```

Si las etiquetas fueran:

```text
(64,)
```

muchas funciones de pérdida generarían errores o realizarían conversiones internas.

---

# Conversión del conjunto de validación

Las siguientes líneas realizan exactamente el mismo procedimiento para validación.

```python
x_val_tensor
```

↓

Tensor de características.

```python
y_val_tensor
```

↓

Tensor de etiquetas.

---

# Guardar el preprocesador

Ahora aparece algo muy importante.

```python
path_preprocessor =
    self.local_models_dir /
    self.preprocessor_filename
```

Se construye la ruta donde se almacenará el preprocesador.

Después:

```python
joblib.dump(
    preprocessor,
    path_preprocessor
)
```

Aquí ya no estamos guardando el modelo.

Estamos guardando el objeto:

```text
ColumnTransformer
```

completo.

---

## ¿Por qué guardar el preprocesador?

Imagina que mañana recibes un nuevo cliente.

```text
Edad = 42

Housing = Own

Purpose = Car
```

La red neuronal espera exactamente las mismas transformaciones que durante el entrenamiento.

Por eso debemos reutilizar el mismo `StandardScaler` y el mismo `OneHotEncoder`.

No podemos crear otros nuevos.

El flujo correcto será:

```text
Entrenamiento

↓

fit()

↓

Guardar preprocessor
```

Luego, durante la inferencia:

```text
Cargar preprocessor

↓

transform()

↓

Predicción
```

Así garantizamos que los datos nuevos se procesen exactamente igual que los datos de entrenamiento.

---

# Retorno

Finalmente:

```python
return (
    x_train_tensor,
    y_train_tensor,
    x_val_tensor,
    y_val_tensor
)
```

El método devuelve todo lo necesario para comenzar el entrenamiento.

A partir de este punto ya no trabajaremos con DataFrames.

Toda la información se encuentra en formato `torch.Tensor`, lista para ser utilizada por la red neuronal.

---

# Resumen

Este método constituye el puente entre el preprocesamiento y el entrenamiento de la red neuronal.

Su flujo completo es:

```text
DataFrame Train
        │
        ▼
fit_preprocessor()
        │
        ▼
Aprender estadísticas
        │
        ▼
process_data()
        │
        ▼
Transformar Train y Validation
        │
        ▼
Convertir a Tensor
        │
        ▼
Guardar preprocesador
        │
        ▼
Retornar:
x_train_tensor
y_train_tensor
x_val_tensor
y_val_tensor
```

La decisión más importante del método es que **el preprocesador solo se ajusta (`fit`) utilizando el conjunto de entrenamiento**, mientras que el conjunto de validación únicamente se transforma (`transform`). Esto evita el **Data Leakage** y garantiza que la evaluación del modelo sea realista y comparable con el comportamiento que tendrá cuando reciba datos completamente nuevos.

# Método `_compute_metrics()`

Este método tiene una única responsabilidad:

> **Calcular todas las métricas de evaluación del modelo a partir de las predicciones realizadas por la red neuronal.**

Es importante entender que este método **no entrena la red neuronal**.

Tampoco modifica los pesos.

Tampoco calcula gradientes.

Su único trabajo consiste en responder una pregunta:

> **¿Qué tan bueno está siendo el modelo?**

Visualmente:

```text
Datos reales
        │
        ▼
Predicciones del modelo
        │
        ▼
Comparar ambos
        │
        ▼
Calcular métricas
```

---

# ¿Por qué es un `@staticmethod`?

Lo primero que aparece es:

```python
@staticmethod
```

Esto significa que el método **no depende del estado interno del objeto**.

Observa que dentro del método nunca aparece:

```python
self.algo
```

No utiliza:

- `self.device`
- `self.history`
- `self.learning_rate`
- `self.model`

Únicamente trabaja con los argumentos que recibe.

Por eso no necesita acceder al objeto completo.

Podría llamarse incluso así:

```python
CreditScoringModelTraining._compute_metrics(...)
```

sin crear una instancia de la clase.

Es simplemente una función agrupada dentro de la clase por organización.

---

# Parámetros de entrada

```python
def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
)
```

Recibe tres cosas.

## 1. `y_true`

```python
y_true
```

Son las etiquetas reales del dataset.

Ejemplo:

```text
Cliente 1 → good

Cliente 2 → bad

Cliente 3 → good

Cliente 4 → bad
```

Después del preprocesamiento:

```text
[1,0,1,0]
```

Estas son las respuestas correctas.

---

## 2. `y_prob`

```python
y_prob
```

Aquí **NO llegan clases**.

Llegan probabilidades.

Por ejemplo:

```text
[

0.92,

0.13,

0.81,

0.44

]
```

Cada número representa:

> Probabilidad de que el cliente pertenezca a la clase positiva ("good").

Todavía no sabemos si será clasificado como 0 o 1.

---

## 3. `threshold`

```python
threshold = 0.5
```

Ya estudiamos este concepto cuando analizamos:

```python
predict()
```

La red produce probabilidades.

Pero muchas métricas necesitan clases binarias.

Por eso utilizamos un umbral.

---

# Conversión de probabilidades en clases

```python
y_pred = (y_prob >= threshold).astype(int)
```

Esta línea es muy elegante porque aprovecha la vectorización de NumPy.

Supongamos:

```python
y_prob =

[

0.81,

0.22,

0.67,

0.49

]
```

Primero:

```python
y_prob >= 0.5
```

produce:

```python
[

True,

False,

True,

False

]
```

Después:

```python
.astype(int)
```

convierte:

```python
True → 1

False → 0
```

Resultado:

```python
[

1,

0,

1,

0

]
```

Ahora sí tenemos clases binarias.

---

# Accuracy

```python
acc = accuracy_score(
    y_true,
    y_pred
)
```

La Accuracy responde a una pregunta muy sencilla.

> ¿Qué porcentaje de predicciones fueron correctas?

Su fórmula es:

```text
Predicciones correctas
────────────────────────
Total de ejemplos
```

Ejemplo.

```text
100 clientes

↓

92 correctos

↓

Accuracy

92%
```

Es la métrica más conocida.

Pero no siempre es la mejor.

---

## Problema de Accuracy

Supongamos:

```text
1000 clientes

990 buenos

10 malos
```

Un modelo que siempre responda:

```text
Bueno
```

obtendrá:

```text
990 aciertos

↓

99%
```

Parece excelente.

Pero jamás detectó un cliente malo.

Por eso existen métricas adicionales.

---

# Precision, Recall y F1

Aquí aparece:

```python
precision_recall_fscore_support(...)
```

Esta función calcula tres métricas al mismo tiempo.

---

# Precision

La Precision responde:

> **Cuando el modelo dice "good", ¿qué porcentaje realmente era "good"?**

Ejemplo.

El modelo predice:

```text
20 clientes buenos
```

Pero solo:

```text
18
```

eran realmente buenos.

Entonces:

```text
Precision

18/20

=

90%
```

Una Precision alta significa que el modelo comete pocos falsos positivos.

---

# Recall

El Recall responde una pregunta distinta.

> **De todos los clientes realmente buenos, ¿cuántos logró encontrar el modelo?**

Ejemplo.

Hay:

```text
100 clientes buenos
```

El modelo encontró:

```text
82
```

Entonces:

```text
Recall

82/100

=

82%
```

Un Recall alto significa que el modelo deja escapar pocos casos positivos.

---

# F1 Score

El F1 Score combina Precision y Recall.

Es una especie de equilibrio entre ambas.

Si una es muy alta y la otra muy baja:

el F1 también será bajo.

Por eso suele utilizarse cuando ambas métricas son importantes.

---

# ¿Qué significa `average="binary"`?

```python
average="binary"
```

La función `precision_recall_fscore_support()` puede trabajar con:

- clasificación binaria;
- clasificación multiclase;
- clasificación multilabel.

Aquí nuestro problema solamente tiene:

```text
Bad

Good
```

Por eso indicamos:

```python
average="binary"
```

Esto le dice a Scikit-Learn que calcule las métricas como un problema de clasificación binaria.

---

# ¿Qué significa `zero_division=0`?

Supongamos un caso extremo.

El modelo predice:

```text
Todo es malo
```

Entonces nunca predijo:

```text
Bueno
```

La Precision sería:

```text
0 / 0
```

Matemáticamente esa división no existe.

En lugar de lanzar un error:

```python
zero_division=0
```

indica:

> Si ocurre esa situación, devuelve 0.

Así el entrenamiento puede continuar normalmente.

---

# ROC-AUC

Después aparece:

```python
auc = roc_auc_score(
    y_true,
    y_prob
)
```

Esta es una de las métricas más importantes en clasificación.

A diferencia de Accuracy, aquí **no utilizamos las clases binarias**.

Utilizamos directamente:

```python
y_prob
```

¿Por qué?

Porque ROC-AUC evalúa **qué tan bien ordena el modelo las probabilidades**, independientemente del umbral elegido.

Imagina dos clientes:

```text
Cliente A

Probabilidad = 0.90

↓

Good
```

```text
Cliente B

Probabilidad = 0.20

↓

Bad
```

El modelo los ordenó correctamente.

ROC-AUC mide precisamente esa capacidad de separar ambas clases.

Su valor siempre está entre:

```text
0

↓

1
```

Interpretación general:

| Valor         | Interpretación                                                                    |
| ------------- | --------------------------------------------------------------------------------- |
| **1.0**       | Clasificación perfecta.                                                           |
| **0.9 - 1.0** | Excelente separación entre clases.                                                |
| **0.8 - 0.9** | Muy buen modelo.                                                                  |
| **0.7 - 0.8** | Aceptable.                                                                        |
| **0.5**       | Equivalente a adivinar al azar.                                                   |
| **< 0.5**     | El modelo clasifica peor que el azar (probablemente está invirtiendo las clases). |

En problemas de riesgo crediticio, medicina o fraude, ROC-AUC suele ser una de las métricas más utilizadas porque no depende de un único umbral de decisión.

---

# ¿Por qué aparece un `try/except`?

```python
try:
    auc = roc_auc_score(...)
except ValueError:
    auc = float("nan")
```

ROC-AUC necesita que existan ejemplos de ambas clases.

Imagina un conjunto de validación donde, por alguna razón, todos los ejemplos son:

```text
Good
```

No existe ninguna comparación posible entre clases.

En ese caso `roc_auc_score()` lanza un `ValueError`.

Para evitar que el entrenamiento se detenga, el código captura esa excepción y asigna:

```python
float("nan")
```

`NaN` significa **Not a Number**, indicando que esa métrica no pudo calcularse para ese conjunto de datos.

---

# Retorno

Finalmente el método devuelve todas las métricas agrupadas en un diccionario.

```python
return {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": auc,
}
```

¿Por qué un diccionario?

Porque posteriormente será muy sencillo acceder a cualquier métrica.

Ejemplo:

```python
metrics = self._compute_metrics(...)

print(metrics["accuracy"])
print(metrics["f1"])
print(metrics["roc_auc"])
```

También resulta muy cómodo registrar estas métricas en herramientas como MLflow o almacenarlas en el historial del entrenamiento.

---

# Resumen

Este método constituye el sistema de evaluación del modelo.

Su flujo completo es:

```text
Probabilidades del modelo
            │
            ▼
Aplicar threshold
            │
            ▼
Obtener clases predichas
            │
            ▼
Compararlas con las etiquetas reales
            │
            ▼
Calcular:
• Accuracy
• Precision
• Recall
• F1 Score
• ROC-AUC
            │
            ▼
Devolver todas las métricas en un diccionario
```

Mientras que la **función de pérdida (Loss Function)** se utiliza para que el optimizador aprenda y actualice los pesos de la red neuronal, este método tiene un propósito completamente distinto: **medir objetivamente la calidad del modelo** durante el entrenamiento y la validación. Gracias a estas métricas es posible comparar distintos modelos, detectar problemas como el desbalance de clases y decidir cuál versión ofrece el mejor rendimiento.

# Método `_setup_loss_function()`

Este es uno de los métodos más importantes de todo el entrenamiento porque aquí se decide **cómo va a medir la red neuronal sus errores**.

Recordemos el ciclo de entrenamiento que ocurre en cada batch:

```text
Forward Pass
        │
        ▼
Predicciones (logits)
        │
        ▼
Loss Function
        │
        ▼
¿Cuánto se equivocó la red?
        │
        ▼
Backpropagation
        │
        ▼
Gradientes
        │
        ▼
Optimizer
        │
        ▼
Actualizar pesos
```

La **Loss Function** es el puente entre las predicciones y el aprendizaje.

Si la pérdida está mal definida, el optimizador recibirá información incorrecta y la red aprenderá mal.

---

# Definición del método

```python
def _setup_loss_function(
    self,
    y_train: torch.Tensor
) -> nn.Module:
```

Recibe únicamente:

```python
y_train
```

¿Por qué necesita las etiquetas de entrenamiento?

Porque este método debe decidir si el dataset está **balanceado** o **desbalanceado**.

Y para saberlo necesita contar cuántos ejemplos existen de cada clase.

---

# Primera decisión

```python
if self.use_pos_weight:
```

Recordemos que este valor viene del YAML.

Por ejemplo:

```yaml
loss_function:
  use_pos_weight: true
```

Si es:

```text
True
```

intentará construir una función de pérdida ponderada.

Si es:

```text
False
```

utilizará la función estándar.

---

# ¿Por qué existe `pos_weight`?

Supongamos un dataset de crédito.

```text
1000 clientes

↓

900 Good

100 Bad
```

Existe un gran desbalance.

Visualmente:

```text
██████████████████████

Good

████

Bad
```

Si ambas clases cuestan exactamente lo mismo durante el entrenamiento:

```text
Error Good = 1

Error Bad = 1
```

la red descubrirá rápidamente que decir siempre:

```text
Good
```

produce muy pocos errores.

No aprenderá realmente a detectar clientes malos.

Por eso necesitamos aumentar el costo de equivocarse en la clase minoritaria.

---

# Conversión del tensor

Primero aparece:

```python
y_train_cpu = (
    y_train
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)
```

Esta línea parece complicada, pero realmente son cuatro operaciones encadenadas.

Vamos una por una.

---

# 1. `detach()`

```python
y_train.detach()
```

Recordemos que PyTorch construye automáticamente un **grafo computacional** durante el entrenamiento.

Ese grafo sirve para calcular gradientes.

Pero aquí no queremos calcular gradientes.

Solo queremos contar etiquetas.

Con:

```python
detach()
```

le decimos a PyTorch:

> "Desconecta este tensor del grafo."

Ahora es simplemente un tensor normal.

---

# 2. `cpu()`

Después:

```python
.cpu()
```

Puede ocurrir que el entrenamiento esté ejecutándose en GPU.

Pero NumPy solamente trabaja sobre memoria de la CPU.

Por eso primero movemos el tensor.

Visualmente:

```text
GPU

↓

CPU

↓

NumPy
```

---

# 3. `numpy()`

Ahora:

```python
.numpy()
```

Convierte el tensor en un arreglo de NumPy.

Antes:

```python
Tensor
```

Después:

```python
ndarray
```

Esto permite utilizar funciones como:

```python
np.sum(...)
```

---

# 4. `reshape(-1)`

Finalmente:

```python
.reshape(-1)
```

Convierte cualquier forma en un vector unidimensional.

Por ejemplo.

Antes:

```text
[[1]

 [0]

 [1]

 [1]]
```

Forma:

```text
(4,1)
```

Después:

```text
[1 0 1 1]
```

Forma:

```text
(4,)
```

Esto facilita realizar comparaciones con NumPy.

---

# Contar ejemplos positivos

Ahora:

```python
pos = float(
    np.sum(
        y_train_cpu == 1
    )
)
```

Aquí ocurre una operación muy elegante.

Supongamos:

```text
[

1,

0,

1,

1,

0

]
```

Primero:

```python
y_train_cpu == 1
```

produce:

```text
[

True,

False,

True,

True,

False

]
```

Después:

```python
np.sum(...)
```

suma los valores booleanos.

En NumPy:

```text
True = 1

False = 0
```

Entonces:

```text
1+0+1+1+0

=

3
```

Resultado:

```python
pos = 3
```

Es decir:

> Número de ejemplos positivos.

---

# Contar negativos

Exactamente igual:

```python
neg = float(
    np.sum(
        y_train_cpu == 0
    )
)
```

Obtiene:

```python
neg = 2
```

Ahora ya conocemos la distribución del dataset.

---

# Validación importante

Después aparece:

```python
if pos == 0 or neg == 0:
```

¿Por qué?

Imagina este caso extremo.

```text
Todos los clientes son Good.
```

Entonces:

```text
Positivos = 1000

Negativos = 0
```

No tiene sentido calcular:

```text
neg / pos
```

porque una clase ni siquiera existe.

Además, una clasificación binaria necesita ejemplos de ambas clases para aprender a diferenciarlas.

Por eso el código decide no utilizar pesos.

---

# Mensaje de advertencia

```python
log.warning(...)
```

Simplemente informa:

```text
No existe una de las clases.

No se utilizará pos_weight.
```

---

# Función estándar

En ese caso retorna:

```python
nn.BCEWithLogitsLoss()
```

Sin ponderación.

---

# Cálculo del peso

Ahora llegamos a la parte más importante.

```python
pos_weight_value = neg / pos
```

Aquí aparece la fórmula.

Supongamos:

```text
900 Good

100 Bad
```

Si:

```text
Good = 1

Bad = 0
```

Entonces:

```text
pos = 900

neg = 100
```

El peso sería:

```text
100 / 900

=

0.11
```

Pero si la clase positiva fuera la minoritaria:

```text
100 positivos

900 negativos
```

Entonces:

```text
900 / 100

=

9
```

Eso significa:

> Equivocarse en un positivo costará nueve veces más.

Y precisamente ese es el objetivo.

---

# Crear el tensor

Después:

```python
pos_weight_tensor = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=self.device
)
```

¿Por qué un tensor?

Porque PyTorch espera:

```python
pos_weight
```

como un tensor.

No acepta un número flotante.

Además debe estar:

- mismo tipo (`float32`)
- mismo dispositivo (`CPU` o `GPU`)

que el resto del modelo.

---

# Registro del peso

```python
log.info(...)
```

Produce algo como:

```text
Using weighted BCE loss

pos_weight = 8.34
```

Muy útil para revisar posteriormente los experimentos.

---

# Crear la Loss Function

Finalmente:

```python
return nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)
```

Aquí sí construimos la función de pérdida.

Recordemos que **no calcula ninguna pérdida todavía**.

Simplemente crea el objeto.

Más adelante aparecerá algo como:

```python
loss = criterion(
    logits,
    labels
)
```

Ahí sí se calculará el error.

---

# ¿Qué es `BCEWithLogitsLoss`?

Es probablemente la función de pérdida más utilizada en problemas de **clasificación binaria**.

Su nombre significa:

```text
Binary Cross Entropy

+

Logits
```

Recordemos que nuestra red termina con:

```python
nn.Linear(...,1)
```

No existe:

```python
Sigmoid()
```

en la última capa.

La salida son:

```text
Logits
```

Por ejemplo:

```text
2.4

-1.8

0.6
```

No probabilidades.

`BCEWithLogitsLoss` realiza internamente dos pasos:

```text
Logits

↓

Sigmoid

↓

Probabilidades

↓

Binary Cross Entropy
```

Todo esto ocurre dentro de la propia función.

---

# ¿Por qué no usar `Sigmoid` antes?

Podríamos hacer:

```python
prob = torch.sigmoid(logits)
loss = BCELoss(prob, y)
```

Pero esto es menos recomendable.

`BCEWithLogitsLoss` utiliza una implementación matemáticamente más estable.

Evita problemas numéricos cuando los logits toman valores muy grandes o muy pequeños.

Por eso en PyTorch la práctica recomendada es:

```text
Última capa

↓

Linear

↓

BCEWithLogitsLoss
```

Y **no**:

```text
Linear

↓

Sigmoid

↓

BCELoss
```

---

# Caso sin ponderación

Si:

```python
self.use_pos_weight
```

es:

```text
False
```

el método termina aquí.

```python
log.info(
    "Using standard BCE loss."
)

return nn.BCEWithLogitsLoss()
```

Es exactamente la misma función de pérdida, pero sin dar mayor importancia a ninguna clase.

---

# Resumen

Este método decide **qué función de pérdida utilizará el entrenamiento**.

Su flujo completo es:

```text
Etiquetas de entrenamiento
            │
            ▼
Contar positivos y negativos
            │
            ▼
¿Usar pos_weight?
      │              │
      │ Sí           │ No
      ▼              ▼
Calcular peso     BCE estándar
      │
      ▼
Crear tensor
      │
      ▼
BCEWithLogitsLoss(pos_weight=...)
      │
      ▼
Retornar función de pérdida
```

La decisión clave es que, cuando el conjunto de entrenamiento está desbalanceado, el método calcula automáticamente un **`pos_weight = negativos / positivos`**. De esta forma, los errores cometidos sobre la clase minoritaria tienen un mayor impacto en la pérdida y el optimizador dedica más esfuerzo a aprender esa clase. Este enfoque suele producir modelos más equilibrados y evita que la red se limite a predecir siempre la clase mayoritaria.

# Método `_run_training_loop() (Primera parte)`

Llegamos al corazón de todo el proyecto.

Hasta ahora hemos preparado todo:

- Leímos el dataset.
- Lo dividimos en entrenamiento y validación.
- Lo preprocesamos.
- Lo convertimos en tensores.
- Creamos la red neuronal.
- Creamos el optimizador.
- Creamos la función de pérdida.

Pero la red todavía **no ha aprendido absolutamente nada**.

Este método es donde realmente ocurre el aprendizaje.

Es aquí donde los pesos de la red neuronal empiezan a modificarse.

---

# Flujo general del entrenamiento

Antes de analizar cada línea, es importante entender el flujo completo que sigue una red neuronal durante el entrenamiento.

```text
               DATOS
                 │
                 ▼
        Forward Pass
                 │
                 ▼
        Calcular Loss
                 │
                 ▼
      Backpropagation
                 │
                 ▼
        Calcular Gradientes
                 │
                 ▼
 Optimizer actualiza pesos
                 │
                 ▼
        Siguiente Batch
```

Este proceso ocurre miles de veces.

Por eso se llama **Training Loop**.

---

# Parámetros del método

```python
def _run_training_loop(
    self,
    model,
    criterion,
    optimizer,
    scheduler,
    x_train,
    y_train,
    x_val,
    y_val,
)
```

Este método recibe prácticamente todo lo necesario para entrenar.

## model

```python
model: nn.Module
```

Es la red neuronal que construimos anteriormente.

En este momento ya contiene:

- capas lineales
- batch normalization
- funciones de activación
- dropout

Pero sus pesos todavía son prácticamente aleatorios.

---

## criterion

```python
criterion
```

Es la función de pérdida.

En nuestro proyecto será:

```python
nn.BCEWithLogitsLoss(...)
```

Su única responsabilidad es responder:

> ¿Qué tan equivocada estuvo la red?

---

## optimizer

```python
optimizer
```

Es el algoritmo encargado de actualizar los pesos.

Puede ser:

```text
Adam

SGD

RMSprop
```

El optimizador nunca calcula el error.

Solo utiliza los gradientes para modificar los pesos.

---

## scheduler

```python
scheduler
```

Controla automáticamente el Learning Rate durante el entrenamiento.

No actualiza pesos.

Solo modifica el tamaño de los pasos que dará el optimizador.

---

## x_train

Las características de entrenamiento.

---

## y_train

Las etiquetas de entrenamiento.

---

## x_val

Características para validación.

---

## y_val

Etiquetas para validación.

---

# Inicialización de variables

Primero aparecen estas líneas.

```python
best_val_loss = float("inf")
```

Esta variable almacenará la mejor pérdida de validación encontrada durante todo el entrenamiento.

Pero...

¿Por qué comienza con:

```python
float("inf")
```

?

---

## ¿Qué es `float("inf")`?

Representa el infinito positivo.

Visualmente:

```text
∞
```

En Python:

```python
float("inf")
```

es un número mayor que cualquier otro número.

Por ejemplo.

```text
∞

>

100000000

>

500

>

20

>

1
```

---

## ¿Por qué utilizar infinito?

Supongamos que la primera época obtiene:

```text
Validation Loss

0.63
```

Entonces:

```python
0.63 < inf
```

Siempre será verdadero.

Eso garantiza que la primera época se convierta automáticamente en la mejor hasta ese momento.

Después:

```text
best_val_loss

↓

0.63
```

Más adelante:

```text
Epoch 2

↓

0.48
```

Como:

```text
0.48 < 0.63
```

se actualiza nuevamente.

---

# patience_counter

Después aparece:

```python
patience_counter = 0
```

Esta variable pertenece al mecanismo de **Early Stopping**.

Su función es contar:

> ¿Cuántas épocas consecutivas llevamos sin mejorar?

Visualmente.

```text
Epoch 1

↓

Mejoró

↓

Counter = 0
```

Luego.

```text
Epoch 2

↓

No mejora

↓

Counter = 1
```

```text
Epoch 3

↓

No mejora

↓

Counter = 2
```

```text
Epoch 4

↓

Mejora

↓

Counter vuelve a 0
```

Cuando llegue al límite configurado:

```yaml
patience: 10
```

el entrenamiento terminará.

---

# epochs_run

Después.

```python
epochs_run = 0
```

Simplemente almacena cuántas épocas realmente se ejecutaron.

Esto es útil porque puede ocurrir que:

```yaml
epochs: 100
```

pero Early Stopping detenga el entrenamiento en:

```text
Epoch 37
```

Entonces:

```python
epochs_run

=

37
```

---

# Ruta del mejor modelo

Ahora aparece:

```python
best_model_path =
    self.local_models_dir /
    self.model_name
```

Aquí simplemente construimos la ruta donde posteriormente se guardará el mejor modelo.

Todavía no guarda nada.

Solo crea algo parecido a:

```text
models/

credit_model.pt
```

---

# Inicio del entrenamiento

Después aparece.

```python
log.info(
    "--- Starting training loop ---"
)
```

Simplemente escribe:

```text
Starting training loop
```

en el log.

---

# Comienza el entrenamiento

Ahora sí llegamos a la parte importante.

```python
for epoch in range(self.epochs):
```

Aquí comienza el ciclo principal.

Supongamos:

```yaml
epochs: 5
```

Entonces:

```text
Epoch 1

↓

Epoch 2

↓

Epoch 3

↓

Epoch 4

↓

Epoch 5
```

Cada vuelta del ciclo representa una **Epoch completa**.

Recordemos.

Una Epoch significa:

> Recorrer completamente el conjunto de entrenamiento una vez.

---

# Modo entrenamiento

Lo primero que ocurre en cada época es:

```python
model.train()
```

Esta línea suele generar muchas dudas.

No significa:

> Entrenar el modelo.

El entrenamiento todavía no ha comenzado.

---

## ¿Qué hace realmente `model.train()`?

Le dice a PyTorch:

> A partir de ahora el modelo trabajará en modo entrenamiento.

¿Por qué existe esto?

Porque algunas capas cambian su comportamiento dependiendo del modo.

Principalmente:

- Dropout
- Batch Normalization

---

## Ejemplo con Dropout

Durante entrenamiento.

```text
Neurona

↓

Se desactiva aleatoriamente
```

Pero durante validación.

```text
Todas las neuronas permanecen activas.
```

Por eso debemos indicar explícitamente el modo.

---

## Batch Normalization

También cambia.

Durante entrenamiento calcula estadísticas del batch.

Durante validación utiliza estadísticas almacenadas.

Todo esto ocurre automáticamente gracias a:

```python
model.train()
```

Más adelante veremos:

```python
model.eval()
```

que activa el modo evaluación.

---

# Registrar la época

Después.

```python
epochs_run = epoch + 1
```

Recordemos que:

```python
range(5)
```

produce:

```text
0

1

2

3

4
```

Pero nosotros queremos mostrar:

```text
Epoch 1

Epoch 2

...

Epoch 5
```

Por eso se suma uno.

---

# Comienza el recorrido por batches

Ahora aparece otra estructura muy importante.

```python
for i in range(
    0,
    len(x_train),
    self.batch_size
):
```

Hasta ahora recorríamos épocas.

Ahora recorremos batches.

Visualmente.

Supongamos.

```text
1000 ejemplos
```

Batch Size:

```text
100
```

Entonces.

```text
Epoch

↓

Batch 1

↓

Batch 2

↓

Batch 3

↓

...

↓

Batch 10
```

Cada batch producirá una actualización de pesos.

---

# ¿Cómo funciona ese `range()`?

Recordemos.

```python
range(
    inicio,
    fin,
    salto
)
```

Aquí.

```python
range(
    0,
    len(x_train),
    self.batch_size
)
```

Supongamos.

```text
1000 ejemplos
```

y

```text
Batch Size = 100
```

Entonces el índice:

```text
i
```

tomará estos valores.

```text
0

100

200

300

...

900
```

Es decir.

El inicio de cada batch.

---

# Extraer el batch

Ahora.

```python
x_batch =
    x_train[
        i :
        i+self.batch_size
    ]
```

Supongamos.

```text
Batch Size = 100
```

Primer batch.

```text
i = 0
```

Obtendremos.

```text
x_train[0:100]
```

Segundo batch.

```text
i =100
```

Obtendremos.

```text
x_train[100:200]
```

Y así sucesivamente.

Lo mismo ocurre para:

```python
y_batch =
    y_train[
        i :
        i+self.batch_size
    ]
```

De esta manera nunca entrenamos utilizando todo el dataset al mismo tiempo.

Siempre trabajamos con pequeños grupos de ejemplos.

---

# ¿Por qué no entrenar con todo el dataset?

Porque sería muy costoso.

Imagina.

```text
1 millón de ejemplos.
```

La memoria necesaria sería enorme.

En cambio.

```text
Batch Size = 64
```

permite entrenar utilizando únicamente 64 ejemplos en cada iteración.

Es mucho más eficiente tanto en CPU como en GPU.

---

# Resumen de esta primera parte

Hasta este punto del método todavía **no se ha actualizado ningún peso**.

Lo que hemos hecho es:

1. Inicializar las variables necesarias para el entrenamiento (`best_val_loss`, `patience_counter`, `epochs_run`).
2. Construir la ruta donde se guardará el mejor modelo.
3. Entrar en el bucle principal de épocas.
4. Poner el modelo en modo entrenamiento con `model.train()`.
5. Comenzar a recorrer el conjunto de entrenamiento por batches.
6. Extraer de cada iteración un `x_batch` y un `y_batch` que serán utilizados en la siguiente etapa.

En la siguiente parte llegaremos al núcleo del aprendizaje, donde ocurre el ciclo fundamental de toda red neuronal:

```text
x_batch
    │
    ▼
Forward Pass
    │
    ▼
Loss
    │
    ▼
Backpropagation
    │
    ▼
Gradientes
    │
    ▼
Optimizer
    │
    ▼
Nuevos pesos
```

Ese bloque es el responsable de que la red neuronal aprenda y es, probablemente, el más importante de todo el proyecto.

### ¿ Como mejorar un modelo ?

#### feature engineering

creacion de nuevas caracteristicas a partir de las existentes. aplicando calculos matematicos y estadisticas.

#### Polinomial features

Es crear caracteristicas polinomiales a variables existentes mas importantes e influenciadoras en la prediccion final. esto se hace con el fin de crear relaciones no lineales entre las variables.
