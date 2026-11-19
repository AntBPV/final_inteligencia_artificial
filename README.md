# Sistema de deteccion y conteo de personas

Proyecto academico de inteligencia artificial para detectar personas en video o
camara web, seguirlas a lo largo de los fotogramas y contar entradas y salidas.

Este repositorio esta preparado para desarrollar el proyecto por fases. La fase 1
corresponde a la deteccion base de personas mediante una CNN.

## Estado actual

- Entorno virtual preparado en `.env`.
- Estructura inicial creada para deteccion, seguimiento, conteo, almacenamiento,
  analisis y visualizacion.
- Dependencias principales instaladas.
- Pendiente la implementacion del modulo de deteccion y su demostracion.

## Tecnologias seleccionadas

| Tecnologia | Razon de uso |
|---|---|
| Python | Permite integrar vision por computador, modelos de aprendizaje profundo y utilidades cientificas. |
| Ultralytics y YOLO | Facilitan la deteccion de objetos mediante una CNN preentrenada y una interfaz sencilla. |
| PyTorch | Motor de tensor y aprendizaje profundo utilizado por los modelos de Ultralytics. |
| OpenCV | Captura fotogramas de archivos y camaras web y permite dibujar resultados. |
| NumPy | Representa imagenes y coordenadas como arreglos numericos eficientes. |
| Pandas | Preparara la consulta y presentacion de estadisticas historicas. |
| Streamlit | Permitira crear posteriormente una interfaz grafica de demostracion. |
| pytest | Permitira verificar con pruebas automatizadas las reglas de deteccion y conteo. |

## Conceptos clave

### Vision por computador

Es el area de inteligencia artificial que permite analizar contenido visual como
imagenes y fotogramas de video. En este proyecto cada fotograma es una imagen que
se entrega al modelo para identificar personas.

### Red neuronal convolucional

Una CNN es una red que aprende patrones visuales como bordes, texturas y formas.
Sus capas convolucionales producen mapas de caracteristicas y permiten reconocer
objetos en posiciones diferentes dentro de una imagen.

No estamos entrenando una CNN desde cero. Usaremos un modelo preentrenado:
aprendio previamente sobre un conjunto grande de imagenes y ya puede detectar
diversas clases, incluida `person`.

### YOLO

YOLO significa *You Only Look Once*. Es una familia de detectores que analiza una
imagen y produce directamente las clases y posiciones de los objetos.

YOLO suele utilizar un modelo pequeno para pruebas rapidas. En este proyecto se
evaluara inicialmente un modelo nano, que ofrece un compromiso razonable entre
velocidad y uso de recursos. Puede reemplazarse por un modelo mas grande si se
requiere mayor precision y el equipo puede procesarlo.

### Deteccion y cuadro delimitador

Una deteccion consiste en:

- una clase, por ejemplo `person`;
- un cuadro delimitador o *bounding box*;
- una puntuacion de confianza;
- opcionalmente, coordenadas de seguimiento en fases posteriores.

Cada cuadro puede representarse como `x1, y1, x2, y2`, donde las coordenadas indican
los bordes izquierdo, superior, derecho e inferior.

### Confianza

La confianza es una probabilidad aproximada de que una deteccion corresponda a la
clase indicada. El umbral reduce las detecciones dudosas, pero no garantiza que todas
las personas sean encontradas. Un umbral alto disminuye falsos positivos y puede
perder personas; un umbral bajo encuentra mas candidatos y puede aumentar falsos
positivos.

### Supresion de detecciones duplicadas

NMS, o supresion no maxima, elimina cuadros superpuestos que representan el mismo
objeto. Conserva principalmente el cuadro con mayor confianza. YOLO aplica esta
logica dentro de su proceso de inferencia.

### Inferencia

Inferencia es el proceso de entregar una imagen ya preparada a un modelo entrenado
para obtener predicciones. El proyecto utilizara inferencia; no realizara
entrenamiento desde cero en la fase inicial.

### Modelos preentrenados y clases

Un modelo preentrenado contiene pesos aprendidos durante su entrenamiento. Los
pesos son numeros que ajustan el comportamiento de las neuronas. Los modelos usados
por Ultralytics suelen incluir clases basadas en datasets como COCO, donde
`person` corresponde a personas.

Los pesos pueden descargarse la primera vez que se ejecuta el modelo. Conviene
guardar el archivo de pesos fuera del control de versiones y no confundirlo con
codigo fuente.

### Fotograma

Un fotograma es una imagen individual dentro de un video. Los videos se procesan
como una secuencia de fotogramas. La tasa de fotogramas por segundo, o FPS, indica
cuantas imagenes se presentan cada segundo.

Para una webcam, OpenCV obtiene un fotograma mediante `VideoCapture`. Para un
archivo MP4, el proceso utiliza el mismo flujo de lectura: abrir el archivo, leer
un fotograma y procesarlo hasta finalizar.

### CPU y GPU

CPU significa unidad central de procesamiento. Es compatible con la mayoría de
equipos y puede ejecutar el sistema, aunque con menor velocidad.

GPU significa unidad de procesamiento grafico. Puede acelerar el calculo tensorial
mediante procesamiento paralelo. CUDA es la plataforma NVIDIA que permite
aprovecharla en muchos equipos compatibles. GPU es opcional; el sistema debe
funcionar con CPU.

### Precision y recall

En una evaluacion formal:

- precision mide cuantas detecciones positivas fueron correctas;
- recall mide cuantas personas reales fueron detectadas;
- F1 combina precision y recall en una sola medida.

Una demostracion visual no sustituye una evaluacion con etiquetas conocidas, pero
permite identificar problemas visibles como oclusiones, iluminacion deficiente y
personas fuera del encuadre.

### Limitaciones

Un detector no garantiza exactitud perfecta. Puede fallar ante:

- oclusion parcial;
- poca iluminacion;
- personas muy pequenas o lejos de la camara;
- movimiento rapido;
- poses poco habituales;
- fondos visualmente complejos;
- cambios de resolucion o angulo.

El seguimiento, aun no implementado, se verá afectado por estas limitaciones.

## Arquitectura actual

```text
detection/      carga y ejecuta el modelo CNN
tracking/       asignara identificadores temporales
counting/       detectara cruces y actualizara ocupacion
analytics/      consultara y resumira estadisticas
storage/        persistira eventos locales
visualization/  dibujara detecciones sobre los fotogramas
docs/           contiene la documentacion ampliada del proyecto
```

Cada modulo debe tener una responsabilidad clara. La deteccion no debe mezclar
codigo de entrada/salida, persistencia o interfaz, lo que facilita pruebas y
modificaciones posteriores.

## Preparacion del entorno

El entorno virtual creado usa el nombre `.env`. Desde la raiz del repositorio:

```powershell
.\.env\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El entorno actualmente probado usa Python 3.14.3. Las versiones previstas
inicialmente eran Python 3.11 o 3.12; para reproducibilidad academica se recomienda
registrar aqui la version final utilizada despues de completar la instalacion.

Si PowerShell impide activar el entorno, puede habilitarse temporalmente la ejecucion
de scripts para el usuario:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Dependencias

Las dependencias principales estan fijadas en [requirements.txt](requirements.txt):

- `ultralytics`: carga e inferencia de YOLO;
- `torch` y `torchvision`: ejecucion del modelo;
- `opencv-python`: lectura y visualizacion de video;
- `numpy`: operaciones con imagenes y coordenadas;
- `pandas`: estructuras de datos para estadisticas;
- `streamlit`: interfaz opcional posterior;
- `pytest`: pruebas automatizadas.

Las versiones transitorias o auxiliares instaladas por pip no requieren ser copiadas
de forma manual a `requirements.txt` si pip las instala como dependencias.

## Planeacion de la fase 1

La fase 1 debe demostrar:

1. carga del modelo preentrenado;
2. lectura de imagen, video o camara;
3. ejecucion de inferencia;
4. filtrado de detecciones a la clase `person`;
5. dibujo de cuadros delimitadores;
6. visualizacion de la confianza;
7. procesamiento estable de varios fotogramas.

La prueba inicial puede realizarse con una imagen antes de utilizar video o la
camara. Esto facilita separar errores del modelo de errores de entrada/salida.

## Ejecutar la demostracion base

Activar el entorno y ejecutar desde la raiz del proyecto:

```powershell
.\.env\Scripts\Activate.ps1
python main.py
```

Por defecto se utiliza la webcam. Para detener la ventana se debe presionar `q`
o la tecla `Esc`.

Procesar un video:

```powershell
python main.py --source data/videos/prueba.mp4
```

Procesar una imagen y guardar el resultado:

```powershell
python main.py --source ruta/a/imagen.jpg --output data/resultados/deteccion.jpg
```

Opciones disponibles:

```text
--model yolo11n.pt   Modelo YOLO preentrenado
--conf 0.40          Umbral de confianza
--device cpu         Dispositivo de inferencia
--source 0           Indice explicito de la webcam
```

La primera ejecucion descarga `yolo11n.pt` automaticamente. El archivo de pesos
se excluye del control de versiones por `.gitignore`.

Ejecutar las pruebas de esta fase:

```powershell
python -m pytest -q
```

## Privacidad y uso academico

El proyecto no utilizara reconocimiento facial ni identificacion nominal. La fase
inicial almacenara solamente resultados visuales temporales. Antes de usar una
camara real se debe contar con autorizacion y explicar el proposito de la captura.

No se deben publicar videos, imagenes o datos personales sin consentimiento. Las
pruebas deben usar material propio, de libre uso o personas que hayan autorizado
su participación.

## Documentacion relacionada

La division completa del proyecto, las fases y las estrategias de seguimiento se
encuentran en [docs/CONTEXTO_PROYECTO.md](docs/CONTEXTO_PROYECTO.md).
