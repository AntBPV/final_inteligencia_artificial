# Contexto del proyecto: sistema de conteo y seguimiento de personas

## 1. Objetivo

Desarrollar un sistema de inteligencia artificial en Python que, a partir de la camara web o de un video, pueda:

- detectar personas mediante una red neuronal convolucional;
- seguir cada persona a lo largo de los fotogramas;
- contar entradas y salidas respecto de una linea virtual;
- calcular la ocupacion actual;
- consultar estadisticas de ocupacion mediante un asistente;
- mostrar visualmente las detecciones y el seguimiento durante una demostracion.

El sistema sera una herramienta de demostracion academica. No se utilizaran
reconocimiento facial, identificacion nominal ni almacenamiento de datos personales.

## 2. Alcance inicial

### Incluido

- Procesamiento de video MP4 o de la camara web.
- Deteccion de personas con una CNN preentrenada.
- Asignacion de un identificador temporal a cada individuo detectado.
- Conteo cuando una persona cruza una linea configurada.
- Registro de entrada, salida y ocupacion historica.
- Interfaz grafica o visualizacion mediante OpenCV.
- Consigna de estadisticas por medio de comandos simples o una interfaz de terminal.
- Documentacion de instalacion, ejecucion y pruebas.

### No incluido inicialmente

- Reconocimiento de identidades.
- Analisis emocional o atributivo.
- Cobro, control de acceso o acciones legales.
- Altas cantidades de camaras distribuidas.
- Capacidad de reidentificar a una persona despues de una desconexion prolongada.

## 3. Tecnologias propuestas

| Tecnologia | Uso en el proyecto | Razon |
|---|---|---|
| Python 3.11 o 3.12 | Lenguaje principal | Compatibilidad amplia con las bibliotecas de vision por computador. |
| YOLO mediante Ultralytics | Deteccion CNN de personas | Detecta objetos en tiempo real y permite extraer directamente personas con clase `person`. |
| OpenCV | Captura de video, dibujo de resultados y ventanas | Facilita trabajar con archivos de video y camaras web. |
| Ultralytics tracking o BoT-SORT | Seguimiento de individuos | Mantiene identificadores entre fotogramas con configuracion sencilla. |
| NumPy | Operaciones numericas y geometria del seguimiento | Base eficiente para datos de imagen y calculos de coordenadas. |
| Pandas | Estadisticas y exportacion historica | Simplifica series temporales y consultas de resumen. |
| SQLite | Persistencia local | Permite guardar eventos sin requerir un servidor ni una base externa. |
| Streamlit | Panel opcional | Presenta ocupacion, tabla de eventos y visualizacion de manera rapida. |
| pytest | Pruebas automatizadas | Permite verificar las reglas de conteo y las operaciones de seguimiento. |
| VS Code | Entorno de desarrollo | Integra terminal, Python, pruebas y depuracion en un solo lugar. |

### Decision sobre YOLO

Se recomienda comenzar con un modelo pequeno como `yolo11n.pt` o `yolov8n.pt`.
Es suficientemente liviano para una demostracion con CPU y puede ajustarse a GPU
CUDA cuando este disponible. Para mayor precision se pueden probar modelos mas
grandes, pero no son necesarios para la primera version.

### Decision sobre CUDA

CUDA es opcional en el entorno inicial. El proyecto debe funcionar en CPU y se puede
activar GPU segun el sistema instalado. La configuracion de PyTorch debe elegirse
segun el sistema operativo, el controlador NVIDIA y si existe soporte CUDA.

## 4. Arquitectura prevista

La implementacion se dividira en modulos independientes:

```text
app/
  main.py                  # Punto de entrada para video o webcam
  config.py                # Configuracion de camara, linea y modelo
  detection/
    detector.py            # Carga y ejecucion del modelo CNN
  tracking/
    tracker.py             # Actualizacion y asignacion de identificadores
  counting/
    counter.py             # Cruce de linea y entrada/salida
  analytics/
    statistics.py          # Ocupacion y consultas historicas
    assistant.py           # Consultas en lenguaje natural o comandos
  storage/
    repository.py          # Persistencia de eventos en SQLite
  visualization/
    renderer.py            # Dibujos sobre cada fotograma
tests/
data/
  videos/                  # Videos locales de prueba
  db/                      # Base de datos local
docs/
  CONTEXTO_PROYECTO.md
```

### Flujo de ejecucion

1. Leer un fotograma desde el video o la camara web.
2. Ejecutar la CNN y conservar las detecciones de clase `person`.
3. Filtrar detecciones por confianza y por regiones invalidas.
4. Actualizar el tracker y obtener una identidad temporal.
5. Comparar la posicion actual con el lado anterior de la linea virtual.
6. Registrar una entrada o salida cuando el cruce sea valido.
7. Actualizar la ocupacion actual.
8. Dibujar cuadros, identificadores, linea y estadisticas.
9. Guardar eventos en SQLite.
10. Permitir consultas a las estadisticas mediante el asistente.

## 5. Estrategia de seguimiento y conteo

Se utilizara el punto inferior central de cada cuadro como representacion de su
posicion. Esto permite comparar movimientos respecto de una linea horizontal sin
depender de que toda la persona este a un lado de ella.

La linea se configurara manualmente con dos puntos normalizados respecto del ancho
y alto del fotograma. De esta forma puede adaptarse a diferentes resoluciones.

Para evitar dobles conteos se aplicaran las siguientes reglas:

- registrar una identidad solamente si mantiene detecciones durante varios fotogramas;
- guardar el ultimo lado conocido de la linea para cada identidad;
- contar cuando cambia de un lado al opuesto;
- confirmar el cruce con una distancia minima o varias detecciones consecutivas;
- desactivar temporalmente el conteo de una identidad despues de registrar el evento;
- conservar el conteo incluso si el tracker pierde momentaneamente al individuo.

La ocupacion sera:

```text
ocupacion = entradas acumuladas - salidas acumuladas
```

El valor se limitara a cero si los datos contienen inconsistencias. Se permitira
corregir o recalibrar la linea sin alterar los eventos guardados.

## 6. Asistente de estadisticas

En una primera version, el asistente respondera consultas locales sin enviar datos
a servicios externos. Puede aceptar lenguaje natural sencillo o comandos claros,
por ejemplo:

- "Cuantas personas hay ahora?"
- "Cuantas entradas hubo hoy?"
- "Cuantas salidas hubo hoy?"
- "Cual fue la ocupacion maxima?"
- "Muestra el total de la ultima hora."

La implementacion inicial puede usar reglas basadas en palabras clave y consultas
SQL. Si el asistente requiere un modelo de lenguaje, se evaluara despues de obtener
los datos y el registro historico; no sera necesario para demostrar el conteo.

## 7. Entorno necesario

### Software base

- Git.
- Python 3.11 o 3.12.
- Visual Studio Code.
- Extension "Python" de Microsoft.
- Camara web, opcional para la demostracion.
- Video MP4 de prueba, recomendado para resultados reproducibles.

### Preparacion del entorno

Desde la raiz del repositorio se recomienda ejecutar:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python numpy pandas streamlit pytest
```

Si el entorno usa Python 3.12, reemplazar `py -3.11` por `py -3.12`.

La primera ejecucion descargara el modelo seleccionado. Se puede colocar el archivo
del modelo dentro de una carpeta local para reutilizarlo y evitar descargas
repetidas.

### Dependencias futuras recomendadas

Cuando exista una interfaz web o persistencia mas avanzada se podran agregar:

```text
sqlalchemy
plotly
streamlit
```

El listado definitivo se fijara en `requirements.txt` o `pyproject.toml` al iniciar
la implementacion.

## 8. Division del trabajo

### Fase 1: configuracion y demostracion base

- Crear el entorno virtual.
- Definir dependencias y estructura de carpetas.
- Descargar o seleccionar un modelo CNN.
- Ejecutar detecciones sobre imagen, video y camara.
- Mostrar cuadros y la clase `person`.

**Resultado:** deteccion funcional visible.

### Fase 2: seguimiento

- Activar el seguimiento integrado.
- Crear una clase de tracker.
- Asignar identificadores temporales.
- Dibujar trayectorias simples.
- Resolver errores con personas que se superponen o salen del cuadro.

**Resultado:** seguimiento estable de personas entre fotogramas.

### Fase 3: conteo de entradas y salidas

- Definir una linea virtual configurable.
- Crear un estado por identidad.
- Detectar cambios de lado.
- Probar cruces simples en ambos sentidos.
- Evitar dobles conteos.
- Exponer el valor de ocupacion actual.

**Resultado:** entradas, salidas y ocupacion calculados en tiempo real.

### Fase 4: persistencia y estadisticas

- Crear la base SQLite.
- Registrar eventos con fecha y hora.
- Crear consultas de entradas, salidas y ocupacion maxima.
- Exportar o presentar resúmenes.
- Implementar el asistente basado en consultas locales.

**Resultado:** estadisticas historicas consultables.

### Fase 5: interfaz de demostracion

- Integrar la salida grafica con OpenCV.
- Agregar panel con Streamlit o una interfaz equivalente.
- Mostrar ocupacion actual y evento mas reciente.
- Permitir ejecutar video o camara desde la interfaz.
- Preparar una muestra reproducible.

**Resultado:** demostracion completa e interactiva.

### Fase 6: pruebas y entrega

- Agregar pruebas unitarias para cruces y ocupacion.
- Agregar una prueba con video corto conocido.
- Medir rendimiento aproximado y errores comunes.
- Documentar ejecucion y configuracion.
- Preparar entrega de codigo y evidencias.

**Resultado:** proyecto verificable y listo para presentar.

## 9. Metricas de aceptacion

La demostracion se considerara funcional cuando cumpla estos criterios:

- procesa al menos una resolucion de 640x480 en tiempo util;
- detecta personas en video y/o camara;
- conserva identificadores temporales mientras una persona atraviesa el cuadro;
- cuenta correctamente un cruce en cada direccion;
- no cuenta el mismo cruce varias veces;
- muestra ocupacion igual a entradas menos salidas;
- permite cambiar o configurar la linea;
- permite consultar al menos ocupacion actual y totales del dia;
- no almacena datos de identidad personal ni imagenes facialmente identificables.

La precision exacta dependra de la calidad del video, la iluminacion, el angulo
de la camara y el modelo utilizado. Se documentaran estos limites.

## 10. Criterios para elegir video de prueba

El video ideal debe cumplir:

- personas visibles de cuerpo completo o al menos del torso;
- camara fija;
- movimiento perpendicular o definido respecto de la linea;
- buena iluminacion;
- resolucion uniforme;
- cruces reales separados entre si;
- numero conocido de entradas y salidas;
- minimo cruce por camara lenta o trayectoria ambigua.

## 11. Privacidad y uso responsable

- El sistema no realizara reconocimiento facial.
- Se conservaran solamente identificadores temporales y eventos de conteo.
- Las imagenes no se enviaran a servicios externos.
- Si se activa almacenamiento de video, sera local y con consentimiento.
- La camara debe utilizarse solo en espacios autorizados.
- Un cartel o aviso identificara las zonas donde existe monitoreo.
- Los datos de prueba deberan ser simulados o consentidos.

## 12. Riesgos conocidos

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Personas superpuestas | Pérdida o intercambio de identificadores | Usar filtros de deteccion, velocidad razonable y tracking robusto. |
| Iluminacion o oclusion | Falsos negativos | Cambiar modelo, iluminacion o posicion de la camara. |
| Camara moviendose | Cruces falsos | Mantener camara fija y permitir recalibrar la linea. |
| Entrada simultanea | Conteo incompleto | Ajustar confirmacion temporal y realizar pruebas visuales. |
| Limitacion de CPU | Bajos FPS | Usar modelo nano, reducir resolucion y optimizar el ciclo. |
| Reidentificacion incompleta | Numero incorrecto por salidas temporales | Mantener estados breves y documentar el comportamiento. |
| Privacidad | Uso indebido de imagen | Limitar datos, desactivar almacenamiento y mantener procesamiento local. |

## 13. Convenciones previstas

- Codigo en Python con nombres y tipos claros.
- Configuracion centralizada en un archivo o en variables de entorno.
- Ninguna ruta absoluta del equipo se escribira en el codigo.
- Los datos locales se excluyeran del control de versiones.
- Las pruebas se ubicaran en `tests/`.
- Los errores se registraran con mensajes entendibles.
- La documentacion y los comentarios deberan actualizarse junto con el codigo.

## 14. Archivos que deben agregarse antes de implementar

El siguiente paso, despues de preparar el entorno, sera crear:

1. `requirements.txt` o `pyproject.toml`.
2. `.gitignore`.
3. `.vscode/settings.json`, si se desea fijar el entorno.
4. `app/main.py`.
5. `app/config.py`.
6. Pruebas iniciales para el contador.
7. Un video corto de prueba o instrucciones para usar la camara.
