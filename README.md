# GDG UAM ML Contest — entrega

Repositorio privado de preparación de la entrega de Álvaro Gutiérrez de
Cabiedes Gonzalvo. Versión seleccionada: **runtime v4**, CPU y un hilo.

El modelo y los pesos están preparados para evaluar: no hay que entrenar nada.
`model.py` expone `Model` y `predict(sample) -> int`, con clases de 0 a 7.
La inferencia utiliza ONNX Runtime y auxiliares compilados con Numba. No requiere
GPU, PyTorch ni conexión a Internet durante la preparación del modelo o la predicción.
Las dependencias se instalan antes de ejecutar el evaluador.

## Archivos

- `model.py`: punto de entrada.
- `model.onnx` y `head.npz`: pesos entrenados y grafo de inferencia.
- `preprocessing.py` y `fused_head.py`: preparación de señales y cabeza del modelo.
- `runtime.json`: configuración de inferencia.
- `requirements.txt`: versiones de las tres dependencias necesarias.
- `manifest.json`: hashes de los siete archivos de inferencia y evidencia de validación.
- `docs/`: comprobación del paquete y decisión de mantener v4 tras el estudio de v5.

## Comprobación desde una copia nueva

Se necesitan los [materiales oficiales](https://github.com/GDG-UAM/ml-contest-materials)
y su fichero `data/test_demo.npz`. Desde un directorio que contenga la carpeta
`ml-contest-materials`, ejecutar:

```bash
git clone https://github.com/alvarogcabiedess/gdg-uam-ml-contest.git entrega-gdg
python -m pip install -r entrega-gdg/requirements.txt
python ml-contest-materials/harness.py evaluate --model entrega-gdg/model.py --source file:ml-contest-materials/data/test_demo.npz --runs 3
```

Usar el harness oficial sin modificaciones y una sesión de Colab con CPU para
comparar rendimiento en el entorno del concurso. `Model()` compila los auxiliares
durante la preparación, antes del tramo cronometrado. No se incluyen datos en este
repositorio; el evaluador proporciona cada muestra por separado.

## Versión seleccionada y límites de la evidencia

Los siete archivos de inferencia coinciden byte a byte con el paquete v4 validado
el 11 de noviembre de 2026. SHA-256 del ZIP de referencia:

```text
a08b025fdfd90ae6794c93e49f928eadc224433f0ebc1e9fadb3f5b8e8dd6d99
```

El modelo pasó las evaluaciones completas de Colab, con un hilo, sin excepciones,
respuestas inválidas ni timeouts. Las comparaciones de velocidad repiten ejemplos
de desarrollo: no son una estimación del test oculto ni de la posición final.
La candidata v5 fue más rápida en promedio, pero no cumplió el criterio previo
de consistencia; este repositorio conserva v4.

El repositorio está en preparación. Todavía no se ha enviado a los organizadores
ni se les ha concedido acceso.
