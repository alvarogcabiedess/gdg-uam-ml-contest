# Decisión final exprés: v4

Se conserva v4: la candidata es más rápida en promedio, pero no supera el criterio previo de estabilidad en las comparaciones oficiales. Se mantiene la entrega validada y se conservan todas las pruebas, incluidas las desfavorables y la ejecución interrumpida por el plazo.

La ejecución conserva los valores de los pesos aprendidos. Se probaron nuevas especializaciones de funciones Numba con pesos constantes y menos argumentos; además se repitió la candidata combinada y se descartó una variante de métodos ligados que resultó más lenta.

## Medición directa local

| Variante | Latencia media, microsegundos |
|---|---:|
| baseline_v4 | 47.753 |
| combined | 44.541 |
| v4_closed | 45.716 |
| v5_closed | 44.135 |
| v4_bound | 49.854 |

70 bloques con orden aleatorio, 200 entradas de validación; etiquetas y partición reservada sin usar. Las tres candidatas principales se comprobaron con 110.230 entradas, sin diferencias de clase, alteración de entradas ni dependencia del historial.

## Evaluador oficial

CPU y un hilo. Tres parejas con orden prefijado, semillas 20261112/13/14 y 60 segundos acumulados de predict por ejecución. El criterio previo exige ganar las tres parejas y una ganancia media de puntuación de al menos 1 %.

### local

Ganancias por pareja: +11.092 %, -8.619 %, +19.475 %.
Reducción media de latencia por pareja: 9.724 %. Criterio superado: no.
Máxima preparación en frío: 11.210 s. Sin excepciones, respuestas inválidas ni timeouts.

### Colab: estudio limitado por el plazo

Se completaron 5 ejecuciones oficiales antes del corte. El estudio no reúne las tres parejas exigidas y no se usa para declarar superado el criterio.
Ganancias de puntuación en las parejas completas: -0.636 %, +9.372 %.
El usuario amplió ocho minutos el plazo de decisión. La continuación mantuvo todas las ejecuciones completadas, candidatos, semillas, orden y criterio de selección; el último intento se detuvo al alcanzar el corte ampliado.
La comprobación de equivalencia también pasó las 110.230 entradas en Colab. Los ensayos incompletos se conservan en el informe.

## Evidencia histórica y límites

La revisión anterior de Colab de la misma candidata combinada dio ganancias por pareja de +5,707 %, -0,090 % y +6,314 %; por ese criterio se había conservado v4. Se mantiene esa evidencia, incluido el resultado desfavorable.
Las cifras de estas pruebas repiten 200 entradas de desarrollo para medir velocidad. No son una evaluación independiente de exactitud ni permiten conocer la puntuación oculta o el puesto en el concurso. Cambiar el orden float32 no garantiza igualdad matemática universal.
En el primer intento local el evaluador terminó correctamente, pero el coordinador buscaba el resultado en una ruta relativa equivocada. Se recuperó y conservó esa primera ejecución sin repetirla ni excluirla. En Colab se corrigió una carpeta de informes ausente antes de iniciar las pruebas. Los scripts del modelo y el harness no cambiaron durante la comparación.

## Artefacto

ZIP SHA-256: `a08b025fdfd90ae6794c93e49f928eadc224433f0ebc1e9fadb3f5b8e8dd6d99`.
Repositorio privado: https://github.com/alvarogcabiedess/gdg-uam-ml-contest . Autor y committer: Álvaro; sin coautores. Los jueces aún no tienen acceso y no se ha enviado la entrega.
Se conservan los informes completos en `reports/runtime_express` y el código de los experimentos en `runtime_express`.
