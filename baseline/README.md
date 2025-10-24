# LAB IA - HPCTools

## Introducción

Los modelos de lenguaje pre-entrenados han mostrado en los últimos años una gran efectividad para mejorar varias tareas de procesamiento de lenguaje natural, 
tanto a nivel de inferencia del lenguaje natural o parafraseo, como a nivel de reconocimiento de entidades o respuesta a preguntas. Entre los diversos modelos 
de lenguaje existentes con sus diferentes características, se propone para esta tarea el empleo de **BERT**.

**BERT** (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje basado en la arquitectura Transformer que se entrena de manera bidireccional, 
considerando simultáneamente el contexto a la izquierda y a la derecha de cada palabra, lo que permite capturar relaciones semánticas más profundas que los modelos unidireccionales tradicionales [1]. Para su entrenamiento, se propone emplear **SQuAD** (Stanford Question Answering Dataset), un conjunto de datos que contiene pares de preguntas y fragmentos de texto extraídos de artículos de Wikipedia, donde cada pregunta tiene una respuesta que puede localizarse directamente en el texto [2]. 

El objetivo de esta tarea es seleccionar una implementación en PyTorch que permita entrenar el modelo con ese set de datos, y medir los tiempos de entrenamiento empleando una única GPU, comparando si es posible su desempeño frente a CPU.

## Metodología

Dado que la implementación desde cero implica una complejidad técnica considerable, emplearemos alguna implementación ya realizada que se encuentra en la red. Una búsqueda poco profunda me lleva a emplear la implementación del modelo BERT propuesta por **alexaapo** que se encuentra en la red [3]. El link a su repositorio es "https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset".

Esta implementación emplea el modelo BERT pre-entrenado **bert-base-uncased** y el dataset SQuAD 2.0. que contiene pares de (i) contexto o párrafo (**context**), (ii) pregunta (**queries**) y (iii) respuesta (**answer**). El modelo aprende a predecir las posiciones de inicio y fin de la respuesta dentro del texto. El código de la implementación en Python se puede ver en `main_baseline.py`. El dataset empleado se encuentra en la carpeta `squad`. Se han realizado algunos ajustes para acomodar la ejecución y la idea es jugar con los hiperparámetros **batch_size**, **learning-rate** y **epochs** para los dispositivos empleados (**1 NVIDIA A100 GPU**, **1 CPU** y si consigo recursos **1 NVIDIA T4 GPU**). Se empleó como optimizador el implementado por defecto **AdamW**.

La implementación requiere de la instalación de **transformers** y se ha ejecutado en el venv environment instalado en la asignatura. En mi caso se encuentra en el supercomputador Finisterra III del CESGA.

        #!/bin/bash
        source $STORE/mypython/bin/activate

Para instalar transformers ejecutar

        $ pip install -r requirements.txt

Se proporciona los scripts bash y SLURM para la ejecución en `run_train.sh`.

        $ sbatch run_train.sh

El tiempo de ejecución reportado se corresponde con una única ejecución. Lo propio hubiera sido lanzar varias ejecuciones y realizar un promedio, pero para efectos prácticos de la tarea creo que puede servir.

## Implementación BASELINE y resultados.

Todos los resultados obtenidos para todas las pruebas se pueden consultar en la carpeta `test` en los ficheros `.out`. En la tabla sólo se muestran los resultados que considero más relevantes. 


La primera ejecución se lanzó con los parámetros que por defecto traía la implementación (`baseline_GPUA100_2epoch_8BS.out`) con el dispositivo NVIDIA A100-PCIE-40GB (ver tabla). Comprobé el efecto del batch_size, reduciéndolo de 8 a 1 y aumentándolo a 16. Para un batch_size de 1 y 2 epochs el tiempo de ejecución fué de 6460.94 s. 

Los primeros test para CPU fallaron por límite de tiempo al emplear batch_sizes muy pequeños que implican muchas más iteraciones por época.



| Device         | Tiempo total de entrenamiento (s)   | Batch_size |Learning rate| epochs |
|----------------|---------------------------------|------------|-------------|------------|
| NVIDIA A100-PCIE-40GB  |   4108.80               |     8      |     5e-5    |      2     |
| 1:CPU          |                                 |            |             |            |
| 1:T4           |                                 |            |             |            |

## Referencias

    - [1] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) (pp. 4171-4186).
    - [2] https://rajpurkar.github.io/SQuAD-explorer/
    - [3] https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset
