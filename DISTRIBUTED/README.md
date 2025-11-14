# LAB IA - HPCTools - Artai Rodríguez Moimenta

## Introducción

El objetivo de esta segunda tarea es realizar el entrenamiento distribuído en paralelo con dos NVIDIA A100-PCIE-40GB y distintas estrategias de paralelización. 

Durante el periodo de documentación para la realización de esta segunda parte de la práctica, me he dado cuenta de que la implementación baseline escogida previamente(ver implementación en `main_baseline.py`) no era la mejor opción. Esta implementación ''casera'' está pensada para jugar de forma secuencial con el entrnamiento de modelos BERT y su estructura está realizada directamente en el código principal. Durante el proceso de documentaciṕn para esta segunda parte, me he encontrado con ejemplos de código más funcionales, con implementaciones encapsuladas por clases, lo que resulta más conveniete para este tipo de códigos. Ha sido una mala decisión la elección de la primera implementación en su momento y me hubiese gustado haber esgocido alguna más adecuada para obtener una experiencia más directa y permitirme controlar un poco más las características de la implementación.

Con todo, y dado el tiempo que puedo dedicar a esete ejericcio, decidí quedarme lo que tengo, modificar las salidas del código para adaptarlo al proceso distribuído y simplemente verificar que la paralelización reporta resultados coherentes. A pesar de todo, no poseeo control sobre lo que está ocurriendo internamente y todo está funcionando como una caja negra para mi. 

## Metodología

Para realizar la paralelización se proponen dos versiones **Pytorch native** o **Pytorch Lightning**. Me decanto por el escenario de Lightning que está más dirigido y creo que puede resultar más sencillo para mi conseguir adaptar el código que una implementación nativa. En concreto encuentro la opción de **Fabric** [1] que me permite realizar la paralelización con muy pocas modificaciones del código base.  

El código distribuído se encuentra en `main_distributed_class.py`. Esta implementación emplea el modelo BERT pre-entrenado **bert-base-uncased** y el dataset SQuAD 2.0. que contiene pares de (i) contexto o párrafo (**context**), (ii) pregunta (**queries**) y (iii) respuesta (**answer**). El dataset empleado se encuentra en la carpeta `squad`. He replicado una de las condiciones del caso baseline para la comparación de tiempos de entrenamiento. 

Se escogieron como hiperparámetros **batch_size=8**, **learning-rate=5e-5** y **epochs=2** para los dispositivos empleados (**2 NVIDIA A100 GPU**). Se empleó como optimizador el implementado por defecto **AdamW**. Como estrategias de paralelización se probaron `dp`, `ddp`, `fsdp` y `deepspeed`. También se realizó una prueba extra con `ddp` y 5 epochs para verificar si la pérdida en validación mejora o si se experimenta overfitting.

La implementación requiere de la instalación de **transformers** y **deppspeed** para una de las estrategias de paralelización. La instalación se realiza automáticamente en el entorno virtual ejecutando:

        #!/bin/bash
        source $STORE/mypython/bin/activate

Para instalar transformers y deepspeed ejecutar

        $ pip install -r requirements.txt

Se proporciona los scripts bash y SLURM para la ejecución en `sbatch_distr.sh`.

        $ sbatch sbatch_distr.sh

Las estrategias de paralelización se deben modificar ''ad hoc'' en el ejecutable `.sh`. Modificar la línea de código `--strategy=<put Strategy>` por `dp`, `ddp`, `fsdp` y `deepspeed`

        $ srun python main_distributed_class.py --strategy="ddp"

## Resultados.

Todos los resultados obtenidos para todas las pruebas se pueden consultar en la carpeta `test` en los ficheros `.out`. En la tabla sólo se muestran los resultados que considero más relevantes. 


La primera ejecución se lanzó con los parámetros que por defecto traía la implementación (`baseline_GPUA100_2epoch_8BS.out`) con el dispositivo NVIDIA A100-PCIE-40GB (ver tabla). Comprobé el efecto del batch_size, reduciéndolo de 8 a 1 y aumentándolo a 64. Para un batch_size (BS) de 1 y 2 epochs el tiempo de ejecución fué de 6460.94 s frente a los 4108.80 y 3758.40 s para un BS de 8 y 64 respectivamente. Para BS pequeños, la perdida por entrenamiento (training loss) se reduce considerablemente entre épocas sin una disminución equivalente en la pérdida en validación (validation loss) lo que me indica que se puede estar experimentando overfitting. Esto se ve de forma clara para el BS=16 (`baseline_GPUA100_2epoch_16BS.out`) pasando de Training Loss= 1.31/ Validation Loss=1.11 en epoch 1 a Training Loss= 0.81/ Validation Loss=1.19 en epoch 2.


Jugar con el learning-rate (lr) me proporcionó problemas en las salidas del entrenamiento. No tengo claro cómo estará realizada la implementación escogida internamente pero observo que si reduzco mucho este parámetro, las salidas del programan muestran nan en los cómputos de validation/training loss. Por esta razón decido fijar un lr=5e-5 para todos los casos. No se han probado diferentes configuraciones del optimizador, seleccionando AdamW por defecto. 

Los primeros test para CPU fallaron por límite de tiempo/memoria al emplear batch_sizes muy pequeños/grandes respectivamente. He lanzado otras ejecuciones ajustando tiempo y memoria, pero no han entrado en el sistema de colas. Dejo la tabla sin completar en el momento del tag, pero espero poder incluir en breves los resultados de tiempo.  

Las ejecuciones con T4 están resultando complicadas de ejecutar, al no disponer de recursos suficientes en el supercomputador. Me gustaría hacer las comparaciones en igualdad de condiciones, pero si no consigo recursos intentaré reducir la carga de entrenamientos en epochs y batch_size. Dejo la tabla sin completar en el momento del tag, pero espero poder incluir en breves los resultados de tiempo.  



| Device         | Strategy   | Tiempo total de entrenamiento (s) |
|----------------|---------------------------------|------------|
| 2 NVIDIA A100-PCIE-40GB  |   4108.80               |     8      |
| 2 NVIDIA A100-PCIE-40GB  |   3758.40               |     8     | 
| 1:CPU          |                                 |            | 
| 1:T4           |                                 |            | 

## Referencias

    - [1] https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    - [2] https://rajpurkar.github.io/SQuAD-explorer/
    - [3] https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset
