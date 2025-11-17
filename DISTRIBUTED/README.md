# LAB IA - HPCTools - Artai Rodríguez Moimenta

## Introducción

El objetivo de esta segunda tarea es realizar el entrenamiento distribuído en paralelo con dos NVIDIA A100-PCIE-40GB y distintas estrategias de paralelización. 

Durante el periodo de documentación para la realización de esta segunda parte de la práctica, me he dado cuenta de que la implementación baseline escogida previamente (ver implementación en `main_baseline.py`) no era la mejor opción. Esta implementación ''casera'' está pensada para jugar de forma secuencial con el entrnamiento de modelos BERT y su estructura está realizada directamente en el código principal. Durante el proceso de documentaciṕn para esta segunda parte, me he encontrado con ejemplos de código más funcionales, con implementaciones encapsuladas por clases, lo que resulta más conveniete para este tipo de códigos. Ha sido una mala decisión la elección de la primera implementación en su momento y me hubiese gustado haber esgocido alguna más adecuada para obtener una experiencia más directa y permitirme controlar un poco más las características de la implementación.

Con todo, y dado el tiempo que puedo dedicar a esete ejericcio, decidí quedarme lo que tengo, modificar las salidas del código para adaptarlo al proceso distribuído y simplemente verificar que la paralelización reporta resultados coherentes. A pesar de todo, no poseeo control sobre lo que está ocurriendo internamente y todo está funcionando como una caja negra para mi. 

## Metodología

Para realizar la paralelización se proponen dos versiones **Pytorch native** o **Pytorch Lightning**. Me decanto por el escenario de Lightning que está más dirigido y creo que puede resultar más sencillo para mi conseguir adaptar el código que una implementación nativa. En concreto encuentro la opción de **Fabric** [1] que me permite realizar la paralelización con muy pocas modificaciones del código base.  

El código distribuído se encuentra en `main_distributed_class.py`. Esta implementación emplea el modelo BERT pre-entrenado **bert-base-uncased** y el dataset SQuAD 2.0. que contiene pares de (i) contexto o párrafo (**context**), (ii) pregunta (**queries**) y (iii) respuesta (**answer**). El dataset empleado se encuentra en la carpeta `squad`. He replicado una de las condiciones del caso baseline para la comparación de tiempos de entrenamiento. 

Se escogieron como hiperparámetros **batch_size=8**, **learning-rate=5e-5** y **epochs=2** para los dispositivos empleados (**2 NVIDIA A100 GPU**). Se empleó como optimizador el implementado por defecto **AdamW**. Como estrategias de paralelización se probaron `dp`, `ddp`, `fsdp` y `deepspeed`. También se realizó una prueba extra con `ddp` y 5 epochs para verificar si la pérdida en validación mejora o si se experimenta overfitting.

La implementación requiere de la instalación de **transformers** y **deepspeed** para una de las estrategias de paralelización. La instalación se realiza automáticamente en el entorno virtual ejecutando:

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

En base a los resultados podemos afirmar que se observa una reducción drásticas en los tiempos de ejecución para las distintas estrategias de paralelización con respecto a la implementación `baseline` con excepción de la estrategia `dp` para la cual no parece observarse un efecto en el tiempo de entrenamiento (ver tabla). `dp` divide el batch entre el número de GPUs y según la documentación [2] no se recominenda su uso si lo que se busca es reducción de tiempos. Entiendo que la reducción del batch no afecta en nuestro caso de forma considerable en los tiempos de entrenamiento y se me ocurre pensar que el overhead que introduce la implementación distribuída no es suficiente para compensar el caso baseline. El resto de estrategias muestran reducciones del orden de 5 veces respecto a baseline lo cual es esperado. `ddp` copia el modelo en cada GPU y divide el dataset en lotes que se reparten para posteriormente sincronizar la información una vez calculao el ''loss'' y los gradientes. Se muestra como la estrategia más eficiente.

Por último he querido verificar ahora wue el tiempo de ejecución es razonable si se obtienen mejoras en l modelo al aumentar el número de epochs. Se lanzó una ejecución con la estrategia más eficiente y con 5 epochs para verificar la pérdida por validación vs entrenamiento. Los resultados muestran (ver `Fabric_epoch5_ddp.out`) que a pesar de que la pérdida por entrenamiento se reduce de 1.60 en la epoch 1 a 0.33 en la epoch 5 no se observa una reducción equivalente en la pérdida por validación, pasando de 1.11 a 1.43 respectivamente. Esto indica que el modelo no puede ser mejorado y que aumentos en el entrenamiento sólo llevarán a un sobreajuste en el entrenamiento sin una mejora equivalente en la validación.



| Device                   | Strategy   | Tiempo total de entrenamiento (s) | epochs| 
|--------------------------|------------|-----------------------------------|-------|
| 2 NVIDIA A100-PCIE-40GB  |   dp       |     5251.74                       |   2   | 
| 2 NVIDIA A100-PCIE-40GB  |   ddp       |     997.45                      |   2   |
| 2 NVIDIA A100-PCIE-40GB  |   fsdp       |     1046.26                       |   2   |
| 2 NVIDIA A100-PCIE-40GB  |   deepspeed  |     1073.72                       |   2   |
| 2 NVIDIA A100-PCIE-40GB  |   ddp  |     2500.92                       |   5   |
| 2 NVIDIA A100-PCIE-40GB  |   baseline  |     5108.80                       |   2   |

## Referencias

    - [1] https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    - [2] https://lightning.ai/docs/pytorch/LTS/accelerators/gpu_intermediate.html
    - [3] https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset
