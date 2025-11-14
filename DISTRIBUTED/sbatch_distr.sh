#!/bin/bash
#SBATCH -J bert-baseline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00

#!/bin/bash

source $STORE/mypython/bin/activate

# Archivo de requirements

REQ_FILE="requirements.txt"

# Comprobar si el archivo existe
if [ ! -f "$REQ_FILE" ]; then
    echo "No se encontró el archivo $REQ_FILE"
    exit 1
fi

# Leer cada línea del archivo
while IFS= read -r package || [ -n "$package" ]; do
    # Comprobar si el paquete está instalado
    pip show "$package" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Instalando paquete: $package"
        pip install "$package"
    else
        echo "Ya está instalado: $package"
    fi
done < "$REQ_FILE"


echo "Native Fabric implementation: $1"
srun python main_distributed_class.py --strategy="ddp"





