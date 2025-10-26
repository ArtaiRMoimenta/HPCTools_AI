#!/bin/bash
#SBATCH --exclusive
#SBATCH -J bert-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Activate environment

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


echo "Ejecutando main_baseline.py"

srun python main_baseline.py
