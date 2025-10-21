#!/bin/bash
#SBATCH --job-name=bertmodel-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00

 

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
