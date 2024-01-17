#pip install gputil
import GPUtil

def obtener_informacion_gpu():
    gpus = GPUtil.getGPUs()

    if not gpus:
        return "No se encontraron GPUs."

    informacion_gpus = []

    for i, gpu in enumerate(gpus):
        informacion_gpus.append(f"GPU {i + 1}:\n"
                                f"Nombre: {gpu.name}\n"
                                f"VRAM total: {gpu.memoryTotal} MB\n"
                                f"VRAM usada: {gpu.memoryUsed} MB\n"
                                f"VRAM libre: {gpu.memoryFree} MB\n"
                                f"Porcentaje de uso: {gpu.load * 100}%\n")

    return "\n".join(informacion_gpus)

print(obtener_informacion_gpu())
##############https://www.youtube.com/watch?v=o14-CklOzMg
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
import torch
import torchvision

# No need for Jupyter-specific initialization
# No need for utils.notebook_init()

print(torch.cuda.is_available())
print(torch.__version__)
print(torchvision.__version__)