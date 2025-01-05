from scipy.io import loadmat
from scipy.io import loadmat
import numpy as np
from PIL import Image
import sys
import  os
from pathlib import Path
import shutil


# HYPERPARAMETERS
# NOTE: ajustar isto para o contexto de execução da sua maquina, caso contratio o programa nao funcionara
X_MATLAB_FILE_PATH = r'C:\Users\USER\Desktop\jupytervscode\LabMems\dead-ends\data\ImCrop.mat'  # caminho para arquivo .mat que contem as fotos do micro-model/rocha
Y_MATLAB_FILE_PATH = r'C:\Users\USER\Desktop\jupytervscode\LabMems\dead-ends\data\DeadEnd.mat' # caminho para o arquivo .mat que contem os recortes desejados
QUANTIDADE_DE_IMAGENS = 70
COUNT_NAME = 'Nfig'


script_path = Path(os.path.realpath(__file__))
current_dir = script_path.parent



print(current_dir)

x = loadmat(X_MATLAB_FILE_PATH)
y = loadmat(Y_MATLAB_FILE_PATH)

arr = y[COUNT_NAME]

fig_list = []

for i in range(0, QUANTIDADE_DE_IMAGENS):
    fig_list.append(int(arr[0][i][0].squeeze()))

fig_list.reverse()
fig_list[0]

# extrair fator acumulativo
for i in range(len(fig_list) - 1):
    fig_list[i] -= fig_list[i+1]

fig_list.reverse()

def maximize(mask):
    '''
    Convert label to 0-1 format
    '''

    # assegurar-se qu imagem tem 3 canais
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)

    c1 = mask[:, :, 0]
    c2 = mask[:, :, 1]
    c3 = mask[:, :, 2]

    out = c1 + c2 + c3
    out[out > 0] = 1
    return out

def sum_masks(m1, m2):
    return maximize(m1) + maximize(m2)

label_list = [] # armazenar mascaras em uma lista

j = 0
for img_range in fig_list:
    accumulate = np.zeros(shape=[200, 200, 3]) # resetar mascara acumulativa
    i = 0
    while i < img_range:
        img = np.array(y['Anew'][0][j][0], dtype=np.float32)
        i += 1
        j += 1
        accumulate = sum_masks(accumulate, img) # acumular mascara sobre iterações
    label_list.append(accumulate) # armazenar

input_list = []
for i in range(QUANTIDADE_DE_IMAGENS):
    img = np.array(x['Acrop'][0][i][0], dtype=np.float32)
    img = img / np.max(img)
    input_list.append(img)

print("----------------------------------------------------")
print()
resp = input("Deseja performar data augmentation? (y/n) ")
print()
print("----------------------------------------------------")

if resp == 'n' or resp == 'N':

    # salvar imagens direto no diretorio, sem nenhuma operação de data augmentation
    os.makedirs(current_dir / "data" / "input", exist_ok=True)
    os.makedirs(current_dir / "data" / "label", exist_ok=True)
    input_folder = current_dir / "data" / "input"
    label_folder = current_dir / "data" / "label"

    for i, img_array in enumerate(input_list):
        if img_array.ndim != 2:
            img_array = img_array.squeeze()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(input_folder, f"{i + 1}.png"))

    for i, img_array in enumerate(label_list):
        if img_array.ndim != 2:
            img_array = img_array.squeeze()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(label_folder, f"{i + 1}.png"))
    
    print(f'Salvando arquivos em diretório {current_dir / "data"} e finalizando programa')
    sys.exit(0)

elif resp == 'Y' or resp == 'y':

    print("performando data augmentation")
    print("operações disponíveis: rotação e espelhamento")
    print("razão de aumento: 8")


    def rotate90(path_original, img_name, rotate_name, path_to_augment):
        '''
        Recebe o caminho de uma imagem PIL `path_original` e salva ela e sua rotação com os respectivos nomes informados em `path_to_augment`

        multiplica por 2 o tamanho do dataset
        '''

        original_image = Image.open(os.path.join(path_original, img_name))
        rotated_image = original_image.rotate(90)
        original_image.save(os.path.join(path_to_augment, img_name))
        rotated_image.save(os.path.join(path_to_augment, rotate_name))


    # salvar imagens em um folder temporaro
    os.makedirs(current_dir / "temp" / "torotate" / "input")
    os.makedirs(current_dir / "temp" / "torotate" / "label")
    input_folder = current_dir / "temp" / "torotate" / "input"
    label_folder = current_dir / "temp" / "torotate" / "label"

    for i, img_array in enumerate(input_list):
        if img_array.ndim != 2:
            img_array = img_array.squeeze()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(input_folder, f"{i + 1}.png"))

    for i, img_array in enumerate(label_list):
        if img_array.ndim != 2:
            img_array = img_array.squeeze()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(label_folder, f"{i + 1}.png"))

    img_number = 1
    input_list_from = os.listdir(input_folder)
    os.makedirs(current_dir / "temp" / "toflip" / "input")
    input_folder_rotated = current_dir / "temp" / "toflip" / "input"
    for img_name in input_list_from:
        rotate_number = img_number + QUANTIDADE_DE_IMAGENS
        rotate_name = str(rotate_number) + ".png"
        rotate90(input_folder, img_name, rotate_name, input_folder_rotated)
        img_number += 1

    img_number = 1
    label_list_from = os.listdir(label_folder)
    os.makedirs(current_dir / "temp" / "toflip" / "label")
    label_folder_rotated = current_dir / "temp" / "toflip" / "label"
    for img_name in label_list_from:
        rotate_number = img_number + QUANTIDADE_DE_IMAGENS
        rotate_name = str(rotate_number) + ".png"
        rotate90(label_folder, img_name, rotate_name, label_folder_rotated)
        img_number += 1
    
    os.makedirs(current_dir / "data" / "input")
    input_folder_final = current_dir / "data" / "input"
    os.makedirs(current_dir / "data" / "label")
    label_folder_final = current_dir / "data" / "label"

    def flip_mirror_flip_and_mirror(path1, path2):
        '''
        Performa data augmentation fazendo, para cada imagem existente em `path1` e `path2`, as 4 operações:

        salva a imagem e mascara original
        salva a imagem e mascara espelhadas
        salva a imagem e mascara de cabeça pra baixo
        salva a imagem e mascara espelhadas de cabeça pra baixo

        ou seja, um dataset com x imagens vira um dataset com 4x imagens
        '''
        
        path1_list = []
        for file in os.listdir(path1):
            path1_list.append(os.path.join(path1, file))
        path2_list = []
        for file in os.listdir(path2):
            path2_list.append(os.path.join(path2, file))

        name = 1
        for file_index in range(len(os.listdir(path1))):
            image_path = path1_list[file_index]
            mask_path = path2_list[file_index]

            # salvar imagem e mascara padrão
            regular_image = Image.open(image_path)
            regular_image.save(os.path.join(input_folder_final, str(name) + '.png'))
            regular_mask = Image.open(mask_path)
            regular_mask.save(os.path.join(label_folder_final, str(name) + '.png'))
            name += 1

            # salvar imgem e mascara espelhadas no eixo X
            mirror_image = regular_image.transpose(Image.FLIP_LEFT_RIGHT)
            mirror_image.save(os.path.join(input_folder_final, str(name) + '.png'))
            mirror_mask = regular_mask.transpose(Image.FLIP_LEFT_RIGHT)
            mirror_mask.save(os.path.join(label_folder_final, str(name) + '.png'))
            name += 1

            # salvar imagem e mascara espelhadas no eixo Y
            flip_image = regular_image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_image.save(os.path.join(input_folder_final, str(name) + '.png'))
            flip_mask = regular_mask.transpose(Image.FLIP_TOP_BOTTOM)
            flip_mask.save(os.path.join(label_folder_final, str(name) + '.png'))
            name += 1

            # salvar imagem e mascara espelhadas no eixo X e Y
            flip_and_mirror_image = mirror_image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_and_mirror_image.save(os.path.join(input_folder_final, str(name) + '.png'))
            flip_and_mirror_mask = mirror_mask.transpose(Image.FLIP_TOP_BOTTOM)
            flip_and_mirror_mask.save(os.path.join(label_folder_final, str(name) + '.png'))
            name += 1

    flip_mirror_flip_and_mirror(input_folder_rotated, label_folder_rotated)
    shutil.rmtree(current_dir / "temp")
    sys.exit(0)