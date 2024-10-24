import os
import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import cv2
from googletrans import Translator  # Biblioteca para tradução
import streamlit as st

# Função para traduzir o texto de português para inglês
def traduzir_para_ingles(texto_portugues):
    translator = Translator()
    traducao = translator.translate(texto_portugues, src='pt', dest='en')
    return traducao.text

# Função para gerar imagem a partir de um prompt
def gerar_imagem(prompt, pipe, output_folder, image_count):
    # Geração da imagem
    image = pipe(prompt).images[0]

    # Caminho completo para salvar a imagem com nome sequencial
    image_path = os.path.join(output_folder, f"{image_count + 1}.png")
    image_count += 1

    # Salvando a imagem gerada
    image.save(image_path)

    return image_path, image_count

# Função para criar um vídeo a partir das imagens geradas
def criar_video(image_count, output_folder, video_path, fps=1):
    # Definindo as dimensões do vídeo com base na primeira imagem
    first_image = Image.open(os.path.join(output_folder, "1.png"))
    height, width = first_image.size

    # Criar um objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Adicionar imagens ao vídeo
    for i in range(image_count):
        img_path = os.path.join(output_folder, f"{i + 1}.png")
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    # Liberar o objeto VideoWriter
    video_writer.release()

# Interface do Streamlit
st.title("Gerador de Imagens e Vídeo a partir de Prompts")

# Campo de texto para o usuário inserir o prompt em português
prompt_pt = st.text_area("Digite o prompt em português:")

# Botão para gerar a imagem
if st.button("Gerar Imagem e Vídeo"):

    # Verifique se a GPU está disponível
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carregue o modelo pré-treinado do Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Traduzir o prompt para inglês
    prompt_en = traduzir_para_ingles(prompt_pt)
    st.write(f"Prompt traduzido: {prompt_en}")

    # Definir a pasta de saída como /content/Imagens
    output_folder = "Imagens"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gerar a imagem a partir do prompt traduzido
    image_count = 0
    image_path, image_count = gerar_imagem(prompt_en, pipe, output_folder, image_count)
    
    # Exibir a imagem gerada no Streamlit
    st.image(image_path, caption="Imagem Gerada")

    # Criar o vídeo a partir das imagens
    video_path = "video_output.avi"  # Caminho para salvar o vídeo
    criar_video(image_count, output_folder, video_path)

    # Exibir o vídeo gerado no Streamlit
    st.video(video_path)

    st.write(f"Vídeo salvo em: {video_path}")
