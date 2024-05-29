SELEÇÃO DE BEAM USANDO UM REDE NEURAL WISARD

Neste repositorio se encontra o codigo que realiza a seleção de Beam com uma rede neural sem peso WiSARD. 
São usados dados do dataset Raymobtime (https://www.lasse.ufpa.br/raymobtime/) especificamente o dataset s008 para treinamento e s009 para teste. 
A seleção de beam é realizada usando dados multimodais (coordenadas e LiDAR) com um preprocessamento que se discute entre as diversas propostas ao longo do repositorio.
Dentre as propostas se encontra: 
  - Eliminacao de Variancias
  - Implementação da técnica Análise de componentes principais (PCA)
  - Balanceamento de Classes: até agora usando o randomoversampling
  - Autoencoder: atualizacao do modelo

No link (https://drive.google.com/drive/folders/1m2-OCTeLE6pwMnNCsiObMW0L-W3XOhHB?usp=sharing) na pasta data se encontram os dados necessarios para compilar este algoritmo corretamente
