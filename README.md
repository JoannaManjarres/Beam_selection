<h1> SELEÇÃO DE BEAM USANDO UMA REDE NEURAL WISARD </h1>

Neste repositorio se encontra o codigo que realiza a seleção de Beam com uma rede neural sem peso WiSARD. 
São usados dados do dataset Raymobtime (https://www.lasse.ufpa.br/raymobtime/) especificamente o dataset s008 para treinamento e s009 para teste. 
A seleção de beam é realizada usando dados multimodais (coordenadas e LiDAR) com um preprocessamento que se discute entre as diversas propostas ao longo do repositorio.
Dentre as propostas se encontra: 
  - Eliminacao de Variancias
  - Implementação da técnica Análise de componentes principais (PCA)
  - Balanceamento de Classes: até agora usando o randomoversampling
  - Autoencoder: atualizacao do modelo

<p> No link (https://drive.google.com/drive/folders/1m2-OCTeLE6pwMnNCsiObMW0L-W3XOhHB?usp=sharing) na pasta data se encontram os dados necessarios para compilar este algoritmo corretamente, fazer download das pastas: <strong> coord, lidar e beams_output </strong></p>
------------------------------------------------------------------------------------------
Antes de realizar a selecao de Beam com a rede WiSARD verifique que os dados de treinamento e teste estao disponiveis 					
																					
* 	 COORDENADAS 			
<br> <pre> data/coord/CoordVehiclesRxPerScene_s008.csv </pre>
<br> <pre> data/coord/CoordVehiclesRxPerScene_s009.csv </pre>
* 	 LiDAR 		
<br> <pre> data/lidar/s008/lidar_train_raymobtime.npz </pre>
<br> <pre> data/lidar/s008/lidar_validation_raymobtime.npz </pre>
<br> <pre> data/lidar/s009/lidar_test_raymobtime.npz </pre>
* 	 BEAMS 			
<br> <pre> data/beam_output_baseline_raymobtime_s008/beams_output_train.npz </pre>
<br> <pre> data/beam_output_baseline_raymobtime_s008/beams_output_test.npz </pre>
<br> <pre> data/beam_output_baseline_raymobtime_s009/beams_output_test.npz </pre>

------------------------------------------------------------------------------------------


<h1> Para salvar os resultados criar as seguintes pastas: </h1>

PARA GUARDAR OS TESTES DO PARADIGMA APRENDIZADO ONLINE:
 
* 	 COORDENADAS 
<br> <pre> results/score/Wisard/online/coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/coord/sliding_window/ </pre>
* 	 LiDAR
<br> <pre> results/score/Wisard/online/lidar/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar/sliding_window/ </pre>
  
* 	 COORDENADAS + LiDAR
<br> <pre> results/score/Wisard/online/lidar_coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar_coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar_coord/sliding_window/ </pre>	  

