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
------------------------------------------------------------------------------------------
Antes de realizar a selecao de Beam com a rede WiSARD verifique: 			
 			 se os dados de treinamento e teste estao disponiveis 					
																					
* 	 COORDENADAS 			
 			 data/coord/CoordVehiclesRxPerScene_s008.csv 	
 			 data/coord/CoordVehiclesRxPerScene_s009.csv 
 	 LiDAR 2D 		
 			 data/lidar/s008/lidar_train_raymobtime.npz 
  			 data/lidar/s009/all_data_lidar_+_rx_like_cube_test.npz 
* 	 BEAMS 			
     		data/beams_output/beam_output_baseline_raymobtime_s008/
 			 beams_output_train.npz 		 beams_output_test.npz 	
 		 data/beams_output/beam_output_baseline_raymobtime_s009/ 
 			 beams_output_test_s009.npz 		
------------------------------------------------------------------------------------------
