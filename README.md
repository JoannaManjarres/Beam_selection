<h1> SELEÇÃO DE BEAM USANDO UMA REDE NEURAL WISARD </h1>

Neste repositorio se encontra o codigo que realiza a seleção de Beam com uma rede neural sem peso WiSARD. 
São usados dados do dataset Raymobtime (https://www.lasse.ufpa.br/raymobtime/) especificamente o dataset s008 para treinamento e s009 para teste. 
A seleção de beam é realizada usando dados multimodais (coordenadas e LiDAR) com um preprocessamento que se discute entre as diversas propostas ao longo do repositorio.
Dentre as propostas se encontra: 
  - Eliminacao de Variancias
  - Implementação da técnica Análise de componentes principais (PCA)
  - Balanceamento de Classes: até agora usando o randomoversampling
  - Autoencoder: atualizacao do modelo

<p> No link (https://drive.google.com/drive/folders/1m2-OCTeLE6pwMnNCsiObMW0L-W3XOhHB?usp=sharing) na pasta data se encontram os dados necessarios para compilar este algoritmo corretamente, fazer download das pastas:  <strong> coord, lidar e beams_output </strong></p>

<hr>

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

<hr>
<h1> Para rodar o codigo </h1>

<p> para simular os testes do paradigma aprendizado online  </p>
<p> rodar o arquivo online_learning_beam_selection_wisard.py --input='coord' --top_k=False </p>

<p> Para simular os testes incluido top-k, apenas basta aticar a variavel top_k como True</p>


<hr>

<h1> Para salvar os resultados  </h1>

Para guardar os testes do paradigma aprendizado online, criar as seguintes pastas:
 
* 	 COORDENADAS 
<br> <pre> results/score/Wisard/online/coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/coord/sliding_window/ </pre>
<br> <pre> results/score/Wisard/online/coord/sliding_window/window_size_var/ </pre>
* 	 LiDAR
<br> <pre> results/score/Wisard/online/lidar/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar/sliding_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar/sliding_window/window_size_var/  </pre>
* 	 COORDENADAS + LiDAR
<br> <pre> results/score/Wisard/online/lidar_coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar_coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar_coord/sliding_window/ </pre>
<br> <pre> results/score/Wisard/online/lidar_coord/sliding_window/window_size_var/ </pre>

Para guardar os testes do paradigma aprendizado online <strong> TOP-K </strong>, criar as seguintes pastas:

* 	 COORDENADAS 
<br> <pre> results/score/Wisard/online/top_k/coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/coord/sliding_window/ </pre>
* 	 LiDAR
<br> <pre> results/score/Wisard/online/top_k/lidar/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/lidar/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/lidar/sliding_window/ </pre>
* 	 COORDENADAS + LiDAR
<br> <pre> results/score/Wisard/online/top_k/lidar_coord/fixed_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/lidar_coord/incremental_window/ </pre>
<br> <pre> results/score/Wisard/online/top_k/lidar_coord/sliding_window/ </pre>


