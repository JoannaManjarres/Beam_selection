import matplotlib.pyplot as plt

# Dados das gerações (sem 5G)
geracoes = ["1G", "2G", "3G", "4G", "5G"]
velocidades = [0.0024, 0.064, 2.0, 100.0, 100000.0]  # Mbps
tecnologias = [
    "AMPS\n(Analógico)",
    "GSM, CDMA \n 825-1900 MHz",
    "UMTS, HSPA\n 1,6-2,5 GHz",
    "LTE, WiMAX\n 2-8 GHz",
    "New Radio (NR)\n 24-300 GHz"
]

# Configurações gerais
plt.figure(figsize=(10, 6))
plt.rcParams["font.family"] = "Arial"  # Fonte Times New Roman
plt.rcParams["font.size"] = 16  # Tamanho da fonte

# Cores para as barras (tons de azul)
#cores_azuis = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "darkblue"]
cores_azuis = ["#cfd8dc","#90a4ae", "#546e7a", "#37474f", "#263238"]#, "darkblue"]
cores = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
c = ["#BCD2E8", "#A9CCE3", "#7FB3D5", "#5DADE2", "#1E3F66"]
a = ["#9DECE6", "#75CBD1", "#4EAABD", "#2789A9", "#006995"]

# Criar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(geracoes, velocidades, color=a)#, alpha=0.7)
ax.set_yscale("log")  # Escala logarítmica para enfatizar os saltos nas velocidades

# Remover o frame de cima e da direita
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

velocidades1 = ["< 2.4 kps", "< 64 kps", "2 Mbps", "100 Mbps", "100 Gbps"]
vel_1G = "< 2.4 kps"
vel2G = "< 64 kps"
vel3G = "2 Mbps"
vel4G = "100 Mbps"
vel5G = "100 Gbps"

pos_vel = [0.0011, 0.03, 0.8, 30.0, 30000.0]

for i in range(5):
    ax.text(i, pos_vel[i], f"{velocidades1[i]}", ha='center', fontsize=16, color='white', fontweight='bold', fontfamily='Arial')
    ax.text(i, pos_vel[i]*(3+i), f"{tecnologias[i]}", ha='center', fontsize=14, fontweight='bold', fontfamily='Arial')

#ax.text(0, 0.0013, f"{vel_1G}", ha='center', fontsize=14, color='white')
#ax.text(1, 0.03, f"{vel2G}", ha='center', fontsize=14, color='white')
#ax.text(2, 1.0, f"{vel3G}", ha='center', fontsize=14, color='white')
#ax.text(3, 50.0, f"{vel4G}", ha='center', fontsize=14, color='white')
#ax.text(4, 50000.0, f"{vel5G}", ha='center', fontsize=14, color='white')


# Adicionar rótulos às barras
#for i, (velocidade, tecnologia) in enumerate(zip(velocidades, tecnologias)):
#    ax.text(i, velocidade * 1.2, f"{velocidade} Mbps\n{tecnologia}", ha='center', fontsize=12)

# Títulos e rótulos
#ax.set_title("Velocidades e Tecnologias nas Gerações de Redes Móveis", fontsize=16)
ax.set_xlabel("Geração", fontsize=16)
ax.set_ylabel("Velocidade (Mbps)", fontsize=16)
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Layout e exibição
plt.tight_layout()
#plt.show()
plt.savefig('../../results/score/plot_for_jornal/introduction_qualificacao.png', dpi=300, bbox_inches='tight')
