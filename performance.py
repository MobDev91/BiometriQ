
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration de la figure
fig = plt.figure(figsize=(16, 12))
fig.suptitle('BiometriQ - Métriques de Performance', fontsize=20, fontweight='bold', y=0.95)

# Couleurs personnalisées
colors = {
    'primary': '#2980b9',
    'secondary': '#3498db', 
    'accent': '#1abc9c',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'success': '#27ae60',
    'purple': '#9b59b6'
}

# ==================== GRAPHIQUE 1: FPS vs Nombre de visages ====================
ax1 = plt.subplot(2, 2, 1)

# Données FPS basées sur les benchmarks BiometriQ
faces = np.array([1, 2, 3, 4, 5, 6])
fps_values = np.array([45.2, 38.7, 28.3, 22.1, 18.5, 15.2])

# Graphique en ligne avec marqueurs
ax1.plot(faces, fps_values, 'o-', linewidth=3, markersize=8, 
         color=colors['primary'], markerfacecolor=colors['secondary'])

# Zone d'acceptable performance (>15 FPS)
ax1.axhline(y=15, color=colors['danger'], linestyle='--', alpha=0.7, 
           label='Seuil Minimum (15 FPS)')
ax1.fill_between(faces, 15, fps_values, where=(fps_values >= 15), 
                alpha=0.3, color=colors['success'], label='Zone Acceptable')

# Configuration des axes
ax1.set_xlabel('Nombre de Visages Simultanés', fontsize=12, fontweight='bold')
ax1.set_ylabel('FPS (Images par Seconde)', fontsize=12, fontweight='bold')
ax1.set_title('Performance FPS vs Charge de Travail', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(0.5, 6.5)
ax1.set_ylim(0, 50)

# Annotations pour les points clés
for i, (x, y) in enumerate(zip(faces, fps_values)):
    ax1.annotate(f'{y} FPS', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontweight='bold')

# ==================== GRAPHIQUE 2: Répartition du temps de traitement ====================
ax2 = plt.subplot(2, 2, 2)

# Données basées sur l'analyse du pipeline BiometriQ
modalites = ['Détection\nVisages', 'Reconnaissance\nÉmotions', 'Analyse\nDémographique', 
            'Classification\nFormes', 'Reconnaissance\nIdentité']
temps_pct = [35, 25, 20, 15, 5]
couleurs_pie = [colors['primary'], colors['success'], colors['warning'], 
               colors['purple'], colors['danger']]

# Créer le camembert avec explosion pour mise en valeur
explode = (0.05, 0.02, 0.02, 0.02, 0.02)
wedges, texts, autotexts = ax2.pie(temps_pct, labels=modalites, autopct='%1.1f%%',
                                  startangle=90, colors=couleurs_pie, explode=explode,
                                  textprops={'fontsize': 10, 'fontweight': 'bold'})

# Améliorer la lisibilité
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax2.set_title('Répartition du Temps de Traitement\npar Modalité', 
             fontsize=14, fontweight='bold')

# ==================== GRAPHIQUE 3: Évolution GPU ====================
ax3 = plt.subplot(2, 1, 2)

# Simulation d'une session d'analyse de 10 minutes
temps_minutes = np.linspace(0, 10, 300)
gpu_utilisation = np.zeros_like(temps_minutes)

# Phase de démarrage (0-1 min): montée progressive
mask_startup = temps_minutes <= 1
gpu_utilisation[mask_startup] = 65 * (temps_minutes[mask_startup] / 1)**0.5

# Phase d'analyse active (1-8 min): utilisation stable avec variations
mask_active = (temps_minutes > 1) & (temps_minutes <= 8)
base_usage = 65
# Ajouter des variations réalistes
variations = 5 * np.sin(temps_minutes[mask_active] * 2) + 3 * np.random.normal(0, 1, np.sum(mask_active))
gpu_utilisation[mask_active] = base_usage + variations

# Phase de ralenti (8-10 min): descente vers usage minimal
mask_idle = temps_minutes > 8
decay_factor = np.exp(-(temps_minutes[mask_idle] - 8) * 2)
gpu_utilisation[mask_idle] = 65 * decay_factor + 15 * (1 - decay_factor)

# Créer le graphique en aire
ax3.fill_between(temps_minutes, 0, gpu_utilisation, alpha=0.7, 
                color=colors['accent'], label='Utilisation GPU')
ax3.plot(temps_minutes, gpu_utilisation, color=colors['primary'], linewidth=2)

# Zones d'analyse
ax3.axvspan(0, 1, alpha=0.2, color=colors['warning'], label='Démarrage')
ax3.axvspan(1, 8, alpha=0.2, color=colors['success'], label='Analyse Active')
ax3.axvspan(8, 10, alpha=0.2, color=colors['danger'], label='Ralenti')

# Configuration
ax3.set_xlabel('Temps (minutes)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Utilisation GPU (%)', fontsize=12, fontweight='bold')
ax3.set_title('Évolution de l\'Utilisation GPU pendant une Session d\'Analyse', 
             fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 100)

# Annotations pour les phases
ax3.annotate('Chargement\ndes modèles', xy=(0.5, 30), xytext=(0.5, 50),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
            ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', alpha=0.8))

ax3.annotate('Analyse\nmultimodale', xy=(4.5, 70), xytext=(4.5, 85),
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
            ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', alpha=0.8))

# ==================== MÉTRIQUES RÉSUMÉES ====================
# Ajouter un encadré avec les métriques clés
textstr = '''Métriques Clés BiometriQ:
• Performance Max: 45.2 FPS (1 visage)
• Utilisation GPU Moyenne: 65%
• Précision Globale: 90.3%
• Latence Moyenne: 22.1 ms'''

props = dict(boxstyle='round', facecolor=colors['accent'], alpha=0.1)
fig.text(0.02, 0.02, textstr, fontsize=11, verticalalignment='bottom',
         bbox=props, fontweight='bold')

# Ajustement de la mise en page
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.12, hspace=0.3, wspace=0.3)

# Sauvegarde et affichage
plt.savefig('biometriq_performance_metrics.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.show()

# ==================== DONNÉES EXPORTÉES ====================
print("=== DONNÉES DE PERFORMANCE BIOMETRIQ ===")
print("\n1. Performance FPS:")
for f, fps in zip(faces, fps_values):
    print(f"   {f} visage(s): {fps} FPS")

print(f"\n2. Répartition du temps de traitement:")
for modalite, pct in zip(modalites, temps_pct):
    print(f"   {modalite.replace(chr(10), ' ')}: {pct}%")

print(f"\n3. Utilisation GPU:")
print(f"   Pic d'utilisation: {np.max(gpu_utilisation):.1f}%")
print(f"   Utilisation moyenne: {np.mean(gpu_utilisation[mask_active]):.1f}%")
print(f"   Utilisation au repos: {np.mean(gpu_utilisation[mask_idle]):.1f}%")