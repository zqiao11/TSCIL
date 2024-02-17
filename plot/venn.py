import matplotlib.pyplot as plt
from matplotlib_venn import venn3


# Choose colorblind-friendly colors: Blue, Green, Orange
venn_colors = ('#377eb8', '#4daf4a', '#ff7f00')  # Colorblind-friendly palette


# Create the Venn diagram
plt.figure(figsize=(8, 8), dpi=128)
v = venn3(subsets=(1, 1, 0.5, 1, 0.5, 0.5, 0.25),
          # set_labels=('Regularization', 'Replay', 'Architecture'),
          set_colors=venn_colors, alpha=0.7)


v.get_label_by_id('100').set_text('\n                      LwF\n\nEWC\n\n                MAS \n\nSI          \n\nOWM')
v.get_label_by_id('010').set_text('\nER                   \n\nGR       DER             \n\n   MIR \n\n Mnemonics  \n\n  ASER')
v.get_label_by_id('001').set_text(' PNN                         \n\n                        ExpertGate \n\nDEN           \n\n')
v.get_label_by_id('110').set_text('iCaRL \n\n GEM')
v.get_label_by_id('011').set_text('iTAML')
v.get_label_by_id('101').set_text('AR1')
v.get_label_by_id('111').set_text('DualNet')

v.get_label_by_id('A').set_text('')
v.get_label_by_id('B').set_text('')
v.get_label_by_id('C').set_text('')


# Increase the font size
for text in v.set_labels:
    text.set_fontsize(24)
for text in v.subset_labels:
    text.set_fontsize(20)

plt.text(-0.24, 0.55, 'Regularization', fontsize=24, color='gray', ha='center', weight='bold')
plt.text(0.24, 0.55, 'Replay', fontsize=24, color='gray', ha='center', weight='bold')
plt.text(0, -0.7, 'Architecture', fontsize=24, color='gray', ha='center', weight='bold')

# plt.text(-0.35, 0.55, 'Regularization', fontsize=24, color='gray', ha='center', weight='bold')
# plt.text(0.36, 0.55, 'Replay', fontsize=24, color='gray', ha='center', weight='bold')
# plt.text(0, -0.7, 'Architecture', fontsize=24, color='gray', ha='center', weight='bold')


# Display the plot

plt.tight_layout(rect=[0, 0, 1, 1])
path = "../result/plots/venn"
plt.savefig(path, bbox_inches='tight')
plt.show()
