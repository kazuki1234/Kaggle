fig, axes = plt.subplots(3,4,figsize=(15,5))
venn2(subsets=(train.shape[0],test.shape[0],0), set_labels=("train", "test"), ax=axes[0,0])
for ax in axes.ravel()[:4]:
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                   bottom=False, left=False, right=False, top=False)
    ax.spines["right"].set_color("none"); ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none");   ax.spines["bottom"].set_color("none")
    ax.patch.set_facecolor('white') 

for ax, col in zip(axes.ravel()[4:], test.columns):
    venn2([set(train[col]), set(test[col])], ax=ax, set_labels=(col, col))
plt.show(plt.tight_layout())
