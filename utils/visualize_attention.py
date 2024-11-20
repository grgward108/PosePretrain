import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(pre_bias_attn, post_bias_attn, save_dir, idx):
    # Pre-bias attention visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(pre_bias_attn[0], cmap="Blues", cbar=True)
    plt.title(f'Pre-Bias Attention (Sample {idx})')
    plt.savefig(os.path.join(save_dir, f"pre_bias_attn_{idx}.png"))
    plt.close()

    # Post-bias attention visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(post_bias_attn[0], cmap="Reds", cbar=True)
    plt.title(f'Post-Bias Attention (Sample {idx})')
    plt.savefig(os.path.join(save_dir, f"post_bias_attn_{idx}.png"))
    plt.close()
