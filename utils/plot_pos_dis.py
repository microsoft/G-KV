import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':

    files = [
        './outputs/amc23_dsllama8b_h2o_info.npy',
        './outputs/amc23_dsllama8b_spankv_info.npy',
        './outputs/amc23_dsllama8b_rkv_info.npy',
        './outputs/alpha/amc23_dsllama8b_rkv_mx09_alpha_08_info.npy',
    ]
    titles = ['Local Attention', 'SnapKV', 'R-KV', 'G-KV']
    save_file = './pdf/llama_pos_dis.pdf'

    budget = 512
    window_size = 16
    kept_len = budget - window_size

    layers = [0, 15, 31]
    fig, axs = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3))

    for i, file in enumerate(files):
        if file.endswith('.json'):
            with open(file, 'r') as f:
                data = json.load(f)
            data['scores'] = [np.array(d) for d in data['scores']]
            for j in range(len(data['pos_ids'])):
                batch_pos_ids = []
                for k in sorted(data['pos_ids'][j].keys()):
                    batch_pos_ids.append(np.array(data['pos_ids'][j][k]))
                data['pos_ids'][j] = np.array(batch_pos_ids)
            data = np.array(data, dtype=object)
        elif file.endswith('.npy'):
            data = np.load(file, allow_pickle=True)

        layer_wise_values = [[] for _ in range(len(layers))]
        for d in data.item()['pos_ids']:
            for k in range(len(layers)):
                layer_d = d[layers[k]][:, :, :kept_len]
                max_v = layer_d.max(axis=-1, keepdims=True)
                d_norm = layer_d / max_v
                layer_wise_values[k].append(d_norm.flatten())

        for l in range(len(layers)):
            layer_wise_values[l] = np.concatenate(layer_wise_values[l])

        for l in range(len(layers)):
            axs[l].violinplot(layer_wise_values[l], positions=[i + 1], showmeans=True, vert=False)

    for l in range(len(layers)):
        axs[l].set_xlabel("Normalized Position of Kept Tokens")
        axs[l].grid(alpha=0.5, linestyle='--')
        axs[l].set_title(f'Layer {layers[l] + 1}')
        if l != 0:
            axs[l].set_yticks([])

    axs[0].set_yticks(list(range(1, len(titles) + 1)), titles)

    plt.tight_layout()
    plt.savefig(save_file, format='pdf', bbox_inches='tight')

