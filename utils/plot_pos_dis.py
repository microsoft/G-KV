import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':

    files=[
        './outputs/amc23_dsqwen7b_h2o_info.json',
        './outputs/amc23_dsqwen7b_spankv_info.json',
        './outputs/amc23_dsqwen7b_rkv_info.json',
    ]
    titles=['H2O','SnapKV','R-KV']
    save_file= './pdf/qwen_pos_dis.pdf'

    
    layers=[0,14,27]
    fig,axs=plt.subplots(1,len(layers),figsize=(4*len(layers),3))

    for i,file in enumerate(files):
        if file.endswith('.json'):
            with open(file,'r') as f:
                data=json.load(f)
            data['scores']=[np.array(d) for d in data['scores']]
            for j in range(len(data['pos_ids'])):
                batch_pos_ids=[]
                for k in sorted(data['pos_ids'][j].keys()):
                    batch_pos_ids.append(np.array(data['pos_ids'][j][k]))
                data['pos_ids'][j]=np.array(batch_pos_ids)
            data=np.array(data,dtype=object)
        elif file.endswith('.npy'):
            data=np.load(file,allow_pickle=True)
        
        layer_wise_values = [[] for _ in range(len(layers))]
        for d in data.item()['pos_ids']:
            print(d.shape)
            for k in range(len(layers)):
                layer_d=d[layers[k]]
                max_v=layer_d.max(axis=-1,keepdims=True)
                d_norm=layer_d/max_v
                layer_wise_values[k].append(d_norm.flatten())

        for l in range(len(layers)):
            layer_wise_values[l]=np.concatenate(layer_wise_values[l])

        for l in range(len(layers)):
            axs[l].violinplot(layer_wise_values[l], positions=[i+1], showmeans=True, vert=False)
    
    for l in range(len(layers)):
        axs[l].set_xlabel("Normalized Position of Kept Tokens")
        axs[l].grid(alpha=0.5,linestyle='--')
        axs[l].set_title(f'Layer {layers[l]}')
    
    axs[0].set_yticks([1,2,3], titles)

    plt.tight_layout()
    plt.savefig(save_file,format='pdf',bbox_inches='tight')

