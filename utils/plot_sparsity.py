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
    save_file= './pdf/qwen_sparsity.pdf'
    rates=[0.01,0.05,0.1]

    num_subplot=len(files)
    fig,axs=plt.subplots(1,num_subplot,figsize=(4*num_subplot,3))

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
        
        all_scores=data.item()['scores']
        all_scores = np.concatenate(all_scores, axis=1)
        max_v=all_scores.max(axis=-1,keepdims=True)
        
        all_s=[]
        for rate in rates:
            all_s.append((all_scores<(max_v*rate)).mean(axis=(1,2,3)))
        sparsities=np.array(all_s).T

        axs[i].set_xticks(rates)
        axs[i].set_xticklabels([str(r) for r in rates])
        axs[i].plot(rates,sparsities[0],label=f'layer 1')
        for j in range(3,sparsities.shape[0],4):
            axs[i].plot(rates,sparsities[j],label=f'layer {j+1}')

        axs[i].legend(fontsize=8)
        axs[i].grid(alpha=0.5)
        axs[i].set_xlabel("Threshold")
        axs[i].set_ylabel("Sparsity")
        axs[i].set_title(titles[i])
    plt.tight_layout()
    plt.savefig(save_file,format='pdf')

