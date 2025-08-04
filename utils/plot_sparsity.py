import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import gc


if __name__ == '__main__':

    files=[
        './outputs/amc23_dsqwen7b_h2o_info.json',
        './outputs/amc23_dsqwen7b_spankv_info.json',
        './outputs/amc23_dsqwen7b_rkv_info.json',
    ]

    full_attn_res_file='./outputs/amc23_dsqwen7b_fullkv.jsonl'

    model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

    window_size=16

    titles=['Full KV','H2O','SnapKV','R-KV']
    save_file= './pdf/qwen_sparsity.pdf'
    rates=[0.01,0.05,0.1,0.2]



    # get fullkv attention scores
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model=model.cuda()
    model.eval()
    res=[]
    with open(full_attn_res_file, "r") as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    
    all_seq=[]
    for i in range(len(res)):
        all_seq.append(tokenizer(res[i]["prompt"]+res[i]["output"]))

    # (len(rates),num_layers,num_seq)
    full_kv_sparsity=[defaultdict(list) for _ in range(len(rates))]
    
    for seq_id in tqdm(range(len(all_seq))):
        torch.cuda.empty_cache()
        gc.collect()
        input_tensor = torch.tensor(all_seq[seq_id]["input_ids"]).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(
                input_tensor,
                output_attentions=True,
                return_dict=True,
            )
        num_layers=len(outputs.attentions)
        for l in range(num_layers):
            # (bsz,n_heads,len,len)
            bsz,attn_head,seq_len,_=outputs.attentions[l].shape
            num_kv_heads=model.config.num_key_value_heads
            num_group=attn_head//num_kv_heads
            attn_score=outputs.attentions[l].reshape(bsz,num_kv_heads,num_group,seq_len,seq_len)
            attn_score=attn_score.max(dim=2).values
            attn_score=attn_score[:,:,-window_size:,:-window_size]
            max_v=attn_score.max(dim=-1,keepdim=True).values
            for i,p in enumerate(rates):
                sparsity=(attn_score<(max_v*p)).float().mean()
                full_kv_sparsity[i][l].append(sparsity.item())
        del outputs


    full_kv_sparsity=np.array([[full_kv_sparsity[i][k] for k in sorted(full_kv_sparsity[i].keys())] for i in range(len(rates))])
    full_kv_sparsity=full_kv_sparsity.mean(axis=2).T


    
    num_subplot=len(files)+1
    fig,axs=plt.subplots(1,num_subplot,figsize=(3*num_subplot,3))
    axs[0].set_xticks(rates)
    axs[0].set_xticklabels([str(r) for r in rates])
    
    axs[0].plot(rates,full_kv_sparsity[0],label=f'layer 1')

    for i in range(3,full_kv_sparsity.shape[0],4):
        axs[0].plot(rates,full_kv_sparsity[i],label=f'layer {i+1}',alpha=0.5)

    
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.5)
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("Sparsity")
    axs[0].set_title("Full KV")

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

        axs[i+1].set_xticks(rates)
        axs[i+1].set_xticklabels([str(r) for r in rates])
        axs[i+1].plot(rates,sparsities[0],label=f'layer 1')
        for j in range(3,sparsities.shape[0],4):
            axs[i+1].plot(rates,sparsities[j],label=f'layer {j+1}')

        axs[i+1].legend(fontsize=8)
        axs[i+1].grid(alpha=0.5)
        axs[i+1].set_xlabel("Threshold")
        axs[i+1].set_title(titles[i+1])

    plt.tight_layout()
    plt.savefig(save_file,format='pdf')

