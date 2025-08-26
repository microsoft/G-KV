import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import gc


if __name__ == '__main__':

    full_attn_res_file='./outputs/amc23_dsllama8b_fullkv.jsonl'

    model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

    window_size=128
    num_window=4
    kept_token=window_size*num_window

    save_file= './pdf/window_consis_llama.pdf'

    # target_layers=[0,13,27]
    target_layers = [0, 15, 31]

    # get fullkv attention scores
    print('loading model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    print('done')
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

    kept_rates=[0.55,0.65,0.75,0.85,0.95]

    
    # (layer,num_window,rate)
    consistency_results=[]
    for i,l in enumerate(target_layers):
        consistency_results.append([])
        for j in range(num_window):
            consistency_results[i].append([])
            for k in range(len(kept_rates)):
                consistency_results[i][j].append([])
        

    for seq_id in tqdm(range(len(all_seq))):
        if len(all_seq[seq_id]["input_ids"])>4096:
            # output_attentions use eager attention which is easy to cause OOM
            # so we need to constrain the sequence length
            all_seq[seq_id]["input_ids"]=all_seq[seq_id]["input_ids"][:4096]
        torch.cuda.empty_cache()
        gc.collect()
        input_tensor = torch.tensor(all_seq[seq_id]["input_ids"]).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(
                input_tensor,
                output_attentions=True,
                return_dict=True,
            )
        for x,l in enumerate(target_layers):
            # (bsz,n_heads,len,len)
            bsz,attn_head,seq_len,_=outputs.attentions[l].shape
            num_kv_heads=model.config.num_key_value_heads
            num_group=attn_head//num_kv_heads
            attn_score=outputs.attentions[l].reshape(bsz,num_kv_heads,num_group,seq_len,seq_len)
            attn_score=attn_score.max(dim=2).values
            # (num_kv_heads,kept_token,prefix_len)
            attn_score=attn_score[:,:,-kept_token:,:-kept_token].squeeze()
            attn_score=attn_score.reshape(num_kv_heads,num_window,window_size,-1).mean(dim=2)
            for y,rate in enumerate(kept_rates):
                k = int(rate * attn_score.shape[-1])
                top_k_indices = torch.topk(attn_score, k, dim=-1).indices  # [num_kv_heads, num_window, k]
                pre_wind_union=[set() for _ in range(attn_score.shape[0])]
                for j in range(attn_score.shape[1] - 1):  # 遍历 window
                    consis_across_heads=[]
                    for i in range(attn_score.shape[0]):  # 遍历 head
                        win1 = set(top_k_indices[i, j].tolist())
                        win2 = set(top_k_indices[i, attn_score.shape[1] - 1].tolist())
                        consistency = len(win1 & win2) / len(win2)
                        pre_wind_union[i]=pre_wind_union[i]|win1
                        consis_across_heads.append(consistency)
                    consistency_results[x][j][y].append(sum(consis_across_heads)/len(consis_across_heads))
                consis_across_heads=[]
                for i in range(attn_score.shape[0]):
                    win2 = set(top_k_indices[i, attn_score.shape[1] - 1].tolist())
                    consistency = len(pre_wind_union[i] & win2) / len(win2)
                    consis_across_heads.append(consistency)
                consistency_results[x][-1][y].append(sum(consis_across_heads)/len(consis_across_heads))
        del outputs

    consistency_results=np.array(consistency_results).mean(axis=-1)

    # 所有子图画在同一行
    fig, axes = plt.subplots(1, len(target_layers), figsize=(4 * len(target_layers), 3), sharey=True)
    labels = [
        '$|S_1 \\cap S_4| / |S_4|$ ',
        '$|S_2 \\cap S_4| / |S_4|$',
        '$|S_3 \\cap S_4| / |S_4|$',
        '$|(S_1 \\cup S_2 \\cup S_3) \\cap S_4| / |S_4|$',
    ]
    if len(target_layers) == 1:
        axes = [axes]
    for j in range( len(target_layers)):
        ax = axes[j]
        for y in range(num_window):
            ax.plot(
                kept_rates,
                consistency_results[j][y],
                label=labels[y],
                marker='o'
            )
        ax.set_title(f'Layer {target_layers[j]+1}')
        ax.set_xlabel('Token Kept Rate')
        ax.grid(True, linestyle='--', alpha=0.5)
        if j == 0:
            ax.set_ylabel('Overlap Ratio')
    plt.tight_layout()
    axes[len(target_layers)-1].legend(
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.,
        frameon=False
    )
    plt.savefig(save_file, format='pdf', bbox_inches='tight')
