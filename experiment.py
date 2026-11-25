import numpy as np
from common import *

def run():
    import torch
    model,tok=causal_model();prompt=tok('The database backup should be',return_tensors='pt').input_ids
    corpus=tok('The database backup should be checked before the service is restarted. The service should be healthy.').input_ids
    counts={}
    for a,b in zip(corpus,corpus[1:]):counts.setdefault(a,{})[b]=counts.setdefault(a,{}).get(b,0)+1
    def draft(last):return max(counts.get(last,{tok.eos_token_id:1}),key=counts.get(last,{tok.eos_token_id:1}).get)
    def greedy():
        seq=prompt.clone()
        with torch.no_grad():
            for _ in range(12):seq=torch.cat([seq,model(seq,use_cache=False).logits[:,-1].argmax(-1,keepdim=True)],1)
        return seq
    def speculative():
        seq=prompt.clone();accepted=0;proposed=0;calls=0
        with torch.no_grad():
            while seq.shape[1]<prompt.shape[1]+12:
                proposals=[];last=int(seq[0,-1])
                for _ in range(min(3,prompt.shape[1]+12-seq.shape[1])):last=draft(last);proposals.append(last)
                extended=torch.cat([seq,torch.tensor([proposals])],1);logits=model(extended,use_cache=False).logits;calls+=1;proposed+=len(proposals)
                base=seq.shape[1];tokens=[]
                for i,p in enumerate(proposals):
                    actual=int(logits[0,base+i-1].argmax());tokens.append(actual)
                    if p!=actual:break
                    accepted+=1
                seq=torch.cat([seq,torch.tensor([tokens])],1)
        return seq,accepted,proposed,calls
    reference=greedy();out,accepted,proposed,calls=speculative()
    return {'accepted_draft_tokens':accepted,'proposed_tokens':proposed,'target_calls':calls,
            'greedy_seconds':timed(greedy,2),'speculative_seconds':timed(speculative,2),
            'checks':[check(torch.equal(reference,out),'Speculative verification exactly matches greedy target output')],
            'scope':'Pretrained SmolLM2 target, fitted token-bigram draft, greedy verification only. No sampling or KV-cache optimization; negative speed results retained.'}
