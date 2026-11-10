"""用本地预训练Qwen模型验证攻击实现，结果不能代替论文完整隐私实验。

分阶段执行以避免同时加载模型、训练网络和多套权重。真实权重抽样由公开种子
确定；预测文件与评分真值分目录保存。已知变换只用于搭建对照，不交给预测函数。
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SEED=20261028
DEFAULT_MODEL=ROOT/'model/Qwen2.5-0.5B-Instruct'
DEFAULT_OUT=ROOT/'outputs/attack_verification_20261028'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def get_weight(folder,name,rows=None):
    """只取所需张量或行，不把完整大模型复制成多份float64权重。"""
    folder=Path(folder);index=folder/'model.safetensors.index.json'
    if index.exists():
        mapping=read(index)['weight_map']
        if name not in mapping and name=='lm_head.weight':name='model.embed_tokens.weight'
        file=folder/mapping[name]
    else:file=folder/'model.safetensors'
    with safe_open(str(file),framework='pt',device='cpu') as handle:
        if name not in handle.keys() and name=='lm_head.weight':name='model.embed_tokens.weight'
        value=handle.get_tensor(name)
        if rows is not None:value=value[torch.as_tensor(rows,dtype=torch.long)]
        return value.detach().float().clone()


def score(predictions,truth):
    """评分只读取已经生成的结果，不参与候选生成、训练或选模型。"""
    from scripts.evaluation.privacy.run_additional_attacks import evaluate_predictions
    return evaluate_predictions(predictions,truth)


def prepare(args):
    from transformers import AutoTokenizer
    import transformers
    out=args.out
    for name in ('inputs','truth','predictions','checkpoints','logs'): (out/name).mkdir(parents=True,exist_ok=True)
    config=read(args.model/'config.json')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    ids=torch.randperm(tokenizer.vocab_size,generator=torch.Generator().manual_seed(SEED))[:128].tolist()
    write(out/'inputs/public_token_ids.json',ids)
    versions=dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,
                  cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    docker=Path(r'C:\Program Files\Docker\Docker\resources\bin\docker.exe')
    docker_probe=subprocess.run([str(docker),'ps','-a','--format','{{json .}}'],capture_output=True,text=True,timeout=20) if docker.exists() else None
    payload=dict(date='2026-10-28',model=str(args.model),model_type=config['model_type'],hidden_size=config['hidden_size'],
        layers=config['num_hidden_layers'],model_vocab_size=config['vocab_size'],tokenizer_vocab_size=tokenizer.vocab_size,
        model_weights_sha256=sha(args.model/'model.safetensors'),versions=versions,seed=SEED,
        weight_probe_vocabulary=128,weight_probe_layers=[0,11,23],
        docker=dict(executable_exists=docker.exists(),returncode=docker_probe.returncode if docker_probe else None,
                    message=(docker_probe.stderr or docker_probe.stdout).strip() if docker_probe else 'Docker executable unavailable'),
        protocol='Controlled implementation verification; real pretrained weights plus known synthetic transforms; not full-vocabulary or full-paper attack success claims')
    write(out/'environment.json',payload)
    print(json.dumps(payload,ensure_ascii=True),flush=True)
    return payload


def load_probe(args,folder=None,rows=None):
    """提取真实模型的128词条、完整头宽/前馈宽和第0、11、23层，非随机初始化模型。"""
    folder=folder or args.model
    rows=read(args.out/'inputs/public_token_ids.json') if rows is None else rows
    config=read(folder/'config.json');state={}
    for key in ('model.embed_tokens.weight','lm_head.weight'):
        state[key]=get_weight(folder,key,rows)
    for key in ('model.norm.weight',):state[key]=get_weight(folder,key)
    for layer in (0,11,23):
        for suffix in ('input_layernorm.weight','post_attention_layernorm.weight',
                       'self_attn.q_proj.weight','self_attn.k_proj.weight','mlp.gate_proj.weight',
                       'mlp.up_proj.weight','mlp.down_proj.weight'):
            key=f'model.layers.{layer}.{suffix}';state[key]=get_weight(folder,key)
    return state,dict(config,vocab_size=len(rows))


def positive_transform(plain,config):
    """已知正确对照：算法1残差变换、FFN置换、GQA头置换与成对可逆Q/K变换。

    不加入会破坏不变量的噪声和FFN非均匀缩放，先检查攻击在成立条件下能否恢复。
    """
    from src.keymat import build_keymat_transform,sample_compatible_inverse
    key=build_keymat_transform(config['hidden_size'],4,.1,SEED,dtype=torch.float64)
    p,q=key.key,key.inverse;g=torch.Generator().manual_seed(SEED+1)
    inv=torch.randperm(128,generator=g)
    server={'model.embed_tokens.weight':plain['model.embed_tokens.weight'].double()[inv]@p,
            'lm_head.weight':plain['lm_head.weight'].double()[inv]@q.T}
    heads,kv=config['num_attention_heads'],config['num_key_value_heads'];width=config['hidden_size']//heads
    kv_order=torch.arange(kv).roll(1)
    q_order=torch.arange(heads).reshape(kv,heads//kv)[kv_order].reshape(-1)
    for layer in (0,11,23):
        prefix=f'model.layers.{layer}'
        ff=plain[prefix+'.mlp.gate_proj.weight'].shape[0];order=torch.randperm(ff,generator=g)
        for kind in ('gate','up'):
            name=prefix+f'.mlp.{kind}_proj.weight'
            inverse=sample_compatible_inverse(key,SEED+layer*10+(1 if kind=='gate' else 2))
            server[name]=plain[name].double()[order]@inverse.T
        name=prefix+'.mlp.down_proj.weight';server[name]=p.T@plain[name].double()[:,order]
        # 同一个二维组的Q/K分别乘A与A^{-T}，既验证列空间特征，也验证QK乘积。
        a=torch.tensor([[1.2,.2],[-.1,.9]],dtype=torch.float64)
        for kind,head_order,matrix in (('q',q_order,a),('k',kv_order,torch.linalg.inv(a).T)):
            name=prefix+f'.self_attn.{kind}_proj.weight'
            inverse=sample_compatible_inverse(key,SEED+layer*10+(3 if kind=='q' else 4))
            weight=plain[name].double()@inverse.T
            weight=weight.view(len(head_order),width,-1)[head_order].clone()
            for i in range(width//2):
                pair=[i,i+width//2]
                weight[:,pair,:]=matrix.T@weight[:,pair,:]
            server[name]=weight.reshape(-1,p.shape[1])
    return server,dict(config,hidden_size=p.shape[1],head_dim=width),inv


def weights(args):
    from src.security_qwen.weight_attacks import recover_weight_mapping,table9_features,attention_leverage_features
    out=args.out;plain,config=load_probe(args)
    server,server_config,truth=positive_transform(plain,config)
    truth=truth.tolist();write(out/'truth/weight_positive.json',truth)
    cases=[]
    methods=[('gate-ia',('gate',)),('attn-ia',('gate',))]+[('vma',(kind,)) for kind in ('gate','up','down_head','embed_head','qk')]
    methods.append(('vma',('gate','up','down_head','embed_head','qk')))
    for attack,combos in methods:
        label=attack+'_'+'_'.join(combos);start=time.perf_counter()
        result=recover_weight_mapping(plain,server,config,server_config,attack=attack,layers=(0,11,23),combinations=combos,topk=5)
        write(out/f'predictions/positive_{label}.json',result)
        scored=score(result,truth)
        wrong=truth[1:]+truth[:1]
        control=score(result,wrong)
        assert scored['token_accuracy']==1.,(label,scored)
        assert control['token_accuracy']==0.
        cases.append(dict(case='positive',attack=attack,combinations=combos,**scored,wrong_truth_accuracy=control['token_accuracy'],seconds=time.perf_counter()-start))
        print('WEIGHTS',label,scored['token_accuracy'],flush=True)
    # 候选故意排除一半正确答案，检查程序是否把它们偷偷补回候选。
    result=recover_weight_mapping(plain,server,config,server_config,attack='vma',layers=(0,11,23),candidate_ids=list(range(64)),topk=5)
    assert all(set(row)<=set(range(64)) for row in result['topk_plain_ids'])
    excluded=[i for i,t in enumerate(truth) if t>=64]
    assert all(result['predicted_plain_ids'][i]!=truth[i] for i in excluded)
    write(out/'predictions/candidate_exclusion.json',result)
    candidates=dict(candidate_count=64,total_observed=128,truth_coverage=.5,**score(result,truth),excluded_truth_recovered=0)
    # 保存攻击输入样本，检查真实文件读取与CLI。它们是权重样本，不是假冒的完整模型。
    for name,state,cfg in (('public_weight_probe',plain,config),('positive_weight_probe',server,server_config)):
        folder=out/'inputs'/name;folder.mkdir(exist_ok=True)
        save_file({k:v.contiguous() for k,v in state.items()},str(folder/'model.safetensors'))
        write(folder/'config.json',cfg)
        (folder/'README.txt').write_text('Selected real pretrained weight tensors for attack verification only; not a complete deployable model.\n',encoding='utf-8')
    cli_out=out/'predictions/positive_gate_cli.json'
    cmd=[sys.executable,str(ROOT/'scripts/evaluation/privacy/run_additional_attacks.py'),'weights','--attack','gate-ia',
        '--baseline-model-dir',str(out/'inputs/public_weight_probe'),'--server-dir',str(out/'inputs/positive_weight_probe'),
        '--layers','0','11','23','--output',str(cli_out)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    assert score(read(cli_out),truth)['token_accuracy']==1.
    # 已有正式同宽/扩维导出：仍只取固定128词条，不能称为全词表实验。
    del server;gc.collect()
    public_ids=read(out/'inputs/public_token_ids.json')
    for variant in ('square_zero_noise','expanded_zero_noise_sharded'):
        checkpoint=ROOT/'artifacts/round3_pretrained'/variant
        secret=torch.load(checkpoint/'client/client_secret.pt',weights_only=True,map_location='cpu')
        order=torch.randperm(128,generator=torch.Generator().manual_seed(SEED+7)).tolist()
        rows=secret['perm_vocab'][torch.tensor(public_ids)[order]].tolist()
        actual,actual_config=load_probe(args,checkpoint/'server',rows)
        actual_truth=order
        write(out/f'truth/{variant}.json',actual_truth)
        for attack,combos in [('gate-ia',('gate',)),('vma',('gate',)),('vma',('gate','up','down_head')),('attn-ia',('gate',))]:
            label=attack+'_'+'_'.join(combos)
            result=recover_weight_mapping(plain,actual,config,actual_config,attack=attack,layers=(0,11,23),
                                         combinations=combos,topk=5,fuse_original_norms=variant.startswith('expanded'))
            write(out/f'predictions/{variant}_{label}.json',result)
            cases.append(dict(case=variant,attack=attack,combinations=combos,**score(result,actual_truth)))
            print('ACTUAL',variant,label,cases[-1]['token_accuracy'],flush=True)
        del actual;gc.collect()
    return dict(cases=cases,candidate_exclusion=candidates,cli_passed=True,
        scope='128 independently sampled real token rows, full hidden/FFN widths, layers 0/11/23. Restricted weight-table verification; not a full-vocabulary attack.')


def pattern_data():
    """六种符号的可学习公开规则，用未重复的序列检验SDA能否真正解码。"""
    generator=torch.Generator().manual_seed(SEED+22);order=[3,0,5,2,1,4]
    data=set()
    while len(data)<224:
        prefix=order.copy()
        suffix=[order[int(i)] for i in torch.randint(6,(8,),generator=generator)]
        data.add(tuple(prefix+suffix))
    items=[list(seq) for seq in sorted(data)];shuffle=torch.randperm(len(items),generator=generator).tolist()
    items=[items[i] for i in shuffle]
    return items[:160],items[160:192],items[192:]


def train_or_load_sda(train,val,*,model_path,**kwargs):
    """同一次实验重试时复用已成功保存的训练结果，不覆盖或重复训练。"""
    from src.security_qwen.sda_transformer import train_sda_transformer
    if not model_path.exists():return train_sda_transformer(train,val,model_path=model_path,**kwargs)
    saved=torch.load(model_path,map_location='cpu',weights_only=True)
    if saved['plain_vocab_size']!=kwargs['vocab_size'] or saved['seed']!=kwargs['seed']:
        raise ValueError('Existing SDA checkpoint does not match the requested experiment')
    return dict(history=saved['history'],best_validation_loss=min(row['validation_loss'] for row in saved['history']))


def sequences(args):
    from src.security_qwen.sda_transformer import train_sda_transformer,recover_sda_transformer,encode_symbols
    from src.security_qwen.frequency_mapping import recover_frequency_mapping
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    out=args.out;device=args.device;results={}
    train,val,test=pattern_data();mapping=[83,11,72,24,96,55]
    observed=[[mapping[t] for t in seq] for seq in test]
    assert not set(map(tuple,train))&set(map(tuple,val+test))
    assert not set(map(tuple,val))&set(map(tuple,test))
    for label,value in [('train',train),('validation',val),('observed',observed)]:write(out/f'inputs/sda_control_{label}.json',value)
    write(out/'truth/sda_control.json',test)
    path=out/'checkpoints/sda_control.pt'
    learned=train_or_load_sda(train,val,vocab_size=6,model_path=path,epochs=40,batch_size=32,hidden=64,device=device,seed=SEED)
    recovered=recover_sda_transformer(path,observed,device=device)
    write(out/'predictions/sda_control.json',recovered)
    renamed=[[t+10000 for t in seq] for seq in observed]
    other=recover_sda_transformer(path,renamed,device=device)
    assert recovered['predicted_sequences']==other['predicted_sequences']
    mapping_found={}
    for obs,pred in zip(observed,recovered['predicted_sequences']):
        for symbol,plain in zip(obs,pred):
            assert symbol not in mapping_found or mapping_found[symbol]==plain
            mapping_found[symbol]=plain
    assert len(set(mapping_found.values()))==len(mapping_found)
    accuracy=score(recovered,test)
    results['sda_control']=dict(**accuracy,train_sequences=len(train),validation_sequences=len(val),test_sequences=len(test),
       initial_train_loss=learned['history'][0]['train_loss'],final_train_loss=learned['history'][-1]['train_loss'],
       best_validation_loss=learned['best_validation_loss'],renaming_invariant=True,global_mapping_consistent=True)
    print('SDA_CONTROL',accuracy,flush=True)
    # TFMA：独立序列但严格频率排名相同的对照，以及没有唯一解的同频情况。
    reference=[[t for t,count in enumerate((90,60,40,25,12,3)) for _ in range(count)]]
    obs=[list(reversed([mapping[t] for t in reference[0]]))]
    prediction=recover_frequency_mapping(reference,obs)
    truth=[mapping.index(token) for token in prediction['observed_ids']]
    write(out/'predictions/tfma_control.json',prediction);write(out/'truth/tfma_control.json',truth)
    assert score(prediction,truth)['token_accuracy']==1.
    tied=recover_frequency_mapping([[0,1,2,3]],[[mapping[0],mapping[1],mapping[2],mapping[3]]])
    assert tied['reference_has_frequency_ties'] and tied['observed_has_frequency_ties']
    unknown=recover_frequency_mapping([[0,0]],[[3,4]],topk=3)
    assert -1 in unknown['predicted_plain_ids']
    results['tfma_control']=dict(**score(prediction,truth),ties_reported=True,unknown_preserved=True,
        control_scope='Known-count functional control; not an independent natural-text privacy success rate.')
    # 真实自然文本：公开SST-2不相交子集，真实Qwen分词、同一随机词表替换。
    parquet=ROOT/'outputs/round3_pretrained/sst2_validation.parquet'
    rows=pq.read_table(parquet).to_pylist()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    tokenized=[];seen=set()
    for row in rows:
        ids=tokenizer.encode(row['sentence'],add_special_tokens=False)[:12]
        if len(ids)>=4 and tuple(ids) not in seen:seen.add(tuple(ids));tokenized.append(ids)
    train,val,test=tokenized[:128],tokenized[128:160],tokenized[160:192]
    assert len(test)==32
    permutation=torch.randperm(tokenizer.vocab_size,generator=torch.Generator().manual_seed(SEED+23))
    observed=[permutation[torch.tensor(seq)].tolist() for seq in test]
    for label,value in [('train',train),('validation',val),('observed',observed)]:write(out/f'inputs/sda_natural_{label}.json',value)
    write(out/'truth/sda_natural.json',test)
    write(out/'inputs/sst2_subset_manifest.json',dict(parquet_sha256=sha(parquet),selection='First unique tokenized rows, at least 4 tokens, truncate to 12; 128 train/32 validation/32 test; mutually disjoint token sequences',train_size=128,validation_size=32,test_size=32))
    path=out/'checkpoints/sda_natural.pt'
    learned=train_or_load_sda(train,val,vocab_size=tokenizer.vocab_size,model_path=path,epochs=3,batch_size=4,hidden=32,device=device,seed=SEED)
    recovered=recover_sda_transformer(path,observed,device=device)
    write(out/'predictions/sda_natural.json',recovered)
    results['sda_natural']=dict(**score(recovered,test),train_sequences=128,validation_sequences=32,test_sequences=32,
         initial_train_loss=learned['history'][0]['train_loss'],final_train_loss=learned['history'][-1]['train_loss'],best_validation_loss=learned['best_validation_loss'],
         scope='Full Qwen tokenizer vocabulary; small three-epoch natural-text run, not a paper-scale trained attack.')
    tfma=recover_frequency_mapping(train,observed)
    inv=torch.empty_like(permutation);inv[permutation]=torch.arange(len(permutation))
    tfma_truth=inv[torch.tensor(tfma['observed_ids'])].tolist()
    write(out/'predictions/tfma_natural.json',tfma);write(out/'truth/tfma_natural.json',tfma_truth)
    recovered_map=dict(zip(tfma['observed_ids'],tfma['predicted_plain_ids']))
    occurrence={'predicted_sequences':[[recovered_map[t] for t in seq] for seq in observed]}
    results['tfma_natural']=dict(type_score=score(tfma,tfma_truth),occurrence_score=score(occurrence,test),
         observed_ties=tfma['observed_has_frequency_ties'],reference_ties=tfma['reference_has_frequency_ties'],
         unknown_count=sum(t==-1 for seq in occurrence['predicted_sequences'] for t in seq))
    cli=out/'predictions/sda_control_cli.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluation/privacy/run_additional_attacks.py'),'sda-predict',
       '--model-path',str(out/'checkpoints/sda_control.pt'),'--observed-json',str(out/'inputs/sda_control_observed.json'),
       '--device',device,'--output',str(cli)],cwd=ROOT,check=True)
    assert read(cli)['predicted_sequences']==read(out/'predictions/sda_control.json')['predicted_sequences']
    results['sda_cli_matches_api']=True
    return results


def ima_predict(model,observations,public_embed,*,topk=10,batch_size=8,standardization=None):
    """只用观测、已训练网络和完整公开词表预测；没有测试答案参数。"""
    from src.security_qwen.ima import _rank_candidate_plain_ids
    from src.security_qwen.ima import normalize_ima_tensor,restore_ima_predictions
    device=next(model.parameters()).device;predictions=[];vectors=[]
    candidates=torch.arange(len(public_embed),device=public_embed.device)
    with torch.inference_mode():
        for start in range(0,len(observations),batch_size):
            inputs=observations[start:start+batch_size].to(device)
            if standardization is not None:inputs=normalize_ima_tensor(inputs,standardization)
            vector=model(inputs)
            if standardization is not None:vector=restore_ima_predictions(vector,standardization)
            vectors.append(vector.cpu())
            ranked=_rank_candidate_plain_ids(predicted_embeddings=vector.reshape(-1,vector.shape[-1]),
                candidate_plain_ids=candidates,baseline_embed=public_embed,topk=topk)
            predictions.append(ranked.reshape(*vector.shape[:2],-1).cpu())
    ranked=torch.cat(predictions);vector=torch.cat(vectors)
    return dict(attack='ima_paper_like_network_control',predicted_sequences=ranked[:,:,0].tolist(),
        topk_plain_ids=ranked.tolist(),candidate_count=len(public_embed),
        prediction_uses_test_truth=False,training_requires_known_pairs=True),vector


def ima(args):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader,TensorDataset
    from src.security_qwen.ima import build_paper_like_inverter_config,_PaperLikeIMAInverter
    from src.security_qwen.ima import fit_ima_standardization,normalize_ima_tensor
    out=args.out;device=args.device;torch.manual_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    public=get_weight(args.model,'model.embed_tokens.weight')[:tokenizer.vocab_size]
    candidates=public.to(device)
    pool=torch.randperm(tokenizer.vocab_size,generator=torch.Generator().manual_seed(SEED+31))[:64]
    seen=pool[:32];unseen=pool[32:]
    g=torch.Generator().manual_seed(SEED+32)
    all_sequences=set()
    while len(all_sequences)<120:all_sequences.add(tuple(seen[torch.randint(32,(8,),generator=g)].tolist()))
    values=[list(row) for row in sorted(all_sequences)]
    order=torch.randperm(120,generator=g).tolist();values=[values[i] for i in order]
    train_ids=torch.tensor(values[:96]);val_ids=torch.tensor(values[96:104]);test_ids=torch.tensor(values[104:])
    unseen_ids=unseen[torch.randint(32,(16,8),generator=g)]
    for name,ids in [('train',train_ids),('validation',val_ids),('test',test_ids),('unseen',unseen_ids)]:
        write(out/f'truth/ima_{name}.json',ids.tolist())
    write(out/'inputs/ima_protocol.json',dict(train_sequences=96,validation_sequences=8,test_sequences=16,sequence_length=8,
        training_token_types=32,unseen_token_types=32,full_candidate_count=len(public),
        sequence_sets_disjoint=True,main_test_reuses_training_token_types_in_new_sequences=True,
        assumption='Known plaintext/observation training pairs under target key. Main test checks learning known words in unseen sequences; separate unseen-vocabulary test is reported.'))
    results=[]
    for variant in ('identity','expanded','identity_shuffled_labels'):
        torch.manual_seed(SEED)
        if variant=='expanded':
            folder=ROOT/'artifacts/round3_pretrained/expanded_zero_noise_sharded'
            secret=torch.load(folder/'client/client_secret.pt',weights_only=True,map_location='cpu')
            rows=secret['perm_vocab'][pool]
            observed_pool=get_weight(folder/'server','model.embed_tokens.weight',rows)
        else:observed_pool=public[pool].clone()
        lookup={int(token):i for i,token in enumerate(pool)}
        observe=lambda ids:observed_pool[torch.tensor([[lookup[int(t)] for t in row] for row in ids])]
        xtrain,xval,xtest,xunseen=map(observe,(train_ids,val_ids,test_ids,unseen_ids))
        training_labels=public[train_ids]
        validation_labels=public[val_ids]
        validation_ids=val_ids
        if variant=='identity_shuffled_labels':
            wrong_map={int(t):int(seen[(i+1)%len(seen)]) for i,t in enumerate(seen)}
            wrong=lambda ids:torch.tensor([[wrong_map[int(t)] for t in row] for row in ids])
            training_labels=public[wrong(train_ids)];validation_ids=wrong(val_ids);validation_labels=public[validation_ids]
        standardization=fit_ima_standardization(xtrain,training_labels) if args.ima_standardization=='train_only' else None
        config=build_paper_like_inverter_config(observed_hidden_size=observed_pool.shape[1],vocab_size=64,baseline_model_dir=str(args.model))
        model=_PaperLikeIMAInverter(backbone_config=config,target_embedding_dim=public.shape[1]).to(device)
        # 此网络使用inputs_embeds；64行未使用的查表不参与计算，输出候选仍为完整151643词表。
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=0.)
        tx=normalize_ima_tensor(xtrain,standardization) if standardization is not None else xtrain
        ty=normalize_ima_tensor(training_labels,standardization,target=True) if standardization is not None else training_labels
        loader=DataLoader(TensorDataset(tx,ty),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(SEED))
        best_score=-1.;best_loss=float('inf');best_state=None;history=[]
        model.eval()
        with torch.no_grad():initial_loss=float((model(tx[:8].to(device))-ty[:8].to(device)).square().mean())
        for epoch in range(20):
            model.train();losses=[]
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=(model(x.to(device))-y.to(device)).square().mean()
                if not torch.isfinite(loss):raise ValueError('Non-finite IMA loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            prediction,vector=ima_predict(model,xval,candidates,standardization=standardization)
            val_score=score(prediction,validation_ids.tolist())['token_accuracy']
            val_loss=float((vector-validation_labels).square().mean())
            history.append(dict(epoch=epoch+1,train_loss=sum(losses)/len(losses),validation_top1=val_score,validation_mse=val_loss))
            if (val_score,-val_loss)>(best_score,-best_loss):
                best_score,best_loss=val_score,val_loss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if (epoch+1)%5==0:print('IMA',variant,epoch+1,history[-1],flush=True)
        model.load_state_dict(best_state);model.eval()
        prediction,_=ima_predict(model,xtest,candidates,standardization=standardization)
        unseen_prediction,_=ima_predict(model,xunseen,candidates,standardization=standardization)
        suffix='_standardized' if standardization is not None else ''
        write(out/f'predictions/ima_{variant}{suffix}.json',prediction)
        write(out/f'predictions/ima_{variant}{suffix}_unseen.json',unseen_prediction)
        path=out/f'checkpoints/ima_{variant}{suffix}.pt'
        torch.save(dict(config=config.to_dict(),target_dim=public.shape[1],state=best_state,history=history,standardization=standardization),path)
        # 重新创建网络加载保存结果；不把测试答案用于选轮次。
        clone=_PaperLikeIMAInverter(backbone_config=config,target_embedding_dim=public.shape[1]).to(device)
        clone.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state']);clone.eval()
        saved=torch.load(path,weights_only=True,map_location='cpu')
        reload_prediction,_=ima_predict(clone,xtest[:2],candidates,standardization=saved['standardization'])
        assert reload_prediction['predicted_sequences']==prediction['predicted_sequences'][:2]
        main_score=score(prediction,test_ids.tolist());new_score=score(unseen_prediction,unseen_ids.tolist())
        top10=sum(t in r for row,truthrow in zip(prediction['topk_plain_ids'],test_ids.tolist()) for r,t in zip(row,truthrow))/test_ids.numel()
        results.append(dict(variant=variant,**main_score,top10_accuracy=top10,unseen_token_score=new_score,initial_mse=initial_loss,
            final_train_loss=history[-1]['train_loss'],best_validation_top1=best_score,save_reload_identical=True,
            observed_width=observed_pool.shape[1],network_width=config.hidden_size,output_width=public.shape[1],
            epochs=20,history=history,standardization=args.ima_standardization))
        print('IMA_RESULT',variant,main_score,flush=True)
        del model,clone,optimizer,best_state;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return dict(cases=results,protocol=read(out/'inputs/ima_protocol.json'),
        scope='Existing two-layer/eight-head inverter and production candidate-ranking module; controlled known-pair training; legacy target-registry CLI is not used.')


def model_digest(model):
    """逐张量计算真实参数摘要，检查ISA只修改输入、没有改模型参数。"""
    digest=hashlib.sha256()
    for name,p in model.state_dict().items():
        digest.update(name.encode());value=p.detach().cpu().contiguous()
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def isa(args):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from src.security_qwen.isa_optimization import run_isa_optimization,observation_features
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
         dtype=torch.float32,attn_implementation='eager').to(args.device).eval()
    texts=['The cat is sleeping on the bed.','I like reading books in the evening.']
    ids=torch.tensor([tokenizer.encode(t,add_special_tokens=False)[:4] for t in texts],device=args.device)
    with torch.no_grad():
        state=model(ids,output_hidden_states=True,use_cache=False).hidden_states
        target0,target1=state[0].clone(),state[1].clone()
    del state
    write(args.out/'truth/isa.json',ids.cpu().tolist())
    before=model_digest(model);flags=[p.requires_grad for p in model.parameters()]
    candidates=list(range(tokenizer.vocab_size));results=[]
    for layer,steps,rate in [(0,160,.01),(1,120,.003)]:
        target=target0 if layer==0 else target1
        prediction=run_isa_optimization(model,target,candidate_ids=candidates,layer=layer,
             initial_embeddings=torch.zeros_like(target0),steps=steps,learning_rate=rate,topk=10)
        write(args.out/f'predictions/isa_layer{layer}.json',prediction)
        results.append(dict(case=f'layer_{layer}_same_coordinates',**score(prediction,ids.cpu().tolist()),
            initial_objective=prediction['history'][0]['objective'],best_objective=prediction['best_objective'],steps=steps))
        print('ISA_RESULT',results[-1],flush=True)
    torch.manual_seed(SEED)
    rotation=torch.linalg.qr(torch.randn(904,896,device=args.device)).Q.T
    rotated=target1@rotation
    gram_error=float((observation_features(target1,'cosine_gram')-observation_features(rotated,'cosine_gram')).abs().max())
    prediction=run_isa_optimization(model,rotated,candidate_ids=candidates,layer=1,
        initial_embeddings=torch.zeros_like(target0),steps=60,learning_rate=.003,topk=10,observation_mode='cosine_gram')
    write(args.out/'predictions/isa_rotated_gram.json',prediction)
    results.append(dict(case='unknown_isometric_expansion_gram',**score(prediction,ids.cpu().tolist()),
        initial_objective=prediction['history'][0]['objective'],best_objective=prediction['best_objective'],steps=60,gram_invariance_max_error=gram_error,
        scope_note='Eight positions provide few pairwise constraints; low loss need not identify tokens. Not a general Algorithm-1 transform.'))
    after=model_digest(model)
    assert before==after and flags==[p.requires_grad for p in model.parameters()] and not model.training
    return dict(cases=results,model_parameters_unchanged=True,model_sha256_before=before,model_sha256_after=after,
                scope='Full pretrained 24-layer model forward; layer0/layer1 observations; zero initialization, full public tokenizer candidates.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','weights','sequences','ima','isa'])
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ima-standardization',choices=['none','train_only'],default='train_only')
    args=parser.parse_args();args.model=args.model.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    start=time.perf_counter()
    try:
        result=globals()[args.stage](args)
        result_name='ima_raw' if args.stage=='ima' and args.ima_standardization=='none' else args.stage
        write(args.out/f'{result_name}_results.json',dict(status='complete',seconds=time.perf_counter()-start,result=result))
    except Exception as exc:
        write(args.out/f'{args.stage}_error.json',dict(status='failed',seconds=time.perf_counter()-start,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
    print('DONE',args.stage,round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':main()
