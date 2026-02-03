dataset=NIPS
model=cof.ckpt

factor=semantic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

dataset=KDD
model=cof.ckpt

factor=semantic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

dataset=SciRepEval
model=cof.ckpt

factor=semantic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

dataset=stelmakh
model=cof.ckpt

factor=semantic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

dataset=wiz1000
model=cof.ckpt

factor=semantic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=topic
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

factor=citation
python get_paper_emb.py --dataset ${dataset} --model ${model} --factor ${factor}

# python3.8 chain_of_factors.py --dataset ${dataset}
