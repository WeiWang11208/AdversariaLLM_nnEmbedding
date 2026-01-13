  python run_attacks.py \
      model=Qwen/Qwen3-8B \
      dataset=refusal_direction, \
      attack=natural_suffix_embedding \
      attacks.natural_suffix_embedding.phase1_enabled=true \
      attacks.natural_suffix_embedding.phase2_num_steps=100 \
      save_dir=/mnt/public/share/users/wangwei/202512/AdversariaLLM-main/outputs/natural_suffix_embedding/ \
      embed_dir=/mnt/public/share/users/wangwei/202512/AdversariaLLM-main/outputs/natural_suffix_embedding_attack