# Multi-TAP


## Main Process
### Step 0. User specific information Pre-generation
profiles/user_info_gen.py Needs following:

Amazon review dataset

Amazon item meta dataset

https://amazon-reviews-2023.github.io/

we also provided sampled review and item meta data ./data/amazon/

### Step 1. Domain Description Generation
should use Openai API Key
python step1_domain_desc_gen.py

### Step 2. Persona Sentence Generation
should use Openai API Key

python step2_persona_gen.py \
--review_dir ./data/amazon/{domain_pair}/filtered_data/f_usr_reviews \
--meta_dir ./data/amazon/{domain_pair}/filtered_data/f_item_meta \
--domain_desc_dir ./profiles/domain_desc/ \
--user_info_dir ./profiles/user_info \
--system ./system_prompt/category_persona_gen.txt \
--tvt_root ./data/amazon/{domain_pair} \
--out_dir ./profiles/persona_sentences/

### Step 3. text embedding
Object is following:
User Persona Sentences
text2emb/t2e_persona.py
Item meta data (Domain Description || Domain Description Keywords || Item's specific category)
text2emb/t2e_itm.py

## Implementation

### Structure of Alt_TAP recommendation system

```
Multi-TAP/
├── Multi_TAP_main.py
├── step1_domain_desc_gen.py
├── step2_persona_gen.py
├── config.py ## <- needs API KEY
└── model/
    └── Multi_TAP/
        └── trainer/
            ├── Multi_TAP.py
            └── trainer.py
```

### Backbone Model
LightGCN (SIGIR, 2020) from opensource repository: https://github.com/Coder-Yu/QRec
