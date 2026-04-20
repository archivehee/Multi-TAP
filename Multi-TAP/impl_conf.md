cd Multi-TAP

# Home-Toys
## Toys->Home

python Multi_TAP_main.py \
  --dataset_root ./data/amazon/home_toys \
  --domain Home_and_Kitchen \
  --source_domain Toys_and_Games \
  --persona_pkl ./data/amazon/home_toys/txt_usr_emb/home_u_pers_emb.pkl \
  --source_persona_pkl ./data/amazon/home_toys/txt_usr_emb/toys_u_pers_emb.pkl \
  --lgn_user_pkl ./pre_tr_emb/home_toys/Home_and_Kitchen/user_emb.pkl \
  --source_lgn_user_pkl ./pre_tr_emb/home_toys/Toys_and_Games/user_emb.pkl \
  --map_dir ./data/amazon/home_toys/Home_and_Kitchen/maps \
  --source_map_dir ./data/amazon/home_toys/Toys_and_Games/maps \
  --lgn_item_pkl ./pre_tr_emb/home_toys/Home_and_Kitchen/item_emb.pkl \
  --txt_item_pkl ./data/amazon/home_toys/txt_itm_emb/home_itm_text_emb.pkl \
  --source_lgn_item_pkl ./pre_tr_emb/home_toys/Toys_and_Games/item_emb.pkl \
  --source_txt_item_pkl ./data/amazon/home_toys/txt_itm_emb/toys_itm_text_emb.pkl \
  --output_dir ./outputs


## Home->Toys
python Multi_TAP_main.py \
  --dataset_root ./data/amazon/home_toys \
  --domain Toys_and_Games \
  --source_domain Home_and_Kitchen \
  --persona_pkl ./data/amazon/home_toys/txt_usr_emb/home_u_pers_emb.pkl \
  --source_persona_pkl ./data/amazon/home_toys/txt_usr_emb/toys_u_pers_emb.pkl \
  --lgn_user_pkl ./pre_tr_emb/home_toys/Toys_and_Games/user_emb.pkl \
  --source_lgn_user_pkl ./pre_tr_emb/home_toys/Home_and_Kitchen/user_emb.pkl \
  --map_dir ./data/amazon/home_toys/Toys_and_Games/maps \
  --source_map_dir ./data/amazon/home_toys/Home_and_Kitchen/maps \
  --lgn_item_pkl ./pre_tr_emb/home_toys/Toys_and_Games/item_emb.pkl \
  --txt_item_pkl ./data/amazon/home_toys/txt_itm_emb/home_itm_text_emb.pkl \
  --source_lgn_item_pkl ./pre_tr_emb/home_toys/Home_and_Kitchen/item_emb.pkl \
  --source_txt_item_pkl ./data/amazon/home_toys/txt_itm_emb/toys_itm_text_emb.pkl \
  --output_dir ./outputs

# Sports-Cloth
## Cloth->Sports
python Multi_TAP_main.py \
  --dataset_root ./data/amazon/sports_cloth \
  --domain Sports_and_Outdoors \
  --source_domain Clothing_Shoes_and_Jewelry \
  --persona_pkl ./data/amazon/sports_cloth/txt_usr_emb/sports_u_pers_emb.pkl \
  --source_persona_pkl ./data/amazon/sports_cloth/txt_usr_emb/cloth_u_pers_emb.pkl \
  --lgn_user_pkl ./pre_tr_emb/sports_cloth/Sports_and_Outdoors/user_emb.pkl \
  --source_lgn_user_pkl ./pre_tr_emb/sports_cloth/Clothing_Shoes_and_Jewelry/user_emb.pkl \
  --map_dir ./data/amazon/sports_cloth/Sports_and_Outdoors/maps \
  --source_map_dir ./data/amazon/sports_cloth/Clothing_Shoes_and_Jewelry/maps \
  --lgn_item_pkl ./pre_tr_emb/sports_cloth/Sports_and_Outdoors/item_emb.pkl \
  --txt_item_pkl ./data/amazon/sports_cloth/txt_itm_emb/sports_itm_text_emb.pkl \
  --source_lgn_item_pkl ./pre_tr_emb/sports_cloth/Clothing_Shoes_and_Jewelry/item_emb.pkl \
  --source_txt_item_pkl ./data/amazon/sports_cloth/txt_itm_emb/cloth_itm_text_emb.pkl \
  --output_dir ./outputs

## Sports->Cloth
python Multi_TAP_main.py \
  --dataset_root ./data/amazon/sports_cloth \
  --domain Clothing_Shoes_and_Jewelry \
  --source_domain Sports_and_Outdoors \
  --persona_pkl ./data/amazon/sports_cloth/txt_usr_emb/cloth_u_pers_emb.pkl \
  --source_persona_pkl ./data/amazon/sports_cloth/txt_usr_emb/sports_u_pers_emb.pkl \
  --lgn_user_pkl ./pre_tr_emb/sports_cloth/Clothing_Shoes_and_Jewelry/user_emb.pkl \
  --source_lgn_user_pkl ./pre_tr_emb/sports_cloth/Sports_and_Outdoors/user_emb.pkl \
  --map_dir ./data/amazon/sports_cloth/Clothing_Shoes_and_Jewelry/maps \
  --source_map_dir ./data/amazon/sports_cloth/Sports_and_Outdoors/maps \
  --lgn_item_pkl ./pre_tr_emb/sports_cloth/Clothing_Shoes_and_Jewelry/item_emb.pkl \
  --txt_item_pkl ./data/amazon/sports_cloth/txt_itm_emb/cloth_itm_text_emb.pkl \
  --source_lgn_item_pkl ./pre_tr_emb/sports_cloth/Sports_and_Outdoors/item_emb.pkl \
  --source_txt_item_pkl ./data/amazon/sports_cloth/txt_itm_emb/sports_itm_text_emb.pkl \
  --output_dir ./outputs
