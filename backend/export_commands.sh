python backend/gallery_backend.py data/datasets/ISIC2017 --export_all_dir frontend/public/datasets/ISIC2017 --methods "" --method pca --default_method color_rgb
python backend/gallery_backend.py data/datasets/VIS30KGUI --export_all_dir frontend/public/datasets/VIS30KGUI --methods "" --method pca --default_method color_rgb
python backend/gallery_backend.py data/datasets/EmoSet --export_all_dir frontend/public/datasets/EmoSet --methods "" --method pca --default_method color_rgb


color_rgb,color_hsv,color_lch,clip,dino,dift_sd_part00,dift_sd_part01,dift_sd_part02,dift_sd_part10,dift_sd_part11,dift_sd_part12,dift_sd_part20,dift_sd_part21,dift_sd_part22


python backend/export_metadata.py data/datasets/ISIC2017 frontend/public/datasets/ISIC2017
python backend/export_metadata.py data/datasets/VIS30KGUI frontend/public/datasets/VIS30KGUI
python backend/export_metadata.py data/datasets/EmoSet frontend/public/datasets/EmoSet