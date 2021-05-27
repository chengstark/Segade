# SegMADe
A Supervised Machine Learning Semantic Segmentation Approach forDetecting Motion Artifacts in Plethysmography Signals from Wearables

# SegMADe Pipeline
![overall_pipeline](https://github.com/chengstark/SegMADe/raw/main/readme_images/overall_pipeline.png)
Step (1): 30-second  signal  segments  from  PPG  DaLiA  were  split  into  train  and  test  set  by randomly chosen subject IDs.\
Step (2): Datasets were uploaded into the web annotation tool for human annotation.\
Steps (3): and (4): The tool transcribes human annotations into binary segmentation label masks (ground truth) and 30-second signal segments were stored into the back-end database.\
Step (5): Ground truth and 30-second signal segments were then used to train the model.\
Step (6): Predictions by the trained model are made on the PPG DaliA test set, WESAD, TROIKA, and UCSF datasets.\
Step (7): The trained model’s predictions are provided for evaluation against human-annotated ground truth.\

# Model
![model](https://github.com/chengstark/SegMADe/raw/main/readme_images/model_U_plot%20v3.png)