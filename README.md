# SegMADe
A Supervised Machine Learning Semantic Segmentation Approach forDetecting Motion Artifacts in Plethysmography Signals from Wearables

# SegMADe Pipeline
![overall_pipeline](https://github.com/chengstark/SegMADe/raw/main/readme_images/overall_pipeline.png)
Step (1): 30-second  signal  segments  from  PPG  DaLiA  were  split  into  train  and  test  set  by randomly chosen subject IDs.\
Step (2): Datasets were uploaded into the [web annotation tool](https://github.com/chengstark/SegMADe-Annotation-Tool) for human annotation. \
Steps (3): and (4): The tool transcribes human annotations into binary segmentation label masks (ground truth) and 30-second signal segments were stored into the back-end database.\
Step (5): Ground truth and 30-second signal segments were then used to train the model.\
Step (6): Predictions by the trained model are made on the PPG DaliA test set, WESAD, TROIKA, and UCSF datasets.\
Step (7): The trained model’s predictions are provided for evaluation against human-annotated ground truth.\

# Model architecture
![model](https://github.com/chengstark/SegMADe/raw/main/readme_images/model_U_plot%20v3.png)

# Results
![results](https://github.com/chengstark/SegMADe/raw/main/readme_images/results.png)

# Results Visualizations
![vis_1](https://github.com/chengstark/SegMADe/raw/main/readme_images/DaLiA_38.jpg)
![vis_2](https://github.com/chengstark/SegMADe/raw/main/readme_images/DaLia_715.jpg)
![vis_3](https://github.com/chengstark/SegMADe/raw/main/readme_images/TROIKA_6.jpg)
![vis_4](https://github.com/chengstark/SegMADe/raw/main/readme_images/WESAD_2880.jpg)

# Installation and Results Reproduction
<pre>
Use environment_no_builds.yml if not on Linux
	*Packages might be different on different OS, if conflicts exists, please delete the confilct packages from yml file, since some packages are required only on Ubuntu/ Linux

Place test set in data/ folder\
	* 	test set folder structure should look like this:
		TESTSETNAME
			|_ processed_dataset
				|_ scaled_ppgs.npy
				|_ seg_labels.npy
		 
		scaled_ppgs.npy has shape (n*1920)
		seg_labels.npy has shape (n*1920), consists of only 1 and 0 integers
		* files must be named scaled_ppgs.npy and seg_labels.npy

cd to model folders (cnn_slider, pulse_template_matching, resnet34, proposed)

Run evaluation with command: python run_evaluations.py TESTSETNAME plot_limiter
	- TESTSETNAME is the same name of the test set folder you placed in the parent data/ folder
	- plot_limiter is an integer, how many plots you want to generate for sampling purpose
	* generate plots will increase evaluation time dramatically, recommend 0 if evaluation is the priority
</pre>