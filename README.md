## Instructions for the BiLSTM models
1. Working Directory: `./supervised/neural_methods/`
2. Place the saved models in their respective folders.
	path: \<parentDirectory>/model/\<yourModel> <br />
	Model Naming Convention : \<model_<dataSetName>_{Embedding name}> <br />
	For Example : <br /> `model_inspec_naive` <br />
					  `model_semeval2017_glove` <br />
					  `model_semeval2010_naive`
	
3. Run using `python3 training.py` from the respective folders. (Note: only use either naive or glove model's .pth file at one time) <br />
	For Example : python training.py \<datasetname>_\<embedding(glove/naive)> --> Provided as Command line argument <br />
	Usage       : `python training.py inspec_glove`
4. If pretrained model is not present it will start running the training.
5. Place the datasets in their respective folders.
	path: \<parentDirectory>/datasetPath/\<datasetName> <br />
	Dataset Naming Convention : \<datasetName>_\<dataset Type>.txt <br />
	For Example : <br /> `inspec_text.txt` <br />
					  `inspec_train.txt` <br />
					  `semeval2010_test.txt`			
		

## Instructions for running pre-trained model files
1. Go to the folder `./supervised/pretrained_models/`
2. Just run using `python3 training.py`
3. Now provide the model name as an input:
   1. To run kbir model, provide the input : `kbir`
   2. To run keybart model, provide the input : `keybart`
   3. To run T5 model, provide the input : `t5`
   4. To run distilbert model, provide the input : `distilbert`
4. After running the Python files, the user will get the classification report for the following pretrained model on the Inspec dataset as an output.


## Instructions for the Unsupervised models
1. Go to the folder `./unsupervised/`
2. Run the file using following command: <br />
   Naming Convention: `python main.py {dataset_input_text_file}>` --> Provided as Command line argument <br />
   For Example : `python3 main.py doc.txt`

## Links to the python notbooks
1. Pretrained Notebooks: https://drive.google.com/drive/folders/1VPHrvSYcWnYpybpaWxY69lkUhii62Xyw?usp=sharing
2. BiLSTM Notebooks    : https://drive.google.com/drive/folders/1GygZWJMXW-sI1wyNIjFUE1G4VjaiWQde?usp=sharing