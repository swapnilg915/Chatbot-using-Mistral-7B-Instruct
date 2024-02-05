This repo implements chatbot using state of the art open source LLM model Mistral 7B instruct-v0.1
This is a multimodal chatbot which can acccept the input in 2 forms: Text and voice
2 LLM models are used here:
	1. Mistral 7B instruct-v0.1
	2. Whisper base (Automatic Speech recognition)

### Steps to run training:
	
	1. Open the notebook from the notebooks folder "Finetuning_Mistral_7b_Using_AutoTrain_EN.ipynb" in any platform like Google colab or Azure ML studio or Data Science VM.

	2. Run every cell sequentially till the end.
	3. Upload the training data CSV file from datasets folder.


Training Algorithm

1. Install the necessary packages required for training
2. Download the dataset. Convert it into a required format by our LLM Mistral 7B instruct. Save in the current working directory.
3. Run the training command using huggingface autotrain with other necessary params.

	!autotrain llm --train --project-name mistral-7b-mj-finetuned --model filipealmeida/Mistral-7B-Instruct-v0.1-sharded --data-path . --use-peft --quantization int4 --lr 2e-4 --batch-size 12 --epochs 3 --trainer sft --target_modules q_proj,v_proj --push-to-hub --token "your Huggingface write token" --repo-id "your huggingface repo name"/"model name you are going to push"

	Here we are using 
		"Mistral-7B-Instruct-v0.1-sharded" as a baseline LLM
		PEFT for finetuning
		int4 quantization
		SFT trainer



### Steps for Inference:

1. Open notebook named "Inference_with_Gradio_text_audio_RAG.ipynb" from notebooks directory 
2. Run all the cells
3. Run appropriate cells for launching Gradio interface for chat and audio (voice).

	


References:

1. chatbot app using streamlit and mistral 7b quantized (only inference)

github - https://github.com/lalanikarim/ai-chatbot?tab=readme-ov-file
quantized - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ

2. fietuning mistral 7b on Tesla V100 32GB GPU
https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8


3. Gradio chat app 

	https://blog.gopenai.com/building-chatbot-streamlit-app-using-mistral-7b-8ba78fac9ee5

4. ASR using Gradio and Whisper

	https://colab.research.google.com/drive/1Mperx1lLC0KrT4Br6Bbfigdjn9WF3_AR#scrollTo=oWWMdu-68rSZ

5. Prompt optimization

	https://medium.com/@averma9838/prompt-optimization-for-llm-b1bd0dede95f


6. Original google colab for finetuning mistral 7B instruct using autotrain
	
	https://colab.research.google.com/drive/1i-gU8HFIiH1hCf_taRHpz7beX_A50RtR?usp=sharing#scrollTo=pRZudTTXs8S8

7. Automatic Speech Recognition (ASR) model

	https://huggingface.co/openai/whisper-base.en