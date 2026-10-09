# How to use this RAG based AI-teaching assistant
## Step-1
Add all videos to the 'videos' folder

## Step-2
Run 'Video_To_Audio' to convert all videos to audio. Then all audio files will be stored to 'audio' folder

## Step-3
Run 'Audio_To_Text' to covert all audio to text and stores the chunks in jsons folder
NOTE:
This process is done locally on your system, so you need to have a good system and whisper installed on it 

## Step-4
Run 'Text_To_Vectors' to convert all text data stored in 'json' folder to vectors and dump in 'joblib' folder
NOTE:
This process also takes place locally on your system, so have ollama bge-3 model installed on your system

## Step-4
Run 'UserQuery_And_LLM' to type your query in the terminal and get response in 'response.txt' file
