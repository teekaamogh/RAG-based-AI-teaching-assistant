import os
import whisper
import json

model = whisper.load_model("large-v3")

audios = os.listdir("audio")

for audio in audios:
  number = audio.split("_")[0]
  title = audio.split("_")[1][:-4]
  print(number, title)
  result = model.transcribe(audio=f"audio/{audio}",  
                            language="hi",
                            task="translate",
                            word_timestamps=False)

  chunks=[]
  for segemet in result["segments"]:
    chunks.append({
      'number': number,
      'title': title,
      'id': segemet['id'],
      'start': segemet['start'],
      'end': segemet['end'],
      'text': segemet['text']
    })

  chunks_with_metadata = {'chunks': chunks, 'text': result['text']}

  with open(f"jsons/{audio}.json", "w") as f:
    json.dump(chunks_with_metadata, f)

