import joblib
import requests
import numpy as np
import pandas as pd
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from config import key

# Function to convert text to embedding 
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
    embedding = r.json()["embeddings"]
    return embedding

# Convert user_query to vectors
incoming_query = input("Ask a question:")
question_embedding = create_embedding([incoming_query])[0]

# Load the vector data to compare with user_query
df = joblib.load('joblib/embeddings.joblib')

# Find similarities of question embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 5
max_indices = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indices]

# Prompt which is given to the LLM
prompt = f'''
I am teaching webdevelopment using sigma webdevelopment course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient="records")}

--------------------------------

"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format it's just for you) where and how much content is taught in which video (in the video and at what timestamp) and guide the user to go to that particular video.
If user asks unrelated question, tell him that you can answer only qustions related to the course
Note: If a user asks any question like "what is HTML" or "what is CSS" or ... just provide the answer instead of giving
timestamp and in which video it was taught
'''


# Gemini API and model for getting response
client = genai.Client(api_key=key)
interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=prompt
)
response = interaction.output_text

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)