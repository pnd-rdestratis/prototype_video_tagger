
import json
import whisper
from openai import OpenAI
import tiktoken

# Initialize the OpenAI client with your API key
client = OpenAI()


def transcribe_video(video_path):
    """
    Transcribes the video at the given path using Whisper.
    """
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]


def generate_tags(transcript,tags_to_return):
    """
    Generates journalist tags from a given transcript using OpenAI.
    """
    # Initialize the tokenizer
    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(transcript)
    total_tokens = len(tokens)
    max_tokens = 100000
    num_chunks = (total_tokens + max_tokens - 1) // max_tokens  # Calculate the number of chunks

    # Split the transcript into chunks
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    tags = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_text = enc.decode(chunk)

        # Adjust the system message to include the number of chunks and tag distribution
        system_message = f"""
        You are a highly skilled language model trained in journalistic analysis. Your task is to examine the provided 
        transcript and generate tags that are highly relevant to the main topics, regions, or people discussed. 
        - Each tag should be no more than two words. 
        - Additionally, each tag should come with a relevance score between 0 and 1, where 1 indicates maximum relevance. 
        - The tags should encapsulate big-picture topics and significant elements mentioned in the transcript.
        - You will get a total of {num_chunks} chunks from the transcript. Please don't provide more than {tags_to_return} tags in total
        - If you get {num_chunks} chunks, you will provide {tags_to_return / num_chunks} tags per chunk. 
        - Ensure each tag is concise, relevant, and no longer than two words
        - You are part of a function that is called multiple times, that's why you don't need to provide all the tags at once unless it is only 1 chunk.
        - The tags should reflect key themes, regions, or notable individuals mentioned in the text.
        - You need to provide it in German. Provide only relevant Words!
        Example Response:
        {{
          "tags": [
            {{
              "tag": "Demokratie",
              "relevance": 0.95
            }},
            {{
              "tag": "Amerika",
              "relevance": 0.88
            }},
            {{
              "tag": "Klimawandel",
              "relevance": 0.90
            }},
            {{
              "tag": "Angela Merkel",
              "relevance": 0.85
            }}
          ]
        }}
        Only return JSON
        """

        chat_completion = client.chat.completions.create(
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": chunk_text}
            ],
            model="gpt-4o"
        )

        # Extract the JSON content correctly
        response_content = chat_completion.choices[0].message.content
        chunk_tags = json.loads(response_content)['tags']

        # Extend the tags list with the tags from the current chunk
        tags.extend(chunk_tags)
    return {"tags": tags}



def main(video_path,tags_to_generate):
    transcript = transcribe_video(video_path)
    tags = generate_tags(transcript,tags_to_generate)
    print(json.dumps(tags, indent=2, ensure_ascii=False))
    tags_json = json.dumps(tags, indent=2, ensure_ascii=False)
    return tags_json





