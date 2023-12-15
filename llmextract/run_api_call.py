import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def main():
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content

if __name__ == '__main__':
    main()
