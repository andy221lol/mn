from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
tkn = os.getenv("HFTKN")
# Initialize Hugging Face Inference client
client = InferenceClient(api_key=tkn)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    conversation = request.json.get('conversation', [])

    # Add user message to conversation
    conversation.append({"role": "user", "content": user_message})

    # Call Hugging Face API for response
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct", 
        messages=conversation,
        temperature=0.5,
        max_tokens=2048,
        top_p=0.7
    )

    ai_response = completion.choices[0].message['content']

    # Add AI response to conversation
    conversation.append({"role": "assistant", "content": ai_response})

    return jsonify({"response": ai_response, "conversation": conversation})

if __name__ == '__main__':
    app.run(debug=True)
