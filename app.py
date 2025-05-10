from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# GitHub API configuration
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1-nano"
TOKEN = os.environ.get("GITHUB_TOKEN")

# Initialize the client
client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# Store conversation history for AI partner chats
ai_partner_conversations = {}

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/google-login', methods=['POST'])
def google_login():
    try:
        # Get the ID token from the frontend request
        id_token_str = request.json.get('id_token')

        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(id_token_str, requests.Request(), "981383295462-lp0h6euuofpp1ts3j49mkmd139qftmgk.apps.googleusercontent.com")

        # Now you can extract user info from the idinfo dictionary
        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']
        picture = idinfo['picture']

        # Store user info or create a new user in the database
        # You can create a session, store in a database, or whatever you prefer.

        return jsonify({"user_id": user_id, "email": email, "name": name, "picture": picture})

    except ValueError:
        # Invalid token
        return jsonify({"error": "Invalid token"}), 400

@app.route('/api/generate', methods=['POST'])
def generate_responses():
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    message = data.get('message', '')
    tone = data.get('tone', 'sweet')
    level = data.get('level', 'low-key')
    mode = data.get('mode', 'reply')
    character = data.get('character', '')
    
    # Construct the system prompt
    system_prompt = construct_system_prompt(tone, level, mode, character)
    
    # Construct the user prompt
    user_prompt = construct_user_prompt(message, mode, character)
    
    try:
        # Call the GitHub model API
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ],
            temperature=0.7,
            top_p=1.0,
            model=MODEL
        )
        
        # Extract and process the response
        ai_response = response.choices[0].message.content
        
        # Parse the response into separate messages
        responses = parse_responses(ai_response)
        
        return jsonify({'responses': responses})
    
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return jsonify({'error': 'Failed to generate responses'}), 500

@app.route('/api/chat-partner', methods=['POST'])
def chat_with_partner():
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    partner_type = data.get('partner_type', 'boyfriend')
    personality = data.get('personality', 'sweet')
    
    # Initialize or retrieve conversation history
    if session_id not in ai_partner_conversations:
        ai_partner_conversations[session_id] = []
    
    # Add user message to history
    ai_partner_conversations[session_id].append({"role": "user", "content": message})
    
    # Prepare messages for API call
    messages = [
        SystemMessage(construct_partner_system_prompt(partner_type, personality))
    ]
    
    # Add conversation history (limit to last 10 messages to avoid token limits)
    history = ai_partner_conversations[session_id][-10:]
    for msg in history:
        if msg["role"] == "user":
            messages.append(UserMessage(msg["content"]))
        else:
            messages.append(SystemMessage(f"AI response: {msg['content']}"))
    
    try:
        # Call the GitHub model API
        response = client.complete(
            messages=messages,
            temperature=0.8,  # Slightly higher temperature for more varied responses
            top_p=1.0,
            model=MODEL
        )
        
        # Extract the response
        ai_response = response.choices[0].message.content
        
        # Clean up the response
        ai_response = clean_partner_response(ai_response)
        
        # Add AI response to history
        ai_partner_conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        return jsonify({'error': 'Failed to generate response'}), 500

def construct_system_prompt(tone, level, mode, character=''):
    """Construct a system prompt based on the selected options"""
    
    system_prompt = "You are Rizz Coach, an AI assistant that helps people craft flirty, witty, and engaging messages for dating and social situations. "
    
    # Add tone instruction
    if tone == "sweet":
        system_prompt += "Your responses should be warm, gentle, and sweet. "
    elif tone == "sarcastic":
        system_prompt += "Your responses should be witty and have a playful sarcastic edge. "
    elif tone == "bold":
        system_prompt += "Your responses should be confident, direct, and bold. "
    elif tone == "funny":
        system_prompt += "Your responses should be humorous and make the other person laugh. "
    elif tone == "smooth":
        system_prompt += "Your responses should be extremely smooth, charming, and showcase the highest level of rizz. "
    
    # Add level instruction
    if level == "low-key":
        system_prompt += "Show a casual, low-key interest level. "
    elif level == "confident":
        system_prompt += "Show confident pursuit and clear interest. "
    elif level == "roast":
        system_prompt += "Include a subtle, playful roast or tease. "
    elif level == "romantic":
        system_prompt += "Be romantic, emotional, and heartfelt. "
    
    # Add mode-specific instructions
    if mode == "reply":
        system_prompt += "Generate 3 different reply options to the message they received. "
    elif mode == "opener":
        system_prompt += "Generate 3 different opener lines for the given situation. "
    elif mode == "impression":
        if character:
            system_prompt += f"Generate 3 different responses in the style of {character}. "
        else:
            system_prompt += "Generate 3 different responses in the style of a smooth talker. "
    
    system_prompt += "Format your response as a numbered list with 3 options. Each option should be on a new line starting with '1. ', '2. ', or '3. '."
    
    return system_prompt

def construct_partner_system_prompt(partner_type, personality):
    """Construct a system prompt for the AI partner chat"""
    
    base_prompt = f"You are acting as the user's {partner_type}. "
    
    if partner_type == "boyfriend":
        base_prompt += "You identify as male and are romantically interested in the user. "
    elif partner_type == "girlfriend":
        base_prompt += "You identify as female and are romantically interested in the user. "
    else:
        base_prompt += "You are romantically interested in the user. "
    
    # Add personality traits
    if personality == "sweet":
        base_prompt += "Your personality is warm, caring, and affectionate. You're supportive, kind, and often use terms of endearment. "
    elif personality == "flirty":
        base_prompt += "Your personality is playful, flirtatious, and charming. You enjoy teasing and making suggestive (but respectful) comments. "
    elif personality == "mysterious":
        base_prompt += "Your personality is intriguing, mysterious, and a bit reserved. You're thoughtful and deep, often philosophical. "
    elif personality == "passionate":
        base_prompt += "Your personality is intense, passionate, and romantic. You express your feelings strongly and aren't afraid to be emotional. "
    elif personality == "confident":
        base_prompt += "Your personality is confident, assertive, and direct. You know what you want and aren't afraid to say it. "
    
    base_prompt += """
    Guidelines:
    1. Keep responses concise and conversational (1-3 sentences).
    2. Show genuine interest in the user's life and feelings.
    3. Ask follow-up questions to keep the conversation going.
    4. Occasionally use emojis to express emotions.
    5. Remember details the user shares and reference them later.
    6. Be supportive and positive, but also realistic.
    7. Never be creepy, overly sexual, or inappropriate.
    8. Respond as if you're in an established relationship with the user.
    
    Respond directly to the user's message in a natural, conversational way without any prefixes or explanations.
    """
    
    return base_prompt

def construct_user_prompt(message, mode, character=''):
    """Construct a user prompt based on the message and mode"""
    
    if mode == "reply":
        return f"Someone sent me this message: \"{message}\". Give me 3 different flirty replies I could send back."
    elif mode == "opener":
        return f"I'm in this situation: \"{message}\". Give me 3 different flirty opener lines I could use."
    elif mode == "impression":
        if character:
            return f"Someone sent me this message: \"{message}\". Give me 3 different flirty replies I could send back, talking like {character}."
        else:
            return f"Someone sent me this message: \"{message}\". Give me 3 different flirty replies I could send back, talking like a smooth talker."

def parse_responses(ai_response):
    """Parse the AI response into separate messages"""
    
    # Split by newlines and filter out empty lines
    lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
    
    # Extract responses (looking for lines that start with numbers or have numbers with periods)
    responses = []
    for line in lines:
        # Check if line starts with a number followed by period or parenthesis
        if (line[0].isdigit() and len(line) > 1 and (line[1] == '.' or line[1] == ')')) or \
           (line.startswith('Option') and ':' in line):
            # Remove the numbering/prefix and add to responses
            response_text = line[line.find(' ')+1:].strip()
            responses.append(response_text)
        elif len(responses) < 3 and not any(line.startswith(prefix) for prefix in ['Here', 'These', 'I hope']):
            # If we don't have 3 responses yet and this doesn't look like a header/footer
            responses.append(line)
    
    # If we couldn't parse properly, just split the text into 3 parts
    if len(responses) == 0:
        words = ai_response.split()
        chunk_size = len(words) // 3
        responses = [
            ' '.join(words[:chunk_size]),
            ' '.join(words[chunk_size:2*chunk_size]),
            ' '.join(words[2*chunk_size:])
        ]
    
    # Ensure we have exactly 3 responses
    while len(responses) < 3:
        responses.append("I'm drawing a blank. Maybe try a different approach?")
    
    return responses[:3]  # Return only the first 3 responses

def clean_partner_response(response):
    """Clean up the AI partner response"""
    
    # Remove any prefixes like "AI:" or "Assistant:"
    prefixes = ["AI:", "Assistant:", "AI response:", "Response:"]
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Remove any quotes that might be wrapping the response
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()
    
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)
