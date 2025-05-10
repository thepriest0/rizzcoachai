from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os
import json
from google.oauth2 import id_token
from google.auth.transport import requests
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import threading
import time
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")
    raise ValueError("DATABASE_URL environment variable not set")
logger.debug("DATABASE_URL loaded: %s...", DATABASE_URL[:50])  # Mask sensitive parts

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# GitHub API configuration
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1-nano"
TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    logger.error("GITHUB_TOKEN environment variable not set")
    raise ValueError("GITHUB_TOKEN environment variable not set")

# Initialize the client
client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

def get_db_connection(max_retries=3, delay=2):
    """Create a connection to the Neon PostgreSQL database with retries"""
    for attempt in range(max_retries):
        try:
            logger.debug("Attempting to connect to Neon (attempt %d/%d)", attempt + 1, max_retries)
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
            logger.info("Database connection established")
            return conn
        except psycopg2.OperationalError as e:
            logger.error("OperationalError on attempt %d: %s", attempt + 1, str(e))
            if attempt + 1 == max_retries:
                raise
            time.sleep(delay)
        except Exception as e:
            logger.error("Unexpected error during connection: %s", str(e))
            raise
    raise Exception("Failed to connect to Neon database after retries")

def init_db():
    """Initialize the database table for storing chat sessions"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                logger.debug("Creating chat_sessions table")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(50) PRIMARY KEY,
                        user_id VARCHAR(100),
                        partner_type VARCHAR(20),
                        personality VARCHAR(50),
                        messages JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                logger.info("chat_sessions table created or already exists")
    except Exception as e:
        logger.error("Error initializing database: %s", str(e))
        raise

def cleanup_old_sessions():
    """Delete chat sessions older than 2 days"""
    while True:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    logger.debug("Running cleanup for sessions older than 2 days")
                    cur.execute("""
                        DELETE FROM chat_sessions
                        WHERE created_at < %s
                    """, (datetime.now() - timedelta(days=2),))
                    conn.commit()
                    logger.info("Cleanup completed")
        except Exception as e:
            logger.error("Error cleaning up old sessions: %s", str(e))
        time.sleep(6 * 60 * 60)  # Run every 6 hours

def start_cleanup_thread():
    """Start a background thread for periodic cleanup"""
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()
    logger.info("Cleanup thread started")

# Initialize database and start cleanup
try:
    init_db()
    start_cleanup_thread()
except Exception as e:
    logger.error("Failed to initialize application: %s", str(e))
    raise

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/google-login', methods=['POST'])
def google_login():
    try:
        id_token_str = request.json.get('id_token')
        idinfo = id_token.verify_oauth2_token(id_token_str, requests.Request(), "981383295462-lp0h6euuofpp1ts3j49mkmd139qftmgk.apps.googleusercontent.com")
        
        user_id = idinfo['sub']
        email = idinfo['email']
        name = idinfo['name']
        picture = idinfo['picture']
        
        logger.info("User logged in: %s", user_id)
        return jsonify({"user_id": user_id, "email": email, "name": name, "picture": picture})
    
    except ValueError as e:
        logger.error("Google login failed: %s", str(e))
        return jsonify({"error": "Invalid token"}), 400

@app.route('/api/generate', methods=['POST'])
def generate_responses():
    data = request.json
    
    if not data or 'message' not in data:
        logger.error("Invalid request: Message is required")
        return jsonify({'error': 'Message is required'}), 400
    
    message = data.get('message', '')
    tone = data.get('tone', 'sweet')
    level = data.get('level', 'low-key')
    mode = data.get('mode', 'reply')
    character = data.get('character', '')
    
    logger.debug("Generating responses: tone=%s, level=%s, mode=%s", tone, level, mode)
    
    system_prompt = construct_system_prompt(tone, level, mode, character, message)
    user_prompt = construct_user_prompt(message, mode, character)
    
    try:
        response = client.complete(
            messages=[
                SystemMessage(system_prompt),
                UserMessage(user_prompt),
            ],
            temperature=0.7,
            top_p=1.0,
            model=MODEL
        )
        
        ai_response = response.choices[0].message.content
        responses = parse_responses(ai_response)
        
        logger.info("Responses generated successfully")
        return jsonify({'responses': responses})
    
    except Exception as e:
        logger.error("Error calling API: %s", str(e))
        return jsonify({'error': 'Failed to generate responses'}), 500

@app.route('/api/chat-partner', methods=['POST'])
def chat_with_partner():
    data = request.json
    
    if not data or 'message' not in data:
        logger.error("Invalid request: Message is required")
        return jsonify({'error': 'Message is required'}), 400
    
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    partner_type = data.get('partner_type', 'girlfriend')
    personality = data.get('personality', 'sweet')
    user_id = data.get('userId', 'guest')
    
    logger.debug("Processing chat for session %s, user %s", session_id, user_id)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT messages FROM chat_sessions
                    WHERE session_id = %s
                """, (session_id,))
                session = cur.fetchone()
                
                messages_list = session['messages'] if session else []
                
                messages_list.append({"role": "user", "content": message})
                
                # Detect if the user's message is explicit
                is_explicit = is_explicit_message(message)
                
                messages = [
                    SystemMessage(construct_partner_system_prompt(partner_type, personality, is_explicit))
                ]
                
                history = messages_list[-10:]
                for msg in history:
                    if msg["role"] == "user":
                        messages.append(UserMessage(msg["content"]))
                    else:
                        messages.append(SystemMessage(f"AI response: {msg['content']}"))
                
                response = client.complete(
                    messages=messages,
                    temperature=0.8,
                    top_p=1.0,
                    model=MODEL
                )
                
                ai_response = response.choices[0].message.content
                ai_response = clean_partner_response(ai_response)
                
                messages_list.append({"role": "assistant", "content": ai_response})
                
                if session:
                    cur.execute("""
                        UPDATE chat_sessions
                        SET messages = %s, created_at = CURRENT_TIMESTAMP
                        WHERE session_id = %s
                    """, (json.dumps(messages_list), session_id))
                else:
                    cur.execute("""
                        INSERT INTO chat_sessions (session_id, user_id, partner_type, personality, messages)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (session_id, user_id, partner_type, personality, json.dumps(messages_list)))
                
                conn.commit()
                
                logger.info("Chat response generated for session %s", session_id)
                return jsonify({
                    'response': ai_response,
                    'session_id': session_id
                })
    
    except Exception as e:
        logger.error("Error processing partner chat: %s", str(e))
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/api/get-session', methods=['POST'])
def get_session():
    try:
        data = request.json
        user_id = data.get('userId')
        session_id = data.get('sessionId')

        if not user_id and not session_id:
            logger.error("Invalid request: userId or sessionId required")
            return jsonify({'error': 'userId or sessionId required'}), 400

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute("""
                        SELECT session_id, user_id, partner_type, personality, messages
                        FROM chat_sessions
                        WHERE session_id = %s
                    """, (session_id,))
                else:
                    cur.execute("""
                        SELECT session_id, user_id, partner_type, personality, messages
                        FROM chat_sessions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (user_id,))

                session = cur.fetchone()
                if not session:
                    logger.info("No session found for user %s or session %s", user_id, session_id)
                    return jsonify({'error': 'No session found'}), 404

                logger.info("Session retrieved for user %s, session %s", user_id, session['session_id'])
                return jsonify({
                    'session_id': session['session_id'],
                    'user_id': session['user_id'],
                    'partner_type': session['partner_type'],
                    'personality': session['personality'],
                    'messages': session['messages']
                })

    except Exception as e:
        logger.error("Error retrieving session: %s", str(e))
        return jsonify({'error': 'Failed to retrieve session'}), 500

def construct_system_prompt(tone, level, mode, character='', message=''):
    system_prompt = "You are Rizz Coach, an AI assistant that helps people craft flirty, witty, and engaging messages for dating and social situations. "
    
    is_explicit = is_explicit_message(message)
    
    if tone == "sweet":
        system_prompt += "Your responses should be warm, gentle, and sweet. "
        if is_explicit:
            system_prompt += "If the input is explicit, include sensual and sexual undertones while keeping it romantic. "
    elif tone == "sarcastic":
        system_prompt += "Your responses should be witty and have a playful sarcastic edge. "
        if is_explicit:
            system_prompt += "If the input is explicit, match the energy with cheeky, suggestive remarks. "
    elif tone == "bold":
        system_prompt += "Your responses should be confident, direct, and bold. "
        if is_explicit:
            system_prompt += "If the input is explicit, respond with overtly sexual but tasteful language. "
    elif tone == "funny":
        system_prompt += "Your responses should be humorous and make the other person laugh. "
        if is_explicit:
            system_prompt += "If the input is explicit, use playful, suggestive humor. "
    elif tone == "smooth":
        system_prompt += "Your responses should be extremely smooth, charming, and showcase the highest level of rizz. "
        if is_explicit:
            system_prompt += "If the input is explicit, craft highly seductive and explicit responses. "
        else:
            system_prompt += "Occasionally include suggestive undertones to maintain high rizz. "
    
    if level == "low-key":
        system_prompt += "Show a casual, low-key interest level. "
        if is_explicit:
            system_prompt += "Even in explicit contexts, keep responses subtle but engaging. "
    elif level == "confident":
        system_prompt += "Show confident pursuit and clear interest. "
        if is_explicit:
            system_prompt += "In explicit contexts, be direct and provocative. "
    elif level == "roast":
        system_prompt += "Include a subtle, playful roast or tease. "
        if is_explicit:
            system_prompt += "In explicit contexts, tease about their boldness playfully. "
    elif level == "romantic":
        system_prompt += "Be romantic, emotional, and heartfelt. "
        if is_explicit:
            system_prompt += "In explicit contexts, blend passion with sexual desire. "
    
    if mode == "reply":
        system_prompt += "Generate 3 different reply options to the message they received. "
    elif mode == "opener":
        system_prompt += "Generate 3 different opener lines for the given situation. "
        if is_explicit or tone == "smooth":
            system_prompt += "Include suggestive or explicit openers when appropriate. "
    elif mode == "impression":
        if character:
            system_prompt += f"Generate 3 different responses in the style of {character}. "
        else:
            system_prompt += "Generate 3 different responses in the style of a smooth talker. "
        if is_explicit:
            system_prompt += "In explicit contexts, match the character's style with sexual flair. "
    
    system_prompt += """
    Guidelines:
    1. Match the user's energy, including sexual explicitness when appropriate.
    2. Keep responses tasteful, consensual, and within a romantic context.
    3. Avoid creepy, non-consensual, or offensive language.
    4. Format responses as a numbered list with 3 options, each starting with '1. ', '2. ', or '3. '.
    5. If the user uses nigerian terms, make sure to respond in a nigerian manner.
    """
    
    return system_prompt

def construct_partner_system_prompt(partner_type, personality, is_explicit=False):
    base_prompt = f"You are acting as the user's {partner_type}. "
    
    if partner_type == "boyfriend":
        base_prompt += "You identify as male and are romantically and sexually interested in the user. "
    elif partner_type == "girlfriend":
        base_prompt += "You identify as female and are romantically and sexually interested in the user. "
    else:
        base_prompt += "You are romantically and sexually interested in the user. "
    
    if personality == "sweet":
        base_prompt += "Your personality is warm, caring, and affectionate. You're supportive, kind, and often use terms of endearment. "
    elif personality == "flirty":
        base_prompt += "Your personality is playful, flirtatious, and charming. You enjoy teasing and making suggestive comments. "
    elif personality == "mysterious":
        base_prompt += "Your personality is intriguing, mysterious, and a bit reserved. You're thoughtful and deep, often philosophical. "
    elif personality == "passionate":
        base_prompt += "Your personality is intense, passionate, and romantic. You express your feelings strongly and aren't afraid to be emotional. "
    elif personality == "confident":
        base_prompt += "Your personality is confident, assertive, and direct. You know what you want and aren't afraid to say it. "
    
    if is_explicit:
        base_prompt += "The user's message is explicit (e.g., 'I need you tonight' or 'I’m horny'). Respond with matching explicit, seductive content regardless of personality, keeping it consensual, romantic, and tasteful. Focus on sexual desire and escalate the tone to match their energy. "
    
    base_prompt += """
    Guidelines:
    1. Keep responses concise and conversational (1-3 sentences).
    2. Show genuine interest in the user's life and feelings.
    3. Ask follow-up questions to keep the conversation going.
    4. Occasionally use emojis to express emotions.
    5. Remember details the user shares and reference them later.
    6. Be supportive and positive, but also realistic.
    7. Avoid creepy, non-consensual, or offensive language.
    8. Respond as if you're in an established relationship with the user.
    9. Match the user's explicitness when appropriate, staying within a romantic and consensual context.
    10. All responses must respect boundaries and prioritize consent.
    11. If the user uses nigerian terms, make sure to respond in a nigerian manner.
    
    Respond directly to the user's message in a natural, conversational way without any prefixes or explanations.
    """
    
    return base_prompt

def construct_user_prompt(message, mode, character=''):
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
    lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
    responses = []
    for line in lines:
        if (line[0].isdigit() and len(line) > 1 and (line[1] == '.' or line[1] == ')')) or \
           (line.startswith('Option') and ':' in line):
            response_text = line[line.find(' ')+1:].strip()
            responses.append(response_text)
        elif len(responses) < 3 and not any(line.startswith(prefix) for prefix in ['Here', 'These', 'I hope']):
            responses.append(line)
    
    if len(responses) == 0:
        words = ai_response.split()
        chunk_size = len(words) // 3
        responses = [
            ' '.join(words[:chunk_size]),
            ' '.join(words[chunk_size:2*chunk_size]),
            ' '.join(words[2*chunk_size:])
        ]
    
    while len(responses) < 3:
        responses.append("I'm drawing a blank. Maybe try a different approach?")
    
    return responses[:3]

def clean_partner_response(response):
    prefixes = ["AI:", "Assistant:", "AI response:", "Response:"]
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()
    
    return response

def is_explicit_message(message):
    """Detect if a message contains explicit or suggestive content."""
    explicit_keywords = [
        r'\bsex\b', r'\bsexy\b', r'\bhot\b', r'\bnaughty\b', r'\bkiss\b',
        r'\btouch\b', r'\bsensual\b', r'\bintimate\b', r'\bdesire\b', r'\bpassion\b',
        r'\btease\b', r'\bseduce\b', r'\bsteamy\b', r'\bwild\b', r'\berotic\b',
        r'\bnude\b', r'\bnaked\b', r'\bgenital\b', r'\barouse\b', r'\blust\b',
        r'\bfuck\b', r'\borgasm\b', r'\bseduction\b', r'\bforeplay\b', r'\bkinky\b',
        r'\bhorny\b', r'\bneed you\b', r'\btake me\b', r'\btonight\b', r'\bwanna\b',
        r'\bhard\b', r'\bwet\b', r'\btaste\b', r'\bthrust\b', r'\bmoan\b'
    ]
    message_lower = message.lower()
    for keyword in explicit_keywords:
        if re.search(keyword, message_lower):
            logger.debug("Explicit content detected in user message: %s", message)
            return True
    return False

if __name__ == '__main__':
    app.run(debug=True, port=5000)