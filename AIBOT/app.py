from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from flask_cors import CORS
# Use google-generativeai package (more stable and widely supported)
try:
    import google.generativeai as genai
    from google.generativeai import types
    GENAI_PACKAGE = "generativeai"
except ImportError:
    try:
        from google import genai
        from google.genai import types
        GENAI_PACKAGE = "genai"
    except ImportError:
        print("❌ Error: No Google AI package found. Please install with: pip install google-generativeai")
        exit(1)
import re
import os
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS for frontend-backend communication
# Configure Flask session
app.secret_key = secrets.token_hex(32)  # Generate a secure secret key
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "*******************************************"  # Replace with your actual API key
MAX_MESSAGE_LENGTH = 4000
RATE_LIMIT_DELAY = 1  # Minimum seconds between requests

# In-memory storage (use proper database in production)
users_db = {}  # {email: {password_hash, name, created_at, last_login}}
conversation_history = {}  # {user_id: [conversations]}
user_sessions = {}  # {session_token: {user_email, created_at, expires_at}}
user_conversations = {}  # {user_id: {conversation_id: {title, messages, created_at, updated_at}}}

class ChatbotConfig:
    def __init__(self):
        # Use the latest available models
        self.model = "gemini-2.0-flash"  # Latest stable model
        self.max_tokens = 1000
        self.temperature = 0.7
        
    def get_system_prompt(self):
        return """You are AskCodzz AI Assistant, a helpful and knowledgeable AI powered by Google Gemini. 
        You are friendly, informative, and always try to provide accurate and helpful responses. 
        Keep your responses conversational and engaging. If you're not sure about something, 
        it's okay to say so."""

config = ChatbotConfig()

def hash_password(password):
    """Hash a password with salt"""
    salt = secrets.token_hex(32)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    try:
        salt, hash_hex = password_hash.split(':')
        password_check = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return password_check.hex() == hash_hex
    except:
        return False

def generate_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def validate_email(email):

    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check session token
        token = request.headers.get('Authorization')
        if not token:
            token = session.get('token')
        
        if not token or token not in user_sessions:
            return jsonify({"error": "Authentication required"}), 401
        
        # Check if token is expired
        user_session = user_sessions[token]
        if datetime.now() > user_session['expires_at']:
            del user_sessions[token]
            return jsonify({"error": "Session expired"}), 401
        
        # Add user info to request context
        request.user_email = user_session['user_email']
        request.user_token = token
        
        return f(*args, **kwargs)
    return decorated_function

def get_user_id():
    """Get current user ID from session"""
    return request.user_email if hasattr(request, 'user_email') else None

def clean_response(text):
    """Clean and format the response text"""
    if not text:
        return ""
    
    # Remove excessive markdown formatting
    text = re.sub(r'\*{3,}', '**', text)  # Replace *** with **
    text = re.sub(r'_{3,}', '__', text)   # Replace ___ with __
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive line breaks
    text = text.strip()
    
    return text

def save_conversation(user_id, user_message, bot_response):
    """Save conversation to history"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response
    })
    
    # Keep only last 20 exchanges to prevent memory issues
    if len(conversation_history[user_id]) > 20:
        conversation_history[user_id] = conversation_history[user_id][-20:]

def create_conversation(user_id, title="New Chat"):
    """Create a new conversation for a user"""
    conversation_id = f"conv_{int(time.time() * 1000)}"
    
    if user_id not in user_conversations:
        user_conversations[user_id] = {}
    
    user_conversations[user_id][conversation_id] = {
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    return conversation_id

def add_message_to_conversation(user_id, conversation_id, message, is_user=True):
    """Add a message to a specific conversation"""
    if user_id not in user_conversations or conversation_id not in user_conversations[user_id]:
        return False
    
    user_conversations[user_id][conversation_id]["messages"].append({
        "content": message,
        "is_user": is_user,
        "timestamp": datetime.now().isoformat()
    })
    
    user_conversations[user_id][conversation_id]["updated_at"] = datetime.now().isoformat()
    
    # Auto-generate title from first user message if still "New Chat"
    if (user_conversations[user_id][conversation_id]["title"] == "New Chat" and 
        is_user and len(user_conversations[user_id][conversation_id]["messages"]) == 1):
        title = message[:50] + ("..." if len(message) > 50 else "")
        user_conversations[user_id][conversation_id]["title"] = title
    
    return True

def get_user_conversations(user_id):
    """Get all conversations for a user"""
    if user_id not in user_conversations:
        return []
    
    conversations = []
    for conv_id, conv_data in user_conversations[user_id].items():
        conversations.append({
            "id": conv_id,
            "title": conv_data["title"],
            "created_at": conv_data["created_at"],
            "updated_at": conv_data["updated_at"],
            "message_count": len(conv_data["messages"])
        })
    
    # Sort by updated_at descending (most recent first)
    conversations.sort(key=lambda x: x["updated_at"], reverse=True)
    return conversations

def get_conversation_messages(user_id, conversation_id):
    """Get messages for a specific conversation"""
    if (user_id not in user_conversations or 
        conversation_id not in user_conversations[user_id]):
        return []
    
    return user_conversations[user_id][conversation_id]["messages"]

def delete_conversation(user_id, conversation_id):
    """Delete a conversation"""
    if (user_id not in user_conversations or 
        conversation_id not in user_conversations[user_id]):
        return False
    
    del user_conversations[user_id][conversation_id]
    return True

def rename_conversation(user_id, conversation_id, new_title):
    """Rename a conversation"""
    if (user_id not in user_conversations or 
        conversation_id not in user_conversations[user_id]):
        return False
    
    user_conversations[user_id][conversation_id]["title"] = new_title
    user_conversations[user_id][conversation_id]["updated_at"] = datetime.now().isoformat()
    return True

def get_conversation_context(user_id, max_exchanges=5):
    """Get recent conversation context"""
    if user_id not in conversation_history:
        return []
    
    recent_history = conversation_history[user_id][-max_exchanges:]
    context = []
    
    for exchange in recent_history:
        if GENAI_PACKAGE == "generativeai":
            # Use google-generativeai format
            context.extend([
                {"role": "user", "parts": [exchange["user_message"]]},
                {"role": "model", "parts": [exchange["bot_response"]]}
            ])
        else:
            # Use google-genai format
            context.extend([
                types.Content(role="user", parts=[types.Part.from_text(text=exchange["user_message"])]),
                types.Content(role="assistant", parts=[types.Part.from_text(text=exchange["bot_response"])])
            ])
    
    return context

def validate_input(question):
    """Validate user input"""
    if not question or not question.strip():
        return False, "Please enter a message."
    
    if len(question) > MAX_MESSAGE_LENGTH:
        return False, f"Message too long. Please keep it under {MAX_MESSAGE_LENGTH} characters."
    
    # Check for potentially harmful content (basic check)
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False, "Invalid input detected."
    
    return True, None

# Routes
@app.route("/")
def index():
    """Redirect to login or chat based on session"""
    token = session.get('token')
    if token and token in user_sessions:
        # Check if token is still valid
        user_session = user_sessions[token]
        if datetime.now() <= user_session['expires_at']:
            return redirect('/chat')
    
    return redirect('/login')

@app.route("/login")
def login_page():
    """Serve the login page"""
    return render_template("login.html")

@app.route("/chat")
def chat_page():
    """Serve the chat page (requires authentication)"""
    token = session.get('token')
    if not token or token not in user_sessions:
        return redirect('/login')
    
    # Check if token is expired
    user_session = user_sessions[token]
    if datetime.now() > user_session['expires_at']:
        del user_sessions[token]
        session.pop('token', None)
        return redirect('/login')
    
    return render_template("chat.html")

@app.route("/signup", methods=["POST"])
def signup():
    """Handle user registration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        
        # Validation
        if not name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400
        
        if len(name) < 2:
            return jsonify({"error": "Name must be at least 2 characters long"}), 400
        
        if not validate_email(email):
            return jsonify({"error": "Please enter a valid email address"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
        
        # Check if user already exists
        if email in users_db:
            return jsonify({"error": "An account with this email already exists"}), 400
        
        # Create new user
        password_hash = hash_password(password)
        users_db[email] = {
            "password_hash": password_hash,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        logger.info(f"New user registered: {email}")
        return jsonify({
            "message": "Account created successfully",
            "user": {"name": name, "email": email}
        })
        
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({"error": "Registration failed. Please try again."}), 500

@app.route("/login", methods=["POST"])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        
        # Validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        # Check if user exists
        if email not in users_db:
            return jsonify({"error": "Invalid email or password"}), 401
        
        user = users_db[email]
        
        # Verify password
        if not verify_password(password, user["password_hash"]):
            return jsonify({"error": "Invalid email or password"}), 401
        
        # Create session
        token = generate_token()
        user_sessions[token] = {
            "user_email": email,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=7)
        }
        
        # Update last login
        users_db[email]["last_login"] = datetime.now().isoformat()
        
        # Set session cookie
        session['token'] = token
        session.permanent = True
        
        logger.info(f"User logged in: {email}")
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "name": user["name"],
                "email": email
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": "Login failed. Please try again."}), 500

@app.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Handle user logout"""
    try:
        token = request.user_token
        
        # Remove session
        if token in user_sessions:
            del user_sessions[token]
        
        session.pop('token', None)
        
        return jsonify({"message": "Logged out successfully"})
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({"error": "Logout failed"}), 500

@app.route("/ask", methods=["POST"])
@require_auth
def ask():
    """Handle chat requests"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get("question", "").strip()
        conversation_id = data.get("conversation_id")
        user_id = get_user_id()
        
        # Create new conversation if none provided
        if not conversation_id:
            conversation_id = create_conversation(user_id)
        
        # Validate input
        is_valid, error_message = validate_input(question)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        
        logger.info(f"Processing request for {user_id}: {question[:50]}...")
        
        # Initialize Gemini client
        try:
            if GENAI_PACKAGE == "generativeai":
                # Use google-generativeai package
                genai.configure(api_key=API_KEY)
                client = genai
            else:
                # Use google-genai package
                client = genai.Client(api_key=API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            return jsonify({"error": "AI service temporarily unavailable"}), 500
        
        # Prepare conversation context
        context_messages = get_conversation_context(user_id)
        
        # Add system prompt and current user message
        if GENAI_PACKAGE == "generativeai":
            # Use google-generativeai format
            messages = [config.get_system_prompt() + "\n\n" + question]
            if context_messages:
                # Add conversation history
                history_text = ""
                for msg in context_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['parts'][0]}\n"
                messages[0] = history_text + "\n" + messages[0]
        else:
            # Use google-genai format
            messages = [
                types.Content(role="user", parts=[types.Part.from_text(text=config.get_system_prompt())])
            ]
            messages.extend(context_messages)
            messages.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        
        # Configure generation parameters
        if GENAI_PACKAGE == "generativeai":
            # Use google-generativeai format
            generation_config = {
                'max_output_tokens': config.max_tokens,
                'temperature': config.temperature,
            }
        else:
            # Use google-genai format
            generation_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                max_output_tokens=config.max_tokens,
                temperature=config.temperature
            )
        
        # Generate response
        response_text = ""
        try:
            if GENAI_PACKAGE == "generativeai":
                # Use google-generativeai package with correct model names
                model_names_to_try = [
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-001", 
                    "gemini-flash-latest",
                    "gemini-2.5-flash",
                    "gemini-pro-latest"
                ]
                
                success = False
                for model_name in model_names_to_try:
                    try:
                        logger.info(f"Trying generativeai model: {model_name}")
                        model = client.GenerativeModel(model_name)
                        response = model.generate_content(
                            messages,
                            generation_config=generation_config
                        )
                        response_text = response.text if response.text else ""
                        success = True
                        logger.info(f"Successfully used generativeai model: {model_name}")
                        break
                    except Exception as model_error:
                        logger.warning(f"Failed with generativeai model {model_name}: {str(model_error)}")
                        continue
                
                if not success:
                    raise Exception("All generativeai model name formats failed")
                    
            else:
                # Use google-genai package
                model_names_to_try = [
                    f"models/{config.model}",
                    config.model,
                    "models/gemini-2.0-flash",
                    "gemini-2.0-flash",
                    "models/gemini-2.5-flash",
                    "gemini-2.5-flash"
                ]
                
                success = False
                for model_name in model_names_to_try:
                    try:
                        logger.info(f"Trying genai model: {model_name}")
                        for chunk in client.models.generate_content_stream(
                            model=model_name, 
                            contents=messages, 
                            config=generation_config
                        ):
                            if chunk.text:
                                response_text += chunk.text
                        success = True
                        logger.info(f"Successfully used genai model: {model_name}")
                        break
                    except Exception as model_error:
                        logger.warning(f"Failed with genai model {model_name}: {str(model_error)}")
                        continue
                
                if not success:
                    raise Exception("All genai model name formats failed")
        
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return jsonify({"error": "Failed to generate response. Please try again."}), 500
        
        # Clean and validate response
        response_text = clean_response(response_text)
        if not response_text:
            response_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # Save conversation to both old and new systems
        save_conversation(user_id, question, response_text)
        
        # Add messages to conversation
        add_message_to_conversation(user_id, conversation_id, question, is_user=True)
        add_message_to_conversation(user_id, conversation_id, response_text, is_user=False)
        
        # Return response
        return jsonify({
            "reply": response_text,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

@app.route("/clear", methods=["POST"])
@require_auth
def clear_conversation():
    """Clear conversation history"""
    try:
        user_id = get_user_id()
        if user_id in conversation_history:
            del conversation_history[user_id]
        
        return jsonify({"message": "Conversation cleared successfully"})
    
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({"error": "Failed to clear conversation"}), 500

@app.route("/profile", methods=["GET"])
@require_auth
def get_profile():
    """Get user profile information"""
    try:
        user_email = get_user_id()
        user = users_db.get(user_email)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Count conversations and messages
        message_count = len(conversation_history.get(user_email, []))
        conversation_count = len(user_conversations.get(user_email, {}))
        
        return jsonify({
            "user": {
                "name": user["name"],
                "email": user_email,
                "created_at": user["created_at"],
                "last_login": user["last_login"],
                "message_count": message_count,
                "conversation_count": conversation_count
            }
        })
        
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({"error": "Failed to get profile"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "authenticated_users": len(user_sessions)
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get basic statistics"""
    total_users = len(users_db)
    active_sessions = len(user_sessions)
    total_conversations = len(conversation_history)
    total_messages = sum(len(history) for history in conversation_history.values())
    
    return jsonify({
        "total_users": total_users,
        "active_sessions": active_sessions,
        "total_conversations": total_conversations,
        "total_messages": total_messages
    })

@app.route("/conversations", methods=["GET"])
@require_auth
def get_conversations():
    """Get all conversations for the current user"""
    try:
        user_id = get_user_id()
        conversations = get_user_conversations(user_id)
        
        return jsonify({
            "conversations": conversations,
            "total": len(conversations)
        })
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({"error": "Failed to get conversations"}), 500

@app.route("/conversations", methods=["POST"])
@require_auth
def create_new_conversation():
    """Create a new conversation"""
    try:
        user_id = get_user_id()
        data = request.get_json()
        title = data.get("title", "New Chat") if data else "New Chat"
        
        conversation_id = create_conversation(user_id, title)
        
        return jsonify({
            "conversation_id": conversation_id,
            "title": title,
            "message": "Conversation created successfully"
        })
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        return jsonify({"error": "Failed to create conversation"}), 500

@app.route("/conversations/<conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    """Get messages for a specific conversation"""
    try:
        user_id = get_user_id()
        messages = get_conversation_messages(user_id, conversation_id)
        
        if messages is None:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({
            "conversation_id": conversation_id,
            "messages": messages,
            "total": len(messages)
        })
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        return jsonify({"error": "Failed to get conversation"}), 500

@app.route("/conversations/<conversation_id>", methods=["DELETE"])
@require_auth
def delete_conversation_endpoint(conversation_id):
    """Delete a conversation"""
    try:
        user_id = get_user_id()
        success = delete_conversation(user_id, conversation_id)
        
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return jsonify({"error": "Failed to delete conversation"}), 500

@app.route("/conversations/<conversation_id>/rename", methods=["POST"])
@require_auth
def rename_conversation_endpoint(conversation_id):
    """Rename a conversation"""
    try:
        user_id = get_user_id()
        data = request.get_json()
        
        if not data or "title" not in data:
            return jsonify({"error": "Title is required"}), 400
        
        new_title = data["title"].strip()
        if not new_title:
            return jsonify({"error": "Title cannot be empty"}), 400
        
        success = rename_conversation(user_id, conversation_id, new_title)
        
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({
            "message": "Conversation renamed successfully",
            "conversation_id": conversation_id,
            "new_title": new_title
        })
        
    except Exception as e:
        logger.error(f"Error renaming conversation: {str(e)}")
        return jsonify({"error": "Failed to rename conversation"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Clean up expired sessions periodically
def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.now()
    expired_tokens = [
        token for token, session_data in user_sessions.items()
        if current_time > session_data['expires_at']
    ]
    
    for token in expired_tokens:
        del user_sessions[token]
        logger.info(f"Cleaned up expired session: {token[:10]}...")

if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Check if API key is set
    if API_KEY == "AIzaSyAbNfGTfN-WGCbDzEFYnY3MVfvortl-J3s" or not API_KEY:
        print("⚠️  WARNING: Please set your Google Gemini API key in the API_KEY variable")
        print("🔗 Get your API key from: https://aistudio.google.com/app/apikey")
    
    # Create some demo users for testing
    demo_users = [
        {"email": "demo@askcodzz.com", "password": "demo123", "name": "Demo User"},
        {"email": "test@example.com", "password": "test123", "name": "Test User"}
    ]
    
    for demo_user in demo_users:
        if demo_user["email"] not in users_db:
            users_db[demo_user["email"]] = {
                "password_hash": hash_password(demo_user["password"]),
                "name": demo_user["name"],
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
    
    print("🚀 Starting AskCodzz Chatbot Server with Authentication...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("👤 Demo login: demo@askcodzz.com / demo123")
    
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True

    )
