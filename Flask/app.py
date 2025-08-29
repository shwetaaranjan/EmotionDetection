from flask import Flask, jsonify, render_template,request
import requests
app = Flask(__name__)
from keras.models import load_model
import train_chatbot

model = None

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)


def load_model0():
    global model
    model = load_model('C:/Users/ishwe/OneDrive/Desktop/finalyearproject/Flask/chatbot_Bert_9090.h5')
    model.summary() 




def intro ():
    import videocap
    face = videocap.meth()
    #face = 'sad'
    if face == 'happy':
        return 'Do you want to share with me the reason that makes you %s ?' % (face)
    elif face == 'neutral':
        return "Today is the same like yesterday, I see that on your face. Nothing is happend, isn't it?"
    elif face == 'sad':
        return 'I understand that, you are not in a good mood. What happened?'
    elif face == 'angry':
        return 'I believe that you need to calm down, you look a bit %s. What makes you feel that way?' % (face)
    elif face == 'surprise':
        return 'What is the reason that you look %s.' % (face)
    else:
        return 'No feeling today...'


GEMINI_API_KEY = 'AIzaSyCTtv0Hdi0i9UMrK3sph76ROuqoisORjE4'
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/exec')
def parse(name=None):
    a = intro()
    return render_template('chatbot.html',kadir = a)


@app.route("/get")
def get_bot_response():

    global model
    if model is None:
        model = load_model('C:/Users/ishwe/OneDrive/Desktop/finalyearproject/chatbot/chatbot_Bert_9090.h5')

    userText = request.args.get('msg')
    result = train_chatbot.chat(str(userText), model)

    return result

@app.route('/ask', methods=['GET'])
def ask_gemini():
    user_msg = request.args.get('msg')
    
    if not user_msg:
        return jsonify({'error': 'Missing query parameter: msg'}), 400

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_msg
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        response.raise_for_status()

        gemini_output = response.json()

        # Extract text response
        generated_text = gemini_output['candidates'][0]['content']['parts'][0]['text']
        # return jsonify({'response': generated_text})
        return generated_text

    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 500
    except (KeyError, IndexError):
        return jsonify({'error': 'Unexpected response format from Gemini API'}), 500


if __name__ == '__main__':
    load_model0()
    app.run(threaded = False)
    app.debug = False