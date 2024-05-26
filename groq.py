from flask import Flask, render_template, request
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

app = Flask(__name__)

conversational_memory_length = 5
memory = ConversationBufferWindowMemory(k=conversational_memory_length)
model_name = 'mixtral-8x7b-32768'
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
conversation = ConversationChain(llm=groq_chat, memory=memory)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_question = request.form['question']
        response = conversation(user_question)['response']

        # Check if the response contains code
        if '```' in response:
            response_type = 'code'
        else:
            response_type = 'text'

        return render_template('groq.html', question=user_question, response=response, response_type=response_type)
    return render_template('groq.html')

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os

# load_dotenv()
# groq_api_key = os.environ['GROQ_API_KEY']

# app = Flask(__name__)

# conversational_memory_length = 5
# memory = ConversationBufferWindowMemory(k=conversational_memory_length)
# model_name = 'mixtral-8x7b-32768'
# groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
# conversation = ConversationChain(llm=groq_chat, memory=memory)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         user_question = request.form['question']
#         response = conversation(user_question)['response']

#         # Wrap the response with triple backticks if it contains code
#         if '```' in response:
#             response = '```\n' + response + '\n```'

#         return render_template('groq.html', question=user_question, response=response)
#     return render_template('groq.html')

# if __name__ == '__main__':
#     app.run(debug=True)
