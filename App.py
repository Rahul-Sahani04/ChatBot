import streamlit as st
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
from streamlit_chat import message

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Function to generate response from the model
def generate_response(user_input, max_length=100):
    input_ids = tokenizer.encode(user_input, return_tensors="pt", max_length=1024)
    response_ids = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("BlenderBot Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input:
        st.text("BlenderBot: " + generate_response(user_input))
    else:
        st.warning("Please enter a message.")
