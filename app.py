import streamlit as st
import torch
from torch.serialization import add_safe_globals
from assignment2_fromscratch import (
    BPETokenizer, 
    UrduChatbotDataset, 
    Transformer, 
    normalize_text,
    generate_text
)

# Set page config
st.set_page_config(
    page_title="Urdu Main BaatCheet",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        # Load full checkpoint (safe only if from trusted source)
        checkpoint = torch.load('best_BLEU_model.pt', map_location='cpu', weights_only=False)

        config = checkpoint['config']
        tokenizer = checkpoint['tokenizer']
        vocab_size = checkpoint['vocab_size']

        from Assignment2_fromscratch import Transformer, UrduChatbotDataset

        model = Transformer(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            max_len=config['max_len'],
            dropout=config['dropout']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        dataset = UrduChatbotDataset([], tokenizer, config['max_len'])

        return model, tokenizer, dataset

    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        st.info("Make sure 'best_model_allinone.pt' exists in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        Troubleshooting steps:
        1. Ensure you have the correct model file
        2. Check if PyTorch version matches training environment
        3. Verify all custom classes are properly imported
        """)
        st.stop()

# Page title and description
st.title("Urdu Main Baat Cheet")
st.markdown("""Built with a custom Transformer architecture trained on Urdu text.""")
st.markdown("""Developed by Abdulrehman and Daniyal Shafiq""")

# Load model
model, tokenizer, dataset = load_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("اردو میں کچھ لکھیں... (Type something in Urdu)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Normalize input text
            normalized_input = normalize_text(prompt)
            
            # Generate response
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                input_text=normalized_input,
                max_length=50,
                temperature=0.8
            )
            
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
