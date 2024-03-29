# Core Pkgs
import streamlit as st 

# NLP Pkgs
import spacy_streamlit
import spacy

models = {
    'transformer' : spacy.load("en_core_web_trf")
    , 'large' : spacy.load("en_core_web_lg")
    , 'small' : spacy.load("en_core_web_sm")
}

import os
from PIL import Image

print(os.getcwd())

def main():
    """A Simple NLP app with Spacy-Streamlit"""

    st.title("Spacy-Streamlit NLP App")
    our_image = Image.open('./US-GeneralServicesAdministration-Logo.png')
    st.image(our_image)

    menu = ["NER", 'Home']
    choice = st.sidebar.selectbox("Menu",menu)

    menu_model = ['small', 'large', 'transformer']
    choice_model = st.sidebar.selectbox("Model", menu_model)

    nlp = models[choice_model]

    demo_text = """The General Services Administration (GSA) is an independent agency of the United States government established in 1949 to help manage and support the basic functioning of federal agencies. GSA supplies products and communications for U.S. government offices, provides transportation and office space to federal employees, and develops government-wide cost-minimizing policies and other management tasks
    GSA employs about 12,000 federal workers. It has an annual operating budget of roughly $33 billion and oversees $66 billion of procurement annually. It contributes to the management of about $500 billion in U.S. federal property, divided chiefly among 8,700 owned and leased buildings and a 215,000 vehicle motor pool. Among the real estate assets it manages are the Ronald Reagan Building and International Trade Center in Washington, D.C., which is the largest U.S. federal building after the Pentagon.
    """

    if choice == "Home":
        st.subheader("Tokenization")
        raw_text = st.text_area("Your Text", demo_text)
        docx = nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])

    elif choice == "NER":
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Your Text",demo_text)
        docx = nlp(raw_text)
        spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)


if __name__ == '__main__':
	main()