# Core Pkgs
import spacy_transformers
import streamlit as st 
# NLP Pkgs
import spacy_streamlit
import spacy
import eng_spacysentiment
from spacytextblob.spacytextblob import SpacyTextBlob

nlp_sentiment = eng_spacysentiment.load()
nlp_sentiment.add_pipe("spacytextblob")

nlp_transformer =  spacy.load("./models/en_core_web_trf-3.7.3/en_core_web_trf/en_core_web_trf-3.7.3")
nlp_large = spacy.load("./models/en_core_web_lg-3.7.1/en_core_web_lg/en_core_web_lg-3.7.1")
nlp_small = spacy.load("./models/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1")


nlp_transformer.add_pipe("spacytextblob")
nlp_large.add_pipe("spacytextblob")
nlp_small.add_pipe("spacytextblob")

email_nlp = spacy.load("./models/output_comments_nov12/model-best")

models = {
    'transformer' : nlp_transformer
    , 'large' : nlp_large
    , 'small' : nlp_small
}

# importing additional packages for visualizing page
import os
from PIL import Image

print(os.getcwd())

def main():
    """A Simple NLP app with Spacy-Streamlit"""

    st.title("Spacy-Streamlit NLP App")
    our_image = Image.open('./US-GeneralServicesAdministration-Logo.png')
    st.image(our_image)

    menu = ["NER", 'TOKENIZE', 'SENTIMENT', 'CLASSIFY']
    choice = st.sidebar.selectbox("Menu",menu)

    menu_model = ['small', 'large', 'transformer']
    choice_model = st.sidebar.selectbox("Model", menu_model)

    nlp = models[choice_model]

    demo_text = """PITTSBURGH – Today, the U.S. General Services Administration (GSA) Administrator Robin Carnahan and other Biden-Harris Administration officials visited Pittsburgh to announce more than $16 million in funds for improvements to the Joseph F. Weis, Jr. U.S. Courthouse as part of President Biden’s Investing in America agenda. The funding for the courthouse–made possible by the Inflation Reduction Act, the largest climate investment in history–will be used for low-embodied carbon (LEC) materials including asphalt, concrete and steel that have fewer greenhouse gas emissions associated with their production. Today’s announcement is part of $63 million that GSA is investing through the Inflation Reduction Act across the Commonwealth of Pennsylvania.

Administrator Carnahan was joined by White House Council on Environmental Quality (CEQ) Federal Chief Sustainability Officer Andrew Mayock, U.S. Rep. Summer Lee (PA-12), and members of the Ironworkers Local Union #3 to make the announcement and tour the project.

“This project is a great example of how President Biden’s Investing in America agenda is making an impact in cities across the country,” said GSA Administrator Robin Carnahan. “From electric vehicles to the materials we use in construction, we’re making important upgrades to critical facilities across the nation, creating good paying jobs, healthier communities and saving money in the process. It’s a triple win for the country.” 

The Weis Courthouse project is one of more than 150 LEC projects that GSA announced last November. This project will complete critical repairs to the building’s loading dock using LEC materials, including asphalt, concrete and steel, to repair and replace corroded steel members, delaminated and spalled concrete, and damaged asphalt.

The Weis Courthouse currently houses the U.S. District Court for the Western District of Pennsylvania, the U.S. Bankruptcy Court, for the Western District of Pennsylvania, and offices of the U.S. Marshals Service and the Social Security Administration.

The announcement furthers the Biden-Harris Administration’s Federal Buy Clean Initiative, under which the federal government is, for the first time, prioritizing the purchase of LEC asphalt, concrete, glass and steel that have lower levels of greenhouse gas emissions associated with their production, use and disposal. These investments aim to expand America’s industrial capacity for manufacturing goods and materials of the future, tackle the climate crisis, and create good-paying jobs for workers in the region.

“We are so excited to host GSA Administrator Robin Carnahan to talk about some of the vital investments we’re bringing home and how it will support jobs for Ironworkers across Western Pennsylvania,” said Rep. Summer Lee. “I am committed to working side-by-side with the Biden Administration and labor unions throughout the region to ensure the benefits of the over $1 billion in federal investments we’ve helped bring home are felt by everybody.”

Earlier this week, GSA Administrator Robin Carnahan also announced $25 million in IRA investments for electric vehicle supply equipment (EVSE), which will install more than 780 charging ports across 33 federal buildings nationwide. That announcement helped kick off the U.S. Department of Energy’s Energy Exchange 2024 conference, a major annual event for energy and sustainability experts focused on improving the sustainability of the federal footprint, taking place this year in Pittsburgh.

The Inflation Reduction Act includes $3.4 billion for GSA to build, modernize, and maintain more sustainable and cost-efficient high-performance facilities. This funding includes $2.15 billion specifically for LEC construction materials. GSA’s IRA projects will implement new technologies and accelerate GSA’s efforts toward achieving a net-zero emissions federal building portfolio by 2045. Through these investments, GSA estimates a total greenhouse gas emissions reduction of 2.3 million metric tons, the same amount that 500,000 gasoline-powered passenger vehicles produce each year.

This announcement is part of President Biden’s Investing in America agenda, focused on growing the American economy from the bottom up and the middle-out – from rebuilding our nation’s infrastructure, to creating a manufacturing and innovation boom, to building a clean-energy economy that will combat climate change and make our communities more resilient."""

    if choice == "TOKENIZE":
        st.subheader("Tokenization")
        raw_text = st.text_area("Your Text", demo_text)
        docx = nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','lemma_', 'pos_','dep_','ent_type_'])

    elif choice == "NER":
        st.subheader("Named Entity Recognition")
        raw_text = st.text_area("Your Text",demo_text)
        docx = nlp(raw_text)
        spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

    elif choice == "SENTIMENT":
        st.subheader("Lexicon-based Sentiment Scores and Sentiment Text Classification")
        raw_text = st.text_area("Your Text",demo_text)
       # sentences = list(nlp(raw_text).sents)
        for para in [p for p in raw_text.split("\n") if p is not '']:
            docx = nlp_sentiment(para)
            spacy_streamlit.visualize_textcat(docx)
            st.success(f"Lexicon Polarity Score: {docx._.polarity}")
            st.success(f"Lexicon Subjectivity Score: {docx._.subjectivity}")
    
    elif choice == "CLASSIFY":
        st.subheader("Email Classifier")
        raw_text = st.text_area("Your Text",demo_text)
        if st.button("Predict"):
            docx = email_nlp(raw_text)
            spacy_streamlit.visualize_textcat(docx)
            

if __name__ == '__main__':
	main()