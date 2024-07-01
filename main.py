# Core Pkgs
import spacy_transformers
import streamlit as st 
# NLP Pkgs
import spacy_streamlit
import spacy
import eng_spacysentiment
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from transformers import pipeline, AutoTokenizer, BartForConditionalGeneration, AutoModelForSequenceClassification
from setfit import AbsaModel


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

def get_top_cat(doc):
    """takes a spacy doc object and returns the category with highest score"""
    cats = doc.cats
    max_score = max(cats.values())
    max_cats = [k for k, v in cats.items() if v == max_score]
    max_cat = max_cats[0]
    return (max_cat, max_score)

def main():
    """A Simple NLP app with Spacy-Streamlit"""

    st.title("GSA Natural Language Processing (NLP) Streamlit App")
    our_image = Image.open('./US-GeneralServicesAdministration-Logo.png')
    st.image(our_image)

    menu = ["NER", "TOKENIZE", "GENERAL SENTIMENT", "ASPECT-LEVEL SENTIMENT", 
	    # "EMOTION",
	    "CLASSIFY EMAIL", "SUMMARIZE"]
    # to add - classify news
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
        st.subheader("Tokenization and processing")
        raw_text = st.text_area("Your Text", demo_text)
        docx = nlp(raw_text)
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','lemma_', 'pos_','dep_','ent_type_'])

    elif choice == "NER":
        st.subheader("Named Entity Recognition")
        st.markdown("> This tool allows you to perform Named Entity Recognition (NER) to extract important entities in text. NER seeks to locate and classify entities using models built on large amounts of text. They don't suffer from some of the issues plaguing traditional approaches: NER is capable of extracting mispelled or previously unseen entities, and is more robust to noise.")
        st.markdown("> This example text comes from a [GSA press release](https://www.gsa.gov/about-us/newsroom/news-releases/gsa-celebrates-over-16-million-for-improvements-t-03272024), but you can test out your own text as well! It works well on a number of different text data such as survey responses to government reports. Try out different models by using the dropdown on the left.")
        raw_text = st.text_area("Your Text",demo_text)
        docx = nlp(raw_text)
        spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

    elif choice == "GENERAL SENTIMENT":
        st.subheader("General Sentiment Analysis")
        # st.markdown("Sentiment analysis can include things")
        st.markdown("> This tool provides lexicon-based sentiment scores and sentiment text classification. Because this pretrained model was trained using short snippets of text, it is applied at the paragraph-level here in this demo on this example text. This text classifier predicts sentiment (postive :smiley: , negative :slightly_frowning_face: , and neutral :neutral_face:) is appropriate to use for short peices of text rather than long texts (e.g., on a paragraph or sentence vs a longer document).")
        raw_text = st.text_area("Your Text",demo_text)
        docx = nlp_sentiment(demo_text)
        st.success(f"Overall Lexicon Polarity Score: {docx._.polarity}")
        st.success(f"Overall Lexicon Subjectivity Score: {docx._.subjectivity}")
        l_dfs = []
        c = 0
        for para in [p for p in raw_text.split("\n") if p != '']:
            c+= 1
            
            docx = nlp_sentiment(para)
            id_ = f'paragraph.{c}'
            # st.markdown(f"  ###### {id_}")
            st.markdown(f"> **{id_}**:   {para}")
           # visualize_textcat(docx)
    
            label_,score_ = get_top_cat(docx)

            d = docx.to_json()['cats']
            d['text'] = para
            d['label'] = label_
            d['text_id'] = id_
            
            df = pd.DataFrame([d])[["text_id", "text", "positive", "negative", "neutral", "label"]]
            l_dfs.append(df)
            # color_ = {'negative': '#faa0a0',
            # 'positive': "#e0f0e3",
            # 'neutral': "#e6e6e6"}.get(label_,  '#faa0a0')
          
            st.markdown(f"  ###### {label_} ({str(score_)[:5]})")
        
        st.dataframe(pd.concat(l_dfs).reset_index(drop=True))
            # spacy_streamlit.visualize_textcat(docx)
    elif choice == "ASPECT-LEVEL SENTIMENT":
        st.subheader("Aspect-level Sentiment Analysis")
        st.markdown("> This tool performs aspect-level sentiment analysis. Sentences with no aspects identified are removed from the prediction table. Example shown is a fake customer feedback response mentioning both positive and negative aspects of their experience. Try it out with your own example.")
        raw_text = st.text_area("Your Text",  "The staff was helpful, but the process was hard to navigate.")
        model = AbsaModel.from_pretrained(
            "./models/setfit-absa-paraphrase-mpnet-base-v2-aspect",
            "./models/setfit-absa-paraphrase-mpnet-base-v2-polarity",
            spacy_model="en_core_web_lg",
            )
        doc = nlp_large(raw_text)
        preds = []
        c = 0
        for sent in doc.sents:
            c+=1
            text = sent.text
            pred = model.predict(text)
            if len(pred[0]) == 0:
                continue
            else: preds.append({"text_id": f"sentence.{c}", "text": text, "pred": pred})
        if len(preds) > 0:
            
            df = pd.DataFrame(preds)
            dfe = df.explode("pred").reset_index(drop=True)
            dfe = dfe[dfe["pred"].notna()]
            
            df_final = dfe.drop(columns=["pred"]).merge(dfe["pred"].apply(pd.Series), right_index=True, left_index=True).set_index(["text_id", "text", "span"])
            st.dataframe(df_final)

        else: st.write("No aspects identified.")
        

    # elif choice == "EMOTION":
    #     st.subheader("Multi-label Email Text Classification")
    #     st.markdown("""> This tool allows you to predicts [Ekman's 6 basic emotions](https://en.wikipedia.org/wiki/Emotion_classification), plus a neutral class (
    # anger 🤬,
    # disgust 🤢,
    # fear 😨,
    # joy 😀,
    # neutral 😐,
    # sadness 😭,
    # surprise 😲)
    # using a multi-label text classification model that will provide probabilities for each label (aka class).""")
    #     raw_text = st.text_area("Your Text",demo_text)
    #     tokenizer = AutoTokenizer.from_pretrained("./models/emotion-english-distilroberta-base")
    #     model =  AutoModelForSequenceClassification.from_pretrained("./models/emotion-english-distilroberta-base")
    #     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    #     st.dataframe(pd.DataFrame(classifier(raw_text)[0]))


    elif choice == "CLASSIFY EMAIL":
        st.subheader("Email Classifier")
        st.markdown("> This is a model trained and developed at GSA from OCFO's collaboration with Department of Labor's Employment Training Administration CareerOneStop program. CareerOneStop is a digital platform that provides resources for career exploration, training, jobs, disaster assistance, and more for a wide range of different types of users. We built a text classifier to automatically categorize emails as an automated email such as from spam or a newsletter versus a user. The categories (also known as "classes") outside of spam were based on CareerOneStop's main user groups that they had previously defined in their survey. This model would not be appropriate to use on use cases that differ greatly from CareerOneStop. This classifier was applied to millions of emails over a span of 7+ years so that we could better understand how different user groups are experiencing the service.")
	st.markdown("> In a multiclass classification algorithm, the output is a set of prediction scores. These scores show how confident the model is that an observation belongs to each class (or category). The predicted class is simply the one with the highest score.")
	
	raw_text = st.text_area("Your Text",demo_text)
        docx = email_nlp(raw_text)
        dfl = []
        for k, v in docx.to_json()['cats'].items():
            dfl.append({'label': k, 'score': v})
        
        df = pd.DataFrame(dfl)
        st.dataframe(df)
        #spacy_streamlit.visualize_textcat(docx)

    elif choice == "CLASSIFY NEWS":
        st.subheader("News Classifier")
        st.markdown("> This is a model trained and developed at GSA from OGP and OCFO's Strategic Atlas collaboration.")
        raw_text = st.text_area("Your Text", demo_text)
        ## add classifier
        ## add visualize te
        # docx = email_nlp(raw_text)
        # spacy_streamlit.visualize_textcat(docx)

    elif choice == "SUMMARIZE":
        st.subheader("Summarize")
        st.markdown(">This tool allows you to apply summarization to your text. We utilize the open-source model published from [Hugging Face](https://huggingface.co/sshleifer/distilbart-cnn-12-6).")
        st.markdown("> This example text comes from a [GSA press release](https://www.gsa.gov/about-us/newsroom/news-releases/gsa-celebrates-over-16-million-for-improvements-t-03272024), but you can test out your own text to summarize as well!")
        raw_text = st.text_area("Your Text", demo_text)
        modelsum = BartForConditionalGeneration.from_pretrained("./models/distilbart-cnn-12-6")
        tokenizersum = AutoTokenizer.from_pretrained("./models/distilbart-cnn-12-6")
        
        textsum = pipeline(task="summarization", model=modelsum, tokenizer=tokenizersum)
        st.write(textsum(raw_text))
    

            

if __name__ == '__main__':
	main()
