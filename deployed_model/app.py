#!/usr/bin/env python
# coding: utf-8

# In[32]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib



# In[33]:


app = Flask(__name__)


# In[34]:


@app.route('/')
def home():
	return render_template('home.html')


# In[35]:


@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('df_nlp.csv', index_col = 0)
    df = df[['last_name', 'text']]
    
    # # remove url, special characters
    # # p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, 
    # #           p.OPT.SMILEY, p.OPT.MENTION, p.OPT.NUMBER)
    # # df['text'] = df['text'].apply(lambda x: p.clean(x))
    
    # # # Use regex to further remove emoji's and emoticons
    # # def remove_emoji(string):
    # #     emoji_pattern = re.compile("["
    # #                            u"\U0001F600-\U0001F64F"  # emoticons
    # #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    # #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
    # #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    # #                            u"\U00002702-\U000027B0"
    # #                            u"\U000024C2-\U0001F251"
    # #                            "]+", flags=re.UNICODE)
    # #     return emoji_pattern.sub(r'', string)

    # # df['text'] = df['text'].apply(lambda x: remove_emoji(x))
    
    # # tokenize tweets
    # # create a list of stop words
    # stop_words = list(spacy.lang.en.stop_words.STOP_WORDS)

    # # create a customerized list of punctuations and special characters
    # symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "--", "&amp", "—", "’", "’s", "'s", "amp"]

    # # create a function to preprocess each tweet with spaCy
    # def spacy_tokenizer(tweet):

    #     # creat a nlp object contains multiple tokenized attributes 
    #     tokens = nlp(tweet)

    #     # Lemmatizing each token and converting each token into lowercase
    #     # and remove any extra space in front of a non-pronounce word
    #     tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" 
    #            else token.lower_ for token in tokens]

    #     # Removing stopwords and punctuations
    #     tokens = [token for token in tokens
    #               if token not in symbols and token not in stop_words]

    #     return tokens   
    
    X = df['text']
    y = df['last_name']
    
    cv = CountVectorizer()
    X = cv.fit_transform(X.values.astype('U')) # Fit the Data
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        # return the highest probability as the prediction
        my_prediction = clf.predict(vect)
        
    return render_template('result.html', prediction = my_prediction)
    


# In[36]:


if __name__ == '__main__':
    app.run(debug=True)

