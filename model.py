import gradio as gr
from nltk.tokenize import RegexpTokenizer as RT
from nltk.stem import PorterStemmer as PS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as CV

cv=CV()
token=RT(r'\w+')
stops=set(stopwords.words('english'))
ps=PS()

def process_review(review):
    if isinstance(review, np.ndarray):
        review = str(review[0])  
    review = review.lower()
    tokens = token.tokenize(review)
    newtokens = [t for t in tokens if t not in stops or t == 'not']
    stemmed = [ps.stem(t) for t in newtokens]
    return ' '.join(stemmed)

def prepare(xarray, training=True):
    if isinstance(xarray, np.ndarray):  
        xarray = xarray.tolist()

    # Process reviews one by one (no parallel processing)
    xarray = [process_review(review) for review in xarray]

    xarray = cv.fit_transform(xarray).toarray() if training else cv.transform(xarray).toarray()
    return xarray
  
xarray=pd.read_csv('...../movie-rating-train/Train.csv').iloc[:5000,0].tolist()
yd=pd.read_csv('......./movie-rating-train/Train.csv').iloc[:5000,-1]
y = np.where(yd == 'pos', 1, 0)
xtest=pd.read_csv('/kaggle/input/testing/Test.csv').iloc[:1000,0].tolist()
xarray=prepare(xarray)
xtest=prepare(xtest,training=False)
import numpy as np

class MultinomialNB:
    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.vocab_size = X.shape[1]
        self.class_priors = class_counts / len(y)

        self.class_word_counts = np.zeros((len(self.classes), self.vocab_size))
        self.class_total_words = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            word_counts = X_c.sum(axis=0)
            self.class_word_counts[idx] = word_counts
            self.class_total_words[idx] = word_counts.sum()

        self.cond_probs = (self.class_word_counts + 1) / (self.class_total_words[:, None] + self.vocab_size)

    def predict(self, X):
        log_prior = np.log(self.class_priors)
        log_cond = np.log(self.cond_probs)
        log_probs = X @ log_cond.T + log_prior
        return self.classes[np.argmax(log_probs, axis=1)]

mnb = MultinomialNB()
mnb.fit(xarray, y)
def predict_review_sentiment(review):
    processed_review = prepare([review], training=False)  
    result = mnb.predict(processed_review)  
    
    ans= "pos" if result[0] == 1 else "neg"
    return ans


def main_function(review):
    processed_review = prepare([review], training=False)  
    result = mnb.predict(processed_review)  
    
    ans= "Positive Review" if result[0] == 1 else "Negative Review"
    return ans

demo=gr.Interface(fn=main_function,inputs="text",outputs="text")
demo.launch()


