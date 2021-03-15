import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
import bert
import bilstm

def F1(ytrue, ypred, eps=1.e-7):
    ypred = np.round(ypred)
    tp = np.sum(ytrue * ypred)
    preci = tp / (np.sum(ypred) + eps)
    recal = tp / (np.sum(ytrue) + eps)
    f1 = (preci * recal) / (preci + recal + eps)
    return f1, preci, recal

class EmailFilterBase():
    def __init__(self):
        self.model = None

    def load_data(self, file_train, file_test):
        self.train = pd.read_csv(file_train)
        self.test = pd.read_csv(file_test)

    def vectorize(self):
        self.vectorizer.fit(self.train['body'].astype(str))
        wordfeat = self.vectorizer.get_feature_names()
        self._wordmap = {k:v for v,k in enumerate(wordfeat)}

    def upsample(self):
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(
            sampling_strategy='minority', random_state=42)
        self.train, _ = sampler.fit_resample(self.train, self.train['target'])
        print("Training samples after resampling: {}".format(self.train.shape[0]))

    def compile(self, loss=None, optimizer=None, metrics=None):
        self.trainset = tf.data.Dataset.from_tensor_slices(
            (self.train['subject'], self.train['target']))
        self.trainset = self.trainset.shuffle(self.buffer_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.cvset = tf.data.Dataset.from_tensor_slices(
            (self.test['subject'], self.test['target']))
        self.cvset = self.cvset.shuffle(self.buffer_size).batch(self.batch_size)

        if (loss is None):
            loss = tf.keras.losses.BinaryCrossentropy()
        if (optimizer is None):
            optimizer = tf.keras.optimizers.Adam(1e-4)
        if (metrics is None):
            metrics = [
                'accuracy',
                tf.keras.metrics.TruePositives(name='tp'),        
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.Precision(name='preci'),
                tf.keras.metrics.Recall(name='recal')
            ] 
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def evaluate(self, X, y, return_false_samples=False):
        ypred = self.model.predict(X).flatten()
        f1, preci, recal = F1(y, ypred)
        print("Precision: {}".format(preci))
        print("Recall: {}".format(recal))
        print("F1 score: {}".format(f1))
        # ROC curve from the test sample
        fpr, tpr, thr = roc_curve(y, ypred)
        self.roc = pd.DataFrame({'FP':fpr, 'TP':tpr, 'thresh':thr})

        if(return_false_samples):
            X['pred'] = ypred
            X['diff'] = np.abs(y - ypred)
            X.sort_values(by='diff', ascending=False, inplace=True)
            fn = X[(X['target']==1) & (X['pred'] < 0.5)]
            fp = X[(X['target']==0) & (X['pred'] > 0.5)]
            return fn, fp
        else:
            return None, None

    def load_roc(self, file_roc):
        self.roc = pd.read_csv(rocfile)
        
class EmailFilterSVC(EmailFilterBase):
    def __init__(self, C=1.0, vectorizer=TfidfVectorizer(), **kwargs):
        from sklearn.svm import SVC    
        super().__init__()
        self.C = C
        self.model = SVC(C=C)
        self.vectorizer = vectorizer
        self.vectorizer.set_params(max_features=10000, max_df=0.9)

    def fit(self, **kwargs):
        X = self.vectorizer.transform(self.train['body'].astype(str))
        y = np.array(self.train['target'])        
        self.history = self.model.fit(X, y)

    def evaluate(self, return_false_samples=False):
        X = self.vectorizer.transform(self.test['body'].astype(str))
        y = np.array(self.test['target'])
        return super().evaluate(X, y, return_false_samples=return_false_samples)

class EmailFilterBayes(EmailFilterBase):
    def __init__(self, alpha=1.0, vectorizer=TfidfVectorizer(), **kwargs):
        from sklearn.naive_bayes import MultinomialNB
        super().__init__()
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)
        self.vectorizer = vectorizer
        self.vectorizer.set_params(max_features=10000, max_df=0.9)

    def fit(self, **kwargs):
        X = self.vectorizer.transform(self.train['body'].astype(str))
        y = np.array(self.train['target'])
        self.history = self.model.fit(X, y)

    def evaluate(self, return_false_samples=False):
        X = self.vectorizer.transform(self.test['body'].astype(str))
        y = np.array(self.test['target'])
        return super().evaluate(X, y, return_false_samples=return_false_samples)

class EmailFilterBERT(EmailFilterBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = bert.build_classifier()
        self.buffer_size = 512
        self.batch_size = 32        

    def load_model(self, file_model):
        import tensorflow_hub as hub
        self.model = tf.keras.models.load_model(
            file_model, custom_objects={'KerasLayer':hub.KerasLayer})

    def fit(self, **kwargs):
        history = self.model.fit(self.trainset,
                                 validation_data=self.cvset,
                                 **kwargs)
        return history
        
    def evaluate(self, return_false_samples=False):
        X = self.test['subject']
        y = np.array(self.test['target'])
        return super().evaluate(X, y, return_false_samples=return_false_samples)

class EmailFilterBiLSTM(EmailFilterBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.buffer_size = 512
        self.batch_size = 32                

    def build_model(self):
        txt = pd.concat([self.train['subject'], self.test['subject']],
                        ignore_index=True)
        txt = txt.astype(str).to_numpy()        
        self.model = bilstm.build_classifier(txt)

    def load_model(self, file_model):
        pass

    def fit(self, **kwargs):
        history = self.model.fit(self.trainset,
                                 validation_data=self.cvset,
                                 **kwargs)
        return history
        
    def evaluate(self, return_false_samples=False):
        X = self.test['subject']
        y = np.array(self.test['target'])
        return super().evaluate(X, y, return_false_samples=return_false_samples)

params_bert = {'epochs':5, 'batch_size':32, 'validation_steps':10}
params_bilstm = {'epochs':20, 'batch_size':32, 'validation_steps':10}

# efilter = EmailFilterBiLSTM()
# efilter.load_data('train.csv','test.csv')
# efilter.build_model()
# efilter.upsample()
# efilter.compile()
# efilter.fit(**params_bilstm)
# efilter.evaluate()

# efilter = EmailFilterBERT()
# efilter.load_data('train.csv','test.csv')
# efilter.load_model('model_bert_epoch5.h5')
# fn, fp = efilter.evaluate(return_false_samples=True)

# efilter = EmailFilterBayes()
# efilter.load_data('train.csv','test.csv')
# efilter.vectorize()
# efilter.upsample() # Makes a big difference for Naive Bayes and BiLSTM
# efilter.fit()
# efilter.evaluate()
