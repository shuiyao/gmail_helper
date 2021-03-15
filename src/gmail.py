from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import base64
import lxml
from bs4 import BeautifulSoup
import textwrap

import pandas as pd
import json

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
PAGESIZE = 500

# Follow this link to enable Gmail API
# https://developers.google.com/gmail/api/quickstart/python

# First time initializing Session() will ask permission from google account

def showdict(d):
    print("Key             Value")
    print("---             -----")
    for k, v in d.items():
        print("{0:<24}{1:<}".format(k,str(v)))

class Session():
    '''
    An Gmail Session. Retrieve and read emails from gmail account.

    At initialization, need either a token file, e.g., token.pickle or a credential, e.g., credentials.json

    Once connect, one can interactively fetch information from the account.

    Example
    -------
    Create a gmail session

    >>> sess = Session() 

    Load the user profile (email address, etc.)

    >>> sess.show_user_profile() 

    Get the 'Labels' of the subfolders such as 'INBOX', 'TRASH' and 
    user created subfolders.

    >>> sess.get_labels() 

    Get a mail **list** containing 3000 mails with specified labels.
    
    >>> sess.get_mails('me', ['Label_3', 'IMPORTANT'], 3000)

    Get the 3rd mail from the selected mail list

    >>> mail = sess.get('me', 3)

    Read (subjects, from/to, body) the 4th mail from the selected mail list.

    >>> sess.read(4)

    Output retrieved emails into a more readable JSON file.

    >>> sess.dump(basename='inbox', textbody=True)

    '''
    def __init__(self):
        self._creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                self._creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not self._creds or not self._creds.valid:
            if self._creds and self._creds.expired and self._creds.refresh_token:
                self._creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', SCOPES)
                except:
                    print("File 'credentials.json' needed.")
                    print("Follow this link to enable Gmail API on your google account:")
                    print("https://developers.google.com/gmail/api/quickstart/python")
                    print("First time initializing Session() will ask for permission.")
                self._creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(self._creds, token)

        self.sess = build('gmail', 'v1', credentials=self._creds)
        self.userId = 'me'
        self.users = self.sess.users()
        self.profile = self.users.getProfile(userId=self.userId)
        print("Connected to account: {}.".format(self.profile.execute()['emailAddress']))

    def show_user_profile(self):
        print("User Profile: ")
        prof = self.profile.execute()
        showdict(prof)

    def get_labels(self, userId='me', full=False, verbose=False):
        '''
        Get all labels from a gmail account.
        Parameters
        ----------
        userId: Identifier for the gmail account. Default: 'me'.
        full: If False, return only id, name and type for each label.
              Default: False.
        '''
        results = self.users.labels().list(userId=userId).execute()
        labels = results.get('labels', [])

        if(verbose):
            if not labels:
                print('No labels found.')
            else:
                print('Labels:')
                for label in labels:
                    print(label['name'])
        if(full):
            self.labels = pd.DataFrame(labels)
        else:
            self.labels = pd.DataFrame(labels)[['id', 'name', 'type']]
        self.labels = self.labels.set_index('id')

    def get_mails(self, userId, labelIds='INBOX', maxResults=PAGESIZE):
        '''
        Get mails from user account.
        The number of mails to retrieve is set by maxResults.
        However, if maxResults > PAGESIZE, the code will move to next page until all mails are read

        Parameters
        ----------
        userId:
        labelIds: list or str. Specify the email label to retrieve
        maxResults: int. Maximum number of email to retrieve. Default: 500
            Note that each page contains a maximum of PAGESIZE emails
        '''

        mails = self.users.messages().list(
            userId=userId,
            labelIds=labelIds,
            maxResults=maxResults).execute()
        nextPageToken = mails.get('nextPageToken')
        try:
            self.mails = pd.DataFrame(mails['messages'])
        except:
            print("No message with label: ", labelIds)

        # keys: 'messages', 'resultSizeEstimate'
        # return mails
        maxResults -= PAGESIZE
        page = 1

        while(maxResults > 0 and nextPageToken):
            page += 1
            print("Loading Page {}...".format(page))
            mails = self.users.messages().list(
                userId=userId,
                labelIds=labelIds,
                pageToken=nextPageToken,
                maxResults=maxResults).execute()
            nextPageToken = mails.get('nextPageToken')
            self.mails = pd.concat([self.mails, pd.DataFrame(mails['messages'])],ignore_index=True)
            maxResults -= PAGESIZE
        
        self.n_mails = self.mails.shape[0]
        print("{0:<4d} mail IDs retrieved.".format(self.n_mails))

    def get(self, userId, mailId, mailFormat='full'):
        '''
        Get one single mail from the server.

        Parameters
        ----------
        userId:
        mailId: int or str. If int, retrieve self.mails['id'][mailId]
        mailFormat: 'minimal' or 'full'

        Return
        ------
        subject: str.
        '''
        assert(mailFormat in ['full', 'minimal']), "mailFormat must be either 'minimal' or 'full'"
        
        if(isinstance(mailId, int)):
            mailId = self.mails['id'][mailId]

        self.mail = self.users.messages().get(
            userId=userId, id=mailId, format=mailFormat).execute()

        return self.mail

    def get_mail_subject(self, mail=None):
        '''
        Get the subject string of a mail

        Parameters
        ----------
        mail: An users.messages() instance. If None, use self.mail

        Return
        ------
        subject: str.
        '''

        if(mail is None): mail = self.mail
        headers = mail['payload']['headers']
        names = [h.get('name') for h in headers]
        subject = [headers[i].get('value') for i, v in enumerate(names) if v == 'Subject']
        return subject[0]

    def get_mail_sender(self, mail=None):
        '''
        Get the subject string of a mail

        Parameters
        ----------
        mail: An users.messages() instance. If None, use self.mail

        Return
        ------
        subject: str.
        '''

        if(mail is None): mail = self.mail        
        headers = mail['payload']['headers']
        names = [h.get('name') for h in headers]
        sender = [headers[i].get('value') for i, v in enumerate(names) if v == 'From']
        return sender[0]

    def get_mail_body(self, mail=None, verbose=False):
        '''
        Get the text body of a mail

        Parameters
        ----------
        mail: An users.messages() instance. If None, use self.mail

        Return
        ------
        body: str.
        '''
        
        if(mail is None): mail = self.mail        
        parts = mail['payload']
        history = "payload "
        while(parts['mimeType'] not in ['text/plain', 'text/html']):
            try:
                parts = parts['parts'][0]
            except:
                return ''
            history += "-> parts"
        if(verbose):
            print("Path to text: {}".format(history))
        body = parts['body']['data'].replace("-","+").replace("_","/")
        body = base64.b64decode(body)
        body = BeautifulSoup(body, 'lxml')
        body = body.get_text()
        return ' '.join(body.split())

    def overview(self, mailId=0, n_mails=10):
        '''
        View a summary of emails starting with mailId up to n_mails
        '''
        n_mails = min(self.n_mails - mailId, n_mails)
        print()
        print("Viewing mailId from {:4d} to {:4d}".format(mailId, mailId + n_mails))
        print("----------------------------------------------------------------")
        for i in range(n_mails):
            if(i >= self.n_mails):
                break
            mail = self.get('me', mailId + i)
            subject = self.get_mail_subject(mail)
            print(subject)

    def read(self, mail=None, textwidth=80):
        '''
        Clean formatted display of the subject, sender, labels, and text body of a mail.

        Parameters
        ----------
        mail: An users.messages() instance. 
              If None, use self.mail. 
              If int or str, pass it as mailId to self.get()
        textwidth: The width for the body text. Auto-wrap.

        Return
        ------
        None
        '''

        if(mail is None): mail = self.mail
        elif(not isinstance(mail, dict)):
            assert(isinstance(mail, int) | isinstance(mail, str)),\
                "Parameter 'mail' must be int, str, dict, or None"
            self.get(userId=self.userId, mailId=mail)
            mail = self.mail
        
        subject = sess.get_mail_subject(mail)        
        sender = sess.get_mail_sender(mail)
        body = sess.get_mail_body(mail)
        labels = sess.labels.loc[mail['labelIds']]['name']
        print()
        print("Subject: {}".format(subject))
        print("From   : {}".format(sender))
        print("Labels : {}".format(labels[0]))
        if(labels.size > 1):
            for lbl in labels[1:]: print("         {}".format(lbl))
        print("--------------------------------")
        print(textwrap.fill(body, width=textwidth))

    def dump(self, basename="emails", batch_size=None, textbody=False):
        '''
        Dump all mails into JSON files.

        Parameters
        ----------
        basename: Base name of the JSON file.
        batch_size: Maximum number of mails in each JSON file.
        textbody: If False, get snippet only. If True, get the whole text body.
        '''
        line = {}
        n_iter = self.n_mails
        for i in range(n_iter):
            if((i % 100) == 0): print("Loading {:4d}-th email.".format(i))
            mailId = self.mails['id'][i]
            mail = self.get(userId=self.userId, mailId=i, mailFormat='full')
            headers = mail['payload']['headers']
            names = [h.get('name') for h in headers]
            try:
                subject = [headers[i].get('value') for i, v in enumerate(names) if v == 'Subject']
                subject = subject[0] if len(subject) > 0 else ""                
                sender = [headers[i].get('value') for i, v in enumerate(names) if v == 'From' or v == 'FROM'][0]
            except:
                print("Error occurred when trying to get subject or sender, check output.")
                return mail

            labels = ', '.join(sess.labels.loc[mail['labelIds']]['name'])
            if(textbody):
                body = sess.get_mail_body(mail)                
                line[mailId] = {'subject':subject,
                                'sender':sender,
                                'labels':labels,
                                'body':body}
            else:
                snippet = mail['snippet']
                line[mailId] = {'subject':subject,
                                'sender':sender,
                                'labels':labels,
                                'snippet':snippet}

            if(batch_size is not None):
                if(i % batch_size == 0 and i > 0):
                    fname = "{}.{:d}.json".format(basename, int(i/batch_size))
                    with open(fname, "w") as outfile:
                        json.dump(line, outfile, indent=2)
                    line = {}

        if(batch_size is None):
            fname = "{}.json".format(basename)
            with open(fname, "w") as outfile:
                json.dump(line, outfile, indent=2)
            
        # use pd.read_json(file, orient='index')

__mode__ = '__load__'
    
if __mode__ == '__example__':
    if 'sess' not in locals():
        sess = Session()
    sess = Session()
    sess.show_user_profile()
    sess.get_labels()
    print(sess.labels)
    sess.get_mails('me', ['Label_3', 'IMPORTANT'])
    mail = sess.get('me', 3)
    sess.read(4)

UserLabeled = [
    'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5',
    'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10',
    'Label_11', 'Label_12', 'Label_13', 'Label_14', 'Label_5',    
    'Label_16', 'Label_17', 'Label_18', 'Label_19', 'Label_20'
]
    
if __mode__ == '__load__':
    if 'sess' not in locals():
        sess = Session()
    sess = Session()
    sess.get_labels()
    # sess.get_mails('me', 'INBOX', 6000)
    for i in range(1, 21):
        sess.get_mails('me', UserLabeled[i], 1000)
        mail = sess.dump(basename='labeled.'+str(i), textbody=True)
    
print("DONE.")
