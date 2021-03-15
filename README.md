# Gmail Helper

**Last update: 2021/03/14 **

## Introduction
<img align="right" width="150" height="150" src="https://github.com/shuiyao/gmail_helper/blob/main/figures/stressful.png"> Though there has been a long history of studying spam classification, in our daily life, one's personal feelings about spam/ham could be very different. Like many others I have a Gmail account used for most web registrations or subscriptions. Over time it has accumulated tons of emails including ads and trivial updates that I would hardly skim over. However, buried under this mountain of emails are the actually IMPORTANT ones: personal messages from family and friends, important account records, tax document, medical bills, etc. I have a habit of archiving these emails for future references but over time I am growing tired of manually filtering out and classifying emails. Moreover, scanning through tens of emails each day, one could mistakenly delete important emails without even noticing. This makes me think, if I can build a automatic email filter based on the thousands of previously labeled or classified emails? Furthermore, by starting using a filter will help one label future emails more consistently which in turn will improve the filter even further.

## Purpose
Load and clean emails from personal Gmail account using the Gmail Python API. Some main functionalities include
- Filter out unimportant emails based on previously labeled emails that reflect user preferences
- Archive and classify important emails in easy-to-read format

## Files
- email_filter.ipynb: A demo of filtering out unimportant emails using various language models and classification algorithms.
- topic_model.ipynb: An example of building topic models using LDA and LSA on 6000+ emails from my personal Gmail account.
- src/gmail.py: A tool for fetching emails from personal Gmail account.
- src/emailfilter.py: Implementation of the email classifiers.
