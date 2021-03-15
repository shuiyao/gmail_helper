# Gmail Helper

**Last update: 2021/03/14 **

<img align="left" width="100" height="100" src="https://github.com/shuiyao/gmail_helper/blob/main/figures/stressful.png">

Load and clean emails from personal Gmail account using the Gmail Python API. Some main functionalities include
- Filter out unimportant emails based on previously labeled emails that reflect user preferences
- Archive and classify important emails in easy-to-read format

## Files
- topic_model.ipynb: An example of building topic models using LDA and LSA on 6000+ emails from my personal Gmail account.
- email_filter.ipynb: A demo of filtering out unimportant emails using various language models and classification algorithms.
- src/gmail.py: A tool for fetching emails from personal Gmail account.
- src/emailfilter.py: Implementation of the email classifiers.
