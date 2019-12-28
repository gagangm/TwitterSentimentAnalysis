
import re
from datetime import datetime


### Creating some data cleaning and processing functions
def get_email_names_and_address(txt):
    '''
    Input: 'per1name <per1@email.com>, per2 lastname\r\n\t<per2@email.com>, per3 lastname <per3@email.com>'
    Output: {name: [], emails: []}
    ## index will match
    '''
    if type(txt) is str:
        cleaned_txt = txt.replace('\r\n\t', ' ')
        email_ids = [ e.split('>')[0] for e in cleaned_txt.split('<') if '>' in e ]
        for e_id in email_ids:
            cleaned_txt = ''.join(cleaned_txt.split(' <'+e_id+'>'))
        names = cleaned_txt.split(' ,')
    else:
        names, email_ids = [], []
    return {'name': names, 'emails': email_ids}
# get_email_names_and_address(msg.to)


## Converting DataTypes: Transforming string to Date ---> usecase can be converting to timestamp
sent_dt_format = '%a, %d %b %Y %H:%M:%S %z'
def convert_senton_to_datetime(dt_str):
    '''
    email sent on comes in a general static format highlighted below
    sent_dt_format = '%a, %d %b %Y %H:%M:%S %z' ## http://strftime.org/
    
    Input: Wed, 27 Feb 2019 16:00:43 +0530 ---> string
    Output: 2019-02-27 16:00:43+05:30 ---> Datetime
    '''
    try:
        dt_sent = datetime.strptime(dt_str, sent_dt_format)
    except ValueError:
        dt_sent = 'Format Not Found : ' + dt_str
    return dt_sent
# convert_senton_to_datetime('Wed, 27 Feb 2019 16:00:43 +0530')


def split_body_msg(txt):
    '''
    split1: forward related msg from the body
    split2: rest all
    '''
    ptx = re.compile('(Subject: .*)')
    result = ptx.search(txt)
    if result is not None:
        split_wrt = result.group(0)
        s1, s2 = txt.split(split_wrt)
        s1 = s1+split_wrt
    else:
        s1, s2 = None, txt
    return {'CommBody': s1, 'MainBody': s2}
# a,b = split_body_msg(email_msgs[e_key]['body'])



# def removing_extra_spaces(txt):
#     ''' to remove extra spaces that might have gotten added '''
#     return ' '.join([ e for e in txt.split(' ') if len(e)>0 ])
# # for email in emails_file_names:
# #     temp = email_msgs[email]['subject']
# #     temp1 = removing_extra_spaces(temp)
# #     print(len(temp), len(temp1))


## formatting the body text
# def clean_escp_seq_in_body(txt):
#     ''' remove: \r seq and also removes repeatition of \n seq'''
#     return '\n'.join([ ele for ele in txt.replace('\r', '').split('\n') if len(ele)>0 ])
