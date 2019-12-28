
'''
Description: 
    This file provide some functionality to clean texts.
Main Function this file Contains:
    - normalize_txt: Used to sequentially clean text based the sequence of cleaning stemp provided to this function.

Package Installation:
    # !pip3 install beautifulsoup4
    # !pip3 install pycontractions -- contraction.py should be in same directory
    # !pip3 install lxml
    # nltk.download('stopwords')
    # python3 -m spacy download en  // python3 -m spacy download en_core_web_md

Reference Links:
    https://www.w3schools.com/python/python_regex.asp
    https://www.regular-expressions.info/optional.html
    https://docs.python.org/3/howto/regex.html
'''

# --------------------------------------------  Loading Libraries  --------------------------------------------- #
import re, unicodedata
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import nltk, spacy
import en_core_web_sm
spacy_nlp = en_core_web_sm.load()
# spacy_nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from wordcloud import STOPWORDS


# ---------------------------------------  Some Data Cleaning Functions  --------------------------------------- #
def remove_accented_chars(text):
    '''
    Input: 'Sómě Áccěntěd těxt'
    Output: Some Accented text
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
# print(remove_accented_chars('Sómě Áccěntěd těxt'))


def strip_html_tags(text):
    '''
    Input: '<html><h2>Some important text</h2></html>'
    Output: Some important text
    '''
    # BeautifulSoup(text, 'lxml').get_text()
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
# print(strip_html_tags('<html><h2>Some important text</h2></html>'))


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    '''
    Input: "Y'all can't expand contractions I'd think"
    Output: "You all cannot expand contractions I would think"
    '''
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
# print(expand_contractions("Y'all can't expand contractions I'd think"))


def remove_special_characters(text):
    '''
    Doesn't remove digits
    Input: "Well this was fun! What do you think?\n 123#@!__ 123_"
    Output: "Well this was fun What do you think\n 123 123"
    '''
    text = text.replace('&', 'and')
    pattern = r'[^a-zA-z0-9\s]|[_\^\\\`\[\]]' #r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]' # [\s] == [ \t\n\r\f\v] True
    text = re.sub(pattern, '', text)
    return text
# print(remove_special_characters("Well this was fun! What do you think?\n 123#@!__ 123_"))


def simple_stemmer(text):
    '''
    Input: "My system keeps crashing his crashed yesterday, ours crashes daily"
    Output: "My system keep crash hi crash yesterday, our crash daili"
    '''
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
# print(simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily"))


def lemmatize_text(text):
    '''
    spacy and Nltk both have this
    Input: "My system keeps crashing! his crashed yesterday, ours crashes daily"
    Output: "My system keep crash ! his crashed yesterday , ours crash daily"
    '''
    text = spacy_nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
# print(lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily"))


def remove_stopwords(text):
    '''
    text should be in lower case
    Input: "The, and, if are stopwords, computer is not"
    Output: ", , stopwords , computer not"
    '''
    stopwords_sklrn = frozenset(text.ENGLISH_STOP_WORDS)
    stopwords_nltk = frozenset(stopwords.words('english'))
    stopwords_wrdcld = frozenset(STOPWORDS)
    all_stopwords = frozenset(pd.Series(list(stopwords_sklrn) + list(stopwords_nltk) + list(stopwords_wrdcld)).unique())
    # print('# of stopwords in each lib: ',len(stopwords_sklrn), len(stopwords_nltk), len(stopwords_wrdcld))
    # print('# of stopwords when aggregated:', len(all_stopwords))

    ## Removing some words from stopwords
    stopword_list = list(all_stopwords)
    excpt_stopword = ['no', 'not']
    for ele in excpt_stopword:
        stopword_list.remove(ele)
    
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
# print(remove_stopwords("The, and, if are stopwords, computer is not"))


def to_lower_case(text):
    '''
    Input: "To THe, AMD"
    Output: "to the, amd"
    '''
    return text.lower()
# print(to_lower_case('To THe, AMD'))


def remove_extra_whitespace(text):
    '''
    Input: "aslsj       alksdla    asdmda    askldalk"
    Output: "aslsj alksdla asdmda askldalk"
    '''
    #return re.sub(' +', ' ', text)
    return ' '.join([ ele for ele in text.split(' ') if len(ele) > 0 ])
# print(remove_extra_whitespace("aslsj       alksdla    asdmda    askldalk"))


def remove_newlines(text):
    '''
    Input: "aslsj \r\n  alksdla   \r \nasdmda \n   \n\n\naskldalk"
    Output: "aslsj alksdla asdmda askldalk"
    '''
    #len(re.sub(r'[\r|\n|\r\n]+', ' ',text)), len(' '.join(' '.join(text.split('\n')).split('\r'))) --> (42, 45)
    return re.sub(r'[\r|\n|\r\n]+', ' ', text)
# print(remove_newlines('aslsj \r\n  alksdla   \r \nasdmda \n   \n\n\naskldalk'))


def remove_extra_meaningless_newlines(text):
    '''
    Input: "word1\r\n word2\t\r\n \r\n  \r\n   word3 \n  \r \n  \n    \n mst"
    Output: "word1\nword2\t\nword3 \nmst" -->removed leading whitespaces in the new line and extra \n also
    '''
    text_cln = text.replace('\r', '\n')
    while True:
        len_ini = len(text_cln)
        text_cln = re.sub('\\n +\\n', '\\n', text_cln)
        if(len(text_cln) == len_ini): break
    text_cln = '\n'.join([ ele for ele in text_cln.split('\n') if len(ele)>0 ])
    text_cln = re.sub('\\n *', '\\n', text_cln)
    # print(text_orig);print('-'*10);print(text)
    return text_cln
# remove_extra_meaningless_newlines('word1\r\n word2\t\r\n \r\n  \r\n   word3 \n  \r \n  \n    \n mst')


def remove_digits(text):
    '''
    Input: 'word1 \n\rword2 word3 @123 from location1 to 2 costed like $18.25 == '
    Output: 'word \n\rword word @ from location to  costed like $. == '
    '''
    ptx =  r'[0-9]' 
    return re.sub(ptx, '', text)
# remove_digits('word1 \n\rword2 word3 @123 from location1 to 2 costed like $18.25 == ')


def remove_whitespace_character(text):
    '''
    Input: 'word1 \n\rword2 word3 @123 from location1 to 2 costed like $18.25 == '
    Output: 'word1   word2 word3  @123 from location1 to 2 costed like $18.25 == '
    '''
    ptx =  r'[\s]' # [\s] == [ \t\n\r\f\v] True
    return re.sub(ptx, ' ', text)
remove_whitespace_character('word1 \n\rword2 word3 \r@123 from location1 to 2 costed like $18.25 == ')


## Removing comma from in between digits
def remove_commas_bw_digits(text):
    '''
    Input: 'asdasd Gamonfsjn $123,123,123'
    Output: 'asdasd Gamonfsjn $123123123'
    '''
    while True:
        ptx = re.compile('\d,\d')
        result = ptx.search(text)
        if result is not None:
            comma_ind = result.start()+1
            text = text[:comma_ind]+text[comma_ind+1:]
        else:
            break
    return text
# remove_commas_bw_digits('asdasd Gamonfsjn $123,123,123 maskal 12,314.12')


def insert_spaces_around_special_character(text):
    '''
    Input: "word1 [word2+(word3-word5) ! word3]"
    Output: "word1    [ word2 +  ( word3 - word5 )     !    word3 ] "
    '''
    return ''.join([ char if re.match('[^A-Za-z0-9]', char) is 
                    None else ' '+char+' ' for char in text ])
# insert_spaces_around_special_character(text)


def remove_attherate_mentions(text):
    '''
    Input: "Hey @qwerty, you are sometime hard to use."
    Output: "Hey , you are sometime hard to use."
    '''
    return re.sub(r'@[A-Za-z0-9_]+', '', text)
# print(remove_attherate_mentions('Hey @qwerty, you are sometime hard to use.'))


def remove_urls(text):
    '''
    Input: "Hey buddy, have a look at this article https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12"
    Output: "Hey buddy, have a look at this article "
    '''
    text = re.sub(r'https?://[^ ]+', '', text)
    return re.sub(r'www.[^ ]+', '', text)
# remove_urls('Hey buddy, have a look at this article https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12')



# -------------------------------------  General Data Cleaning - Executor  ------------------------------------- #
def normalize_txt(txt, clean_steps = ['html_stripping', 'remove_urls', 'contraction_expansion','accented_char_removal', 
                                      'text_lemmatization', 'special_char_removal', 'remove_extra_whitespaces',
                                      'remove_commas_bw_digits', 'remove_extra_meaningless_newlines'], 
                 msg = False):
    '''
    Apply multiple operation in sequence
    Options: 
        1. accented_char_removal, 
        2. html_stripping,
        3. contraction_expansion
        4. special_char_removal
        5. text_stemmer
        6. text_lemmatization
        7. stopword_removal
        8. text_lower_case
        9. remove_extra_whitespaces
        10. remove_newlines
        11. remove_extra_meaningless_newlines
        12. remove_digits
        13. remove_commas_bw_digits
        14. insert_spaces_around_special_character
        15. remove_attherate_mentions
        16. remove_urls
        17. remove_whitespace_character : replace '\n', \t', '\r' etc with ' '
    '''
    cleaning_steps_mapping = {
        # --------------------------------------- General Lib
        # remove accented characters
        'accented_char_removal': remove_accented_chars,
        # strip HTML
        'html_stripping': strip_html_tags,
        # expand contractions  
        'contraction_expansion': expand_contractions,
        # remove special character including '_'
        'special_char_removal': remove_special_characters, 
        # stem text
        'text_stemmer': simple_stemmer,
        # lemmatize text
        'text_lemmatization': lemmatize_text,
        # remove stopwords
        'stopword_removal': remove_stopwords,
        # lowercase the text 
        'text_lower_case': to_lower_case,
        # remove extra whitespace
        'remove_extra_whitespaces': remove_extra_whitespace,
        # remove newlines (\n)
        'remove_newlines': remove_newlines,
        # remove extra newlines as well as leading whitespaces
        'remove_extra_meaningless_newlines': remove_extra_meaningless_newlines,
        # remove digits
        'remove_digits': remove_digits,
        # remove commas that is present between digits
        'remove_commas_bw_digits': remove_commas_bw_digits,
        # insert spaces around special character
        'insert_spaces_around_special_character': insert_spaces_around_special_character,
        # removing the mentions i.e. @string types
        'remove_attherate_mentions': remove_attherate_mentions,
        # removing the urls that are present in the text
        'remove_urls': remove_urls,
        # removing whitespace character and replacing with blankspace
        'remove_whitespace_character': remove_whitespace_character
        
        # --------------------------------------- Specific Lib
#         # from emails txt get names and address
#         'spec_get_email_names_and_address': get_email_names_and_address,
#         # convert_senton_to_datetime
#         'spec_convert_sent_time_to_datetime': convert_senton_to_datetime,
#         # split_body_msg
#         'spec_split_body_msg': split_body_msg
    }
    
    if type(clean_steps) is list:
        if msg: print('Performing Sequential Data Cleaning')
        for step in clean_steps:
            if msg: print('| Performing: {}'.format(step))
            txt = cleaning_steps_mapping[step](txt)
        if msg: print('|________Process Complete\n')
    else:
        raise Exception('Exception: Unexpected input given.')
    return txt
    # -------------------------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------------------------------------- #
# negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
#                 "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
#                 "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
#                 "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
#                 "mustn't":"must not"}
# neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
# neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)


# txt = '''\nTHOMAS/Michael\n \nFRACTAL MARINE CYPRUS TONNAGE\n \nMV Darya Lok - OPEN HOUSTON 10-15 MARCH\n \nTRY ANY\n \n+++\n \nMV "Darya Lok"\nIMO : 9595670\nType : Panamax\nSub type : Bulker\nGears : Gearless\nCO2 fitted in engine room\nBuilt : 2012 by DAEWOO\nFlag : Hong Kong\nClass : Lloyds Register\nDWT/Draft: 81,874 MT DWT / 14.52 m SSW\nTPC / TPI : 71.50 MT / 181.61 LT at full summer draft\nLOA/Beam : 229.00 m (loa) / 32.26 m (beam)\nLBP : 225 m\nInt\'l tonnage : 44,325 GT / 26,919 NT\nSuez : 45,328 GT / 40,479 NT\nPanama : 37,344 NT\nGrain/Bale : 96,155 cbm Grain / 91,780 cbm Bale\nHo/Ha : 7/7\nHatchcover : Mac Gregor\nStrengthened for heavy cargo\nCO2 fitted\nSpeed/cons :\nLaden : abt 13.5 kts on IFO 36 MT (380CST) +MD/GO 0.1 MT at sea\nBallast: abt 14 kts on IFO 36MT (380CST) +MD/GO 0.1 MT at sea\n+\nLaden : abt 11.5 kts on IFO 24 MT (380CST) +MD/GO 0.1 MT at sea\nBallast : abt 12 kts on IFO 24 MT (380CST) +MD/GO 0.1 MT at sea\n+\nPort idle : 2.5 IFO\nPort working : 4.0 IFO\nAbove speed / consumption in good weather conditions \nupto/incl Beaufort Force 4 and Douglas Sea State 3 with\nno adverse current and no negative influence of swell. \n \nAll details given in good faith and without guarantee\n \n+++\n \nBest Regards,\n \n \nwww.brsbrokers.com <http://www.brsbrokers.com/> \nMichael Boni \nDry Bulk Chartering \nT +44 (0) 203 216 1034\nM +44 (0) 7733 182 760\nmichael.boni@brsbrokers.com <mailto:michael.boni@brsbrokers.com> \n \nskype: michael_boni\nBARRY ROGLIANO SALLES\nSuite 465, 2nd Floor, Salisbury House, \n99 London Wall, London. EC2M 5QQ\n \n \n \n________________________________\nThe information contained in this email message may be privileged and confidential. If the reader is not the intended recipient, or the agent of the intended recipient, any unauthorised use, disclosure, copying, distribution or dissemination is strictly prohibited. If you have received this communication in error, please notify the sender immediately by telephoning +33141921234 or it@brsbrokers.com and return this message to the above address. BRS works in accordance with terms and conditions set out in our web site www.brsbrokers.com. BRS or one of their group companies may or may not record conversations.'''
# # remove_extra_meaningless_newlines(remove_special_characters(txt))
# print(normalize_txt(txt, clean_steps = ['html_stripping', 'contraction_expansion','accented_char_removal', 
#                                         'remove_extra_whitespaces', 'remove_commas_bw_digits', 
#                                         'remove_extra_meaningless_newlines']))

# IsStopWord_list =[]
# for ind in range(term_freq_df.shape[0]):
#     CheckWord = frozenset(list(term_freq_df.iloc[ind:(ind+1)].index))
#     print(CheckWord)
#     AllStopword = text.ENGLISH_STOP_WORDS
#     IsStopWord_list.append(set(CheckWord).issubset(set(AllStopword)))