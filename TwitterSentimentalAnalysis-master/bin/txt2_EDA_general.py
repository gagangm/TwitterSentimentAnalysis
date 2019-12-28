
'''
Description:
    This file contains the function that provide some functionality to generate insight about the text and helps in data cleaning
Main Function:
    - plot_wordcloud: Plot Wordcloud graph 
    - txt_char_level_breakdown: To get character level information from a Text. Information can include the breakdown into Upper Case, Lower Case, Numeric, Whitespace, & Special Character
    - plot_distribution: Used to plot distribution of a series/list 
    - plot_chartext_informations: Use to plot graphs when you have a series of texts, also get the cases that have unique constitution
    - cnt_words: Count the number of words belonging to particular class 
    - zipf_law_plot: Use to plot general graph related to zipf law
    - visualize_word_count_wrt_class: use to plot graphs on the table generated using cnt_words function

Package Installation:
    # pip3 install 

Reference Link: 
    
    
'''

# --------------------------------------------  Loading Libraries  --------------------------------------------- #
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# color = sns.color_palette()
# plt.style.use('fivethirtyeight')
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction import text
from nltk.corpus import stopwords



# ------------------------------------------  Aggregating Stopwords  ------------------------------------------- #
stopwords_sklrn = frozenset(text.ENGLISH_STOP_WORDS)
stopwords_nltk = frozenset(stopwords.words('english'))
stopwords_wrdcld = frozenset(STOPWORDS)
all_stopwords = frozenset(pd.Series(list(stopwords_sklrn) + list(stopwords_nltk) + list(stopwords_wrdcld)).unique())

print('# of stopwords in each lib: ',len(stopwords_sklrn), len(stopwords_nltk), len(stopwords_wrdcld))
print('# of stopwords when aggregated:', len(all_stopwords))

## Removing some words from stopwords
stopword_list = list(all_stopwords)
excpt_stopword = ['no', 'not']
for ele in excpt_stopword:
    stopword_list.remove(ele)
# -------------------------------------------------------------------------------------------- #



# ----------------------------------------------  plot_wordcloud  ---------------------------------------------- #
def plot_wordcloud(txt_list, title = None, remove_stopword=True, clr_map = None, bg='black'):
    '''
    A function for plotting wordcloud for the provided list of string
    txt_list = ['asd', 'asd', 'asdads']
    clr_map = 'magma', 
    
    ## https://mubaris.com/posts/dataviz-wordcloud/ ---> Amazing
    ## https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
    '''
    single_pooled_str = ' '.join(txt_list)
    # print('Type: {} \t String Length: {}'.format(type(single_pooled_str), len(single_pooled_str)))
    print('_'*110)
    stopwords = set(stopword_list) if remove_stopword else None
    plt.figure(figsize=(16,6))
    wordcloud = WordCloud(
                        width= 1600, # default: 400,
                        height= 600, # default: 200,
                        max_words=200,
                        min_font_size=4,
                        max_font_size= None, # default: None,
                        colormap=clr_map, # default: None,
                        background_color= bg,  # default: 'black',
                        stopwords=stopwords, # default: None,
                        scale=1, # default: 1,
                        random_state= 123, # default: None,
                        font_path=None,
                        margin=2,
                        ranks_only=None,
                        prefer_horizontal=0.9,
                        mask=None,
                        color_func=None,
                        font_step=1,
                        mode='RGB',
                        relative_scaling='auto',
                        regexp=None,
                        collocations=True,
                        normalize_plurals=True,
                        contour_width=0,
                        contour_color='black',
                        repeat=False,
                         ).generate(single_pooled_str)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14)
        #plt.suptitle(title, fontsize=20)
        #plt.subplots_adjust(top=2.3)
    plt.show()
    # -------------------------------------------------------------------------------------------- #
# ## Developing Wordcloud for NEGATIVE sentiments
# string_list = train_DF['text'].loc[train_DF['sentiment_class'] == 0].tolist()
# plot_wordcloud(string_list, title = 'Wordcloud Visualization of Negative Sentiments', remove_stopword=True, clr_map = 'magma', bg='white')



# ------------------------------  Function to understand text at character level  ------------------------------ #
def txt_char_level_breakdown(txt, msg=True):
    '''
    Get info at the character level for the provided text
    '''
    ptx_aplha_lower =  r'[a-z]'
    ptx_aplha_upper =  r'[A-Z]'
    ptx_digit =  r'[\d]'
    ptx_whitespace =  r'[\s]'
    ptx_special =  r'[^a-zA-z0-9\s]|[_\^\\\`\[\]]'
    temp_dict = {
        'total_len': len(txt),
        'upper': sum([ ele.isupper() for ele in txt ]), #ele.isupper()
        'lower': sum([ ele.islower() for ele in txt ]), #ele.islower()
        'numeric': sum([ re.search(ptx_digit, ele) is not None for ele in txt ]), #ele.isnumeric()
        'whitespace': sum([ re.search(ptx_whitespace, ele) is not None for ele in txt ]),
        'special': sum([ re.search(ptx_special, ele) is not None for ele in txt ]), 
    }
    temp_dict['StringLenMatch'] = temp_dict['total_len'] == temp_dict['upper'] + temp_dict['lower'] + temp_dict['numeric'] + temp_dict['whitespace'] + temp_dict['special'] 
    
    if msg:
        print('|\tTotal Number of character :', temp_dict['total_len'])
        print('|\tNumber of Upper case character :', temp_dict['upper'])
        print('|\tNumber of Lower case character :', temp_dict['lower'])
        print('|\tNumber of Numeric character :', temp_dict['numeric'])
        print('|\tNumber of Whitespace character :', temp_dict['whitespace'])
        print('|\tNumber of special character :', temp_dict['special'])
        print('|\tLength Matching :', temp_dict['StringLenMatch'])
        print('|\t--------')
    return temp_dict
    # -------------------------------------------------------------------------------------------- #
# txt = 'asdf adnkal Alsa AlLLLLLJ@123 #$^*fsl|||\'{}'
# txt_char_level_breakdown(txt, msg=True)


# --------------------------------------  Function to plot distribution  --------------------------------------- #
def plot_distribution(series, title, clr):
    ''' 
    Used to plot distribution of a series/list 
    '''
    print('Plotting{}'.format(' : '+title))
    try:
        f, (ax_box, ax_hist) = plt.subplots(figsize=(12, 3), nrows=2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        bp = sns.boxplot(series, ax=ax_box, color=clr)
        bp.set_title(title, fontsize=15)
        dp = sns.distplot(series, ax=ax_hist, color=clr) #, bins = nbins
#     except LinAlgError:
#         print('LinAlgError Exception: with {}'.format(title))
    except Exception as e:
        print('Exception: '+str(e))
    plt.xlim(0,)
    ax_box.set(xlabel='')
    plt.show()
    # -------------------------------------------------------------------------------------------- #


# -----------------------------------  Function for EDA of series of texts  ------------------------------------ #
def plot_chartext_informations(series_text):
    '''
    Use to plot graphs when you have a series of texts
    '''
    
    ## Extracting Information about texts and generating lists
    tot_li, up_li, lo_li, num_li, whi_li, spe_li, mat_li = [],[],[],[],[],[],[]
    for txt in series_text:
        char_dict = txt_char_level_breakdown(txt, False)
        tot_li.append(char_dict['total_len'])
        up_li.append(char_dict['upper'])
        lo_li.append(char_dict['lower'])
        num_li.append(char_dict['numeric'])
        whi_li.append(char_dict['whitespace'])
        spe_li.append(char_dict['special'])
        mat_li.append(char_dict['StringLenMatch'])
    
    ## Plotting Graphs
    plot_distribution(tot_li, 'Number of characters in texts', 'grey')
    plot_distribution(up_li, 'Number of character that are in upper case', 'darkorange')
    plot_distribution(lo_li, 'Number of character that are in lower case', 'green')
    plot_distribution(num_li, 'Number of character that are Numeric', 'blue')
    plot_distribution(whi_li, 'Number of character that are Whitespaces', 'red')
    plot_distribution(spe_li, 'Number of character that are Special Character', 'gold')
    
    ## Getting Peculiar Cases
    peculiar_cases = [ i for i in range(len(mat_li)) if mat_li[i] == False ]
    print('\n|\tNumber of texts whose character cataloging didn\'t happen properly {} / {}'.format(len(peculiar_cases), len(mat_li)))
    
    ## Plotting all Graphs in one
    height = 1
    samples_to_handle = 500
    samples = samples_to_handle if len(tot_li) > samples_to_handle else len(tot_li)
    ind = np.arange(len(tot_li))
    ind, up_li, lo_li, num_li, whi_li, spe_li = ind[:samples], up_li[:samples], lo_li[:samples], num_li[:samples], whi_li[:samples], spe_li[:samples]
    f, ax = plt.subplots(figsize=(14, 12))
    #p2 = plt.barh(ind, lo_li, width, bottom= up_li, label='Lower Case', color= 'olive') #, yerr=menStd
    p1 = plt.barh(ind, up_li, height, label='Upper Case', color= 'darkorange')
    p2 = plt.barh(ind, lo_li, height, left= up_li, label='Lower Case', color= 'green')
    p3 = plt.barh(ind, num_li, height, left= lo_li, label='Numeric', color= 'blue')
    p4 = plt.barh(ind, whi_li, height, left= num_li, label='Whitespaces', color= 'red')
    p5 = plt.barh(ind, spe_li, height, left= whi_li, label='Special', color= 'gold')
    plt.legend()
    plt.title('Understanding the distribution between Cases in texts')
    plt.ylim(0,samples_to_handle)
    plt.show()

    return peculiar_cases
    # -------------------------------------------------------------------------------------------- #
# peculiar_cases = plot_chartext_informations(train_DF['text'])
# if len(peculiar_cases) > 0: print(train_DF['text'][peculiar_cases])



# ------------------------------------  Count words in particular classes  ------------------------------------- #
def cnt_words(vectorizer, txt_arr, critical_class_arr=None, msg=True):
    '''
    Input: Text_Series/list and critical class if available 
    Output: Word Count
    '''
    txt_ser = pd.Series(txt_arr)
    critical_class_ser = ['-']*len(txt_ser) if critical_class_arr is None else pd.Series(critical_class_arr)
    critical_class = list(critical_class_ser.unique())

    ## Checking is length of Series is matching
    if len(txt_ser) == len(critical_class_arr):
        temp_liofli = []
        for cls in critical_class:
            if msg: print('Working with the class "{}"'.format(cls))
            temp_txt_ser = txt_ser[critical_class_ser == cls]

            ## Count the number of words and put the value accordingly 
            sparse_matrix = vectorizer.transform(temp_txt_ser)
            if msg:
                print('|\tSeries of Texts with length "{}" is fed to a trained Vectorizer having "{}" words.'.format(len(temp_txt_ser), len(CntVec.get_feature_names())))
                print('|\tThe Sparse Matrix width is the respective counts of the words on which it was fitted.')
                print('|\tType "{}"'.format(type(sparse_matrix)))
                print('|\tThe shape of this matrix on feeding data to it has now become "{}"'.format(sparse_matrix.shape))

            ## Summing the number of instances of a word across txts
            '''
            # axis=0 means along "indexes". It's a row-wise operation.   ---> check seems different for array
            # axis=1 means along "columns". It's a column-wise operation
            '''
            wrd_cnt = np.sum(sparse_matrix,axis=0)
            if msg:
                print('\n|\tNumber of occurances of the word is given by summing the occurances along the vertical axis.')
                print('|\tType "{}"\tShape of this matrix is "{}"'.format(type(wrd_cnt), wrd_cnt.shape))
                print('|\tTotal Number of Word are "{}"'.format(array(wrd_cnt.sum(axis=1))[0][0]))

            ## Removing the single dimension Entry from the array
            arr = np.squeeze(np.asarray(wrd_cnt)) #np.squeeze(array(wrd_cnt))
            ## Or np.asarray(wrd_cnt)[0]
            if msg:
                print('\n|\tChanging the shape of the array to get 1D array.')
                print('|\tType "{}"\tShape of this matrix is "{}"'.format(type(arr), arr.shape))

            temp_liofli.append(arr)

        ## Generating a DF for better readability
        term_freq_df = pd.DataFrame(temp_liofli, columns=vectorizer.get_feature_names()).transpose()
        
        ## Renaming Columns which contains critical class
        dict_col = {}
        for col in term_freq_df.columns:
            dict_col[col]= str(col) + '_class'
        term_freq_df.rename(columns=dict_col, inplace=True)
        
        ## Calculating Total Frequency
        total_freq_ser = pd.Series([0]*len(term_freq_df), index=term_freq_df.index)
        for col in term_freq_df.columns:
            total_freq_ser = total_freq_ser + term_freq_df[col]
        term_freq_df['TotalFreq'] = total_freq_ser
        # term_freq_df['TotalFreq'] = term_freq_df['Negative'] + term_freq_df['Positive']
        
        ## Adding a Column to Stopwords
        term_freq_df['IsStopword'] = [ ele in stopword_list for ele in term_freq_df.index ] 
        
        ## Reordering the Columns
        new_col_order = ['IsStopword'] + [col for col in term_freq_df.columns if col not in ['IsStopword', 'TotalFreq'] ] + ['TotalFreq']
        term_freq_df = term_freq_df[new_col_order]
        tot_wrd, stp_wrd = len(term_freq_df['IsStopword']), term_freq_df['IsStopword'].sum()
        #if msg: 
        print('\nTotal Words = Stops Words + Other Words\n \t{}  =  {}  +  {}'.format(tot_wrd, stp_wrd, tot_wrd-stp_wrd))
        
        print('\nA new DataFrame containg term freq has been created and its shape is {}'.format(term_freq_df.shape))
        
        return term_freq_df.sort_values(by='TotalFreq', ascending=False)
    else:
        msg = 'Series length is not matching. txt_ser Length {} != class_ser Length {}. \nHence, Raising Exception'.format(len(txt_ser), len(critical_class))
        print(msg); raise Exception(msg)
    # -------------------------------------------------------------------------------------------- #
# term_freq_df = cnt_words(CntVec, train_DF['text'], critical_class_arr = train_DF['sentiment_class'], msg=False)
# term_freq_df.rename(columns = {'0_class':'Negative_class', '1_class':'Positive_class'}, inplace= True)



# ----------------------------------------------  zipf_law_plot  ----------------------------------------------- #
def zipf_law_plot(text_series, top_ranks_to_view = 200):
    '''
    Input: text_series where index of this series should be words and the series should have the 
    counts of the words
    
     Zipf's Law states that a small number of words are used all the time, while the vast majority 
     are used very rarely.
     
     the rth most frequent word has a frequency f(r) that scales according to 
     ${f(r)} \propto \frac{1}{r^\alpha}$ for $\alpha \approx {1}$
     
     Visualizing this only
    '''
    exponents = [ 1, 0.75, 1.25, 0.55, 1.45]
    constant_of_proportionality = max(text_series)
    plt.figure(figsize=(13,6))
    clr = ['k', 'red', 'blue', 'orange', 'skyblue']
    
    word_rank = np.arange(start = 1, stop = top_ranks_to_view + 1, step=1)
    txt_freq = text_series.sort_values(ascending=False)[:top_ranks_to_view]
    
    ## Plotting Exponent Lines
    for exp_i in range(len(exponents)):
        expected_zipf = [np.around(constant_of_proportionality*(1/r**exponents[exp_i]), 
                                   decimals=5) for r in word_rank]
        plt.plot(word_rank, expected_zipf, color= clr[exp_i], linestyle='--', linewidth=2, 
                 alpha= 0.9, label = 'exponent = {}'.format(exponents[exp_i]))

    ## Plotting Vertical bars representing frequencies
    plt.bar(word_rank, txt_freq, width = 1, align= 'center', color='green', alpha=0.8, 
            label = 'Actual')

    plt.ylabel('Frequency')
    plt.xlabel('Rank')
    plt.title('Zipf Law: Plotting Top {} tokens present in Texts'.format(top_ranks_to_view))
    plt.legend()
    plt.grid(True)
    
    text_series = text_series.sort_values(ascending=False)
    tokens, word_rank = text_series.index, np.arange(1, len(text_series)+1)
    indices = np.argsort(-text_series)
    wrd_frequencies = text_series[indices]
    plt.figure(figsize=(13,6))

    ## Make a plot with log scaling on both the x and y axis
    plt.loglog(word_rank, wrd_frequencies, marker=".", color ='green')

    ## Plotting a line
    plt.plot([1,max(wrd_frequencies)],[max(wrd_frequencies),1],color='k')

    ## Adding Text on the Plots
    for n in list(np.logspace(-0.5, np.log10(len(text_series)-2), 25).astype(int)):
        dummy = plt.text(word_rank[n], wrd_frequencies[n], " " + tokens[indices[n]], 
                     verticalalignment= 'bottom', horizontalalignment= 'left')

    plt.title('Zipf Law Plot: In log10 scale for tokens in text')
    plt.xlabel('log(Rank)') #Frequency rank of token (scale: log10)
    plt.ylabel('log(Frequency)') #Absolute frequency of token (scale: log10)
    plt.ylim(1,10**5.5)
    plt.xlim(1,10**5.5)
    plt.grid(True)
    plt.show()
    # -------------------------------------------------------------------------------------------- #
# zipf_law_plot(term_freq_df['TotalFreq'], top_ranks_to_view=100)


# --------------------------------------  visualize_word_count_wrt_class  -------------------------------------- #
def visualize_word_count_wrt_class(term_freq_df, top_ranks_to_view= 50):
    '''
    Input: takes term_frequency_dataframe as input and plots the graph where columns have 
            '_class' in the column name
    Output: will be able to understand which words are important and associated with which class
    
    '''
    rank = np.arange(top_ranks_to_view)
    sub_df = term_freq_df.filter(like='_class')

    plt.figure(figsize=(14,10))
    class_cols = sub_df.columns
    color_li = ['red', 'green', 'blue', 'yellow']
    for cls_i in range(len(class_cols)):
        class_name = class_cols[cls_i].split('_class')[0]
        plt_loc = int('1'+str(len(class_cols))+str(cls_i+1))
        plt.subplot(plt_loc)
        DataToUse = sub_df.sort_values(by=class_cols[cls_i], ascending=False)[class_cols[cls_i]][:top_ranks_to_view]
        plt.barh(rank, DataToUse, align='center', alpha=0.8, color = color_li[cls_i])
        plt.yticks(rank, DataToUse.index,rotation='horizontal', fontsize=11)
        plt.xlabel('Frequency')
        plt.ylabel('Top {} tokens in class {}'.format(top_ranks_to_view, class_name))
        plt.title('Top {} tokens in {} texts'.format(top_ranks_to_view, class_name))
        plt.gca().invert_yaxis()

    plt.show()
    # -------------------------------------------------------------------------------------------- #



# -------------------------------------------------------------------------------------------------------------- #
# IsStopWord_list =[]
# for ind in range(term_freq_df.shape[0]):
#     CheckWord = frozenset(list(term_freq_df.iloc[ind:(ind+1)].index))
#     print(CheckWord)
#     AllStopword = text.ENGLISH_STOP_WORDS
#     IsStopWord_list.append(set(CheckWord).issubset(set(AllStopword)))
