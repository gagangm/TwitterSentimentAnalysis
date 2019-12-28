
## Lib
import extract_msg, os, glob
import pandas as pd
from datetime import datetime
from txt0_processing_general_lib_executor import normalize_txt
from txt0_processing_specific_func import get_email_names_and_address, convert_senton_to_datetime, split_body_msg



def read_emails_files_and_clean_txt(config):
    '''
    read the '.msg' files and return msgs as dict with clean text
    '''
    ## Reading Emails
    raw_input_path = config['paths']['msg_files_path']
    if os.path.exists(raw_input_path) is False:
        raise Exception('Exception: Provided path doesn\'t exist')
    email_msg_files = glob.glob(raw_input_path + '*.msg')


    print('\nFrom ".msg" file to text')
    email_msgs, today_identifier = {}, datetime.today().strftime('%Y%m%d-')
    for efile in email_msg_files:
        '''
        Duplicacy of the file name can be a issue when the data will come as a stream 
        --- adding program runtime to add identifier
        '''
        msg = extract_msg.Message(efile)
        key = today_identifier + efile.split('/')[-1].split('.msg')[0]
        # print('Processing: {}.msg'.format(key))
        email_msgs[key] = {}
        email_msgs[key]['sender'] = get_email_names_and_address(msg.sender)
        email_msgs[key]['to'] = get_email_names_and_address(msg.to) # returns to field if exists.
        email_msgs[key]['cc'] = get_email_names_and_address(msg.cc) # returns cc if exists
        email_msgs[key]['sent_on'] = convert_senton_to_datetime(msg.date) # returns send date if exists
        email_msgs[key]['subject'] = normalize_txt(msg.subject, clean_steps = ['remove_extra_whitespaces'])
        email_msgs[key]['body'] = split_body_msg(normalize_txt(msg.body, clean_steps = ['html_stripping', 
                                            'contraction_expansion','accented_char_removal', 
                                            'remove_extra_whitespaces', 'remove_commas_bw_digits', 
                                            'remove_extra_meaningless_newlines'])) # returns msg body if exists
        # msg.attachments # returns a list of all attachment
    print('|________Process Complete\n')
    
    
    ## Checking which all sub keys are added
    print('\nStructure of the Dictionary')
    for e_key in list(email_msgs.keys())[:1]:
        print('| Top Key : {}'.format(e_key))
        for sub_e_key in email_msgs[e_key].keys():
            print('|    Sub Key : {} '.format(sub_e_key + ('  ----> SubSub Key: {}'.format(list(email_msgs[e_key][sub_e_key].keys())) if type(email_msgs[e_key][sub_e_key]) is dict else '' )))
    print('|________Process Complete\n')
    
    return email_msgs
# email_msgs = read_emails_files_and_clean_txt(config_settings)


def save_msg_as_dataframe(df, saving_path):
    ## Saving the Information in a TSV
    if (saving_path is not None) & os.path.exists(saving_path):
        print('\nSaving the data in a TSV \n|\t location : {}'.format(saving_path))
        df.to_csv(saving_path, sep='\t',index=False)
        print('|________Process Complete\n')

def convert_to_relational_df(config, msg_dict):
    '''
    convert the dictionary to structure format
    + split the columns that contain dict
    '''
    ## Converting to Relation Data Format
    print('\nConverting the Dictionary to Relational Data')
    df = pd.DataFrame(msg_dict).T.reset_index().rename(columns={'index':'Email File Name'})
    print('|________Process Complete\n')

    ## Changing the format of the df to sub divide the columns
    print('\nUnfolding the information that is present in this DataFrame')
    print('| Columns that are currenlty present : \n|\t{}'.format(list(df.columns)))
    for feat in df.columns:
        if type(df[feat].head(1).values[0]) is dict:
            for subfeat in list(df[feat].head(1).values[0].keys()):
                df[feat+'_'+subfeat] = [ ele[subfeat] for ele in df[feat] ]
            df.drop(columns=[feat], inplace=True)
    print('| Columns that are finally present : \n|\t{}'.format(list(df.columns)))
    print('|________Process Complete\n')
    
    ## Saving the Information in a TSV
    save_msg_as_dataframe(df, saving_path = config['paths']['structure_msg_file_name'])
    
    return df
# semi_clean_df = convert_to_relational_df(config_settings, email_msgs)

