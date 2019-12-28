
'''
Description: 
    This file provide some function that are for general use cases.
Function this file Contains:
    - levprint: It is used to print statement based on the level. i.e.  function level.
    - print_system_info: Used to print the system configurations
    - generate_highlighted_heading: Used to generate some comment based heading / seperators
    - get_abs_path_from_relative_path: Used to get absolute path based on the provided relative path.
    - time_cataloging: To generate a REPORT file for the runtime/crashes of parts of the code.
    - create_key: Used to combine columns to return combined value which can be used as KEY/Unique identifier.
    - add_recommendation: Used to add messages to the recommendation file. When all recommendation are 
                          followed delete this file.

    - is_Valid: To check if __ is a valid __. -- Later

Package Installation:
# !pip3 install py-cpuinfo
'''

# --------------------------------------------  Loading Libraries  --------------------------------------------- #
import pandas as pd
import time, os, sys, ast
import cpuinfo, platform, multiprocessing
from psutil import virtual_memory




def custom_ast_lit_eval(content):
    '''
    Sometimes content are not read by ast.literal_eval hence this function sort some of 
    the error to make it work
    '''
    things_to_remove = ['\n']
    for ele in things_to_remove:
        content = str(content).replace(ele, ' ')
    st = ' '.join([ele for ele in content.split(' ') if len(ele)>0])
    return ast.literal_eval(st)
# custom_ast_lit_eval(config['model_preparation']['grid_search_combinations']['comb1'])


# --------------------------------------------  Level Based Print  --------------------------------------------- #
def levprint(txt='', level=0, StartOrEnd=0):
    '''
    Use to print statement based on levels
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>------------ #
    txt = str(txt)
    if StartOrEnd != 0:
        ## expecting '\t' len = 8 spaces
        print('', '+'+'-'*(112 - 8*level),sep= '\t'*level) ## python 3.x
        # print ''+'\t'*level+'+'+'-'*(112 - 8*level) ## python 2.x
    if len(txt) != 0: print('', txt,sep= '\t'*level + '|'+' ')## python 3.x
    # if len(txt) != 0: print ''+'\t'*level + '|'+' '+txt ## python 2.x
    # -------------------------------------------------------------------------------------------- #



# --------------------------------------------  Print System Info  --------------------------------------------- #
def print_system_info():
    '''
    function that print the configuration of the system/device
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>------------ #
    levprint('Inside "'+print_system_info.__name__+'" function.',0,1)
    
    mem = virtual_memory()
    total_mem = str(round(mem.total/(1024.**3), 2)) + ' GB'
    total_avail = str(round(mem.available/(1024.**3), 2)) + ' GB'
    
    levprint('Python Version: {0}.{1}.{2}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]),0)
    levprint('Machine Name: {}'.format(platform.node()),0)
    levprint('OS: {}'.format(platform.platform()),0)
    levprint('Hardware: \t CPU brand: {}'.format(cpuinfo.get_cpu_info()['brand']),0)
    levprint('\t\t CPU # of cores: {}'.format(multiprocessing.cpu_count()),0)
    levprint('\t\t RAM: Total = {}, Available = {}'.format(total_mem, total_avail),0)
    levprint('Current Working Directory: {}'.format(os.getcwd()),0)
    levprint(level=0,StartOrEnd=1)
    # -------------------------------------------------------------------------------------------- #



# ----------------------------------  Generate Highlighted Heading/Seperator  ---------------------------------- #
def generate_highlighted_heading(msg):
    '''
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>------------ #
    #levprint('Inside "'+generate_highlighted_heading.__name__+'" function.',0,1)
    
    msg = '  '+msg+'  '
    msg_len = len(msg)
    start, end = '# ', ' #'    
    length = [114, 96, 70]
    septors = ['-', ' ']#, '='
    ctr_highlightor = [('',''), ('<<<','>>>'), (' <<<<[[[[', ']]]]>>>> ')]
    
    for sep in septors:
        for le in length:
            for ctr in ctr_highlightor:
                lt_sep_cnt = int(le/2) - len(ctr[0]) - len(start) - (int(len(msg)/2) if len(msg)%2 == 0 else int((len(msg)+1)/2))
                rt_sep_cnt = int(le/2) - len(ctr[1]) - len(end) - (int(len(msg)/2) if len(msg)%2 == 0 else int((len(msg)-1)/2))
                print(start+sep*lt_sep_cnt+ctr[0]+msg+ctr[1]+sep*rt_sep_cnt+end)
            print(start+sep*(le-4)+end)
    # -------------------------------------------------------------------------------------------- #



# -----------------------------------  Get Absolute Path From Relative Path  ----------------------------------- #
def get_abs_path_from_relative_path(RelPath, msg = False):
    '''
    DirToMoveTo = ../../../A/B/
    CurrentAbsDir = /X/Y/Z  ##abs path
    
    returns /X/Y/Z, /X/Y/A/B/
    i.e. returns original and new path
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>------------ #
    
    curr = str(os.getcwd())
    curr0 = curr
    path = RelPath
    DirToGoTo = path.split('/')
    for dirspli in DirToGoTo:
        if dirspli == '..':
            curr = '/'.join(curr.split('/')[0:-1])
        else:
            curr += '/' + dirspli
    if msg is True: print('Current directory where code is executed :', curr0)
    if msg is True: print('New directory path which was mentioned :', curr)
    return curr0, curr
    # -------------------------------------------------------------------------------------------- #



# ---------------------------------------------  Time Cataloging  ---------------------------------------------- #
def time_cataloging(config, Key, Value, First = 'Off'):
    '''
    To generate a REPORT file for the runtime/crashes of parts of the code.
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>------------ #
    if First == 'On':
        ExecTime = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime())
        ## same as in  Iteration file --> copied  from there
        #NpD = int(1440 / int(config('BlacklistingLimits','TimeDiffBwEachIteration_inMin')))
        #IterFile = config('OutputPaths', 'IterationSeqNumberFile')
        #_, absPathIterFile = GetBackSomeDirectoryAndGetAbsPath(IterFile)
        #if os.path.exists(absPathIterFile): ## When Not a first iteration
        #    tempDF = pd.read_csv(absPathIterFile)
        #    DayNo, IterationNo = tempDF['DayNo'][0], tempDF['IterationNo'][0] + 1
        #    if IterationNo == (NpD + 1): ## Day Rotation if all iteration for a days get completed
        #        DayNo, IterationNo = tempDF['DayNo'][0] + 1, 1
        #else: ## When first iteration
        #    DayNo, IterationNo = 1, 1
        
        TimeConsumedReport = {
                'ExecutionTime': ExecTime,
                'ExecTimestamp': int(time.time()),
                'DayNo': DayNo,
                'Iteration': IterationNo,
                'ImportInput': '-',
                'ProcessingData': '-',
                'CombineDataStrems': '-',
                'UpdatingLogs': '-',
                'WholeExecutionTime': '-'
            }
    ## Creating a DataFrame Containing Execution Time Results
    ExecTimePath = path# config('LogPaths','ExecutionTimeTaken')
    col = ['ExecutionTime', 'ExecTimestamp', 'DayNo', 'Iteration', 'ImportInput', 'ProcessingData',
           'CombineDataStrems', 'UpdatingLogs', 'UpdatingBlacklistLogs', 'WholeExecutionTime']
    if(os.path.exists(ExecTimePath) is False):
        tempDF = pd.DataFrame(TimeConsumedReport, columns = col, index = [0]) #TimeConsumedReport.keys()
    else:
        tempDF = pd.read_csv(ExecTimePath)
        if First == 'On':
            tempDF = tempDF.append(TimeConsumedReport, ignore_index=True)
    ## Updating Entries
    try:
        tempDF.iloc[(len(tempDF)-1), tempDF.columns.get_loc(Key)] = Value
    except:
        print('Passed Key Doesn\'t Exist in Present Structure')
    ## Saving Locally
    tempDF.to_csv(ExecTimePath, index=False)
    if Key == 'WholeExecutionTime':
        return tempDF.iloc[len(tempDF)-1,:].to_dict()
    # -------------------------------------------------------------------------------------------- #



# ------------------------------------------------  Create Key  ------------------------------------------------ #
def create_key(DF, Key_ColToUse):
    '''
    Use to combine columns to generate a key which is seperated by '|'
    eg. Key_ColToUse = X, Y & Z ==> return X|Y|Z 
    '''
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index
    # -------------------------------------------------------------------------------------------- #


# -------------------------------------------  Add Recommendations  -------------------------------------------- #
def add_recommendation(msgToAdd, path):
    '''
    Used for adding recommendations inside a single recommendation File
    Delete this file after recommendationhas been followed.
    '''
    filePath = path
    _, absPathRecommFile = GetBackSomeDirectoryAndGetAbsPath(filePath)
    NewDf = pd.DataFrame({'Recommendation': msgToAdd}, columns=['Recommendation'], index=[0])
    
    if os.path.exists(absPathRecommFile):
        df = pd.read_csv(filePath)
        if msgToAdd not in list(df['Recommendation'].unique()):
            print('New Recommendation has been added')
            df = pd.concat([df,NewDf], ignore_index=True, sort=False)
        else:
            print('This recommendation is already present, hence not adding.')
    else:
        print('First Recommendation has been added')
        df = NewDf.copy()
    df.to_csv(absPathRecommFile, index = False)
    # -------------------------------------------------------------------------------------------- #



# -------------------------------------------------------------------------------------------------------------- #
# levprint('Let\'s do it', level=2, StartOrEnd=1)
# print_system_info()
# generate_highlighted_heading('Level Based Print')
# get_abs_path_from_relative_path('../../../', msg=True)

# print(platform.version())
# print(platform.platform()) #platform.system(), platform.processor(), platform.machine()
# print(platform.uname())
# mem.used /(1024.**3), mem.active /(1024.**3), mem.free /(1024.**3), mem.available /(1024.**3)

# def isIP_Valid(IP_str):
#     """
#     check if ip_str is a valid IP address
#     TODO: need a more extensive check
#     """
#     if (len(IP_str.split('.')) == 4) or (len(IP_str.split(':')) == 8):
#         return True
#     else:
#         return False
# # assert (is_valid('0'),is_valid(''),is_valid('84.214.142.44'), is_valid('2a02:587:c453:d900:84a6:f479:d7f5:3ee2')) == (False,False,True,True)
