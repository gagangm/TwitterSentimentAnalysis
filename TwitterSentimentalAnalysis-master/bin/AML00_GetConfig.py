## Config: Saving Method: Dict, Json, INI


'''
Description: 
    This file is to be used for initializing main execution file.
Function this file Contains:
    - 
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
# pip3 install ConfigParser
import configparser
import json,os,ast
from txt0_GeneralFunc import custom_ast_lit_eval
# from _editable_configuration_dict import config as config_dict

class ProcessConfigDict:
    
    def __init__(self, config_dict):
        self.conf_dict = config_dict
        
    @staticmethod
    def config_string_cleaner(content):
        '''
        Sometimes content are not read by ast.literal_eval hence this function sort some of 
        the error to make it work
        '''
        if isinstance(content, str):
            # print('>>>>>Cleaning the string')
            things_to_remove = ['\n']
            for ele in things_to_remove:
                content = str(content).replace(ele, ' ')
            return ' '.join([ele for ele in content.split(' ') if len(ele)>0])
        else:
            return content

    @staticmethod
    def check_dict_and_get_keys(variable):
        if isinstance(variable, dict):
            return True, list(variable.keys())
        else:
            return False, ProcessConfigDict.config_string_cleaner(variable)

    def generate_mod_dict(self):
        '''
        modifying the dictory so that it will be easy to work with json or ini 
        Input: takes multi level Dict
        Output: Single Level Dictionary where keys has been concatinated with '.' as seperator

        Sample:
        {
            'a': 1,
            'b': {
                'c': "[1,2,3]",
                'd': 'asd'
            },
            'e': 'qwerty'
        }

        to

        {
            'a': 1,
            'b.c': "[1,2,3]",
            'b.d': 'asd',
            'e': 'qwerty'
        }

        '''
        key_path, mconf_dict = [], {}
        c0 = self.check_dict_and_get_keys(self.conf_dict)
        if c0[0]:
            for k0 in c0[1]:
                key_path.append(k0)
                c1 = self.check_dict_and_get_keys(self.conf_dict[k0])
                if c1[0]:
                    for k1 in c1[1]:
                        key_path.append(k1)
                        c2 = self.check_dict_and_get_keys(self.conf_dict[k0][k1])
                        if c2[0]:
                            for k2 in c2[1]:
                                key_path.append(k2)
                                c3 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2])
                                if c3[0]:
                                    for k3 in c3[1]:
                                        key_path.append(k3)
                                        c4 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2][k3])
                                        if c4[0]:
                                            for k4 in c4[1]:
                                                key_path.append(k4)
                                                c5 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2][k3][k4])
                                                if c5[0]:
                                                    for k5 in c5[1]:
                                                        key_path.append(k5)
                                                        c6 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2][k3][k4][k5])
                                                        if c6[0]:
                                                            for k6 in c6[1]:
                                                                key_path.append(k6)
                                                                c7 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2][k3][k4][k5][k6])
                                                                if c7[0]:
                                                                    for k7 in c7[1]:
                                                                        key_path.append(k7)
                                                                        c8 = self.check_dict_and_get_keys(self.conf_dict[k0][k1][k2][k3][k4][k5][k6][k7])
                                                                        if c8[0]:
                                                                            msg = 'Dictionary contains More than 8 Levels. WHich is not supported Hence Raising Error'
                                                                            raise Exception(msg)
                                                                        else:
                                                                            print('.'.join(key_path))
                                                                            mconf_dict['.'.join(key_path)] = c8[1]
                                                                        key_path.pop()
                                                                else:
                                                                    mconf_dict['.'.join(key_path)] = c7[1]
                                                                key_path.pop()
                                                        else:
                                                            mconf_dict['.'.join(key_path)] = c6[1]
                                                        key_path.pop()
                                                else:
                                                    mconf_dict['.'.join(key_path)] = c5[1]
                                                key_path.pop()
                                        else:
                                            mconf_dict['.'.join(key_path)] = c4[1]
                                        key_path.pop()
                                else:
                                    mconf_dict['.'.join(key_path)] = c3[1]
                                key_path.pop()
                        else:
                            mconf_dict['.'.join(key_path)] = c2[1]
                        key_path.pop()
                else:
                    mconf_dict['.'.join(key_path)] = c1[1]
                key_path.pop()
        else:
            mconf_dict['.'.join(key_path)] = c0[1]
        
        ## Changing the case of keys so that it matched the config_ini
        new_mod_dict = {}
        for key in mconf_dict.keys():
            new_mod_dict[key.lower()] = mconf_dict[key]
            
        self.modified_config_dict = new_mod_dict
        # return mconf_dict
    
    
    def get_original_config_dict(self):
        return self.conf_dict
    
    def get_modified_config_dict(self):
        self.generate_mod_dict()
        return self.modified_config_dict
    
    def get_config_json(self):
        '''
        if you want to reformat and save the json 
        ## Validate the Json at https://jsonlint.com/
        '''
        print('https://jsonlint.com/')
        self.generate_mod_dict()
        #print(json.dumps(self.modified_config_dict))
        return json.dumps(self.modified_config_dict)
    
    def write_json_config(self, loc = '../config/configuration_json.json'):
        '''
        get the modified_config_dict and write a json config file 
        '''
        if os.path.exists('/'.join(loc.split('/')[:-1])):
            self.generate_mod_dict()
            json.dump(self.modified_config_dict, open(loc, 'w'))
        else:
            raise Exception('Provided path doesn\'t exist.')
    
    def write_ini_config(self, loc = '../config/configuration_ini.ini'):
        '''
        get the modified_config_dict and write a json config file 
        '''
        if os.path.exists('/'.join(loc.split('/')[:-1])):
            self.generate_mod_dict()

            parser = configparser.ConfigParser()
            parser.add_section('config')
            for key in self.modified_config_dict.keys():
                parser.set('config', key, self.modified_config_dict[key])
            parser.write(open(loc, 'w'))
        else:
            raise Exception('Provided path doesn\'t exist.')
        #for section in config.keys():
        #    parser.add_section(section)
        #    for key in config[section].keys():
        #        parser.set(section, key, str(config[section][key]))

# config_instance = ProcessConfigDict(config_dict)
# # config_instance.get_original_config_dict()
# # config_instance.get_modified_config_dict()
# # print(config_instance.get_config_json())
# config_instance.write_json_config()
# config_instance.write_ini_config()



class Configuration:
    '''
    Get a list containing the location of the config files (py/json/ini) read the content
    and convert the content to dictionary and make the content available to be accessed
    
    ## https://realpython.com/instance-class-and-static-methods-demystified/
    '''
    
    def __init__(self, config_file_paths_li=['../config/configuration_json.json'], 
                 raise_key_not_found_Error=True, try_using_ast = True):
        for ele in config_file_paths_li:
            if os.path.exists(ele) is False:
                raise exception('THIS PATH "{}" DOES\'T EXIST'.format(ele))
        self.config_file_paths_li = config_file_paths_li
        self.raise_key_not_found_Error = raise_key_not_found_Error
        self.try_using_ast = try_using_ast
        # self._config = config # set it to conf
    
    
    @staticmethod
    def _load_ini_file(path):
        '''  
        load single ini config file 
        ----> dictionary is not well structure when the file is having multiple nested dict
        '''
        temp_dict = {}
        config = configparser.ConfigParser()
        config.read(path)
        for section in config.sections():
            temp_dict[section] = dict(config[section])

        return temp_dict['config']
    
    
    @staticmethod
    def _load_json_file(path):
        '''  load single json config file
        '''
        return json.load(open(path))
    
    """
    @staticmethod
    def load_dict_file(path):
        '''  load single python config file containing dictionary
        ## Link: https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html
        # from config.configuration_dict import config
        
        ##https://stackoverflow.com/questions/2220699/whats-the-difference-between-eval-exec-and-compile
        '''
        with open(path) as file:
            exec(compile(file.read(), '<string>', 'exec'))
            return config
    """
    
    
    def load_config_files(self):
        '''
        Loading the files provided in the file path list; raise error if some issue is there
        '''
        all_config_dict = {}
        
        print('\nNote:\tWhen multiple configuration files will be provided and they might have some duplicate keys\
        \n\tthen then the priority to the config will be provided to the one configwhich was provided\
        \n\tat the start of the "config_file_paths_li".\n')
        
        for path in self.config_file_paths_li[::-1]: ## reversing the list so that the first element can be given more priority
            if os.path.exists(path):
                file_type = path.split('.')[-1]
                # if 'py' == file_type:
                #     temp_conf = load_dict_file(path)
                if 'json' == file_type:
                    temp_conf = Configuration._load_json_file(path)
                elif 'ini' == file_type:
                    temp_conf = Configuration._load_ini_file(path)
                else:
                    raise Exception('Config file format not defined')
                    
                if isinstance(temp_conf, dict):
                    for key in temp_conf.keys():
                        all_config_dict[key] = temp_conf[key]
            else:
                print('File Path Doesn\'t exist')
        
        self._config = all_config_dict
        
        
    def get_config(self, *which_property):
        '''
        if raise_key_not_found_Error == False i.e. key is not present in config then returns None
        '''
        #self._load_config_files()
        config_prop = '.'.join([ ele.lower() for ele in which_property])
        
        if config_prop in list(self._config.keys()):
            if self.try_using_ast:
                try:
                    custom_ast_lit_eval(self._config[config_prop])
                except SyntaxError:
                    return self._config[config_prop]
                    pass#print('Syntax Error. Therefore can\'t return the value')
            else:
                return self._config[config_prop]
        else:
            msg = 'Key path Exception: NO property is present at this path: \n\t\t{}'.format('> '+config_prop)
            print(msg)
            if self.raise_key_not_found_Error:
                raise Exception(msg)
            else:
                return None
        # ## When It was initailly nested dictionaries
        # which_property, traversed_path = self._config, []
        # for prop in list(which_property):
        #     traversed_path.append(prop)
        #     if prop in prop_to_return.keys():
        #         prop_to_return = prop_to_return[prop]
        #     else:
        #         msg = 'Key path Exception: NO property is present at this path: \n\t\t{}'.format('> '+'\\'.join(traversed_path))
        #         print(msg)
        #         if self.raise_key_not_found_Error:
        #             raise Exception(msg)
        #         else:
        #             return None
        # return prop_to_return
        
    def get_keys_in_config(self):
        '''
        Return all the keys that are present in the config file
        '''
        #self._load_config_files()
        return list(self._config.keys())


# conf = Configuration(raise_key_not_found_Error=True)
# print(conf.get_keys_in_config())
# conf.get_config('paths', 'raw_training_data_file')



#-------------------------------------------------------

# from AML00_GetConfig import ProcessConfigDict, Configuration
# from _editable_configuration_dict import config as config_dict

# stillworkingwithconfig = True
# if stillworkingwithconfig:
#     config_instance = ProcessConfigDict(config_dict)
#     # config_instance.get_original_config_dict()
#     # config_instance.get_modified_config_dict()
#     # print(config_instance.get_config_json())
#     config_instance.write_json_config()
#     config_instance.write_ini_config()

# conf = Configuration(['../config/configuration_json.json'], True,True)
# conf.load_config_files()
# #print(conf.get_keys_in_config())
# conf.get_config('paths', 'raw_training_data_file')






# def GetConfig(X, Y, msg = False):
#     '''
#     To get config value from config.ini Basic or Advance, config.ini is expected to have 2D structure
#     config_li <-- is a  global list containing multiple config files.
#     No Error is raised when X == Config
#     msg == True: print execution messages
#     '''
#     val = []
# #     print(self.config_li)
# #     for conf in self.config_li:
# #     print(config_li)
#     for conf in config_li:
#         try:
#             val.append(conf[X][Y])
#             if msg is True: print('Using', conf['Config']['Type'], 'config, specificallly the pair ', X, Y)
#         except:
# #             print('NoVal')
#             pass
#     if X != 'Config':
#         if len(val) == 0: 
#             raise Exception('configuration value NOT present in any file.')
#         elif len(val) > 1: 
#             raise Exception('configuration value present in MULTIPLE file.')
#         else:
#             return val[0] ## first element in the list i.e. the value
#     return val ## used Just to return value when X=Config is used 

# # if __name__ == '__main':
    
#     # config_bas = configparser.ConfigParser()
#     # config_adv = configparser.ConfigParser()
#     # try:
#     #     config_bas.read('../config/AAT_Config(basic).ini')
#     #     config_adv.read('../config/AAT_Config(advance).ini')
#     #     config_li = [config_adv, config_bas]
#     #     print('Successfully read the configuration files :', GetConfig('Config', 'Type'))
#     # except:
#     #     print('Unable to read config files. Hence Exiting.')
#     #     sys.exit(1)