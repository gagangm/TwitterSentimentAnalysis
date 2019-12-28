# ---------------------------------------------------------------------------------------------------------- #
#                                     <<<<[[[[  CONFIGURATION  ]]]]>>>>                                      #
# ---------------------------------------------------------------------------------------------------------- #

## Configuration File written as dictionary in python 
## this will be ingested by GetConfiguration.py where that function in 
## first run will convert this configuration dict to JSON or INI and 
## Later that JSON or INI will be used for the configuration

config = {
    'paths': {
        'RAW_training_data_file': '/home/ubuntu/Documents/JobWorkSpace/TwitterSentimentAnalysis-RickyTim/TwitterSentimentalAnalysis/data/inputs/training.1600000.processed.noemoticon.csv',
        'RAW_test_data_file': '/home/ubuntu/Documents/JobWorkSpace/TwitterSentimentAnalysis-RickyTim/TwitterSentimentalAnalysis/data/inputs/testdata.manual.2009.06.14.csv',
        'cleaned_train_data_file': '/home/ubuntu/Documents/JobWorkSpace/TwitterSentimentAnalysis-RickyTim/TwitterSentimentalAnalysis/data/inputs/cleaned_train_data.tsv',
        'cleaned_test_data_file': '/home/ubuntu/Documents/JobWorkSpace/TwitterSentimentAnalysis-RickyTim/TwitterSentimentalAnalysis/data/inputs/cleaned_test_data.tsv',
    },
    'data_preparation': {
        'All_Features': '''['sentiment_class','id','date','query_string','user','text']''', ## will be used to define/give names to the columns
        'FeatureToDrop': '''['id','date','query_string','user']''',
        'frac_in_sample_to_take': '0.025',
        'text_cleaning_steps' : '''['accented_char_removal', 'remove_attherate_mentions', 'remove_urls', 'html_stripping', 'contraction_expansion', 'text_lemmatization', 'special_char_removal', 'remove_digits', 'remove_whitespace_character', 'remove_extra_whitespaces', 'remove_commas_bw_digits', 'remove_extra_meaningless_newlines', 'text_lower_case']''',
        'temp': 'temp'
    },
    'model_preparation': {
        'train-val-test_split': '80-10-10',
        'tasks_to_perform': '''['train', 'GridSearch', 'Evaluate']''',
        'grid_search_combinations': {
            'comb0': '''{
                'which_vectorizer': {
                    'mod': 'CountVectorizer',
                    'pc': 'param_config_0',
                },
                'which_classifier': {
                    'mod': 'LogisticRegression',
                    'pc': 'param_config_0',
                }, 
                'params': {
                    'which_vectorizer__max_features': [10000], 
                    'which_vectorizer__stop_words': [None, 'english'], 
                    'which_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], 
                    'which_classifier__fit_intercept': [True] 
                }
            }''',
            'comb1': '''{
                'which_vectorizer': {
                    'mod': 'TfidfVectorizer',
                    'pc': 'param_config_0',
                },
                'which_classifier': {
                    'mod': 'LogisticRegression',
                    'pc': 'param_config_0',
                }, 
                'params': {
                    'which_vectorizer__max_features': [10000], 
                    'which_vectorizer__stop_words': [None, 'english'], 
                    'which_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], 
                    'which_classifier__fit_intercept': [True] 
                }
            }''',
            'comb2': '''{
                'which_vectorizer': {
                    'mod': 'HashingVectorizer',
                    'pc': 'param_config_0',
                },
                'which_classifier': {
                    'mod': 'LogisticRegression',
                    'pc': 'param_config_0',
                }, 
                'params': {
                    'which_vectorizer__max_features': [10000], 
                    'which_vectorizer__stop_words': [None, 'english'], 
                    'which_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], 
                    'which_classifier__fit_intercept': [True] 
                }
            }''',
            'comb3': '''{
                'which_vectorizer': {
                    'mod': 'TfidfVectorizer',
                    'pc': 'param_config_0',
                },
                'which_classifier': {
                    'mod': 'XGBClassifier',
                    'pc': 'param_config_0',
                }, 
                'params': {
                    'which_vectorizer__max_features': [10000], 
                    'which_vectorizer__stop_words': [None, 'english'], 
                    'which_vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], 
                }
            }''',
        },
        
        'models_to_train': {
            'comb0': '''{'which_vectorizer': 'CountVectorizer', 'which_classifier': 'LogisticRegression'} ''',
            'comb1': '''{'which_vectorizer': 'CountVectorizer', 'which_classifier': 'LogisticRegression'} ''',
            'comb2': '''{'which_vectorizer': 'CountVectorizer', 'which_classifier': 'LogisticRegression'} ''',
        },
        
        'ml_mod_params': {
            'CountVectorizer': {
                'param_config_0': '''{}''',
                'param_config_1': '''{
                    'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None, 
                    'lowercase': True, 'preprocessor': None, 'tokenizer': None, 'stop_words': None
                }'''
            },
            'TfidfVectorizer': {
                'param_config_0': '''{}''',
                'param_config_1': '''{
                    'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None, 
                    'lowercase': True, 'preprocessor': None, 'tokenizer': None, 'stop_words': None
                }'''
            },
            'HashingVectorizer': {
                'param_config_0': '''{}''',
                'param_config_1': '''{
                    'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None, 
                    'lowercase': True, 'preprocessor': None, 'tokenizer': None, 'stop_words': None
                }'''
            },
            'LogisticRegression': {
                'param_config_0': '''{}'''
            },
            'LinearSVC': {
                'param_config_0': '''{}'''
            },
            'MultinomialNB': {
                'param_config_0': '''{}'''
            },
            'BernoulliNB': {
                'param_config_0': '''{}'''
            },
            'RidgeClassifier': {
                'param_config_0': '''{}'''
            },
            'AdaBoostClassifier': {
                'param_config_0': '''{}'''
            },
            'Perceptron': {
                'param_config_0': '''{}'''
            },
            'NearestCentroid': {
                'param_config_0': '''{}'''
            },
            'XGBClassifier': {
                'param_config_0': '''{}'''
            },
        }
    }
    
 
    
    
#     'training_file_cleaned': 'CleanedTrainingData.csv',
#     'test_file_cleaned': 'CleanedTestData.csv',
#     'term_frequency_file': "TermFrequency.csv",
#     'Train:Val:TestSplit': '98:1:1'

#     'DimensionalityTransformationAlgo':['PCA', 'ICA'],

#     'bq_env': {'edit_query': 'Yes', 
#                'bq_query_template_file': 'QueryTemplateClustering.txt', 
#                'sid': ['1071'] ,
#                'date': ['010218'],
#                'MaxNoObsToGet': '1000000'},

#     'Trial':["ABC","DEF","GHI"],
}







