
import glob, os 
import dill
# print("CSV Files present in the directory:\n", '\n'.join(glob.glob(CheckPoint_Dir + '*.{}'.format('CheckPt')))) #../input

def WorkspaceBasedCheckPt(CheckPtPosition = 0, AccessRecentOrNewWorkSpaceImage = False, Force = {'access':False, 'task':'load'}, CheckPoint_Dir = "NotebookCheckpoints/"):
    if not os.path.exists(CheckPoint_Dir):
        os.makedirs(CheckPoint_Dir)
        file= open((CheckPoint_Dir + '0__CheckPt.db'),"w+")
        file.close()
    LastCheckPt = max([int(CheckPoint.split('/')[len(CheckPoint.split('/')) -1].split('__')[0]) for CheckPoint in glob.glob(CheckPoint_Dir + '*{}'.format('CheckPt.db'))])
    
    ## Force can be used when proceding in non serialized manner
    if Force['access']: ## could have been compressed with chunk below using boolean algebra to make code more smaller
        AccessRecentOrNewWorkSpaceImage = False
        if Force['task'] == 'load':
            print("Checkpoint ", CheckPtPosition, "is to be loaded")
            if os.path.exists(CheckPoint_Dir + str(CheckPtPosition) + '__CheckPt.db'):
                dill.load_session(CheckPoint_Dir + str(CheckPtPosition) + '__CheckPt.db') # To restore a session
            else:
                print("This checkpoint doesn't exist, hence won't be loaded.")
        elif Force['task'] == 'save':
            print("Checkpoint ", CheckPtPosition, "is to be Saved")
            dill.dump_session(CheckPoint_Dir + str(CheckPtPosition) + '__CheckPt.db') # To Save a session
        return "Force used to {} workspace checkpoint_{}.".format(Force['task'], CheckPtPosition) ## exit here only
    
    
    ## This Code below is used to handle the actions on returning the value to run/not run the cell
    ### = is set so that the current check point cell code are able to run
    if ((AccessRecentOrNewWorkSpaceImage == False) and (CheckPtPosition <= LastCheckPt)):
        print('Most Recent Checkpoint is : {} \nHence, cells won\'t be running content untill most recent checkpoint is crossed.'.format(LastCheckPt))
        return False
    elif ((AccessRecentOrNewWorkSpaceImage == False) and (CheckPtPosition == (LastCheckPt +1))):
        print('Running this cell')
        return True
    elif ((AccessRecentOrNewWorkSpaceImage == False) and (CheckPtPosition > (LastCheckPt +1))):
        print("You have skipped over a checkpoint. Still running this cell")
        return True
    
    ## This Code below is used to handle the actions on saving/loading the workspace images
    if (AccessRecentOrNewWorkSpaceImage and (CheckPtPosition == 0)):
        print("Initial Phase, hence not saving workspace.")
    elif (AccessRecentOrNewWorkSpaceImage and (LastCheckPt > CheckPtPosition)):
        print("This is not the most recent checkpoint, hence not loading it. [Use Force to force load a checkpoint]")
    elif (AccessRecentOrNewWorkSpaceImage and (LastCheckPt == CheckPtPosition)):
        dill.load_session(CheckPoint_Dir + str(CheckPtPosition) + '__CheckPt.db') # To restore a session
        print("This is the most recent checkpoint, hence loading it.")
    elif (AccessRecentOrNewWorkSpaceImage and ((LastCheckPt +1) == CheckPtPosition)):
        dill.dump_session(CheckPoint_Dir + str(CheckPtPosition) + '__CheckPt.db') # To Save a session
        print("Congrats, on reaching a new checkpoint, saving it.")
    elif (AccessRecentOrNewWorkSpaceImage and ((LastCheckPt +1) < CheckPtPosition)):
        print("You have skipped over a checkpoint. Hence not Saving anything.")
    
    
# https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook/29665452

WorkspaceBasedCheckPt(1, True)
ChPt = 2
if WorkspaceBasedCheckPt(ChPt):
    print('to run if exist')
    
#WorkspaceBasedCheckPt(103, False, {'access':True, 'task':'save'})

%reset
%whos
