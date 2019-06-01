import copy

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}
           
transition_matrix = {}           
surgical_phase = {'next_labels':[], 'curr_duration':0, 'min_duration':4, 'max_duration':10000, 'min_probability':0.1} 

def allowed_transition(history, new_label, prev_label, timeout_started):
    
  if ((history[0] == new_label) and (history[1]==new_label)): # and (prev_label=new_label)):
     return(True)
  
  allowed_labels_1 = transition_matrix[prev_label]['next_labels']
  allowed_labels_2 = transition_matrix[history[0]]['next_labels']
  allowed_labels_3 = transition_matrix[history[1]]['next_labels']
    
  if(timeout_started):
    if((new_label in allowed_labels_2) or (new_label in allowed_labels_3)): 
                                         #or (new_label in allowed_labels_3))):
      return(True)
     
  return(False)
  
def initialize_trans_matrix():

  
 

  for label in class_labels:
    transition_matrix[label] = copy.deepcopy(surgical_phase)

  #initialize surgical phases
  transition_matrix["Preparation"]["next_labels"] = ["CalotTriangleDissection"]
  transition_matrix["CalotTriangleDissection"]["next_labels"] = ["ClippingCutting"]
  transition_matrix["ClippingCutting"]["next_labels"] = ["GallbladderDissection"]
  transition_matrix["GallbladderDissection"]["next_labels"] = ["CleaningCoagulation","GallbladderPackaging"]
  transition_matrix["GallbladderPackaging"]["next_labels"] = ["CleaningCoagulation", "GallbladderRetraction"]
  transition_matrix["CleaningCoagulation"]["next_labels"] = ["GallbladderPackaging", "GallbladderRetraction"]
  transition_matrix["CalotTriangleDissection"]["min_duration"] = 10
  transition_matrix["GallbladderDissection"]["min_duration"] = 10
  transition_matrix["GallbladderRetraction"]["min_duration"] = 4
  
  return
  
  
def predict_next_label(new_labels):
  pred_labels = list()
  prev_label = "Preparation"
  
  # Use two level deep history
  label_history = ["Preparation", "Preparation"]
  for label in new_labels:
    
    #print(transition_matrix[prev_label]['curr_duration'])
    timeout_started= (transition_matrix[prev_label]['curr_duration'] > transition_matrix[prev_label]['min_duration'])
   
    if(allowed_transition(label_history,label, prev_label, timeout_started)):
        prev_label  = label
    
    transition_matrix[prev_label]['curr_duration'] = transition_matrix[prev_label]['curr_duration'] + 1
    pred_labels.append(prev_label)
      
    
    label_history[1] = label_history[0]
    label_history[0] = label    
    
          
  return(pred_labels)    
  
  
