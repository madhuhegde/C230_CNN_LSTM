import copy

class_labels = {"Preparation":0, "CalotTriangleDissection":1, "ClippingCutting":2, 
           "GallbladderDissection":3, "GallbladderPackaging":4, "CleaningCoagulation":5, "GallbladderRetraction":6}
           
surgical_phase = {'curr_duration':0, 'min_duration':4, 'max_duration':10000, 'min_probability':0.1} 
           
class surgical_flow():
 
  def __init__(self):
    self.history = "Preparation"
    self.curr_label = "Preparation"
    self.early_phase_count =0
    self.late_phase_count = 0
    self.early_phases = ["Preparation", "CalotTriangleDissection"]
    self.late_phases = ["GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
    self.late_phase = False
    self.early_phase = False  
    self.surgical_phases = {}  
    for label in class_labels:
       self.surgical_phases[label] = copy.deepcopy(surgical_phase)
               
    return
    
    
  def set_curr_phase(self, label): 
     if (self.history == label): 
        if(self.curr_label != label):
                self.curr_label = label
                
     self.history = label  
        
     return(self.curr_label)
     
  def reset_early_late_phase(self):
     self.late_phase = False   
     self.early_phase = False
     self.early_phase_count =0
     self.late_phase_count = 0
     print("Reset count\n")
     return
     
  def  set_early_late_phase(self, label):
      if(label == "CalotTriangleDissection"):
         self.early_phase_count += 1
         
      if(label == "GallbladderDissection"):
         self.late_phase_count += 1   
         
      if(self.early_phase_count > 20):
           self.early_phase = True
           
      if(self.early_phase_count > 30) and (self.late_phase_count > 20):
           self.late_phase = True
      return   
      
  def predict_phase(self, label, second_label, label_gt):
      
      curr_label =  self.set_curr_phase(label)   
      pred_label = curr_label
      
      if (self.late_phase):
        if (curr_label not in self.late_phases):
          if(second_label in self.late_phases):
             pred_label = second_label
             print(label, curr_label, second_label, label_gt)
      else:
         if(self.early_phase == False) and (curr_label not in self.early_phases):
           if(second_label in self.early_phases):
             pred_label = second_label
             print(label, curr_label, second_label, label_gt)   
                 
      
      self.set_early_late_phase(pred_label)      
      
      return(pred_label)
      
      
  def not_used(self):    
      if(self.late_phase):
        if(self.early_phase):
           if (curr_label in self.late_phases):
              pred_label = curr_label
           else:
               pred_label = second_label
               
        else:
           pred_label = curr_label      
         
      else:
        if(self.early_phase):
           pred_label = curr_label
          
        else:
           if (curr_label in self.early_phases):
              pred_label = curr_label
           else:
               pred_label = second_label   
               
      self.set_early_late_phase(pred_label)        
               
      return(pred_label) 
               
                 
         
     #self.surgical_phases[label].curr_duration += 1      
         



#if __name__==__main__:
  
  
  
