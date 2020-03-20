# RetrosyntheticRNN
Retrosynthetic Analysis by using RNN
This code was written for Merck-Kesci organized Game to predict retrosynthetic reactions.
Part of the Codes referred from the baseline comtributed by the organizer or other contributers, 
while the core concept of the Algorithm is shown as follows:

#################################
##### Our main point is that the computer learns slow,
### We teach them something based on our chemical knowledge.
##### For example the solvent is protic or not,
### the compound is saturation or not
##### This part we called the knowledge-based part.
#################################

This knowledge-based part is merged into embedding layer, and that can speed up to increase acc in the first epoch.
However, it seems the acc will decrease with the training data increase.


