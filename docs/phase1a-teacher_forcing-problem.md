Problem Statement                                                                                                                                       
                                                                                                                                                          
  Context: We are optimizing soft prompt embeddings (K vectors of dimension d) against a frozen LLM to reconstruct specific target text sequences. The    
  goal is to find embeddings c* such that greedy decoding from c* produces the target tokens exactly.                                                     
                                                                                                                                                          
  Optimization objective:                                                                                                                                 
  c* = argmin_c [ CrossEntropy(target | c) ]                                                                                                              
                                                                                                                                                          
  The Problem: Optimization achieves 200-300x lower cross-entropy loss than random embeddings, yet greedy decoding produces poor reconstructions—typically
   repetitive loops or scrambled tokens that share vocabulary with the target but lack correct structure.                                                 
                                                                                                                                                          
  Root cause hypothesis: The cross-entropy loss uses teacher forcing (each position conditioned on ground-truth previous tokens), but inference is        
  autoregressive (each position conditioned on model's own outputs). The optimized embeddings minimize loss under teacher forcing but don't properly      
  constrain the autoregressive generation trajectory.                                                                                                     
                                                                                                                                                          
  Observed failure modes:                                                                                                                                 
  - Repetition loops: "London London London..."                                                                                                           
  - Token scrambling: "ATSC" → "SCSCATATSCATAT"                                                                                                           
  - Partial capture with drift: captures key words but wrong order/structure                                                                              
                                                                                                                                                          
  Constraints:                                                                                                                                            
  - Generator (7B parameter LLM) must remain frozen                                                                                                       
  - Soft prompts are the only learnable parameters                                                                                                        
  - Target sequences are short (2-50 tokens typically)                                                                                                    
  - Must work at inference time with standard greedy or beam decoding                                                                                     
                                                                                                                                                          
  Question: What modifications to the optimization objective, training procedure, or inference strategy would close the teacher-forcing/autoregressive gap
   and achieve reliable exact reconstruction?