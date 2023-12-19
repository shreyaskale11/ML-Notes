# Autocorrect 

What is autocorrect?
- Building the model
- Minimum edit distance
- Minimum edit distance algorithm


To implement autocorrect, you have to follow these steps: 
- Identify a misspelled word
- Find strings n edit distance away: (these could be random strings)
- Filter candidates: (keep only the real words from the previous steps)
- Calculate word probabilities: (choose the word that is most likely to occur in that context)

Minimum edit distance algorithm II

There are three equations: 

D[i,j] = D[i-1, j] + del_cost: this indicates you want to populate the current cell (i,j) by using the cost in the cell found directly above. 
D[i,j] = D[i, j-1] + ins_cost: this indicates you want to populate the current cell (i,j) by using the cost in the cell found directly to its left. 
D[i,j] = D[i-1, j-1] + rep_cost: the rep cost can be 2 or 0 depending if you are going to actually replace it or not. 

At every time step you check the three possible paths where you can come from and you select the least expensive one. 


To summarize, you have seen the levenshtein distance which specifies the cost per operation. If you need to reconstruct the path of how you got from one string to the other, you can use a backtrace. You should keep a simple pointer in each cell letting you know where you came from to get there. So you know the path taken across the table from the top left corner, to the bottom right corner. You can then reconstruct it. 

This method for computation instead of brute force is a technique known as dynamic programming. You first solve the smallest subproblem first and then reusing that result you solve the next biggest subproblem, saving that result, reusing it again, and so on. This is exactly what you did by populating each cell from the top right to the bottom left. It’s a well-known technique in computer science! 
