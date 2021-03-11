[TOC]



# EuLo





## Idea

The basic assumption for trying to use Artificial intelligence in order (to try) to predict lottery numbers is that either:

1. There should be a pattern within the draw history that can repeat itself in the future
2. We may be able to use statistics and probability theory to determine numbers that are more likely to be drawn.
   1. The idea behind this is that we know that when rolling six-sided dices, if we just got the 6 side two times in a row, it appears less likely that the 6 will be the result of the next roll. Thus, we estimate the probability of it appearing should be lower than the probability of other sides.
      [However, probability theory tells us that such assumptions are incorrect in theory.]



The approach I tried to solve in this project is the probability-oriented one.



## Approach







The approach I took is to generate features 

### Goals

The main goal of this program is not to predict the exact symbols / numbers that should be drawn in the next draw.

- nor should it predict symbols for every single draw.



Instead, one of the goals is to give a probability score for the prediction. The idea behind this is that sometimes we get results that are standard, and sometimes the results deviate from what they should theoretically be.

 





## Next steps



- Feature engineering: Generate meta-features (i.e. features of features).
- Problem:
  Very rare and rare events can occur, even though their probabilities are small. When visualizing things on a scatter plot, it can become messy.
  - Visualization:
    - Plot using shades/color maps
    - Or: Make scatter plots and map the number of cases at each point to the size of the point.
    - Plot using other types of graphics
  - Develop/Use a model that can capture the distribution of features in the features space, and make probabilistic predictions based on that.






## contribute



Credits: Jeffrey Mvutu Mabilama



## NOTICE



**THE SOFTWARE IS PROVIDED** "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

