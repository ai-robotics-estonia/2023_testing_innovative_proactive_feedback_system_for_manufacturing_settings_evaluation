# Proactive Feedback System for Manufacturing Settings Evaluation
## Summary

| Company name | [Krah Pipes OÜ](https://www.krah-pipes.ee/) |
| :--- | :--- |
| Development Team Lead Name | [Vahur Kotkas](https://taltech.ee/en/department-of-software-science/cooperation/applied-artificial-intelligence-group) |
| Development Team Lead e-mail | [vahur.kotkas@taltech.ee](mailto:vahur.kotkas@taltech.ee) |
| Development Team Member | [Priit Järv](https://taltech.ee/en/department-of-software-science/cooperation/applied-artificial-intelligence-group) |
| Development Team Member | [Kristiina Kindel](https://taltech.ee/en/department-of-software-science/cooperation/applied-artificial-intelligence-group) 

## Objectives of the Demonstration Project
The objective of the project was to create and validate a proactive feedback system, which evaluates the process setting parameters and estimates the geometric dimensions that can be achieved on the final product (compliance with the given dimensions) based on these. The feedback system must warn the operator when the process setting parameters seem to be inappropriate for a target set. The idea was to validate the possible correlations between process inputs and end product dimensions.

## Activities and results of demonstration project
*Activities - The challenges addressed*
-  *a) Collect, evaluate and clean the data from the manufacturing process*
-  *b) Analyse the data for possible dependencies/correlations detection*
-  *c) Evaluate most promising ML algorithms for learning the relations between process setting parameters and their match to the projected*
-  *d) Test the learnt model on the real data*

*Data sources (which data was used for the technological solution)*
-  *a) Data from pipes’ production line sensors and automated quality control station was used.*

*Description and justifictaion of used AI technology*
-  *a) Statistical analysis of production data, to identify correlation to end product quality*
-  *b) Simple models like linear regression, to provide a baseline*
-  *c) Ensemble models, because they have demonstrated high performance in practical settings with limited data*
-  *d) Deep sequential neural models, because they are considered the current state of the art*
-  *e) XAI methods (SHAP) to provide explanations of product defects*

*Results of testing and validating technological solution*
-  *a) It was validated that correlations between process inputs and end product dimensions are present. Any illogical relations were not detected.
Although the accuracy of predictions was lower than anticipated, we also determined the possible reasons for that (measurement accuracy, data cleaning and aggregation methodology).*

*Technical architecture (presented graphically, where can also be seen how the technical solution integrates with the existing system)*
-  *a) The main goal of the demoproject was to perform a data analysis and to train and validate a model for quality parameters prediction, hence there is no technical architecture of the technological solution to present.*

*Potential areas of use of technical solution*
-  *a) pipe production*
-  *b) manufacturing industry*

*Lessons learned (i.e. assessment whether the technological solution actually solved the initial challenge)*
-  *a) Collected data reliability is one of the key issues. The project pointed out that improvements are needed in final products measuring process.*
-  *b) The amount of analysed data for the basis of ML model is of significant importance. Due to relatively short training period this amount was limited. Longer training period contributes to the accuracy of the ML model.*
-  *c) Adding data points to the process contributes also to the accuracy of the model.*

### Description of User Interface
-  *a) There is no user interface developed. The solution needs to be integrated into the production station. For execution of the code one needs python3*
