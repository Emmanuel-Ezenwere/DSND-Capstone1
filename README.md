# The Udacity Data Science Capstone 1 - Project Readme #

The motivation for this project is to apply data science methods to the Stack Overflow Survey 2023 to enable us derive insight from the data and answer a few interesting questions from the data. 

Files in this repository : 

- StackOverflow-data
  - data-dir
     - stack-overflow-developer-survey-2023
         - README_2023.txt
         - so_survey_2023.pdf
         - survey_results_public.csv
         - survey_results_public.csv

- Data
- stackoverflow-survey.ipynb
- stackoverflow-survey.py
- requirements.txt
- README.md

  
A requirements.txt file has been included in the repository for installation of required libraries.


## List of libraries used: 
- pandas
- numpy
- matplotlib
- pprint
- sklearn
- seaborn


## Summary of the results of the analysis; 

------------------------------------------------------
Top Earning Respondents - Case Study
------------------------------------------------------


ConvertedCompYearly:
350000.0 -------- 10.0%

Age:
35-44 -------- 37.8%

WorkExp:
10.0 -------- 8.25%

OrgSize:
10,000+ -------- 40.98%

RemoteWork:
Remote -------- 49.45%

EdLevel:
B.Sc/B.Eng/B.A -------- 49.3%

LearnCode:
Other online resources (e.g., videos, blogs, forum) -------- 5.21%

LearnCodeOnline:
Formal documentation provided by the owner of the tech;Blogs with tips and tricks;Written Tutorials;Stack Overflow -------- 3.22%

LearnCodeCoursesCert:
Coursera -------- 10.31%

YearsCode:
20 -------- 9.2%

YearsCodePro:
10 -------- 8.95%

Currency:
US$ -------- 78.6%

DevType:
Developer, back-end -------- 23.26%

LanguageHaveWorkedWith:
C++;Python -------- 0.9%

Industry:
IT -------- 53.65%

Country:
United States of America -------- 76.8%


Q3) Who are the Least X earning respondents : 




------------------------------------------------------
Least earning Respondents
------------------------------------------------------


Q4) What is common amongst the Least X earning respondents : 


ConvertedCompYearly:
1.0 -------- 1.8%
1212.0 -------- 1.2%

Age:
25-34 -------- 47.3%

WorkExp:
2.0 -------- 13.42%

OrgSize:
20-99 -------- 24.4%

RemoteWork:
Remote -------- 44.49%

EdLevel:
B.Sc/B.Eng/B.A -------- 51.2%

LearnCode:
Other online resources (e.g., videos, blogs, forum) -------- 4.81%

LearnCodeOnline:
Formal documentation provided by the owner of the tech;Blogs with tips and tricks;How-to videos;Written Tutorials;Stack Overflow -------- 1.18%

LearnCodeCoursesCert:
Udemy -------- 22.6%

YearsCode:
6 -------- 10.36%

YearsCodePro:
2 -------- 14.62%

Currency:
US$ -------- 8.8%

DevType:
Developer, full-stack -------- 38.13%

LanguageHaveWorkedWith:
HTML/CSS;JavaScript;TypeScript -------- 3.62%

Industry:
IT -------- 55.38%

Country:
India -------- 7.7%

------------------------------------------------------
Top Young Earning Respondents
------------------------------------------------------
Who are the top young earners, (Age: 25-34') and what do they have in common : 

ConvertedCompYearly:
250000.0 -------- 11.0%

Age:
25-34 -------- 100.0%

WorkExp:
10.0 -------- 14.62%

OrgSize:
10,000+ -------- 37.14%

RemoteWork:
Remote -------- 50.25%

EdLevel:
B.Sc/B.Eng/B.A -------- 61.1%

LearnCode:
On the job training;Other online resources (e.g., videos, blogs, forum);School (i.e., University, College, etc) -------- 5.42%

LearnCodeOnline:
Formal documentation provided by the owner of the tech;Blogs with tips and tricks;Written Tutorials;Click to write Choice 20;Stack Overflow -------- 2.96%

LearnCodeCoursesCert:
Udemy -------- 10.8%

YearsCode:
10 -------- 11.11%

YearsCodePro:
10 -------- 13.78%

Currency:
US$ -------- 80.6%

DevType:
Developer, back-end -------- 26.15%

LanguageHaveWorkedWith:
HTML/CSS;JavaScript;TypeScript -------- 1.2%

Industry:
IT -------- 51.19%

Country:
United States of America -------- 78.8%



------------------------------------------------------
Abnormal Earnings - Case Study
------------------------------------------------------

- Appearance of an outlier / abnormal data : $74,351,432 -- Respondent : 53268
- Age : 18-24
- CompTotal : 100000000.0
- Country : Canada
- Currency : CAD	Canadian dollar
- DevType : Developer, full-stack
- EdLevel : Doctorate
- WorkExp : 7.0
- PurchaseInfluence : I have a great deal of influence
- RemoteWork : Hybrid
- ResponseId : 53269
- YearsCode : 3
- YearsCodePro : <1
- DatabaseWantToWorkWith : nan



------------------------------------------------------
No Degree Earnings - Case Study
------------------------------------------------------

ConvertedCompYearly:
150000.0 -------- 6.0%

Age:
35-44 -------- 35.5%

WorkExp:
15.0 -------- 6.76%

OrgSize:
100-499 -------- 19.72%

RemoteWork:
Remote -------- 67.61%

EdLevel:
DropOut -------- 100.0%

LearnCode:
Books / Physical media;Other online resources (e.g., videos, blogs, forum) -------- 6.93%

LearnCodeOnline:
Formal documentation provided by the owner of the tech;Blogs with tips and tricks;Written Tutorials;Stack Overflow -------- 2.42%

LearnCodeCoursesCert:
Udemy -------- 13.52%

YearsCode:
20 -------- 7.0%

YearsCodePro:
10 -------- 6.61%

Currency:
US$ -------- 75.0%

DevType:
Developer, full-stack -------- 36.39%

LanguageHaveWorkedWith:
C#;HTML/CSS;JavaScript;PowerShell;SQL;TypeScript -------- 1.2%

Industry:
IT -------- 46.13%

Country:
United States of America -------- 72.5%



------------------------------------------------------
Predicting Annual Compensation from Work Experience - Model Evaluation
------------------------------------------------------

The r-squared score and mean squared error value for the model using only quant variables are 0.4 and 2112703815.8847256, respectively on 74 values.




## Acknowledgements : 
- Stack Overflow - Dataset
- Udacity - Coursework
  









