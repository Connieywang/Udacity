# FordGoBike Data Exploration


## Dataset

The data consists of 94802 individual rides made in Ford GoBike covering the greater San Francisco Bay area in January. The dataset has 16 features, such as duration_sec,start_station_id, user_type,etc. It can be found in this website: https://s3.amazonaws.com/fordgobike-data/index.html.

## Summary of Findings

In the exploration, I have to approach different perspective to analyze the features of majority GoBike users. I found out that the ratio of male and female users is about 3:1 and the ratio of subscriber and customer is about 8:1. Thus, we know male subscriber have huge impact on bike usage in January.The most interesting feature is that bike users were more likely use bike at 8 am and 5 pm. The usage at this two hours is higher than any other times. Weekdays usage is also much higher than weekend usage. After doing more research, I found out the average duration_min for trips in the weekdays is actually shorter than weekend.GoBike users were using bike mainly on weekday at 8 am and 5 pm for commuting to work. 

Outside of the main variables of interest, I realized that the average duration_min for trips of female users is actually longer than male users. Also, most of the GoBike users did not apply for the bike share for all program, which provides discount for low-income resident users.


## Key Insights for Presentation

For the presentation, I focus on some important features like duration_min, hour, weekday and start_station_name. First, I want to show that the majority users are subscribers and peak hours are 8 am and 5 pm. I showed the ratio by using bar chart and distribution plot. This can clearly show the usage from hour to hour. Afterwards, I focus on the usage on weekday. I introduced the relationship between duration_min and user_type vs. weekday. I want to show audiences what people were using Gobike for. People were mainly using bike from Monday to Friday and trips were usually around 10 mins. Combined all the features we found out, GoBike users mainly use bike for commuting to work. By researching the popular start stations, we know that bike users were coming from the outskrits of the city and end up stopping in downtown San Francisco or Financial District. Then, I showed some interesting facts like female users were tend to use bike for longer time and custombers were tend to use bikes during the weekend.


