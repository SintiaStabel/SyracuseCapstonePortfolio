# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:25:33 2023

@author: spitum1
"""

from nba_api.stats.endpoints import drafthistory, commonplayerinfo
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
import requests
from lxml import html
from bs4 import BeautifulSoup
import re
import time
import os
import pandas as pd


# Get the players' IDs
draft_info = drafthistory.DraftHistory()
draft_data = draft_info.get_data_frames()[0]

#######only want to go back to the year 2000, since that seems to be the farthest back we have any chance of getting player info

#convert the year column to ints to make it easy to get rid of unwanted years
draft_post2000=draft_data.copy(deep=True)
draft_post2000=draft_post2000.astype({'SEASON':'int'})
draft_post2000.drop(draft_post2000[draft_post2000['SEASON'] < 2000].index,inplace=True)

#remove any periods from a name, it looks like other punctuation can stay in a name
for i in draft_post2000.index:
    draft_post2000.loc[i,'PLAYER_NAME']=re.sub('\.','',draft_post2000.loc[i,'PLAYER_NAME'])

####seperate first name and last name into seperate columns, only first and last name will mostly be needed
#draft_post2000.PLAYER_NAME.str.split(" ", expand=True) <---this results in 4 columns
draft_post2000[['First Name','Last Name','Suffix1', 'Suffix2']]=draft_post2000.PLAYER_NAME.str.split(" ", expand=True)

####manually update the last name for the player Nene
draft_post2000.loc[1259,'Last Name']='Hilario'

########now test how it will look if we get the analysis from the page
#temptextURL='https://www.nbadraft.net/players/Tiago-Splitter/'
#temptextURL='https://www.nbadraft.net/players/Luol-Deng/'
#temptextURL='https://www.nbadraft.net/players/jaylen-brown/'
#temptextURL='https://www.nbadraft.net/players/Ausar-Thompson/'
temptextURL='https://www.nbadraft.net/players/chris-livingston/'

'''linktest = requests.get(temptextURL)
if linktest.status_code == 200: #tests if the link actually works
    temptexthtml=requests.get(temptextURL)
    temptextcontent=html.fromstring(temptexthtml.content)
    temptext=str(temptextcontent.xpath('//*[@id="analysis"]/div[2]/div/div/text()'))'''
###this does not work to get the text

#trying beautiful soup
temptexthtml=requests.get(temptextURL)
analysis_page=BeautifulSoup(temptexthtml.text,'html.parser')
analysis_text=analysis_page.find(class_='wpb_wrapper') #turns out there are multiple of these classes, can't use this, but can use it to get the overall pre-draft score of a player
analysis_overall=analysis_text.find(class_='value')
analysis_overall.text #This does not work with Luoul-Deng, does work with Tiago Splitter
analysis_body=analysis_page.find(class_='vc_tta-panel-body')
analysis_body.text
###this will probably need additional cleanup, but it is a good start

############now put together a script for the overall collection

##Run the below os.remove command if you have had to rerun the predraft analysis script, this will erase the player_dict.txt file
#os.remove('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/player_dict.txt')

#for i in draft_post2000.index:
for i in range(0,5):
    temptextURL=f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}'
    temptexthtml=requests.get(temptextURL)
    #the predraft text has 5 potential sections: nba comparison, strengths, weaknesses, outlook, notes. Collecting the text broken up into those 5 sections.
    overall_val,str_nba, str_str, str_wkn, str_out ,str_note= str(),str(),str(),str(),str(),str()
    if temptexthtml.status_code == 200:
        analysis_page=BeautifulSoup(temptexthtml.text,'html.parser')
        analysis_text=analysis_page.find(class_='wpb_wrapper')
        if analysis_text.find(class_='value'):
            temp_overall=analysis_text.find_all(class_='value')
            overall_val=(temp_overall[len(temp_overall)-1].text)
        else:
            overall_val=('')
        analysis_body=analysis_page.find_all(class_='vc_tta-panel-body')
        for b in analysis_body:
            #if the page has this twitter-timeline section it will provide unwanted text, so avoiding using it
            if "class=\"twitter-timeline\"" not in str(b):
                #the 'NBA Comparison' attribute may live outside of a <p>, so this helps to capture it  if that's the case
                if b.find('h3') and re.match('^NBA Comparison: ',b.find('h3').text):
                    str_nba=str_nba+re.sub('NBA Comparison: ','',b.find('h3').text)
                correct_text=b.find_all('p')
                for p in range(0,len(correct_text)):
                    if re.match('^NBA Comparison:',correct_text[p].text):
                        str_nba=str_nba+re.sub('NBA Comparison: ','',correct_text[p].text)
                    if re.match('^Strengths:',correct_text[p].text):
                        str_str=str_str+re.sub('^Strengths: ','',correct_text[p].text)
                    if re.match('^Weaknesses:',correct_text[p].text):
                        str_wkn=str_wkn+re.sub('^Weaknesses: ','',correct_text[p].text)
                    if re.match('^Outlook:',correct_text[p].text):
                        str_out=str_out+re.sub('^Outlook: ','',correct_text[p].text)
                    if re.match('^Notes:',correct_text[p].text):
                        str_note=str_note+re.sub('^Notes: ','',correct_text[p].text)
        #output the currently collected data to a file in case the script gets interupted so that it isnt lost
        player_dict={}
        player_dict[f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}']={'Player Index':i,'Overall':overall_val,'NBA Comparison':str_nba,'Strengths':str_str,'Weaknesses':str_wkn,'Outlook':str_out,'Notes':str_note}
        with open('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/player_dict.txt', 'a',encoding='utf-8') as file:
            file.write(str(player_dict) + '\n')
        #update the dataframe
        draft_post2000.loc[i,'Overall']=overall_val
        draft_post2000.loc[i,'NBA Comparison']=str_nba
        draft_post2000.loc[i,'Strengths']=str_str
        draft_post2000.loc[i,'Weaknesses']=str_wkn
        draft_post2000.loc[i,'Outlook']=str_out
        draft_post2000.loc[i,'Notes']=str_note
        print('Just finished ' + f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}' + ', index: ' + str(i) + '. Now sleeping for 60 seconds...')
        #now wait 60 seconds before doing the next player
        time.sleep(60)
    else:
        player_dict={}
        player_dict[f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}']={'Player Index':i,'Overall':overall_val,'NBA Comparison':str_nba,'Strengths':str_str,'Weaknesses':str_wkn,'Outlook':str_out,'Notes':str_note}
        with open('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/player_dict.txt', 'a',encoding='utf-8') as file:
            file.write(str(player_dict) + '\n')
        #update the dataframe
        draft_post2000.loc[i,'Overall']=overall_val
        draft_post2000.loc[i,'NBA Comparison']=str_nba
        draft_post2000.loc[i,'Strengths']=str_str
        draft_post2000.loc[i,'Weaknesses']=str_wkn
        draft_post2000.loc[i,'Outlook']=str_out
        draft_post2000.loc[i,'Notes']=str_note
        print(f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]} does not have a page.' +', index: ' + str(i) + '. On to the next one...')

####output the dataframe
#draft_post2000.to_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv',index=None,header=True)

###input the dataframe
#draft_post2000=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv')

#see the number of NAs for each column:
draft_post2000['Overall'].isna().sum() #338
draft_post2000['NBA Comparison'].isna().sum() #616
draft_post2000['Strengths'].isna().sum() #599
draft_post2000['Weaknesses'].isna().sum() #465
draft_post2000['Outlook'].isna().sum() #1206
draft_post2000['Notes'].isna().sum() # 742
        
######rerun one more time to try to get players with text outside of a <p> element, and also add the 'overall' text column, since some players had a section in their pre-draft analysis titled "Overall". Also add a column identifying if a player had a page or not.     
for i in draft_post2000.index: 
#for i in range(876,877):
    temptextURL=f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}'
    temptexthtml=requests.get(temptextURL)
    #the predraft text has 5 potential sections: nba comparison, strengths, weaknesses, outlook, notes. Collecting the text broken up into those 5 sections.
    overall_val,str_nba, str_str, str_wkn, str_out ,str_note,str_over= str(),str(),str(),str(),str(),str(),str()
    if temptexthtml.status_code == 200:
        analysis_page=BeautifulSoup(temptexthtml.text,'html.parser')
        analysis_text=analysis_page.find(class_='wpb_wrapper')
        if analysis_text.find(class_='value'):
            temp_overall=analysis_text.find_all(class_='value')
            overall_val=(temp_overall[len(temp_overall)-1].text)
        else:
            overall_val=('')
        analysis_body=analysis_page.find_all(class_='vc_tta-panel-body')
        for b in analysis_body:
            correct_text=b.find_all('div',class_='wpb_wrapper')
            for p in range(0,len(correct_text)):
                #this hopefully fixes blank strengths
                if pd.isna(draft_post2000.loc[i,'Strengths']):
                    if re.findall('Strength:',correct_text[p].text):
                        str_str=str_str+re.findall('(?<=Strength:)(.*)(?=Weaknesses:)',correct_text[p].text)[0]
                    elif re.findall('Strengths:',correct_text[p].text):
                        if re.findall('Weaknesses:',correct_text[p].text):
                            str_str=str_str+re.findall('(?<=Strengths:)(.*)(?=Weaknesses:)',correct_text[p].text)[0]
                    elif re.findall('Strengths:',correct_text[p].text):
                        str_str=str_str+re.findall('(?<=Strengths:)(.*)(?=Weakness:)',correct_text[p].text)[0]
                    draft_post2000.loc[i,'Strengths']=str_str
                #this hopefully fixes blank weaknesses
                if pd.isna(draft_post2000.loc[i,'Weaknesses']):
                    if re.findall(r'Weaknesses:(.*?)Notes:',correct_text[p].text):
                        str_wkn=str_wkn+re.findall(r'Weaknesses:(.*?)Notes:',correct_text[p].text)[0]
                    elif re.findall(r'Weaknesses:(.*?)Overall:',correct_text[p].text):
                        str_wkn=str_wkn+re.findall(r'Weaknesses:(.*?)Overall:',correct_text[p].text)[0]
                    elif re.findall(r'Weaknesses:(.*?)Outlook:',correct_text[p].text):
                        str_wkn=str_wkn+re.findall(r'Weaknesses:(.*?)Outlook:',correct_text[p].text)[0]
                    elif re.findall(r'Weaknesses:(.*?).$',correct_text[p].text):
                        str_wkn=str_wkn+re.findall(r'Weaknesses:(.*?).$',correct_text[p].text)[0]
                    draft_post2000.loc[i,'Weaknesses']=str_wkn
                #may fix blank notes, but these don't always exist
                if pd.isna(draft_post2000.loc[i,'Notes']):
                    if re.findall(r'Notes:(.*?)Strengths:',correct_text[p].text):
                        str_note=str_note+re.findall(r'Notes:(.*?)Strengths:',correct_text[p].text)[0]
                    elif re.findall(r'Notes:(.*?)Weaknesses:',correct_text[p].text):
                        str_note=str_note+re.findall(r'Notes:(.*?)Weaknesses:',correct_text[p].text)[0]
                    elif re.findall(r'Notes:(.*?)Outlook:',correct_text[p].text):
                        str_note=str_note+re.findall(r'Notes:(.*?)Outlook:',correct_text[p].text)[0]
                    elif re.findall(r'Notes:(.*?)Overall:',correct_text[p].text):
                        str_note=str_note+re.findall(r'Notes:(.*?)Overall:',correct_text[p].text)[0]
                    elif re.findall(r'Notes:(.*?).$',correct_text[p].text):
                        str_note=str_note+re.findall(r'Notes:(.*?).$',correct_text[p].text)[0]
                    draft_post2000.loc[i,'Notes']=str_note
                #may fix blank Outlook, but these don't always exist
                if pd.isna(draft_post2000.loc[i,'Outlook']):
                    if re.findall(r'Outlook:(.*?)Strengths:',correct_text[p].text):
                        str_out=str_out+re.findall(r'Outlook:(.*?)Strengths:',correct_text[p].text)[0]
                    elif re.findall(r'Outlook:(.*?)Weaknesses:',correct_text[p].text):
                        str_out=str_out+re.findall(r'Outlook:(.*?)Weaknesses:',correct_text[p].text)[0]
                    elif re.findall(r'Outlook:(.*?)Notes:',correct_text[p].text):
                        str_out=str_out+re.findall(r'Outlook:(.*?)Notes:',correct_text[p].text)[0]
                    elif re.findall(r'Outlook:(.*?)Overall:',correct_text[p].text):
                        str_out=str_out+re.findall(r'Outlook:(.*?)Overall:',correct_text[p].text)[0]
                    elif re.findall(r'Outlook:(.*?).$',correct_text[p].text):
                        str_out=str_out+re.findall(r'Outlook:(.*?).$',correct_text[p].text)[0]
                    draft_post2000.loc[i,'Outlook']=str_out
                #add the "overall" text category
                if re.findall(r'Overall:(.*?)Strengths:',correct_text[p].text):
                    str_over=str_over+re.findall(r'Overall:(.*?)Strengths:',correct_text[p].text)[0]
                elif re.findall(r'Overall:(.*?)Weaknesses:',correct_text[p].text):
                    str_over=str_over+re.findall(r'Overall:(.*?)Weaknesses:',correct_text[p].text)[0]
                elif re.findall(r'Overall:(.*?)Notes:',correct_text[p].text):
                    str_over=str_over+re.findall(r'Overall:(.*?)Notes:',correct_text[p].text)[0]
                elif re.findall(r'Overall:(.*?)Outlook:',correct_text[p].text):
                    str_over=str_over+re.findall(r'Overall:(.*?)Outlook:',correct_text[p].text)[0]
                elif re.findall(r'Overall:(.*?).$',correct_text[p].text):
                    str_over=str_over+re.findall(r'Overall:(.*?).$',correct_text[p].text)[0]
        #update the dataframe
        draft_post2000.loc[i,'Overall']=overall_val
        draft_post2000.loc[i,'OverTxt']=str_over
        draft_post2000.loc[i,'Has_Page']='yes'
        print('Just finished ' + f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}' + ', index: ' + str(i) + '. Now sleeping for 45 seconds...')
        #now wait 45 seconds before doing the next player
        time.sleep(45)
    else:
        draft_post2000.loc[i,'Overall']=overall_val
        draft_post2000.loc[i,'NBA Comparison']=str_nba
        draft_post2000.loc[i,'Strengths']=str_str
        draft_post2000.loc[i,'Weaknesses']=str_wkn
        draft_post2000.loc[i,'Outlook']=str_out
        draft_post2000.loc[i,'Notes']=str_note
        draft_post2000.loc[i,'OverTxt']=str_over
        draft_post2000.loc[i,'Has_Page']='no'
        print(f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]} does not have a page.' +', index: ' + str(i) + '. On to the next one...')

#check the total number of NAs again
draft_post2000['Overall'].isna().sum() #372
draft_post2000['NBA Comparison'].isna().sum() #616
draft_post2000['Strengths'].isna().sum() #395, was 599
draft_post2000['Weaknesses'].isna().sum() #399, was 465
draft_post2000['Outlook'].isna().sum() #1192
draft_post2000['Notes'].isna().sum() #716
draft_post2000['OverTxt'].isna().sum() #1082
draft_post2000['Has_Page'].isna().sum() #0

####output the dataframe
#draft_post2000.to_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv',index=None,header=True)

###input the dataframe
#draft_post2000=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv')

########There are some manual updates I would like to do...

###Giannis Antetokounmpo is one of the best players in the league but he was drafted 15th overall, not even top ten. His web page used a different last name so his info didn't work. Manually updating that info:
#Giannis page: https://www.nbadraft.net/players/giannis-adetokoubo/
draft_post2000.loc[610,'PLAYER_NAME'] #confirm its the correct row
draft_post2000.loc[610,'Overall']=92
draft_post2000.loc[610,'NBA Comparison']='Nicolas Batum'
draft_post2000.loc[610,'Strengths']='Athletic wing with a remarkable 7’3” wingspan of and incredibly huge hands … He’s still in the development stage and has grown three inches since last year … Standing 6’9”, he has amazing mobility and body control for a guy his height … He’s able to change directions off the dribble and with the ball in his hands with incredible smoothness and quickness, getting to the rim while maintaining excellent balance … His athleticism and wingspan make him able to cover 4 positions on the floor, showing great versatility … He has a natural feel for the game and a good basketball IQ, with good passing skills and instincts and the potential to become a point forward at the next level … Thanks to his big hands and his creativity, he’s an amazing ball handler for his position … He’s unstoppable with momentum, especially during secondary transition, when he can exploit his amazing mobility starting from the dribble … This year during youth games and Greek second division games, he showed the ability to start from one end after the rebound and go coast to coast and get to the rim with 2-3 dribbles, with incredible smoothness and speed … He shows good potential to improve as a shooter even if it’s his main shortcoming at the moment … On the defensive end he has great instincts in the passing lanes and in help situations, often with perfect timing for blocks from the blind side … With his physical tools he could easily guard opposing wings, showing the potential to defend both guards and power forwards, depending match ups. He’s still raw in many perspectives, but his ceiling and upside are the highest among the internationals of this 2013 draft …'
draft_post2000.loc[610,'Weaknesses']='His level of competition is a big concern, because it makes judging his talent and current level very difficult. Without question he has a large basement to go with his large ceiling … Despite his athletic abilities he lacks elite explosiveness … He has to bulk up, working especially in the lower body since he’s definitely too skinny to face NBA opponents at the moment … The concern is how to develop him correctly from the muscular standpoint, in order to avoid loss of speed and mobility (Boris Diaw) … His game off the ball is rather weak, especially related to spacing and use of screens and cuts, in fact most of his offensive production happens with the ball in his hands … The only exception is when he’s slashing to the basket for put backs or on the break … Plus he basically has no mid-range game, he tends to attack the rim without considering the option of the pull up jumper … He shot 31.3 % from three point line this year, showing good potential, but he lacks consistency at this point, his mechanics seems unnatural and not fluent … On the defensive side, he needs to learn the basis, since he’s beaten by the opponents due a lack of proper positioning and comprehension … The overall impression is of a raw prospect from basketball comprehension standpoint, whose is based on instincts, talent, physical gifts and natural feel for the game. For this reason he needs to be tested at a higher level of competition than Greek second division, since his level of experience is definitely low.'
draft_post2000.loc[610,'Notes']='Also known as: Giannis Antetokounmpo … The youngest player in this year’s NBA draft. Won’t be 19 until December … Son of Nigerian immigrants, he was enrolled with his older brother (20 years old) Athanasios Adetokoubo by Filiathlitikos Academy, an ambitious club from Athens Area. He averaged 8 points and 5 rebounds in Greek A2 this year, when he came out as he came out of obscurity to become a potential lottery pick … This June he could be part of the 2013 adidas Eurocamp, since he has just gotten his Nigerian passport.'
draft_post2000.loc[610,'Has_Page']='yes'

###Zion Williamson was one of the most anticipated prospects since Lebron, but so far has been a big bust and his page did not work for some reason, going to manually input his stats too since he should be important to the analysis...
#Zion page: https://www.nbadraft.net/players/zion-williamson/
draft_post2000.loc[236,'PLAYER_NAME']
draft_post2000.loc[236,'NBA Comparison']='Charles Barkley/Blake Griffin'
draft_post2000.loc[236,'Strengths']='Incredibly unique prospect from a physical attribute standpoint … Strength and explosiveness to finish through contact easily and hang in the air … Impacts the game on both ends of the floor … Foot work and body control are like a guard’s at 280 lbs … Has an incredibly explosive second jump with the touch and concentration to finish … Underrated length with a near 6’11 wingspan … Tremendous anticipation, which allows him the ability to play a step ahead of opponents … A nightmare to defend in space … Very well coordinated movement for someone with his size and youth. Incredibly smooth and light on his feet at size. Agile and nimble … Ability to catch the ball, gather and quickly change direction, avoid defender … Excellent acceleration with the ball in hands, blows by guys playing off him, beats guys to spot … Good motor on offensive glass, strong along with leaping ability … Good hands … Catches tough passes, gets 50/50 balls … Wide frame makes it hard to get around him for defenders to contest, able to finish around and through defenders … He is a freight train in the open court and nearly impossible to stop with his combination of size, speed and ball skills … Often beats double and triple teams with both strength and finesse … Strong enough to finish through contact, able to get to rim at a high rate … Good court vision and solid decision making … Imaginative passer and play maker … Solid recognition of defensive schemes, unselfish, looks for teammates … Recognizes traps and double teams, doesn’t panic, moves the ball intelligently … Ball handling moves and shiftiness is super high level at size/age … Very coachable. Absorbed a lot of instruction and improved his skill set at Duke … Good hand eye coordination, able to split traps, keep dribble low, very quick hands … Very effective in and out dribble in transition … Has a solid spin move off the dribble and a quick behind the back dribble as well. Natural instincts for where defense is coming from and going to, instinctual dribble counters … Nice outlets, keeps his head up in transition Defensive instincts … Good anticipation on blocks, pins shots w/o goal tending or fouling and allows for team to recover … When he tries, moves feet well on perimeter, quick hands, changes direction well … Plays defensive back well, baiting opposing guards to throw a pass to his guy and then jumping the pass for an easy steal with his great closing speed … Quick hands, gets a lot of strips and deflections … Has shown improvement on pull ups and step back jumpers, looks fluid, getting squared, going straight up but hasn’t made them with great accuracy so far … Magnetic personality, natural entertainer, great kid off the court by all accounts … As long as he continues to develop, intangibles are in place to be a future star … Great conditioning, especially considering his body type … 6’6 explosive forward … Good length with a 6’10.5 wingspan and 8’7 standing reach … Outlier athlete with his insanely explosive leaping ability … His body type and ability to jump defy physics… Can pop off the ground quickly off one or two feet … Doesn’t need a gather to jump, can explode from a standstill … Can take off from a long ways away from the rim off two feet and finish with a highlight dunk … Finishes jams with either hand. Great transition finisher … Tremendous body control and hang time … Quick first step … High level competitor who seems to raise his intensity level above opponents and out quick them on plays around the basket … Dangerous scorer from midrange and around the rim … Good touch on jump hook/floater at different angles and heights, which he gets off in heavy traffic … Shoots efficiently from the floor. Shot 82.1% (23-28) at adidas Nations in 2016 and 64.5% in 2017 while finishing second in scoring with 22.5 points per game … Led the adidas Gauntlet in scoring with 27.1 points per game while shooting 56.2% from the floor … Aggressive mindset … Good strength, overpowers other perimeter players. Doesn’t shy away from physicality. Handles contact well, can absorb while still elevating to get off a clean and accurate shot. Rare combination of power and finesse … Effective spin move in traffic … Draws a lot of fouls. Shot 8.9 free throws per game over the adidas Gauntlet and 8.5 per game at adidas Nations … Despite the criticism of his competition throughout his high school career, he’s able to step up and play well against bigger and better competition without being behind the curve … Had great showings when facing top players: NBPA Top 100, adidas Nations, USA, etc. … Good rebounder … Led adidas Gauntlet with 11.3 per game and finished top five at adidas Nations with 7.2 per game … Sees the floor well and has solid passing ability … Unselfish and can pass on the move … Good tools, athleticism, and anticipation/timing as a defender on and off the ball. Averaged 2.3 steals and 1.9 blocks per game over the adidas Gauntlet … Started to show emphasis on getting into shape and improving conditioning. Weighed 272 pounds at the USA Basketball Minicamp and noticeably lost weight, getting down to 250 pounds … Younger for his class, will turn 18 in July going into his freshman year of college … Has done a solid job of toning his excess weight (butt) over the past year, but still has constant work on it …'
draft_post2000.loc[236,'Weaknesses']='Playing style and weight puts a lot of pressure on his legs and feet. The question begs to be asked whether he can stay on the floor for a lengthy NBA career … He will likely need to drop another 15-20 lbs in order to sustain his playing style … 250-260 lbs would be ideal. But may not be possible … Shot is somewhat flat with awkward release position, from the side of his head … Is developing into a capable shooter but still needs refinement … Plays with a great deal of emotion, but at times that emotion can get the best of him as he struggles to contain it … Will need to work on not getting too emotional as the 82 game NBA season is a long one and can burn out players that put too much into each and every game … Can get a little out of control at times, needs perimeter game to develop to be less predictable … A good portion of his points come through overpowering less physically developed players, which will not work as well at the next level … Jumper is flat, needs more arch … Struggles to shoot from the perimeter in rhythm … Is much more adept at spotting up when defenders sag off of him than creating shots with defenders on him … Needs to work on polishing a pull up jumper … Poor effort defensively at times and does not always get in a low stance, even though he can sometimes make up for it with his unique athleticism … Loses focus defensively at times, gambles for blocks and steals … Not always in great position, at times late on rotations … Back to the basket game could use some work … Although he hasn’t needed it at the high school level, he needs a more expanded half-court game as it will be predictable moving forward … Often resorts to bully ball … Struggles against a set defense … May struggle against help defense at the highest level as he does a good job of getting into his defenders chest and clearing space … Reading the game as a perimeter player will be an adjustment at the next levels … A bit of a tweener as he lacks the perimeter game of a three and ideal size of a four … While he shows solid shooting mechanics, he doesn’t show much shooting or perimeter skills … Shot could use more arc and he lacks the reps at game speed/pressure … Doesn;t seem to develop much rhythm in space on the perimeter, and seems to have more success and balance on shots with contact … Missed his only two three-point attempts at adidas Nations over six games … Shot 19.2% (5-26) from three and 62.9% from the free throw line over the adidas Gauntlet … Finished with a negative assist/turnover ratio with 1.9/2.3 over the adidas Gauntlet … Injury concerns persist due to his thick legs and the extreme amount of stress his incredible leaping ability puts on his lower body (knees, ankles and feet)  … Any injury could be detrimental as his weight could be affected by extended time off the floor …'
draft_post2000.loc[236,'Notes']='Measured 6’6 with a 6’10.5 wingspan and 8’7 standing reach in high school events … Full name is Zion Lateef Williamson … Born on July 6, 2000, in Salisbury, S.C. … His mother is Sharonda Williamson and stepfather is Lee Anderson … Has a brother, Noah … His mother ran track at Livingstone College, and his stepfather played basketball at Clemson …'
draft_post2000.loc[236,'Outlook']='Incoming Duke freshman… Played in the McDonald’s All-American Game… Named to the Jordan Brand Classic… Named to the 2018 Nike Hoop Summit but sustained an injury a week earlier to his thumb preventing him from playing in the game. Attended the Hoop Summit to support his teammates… Led Spartanburg Day to three consecutive state titles… Averaged 36.4 points, 11.4 rebounds, and 3.5 assists per game his senior year… A YouTube sensation and one of the biggest names in high school basketball… Among the elite prospects of the 2018 high school class… MVP of adidas Nations with 28 points and ten rebounds in the championship game… MVP of the Under Armour Elite 24 in 2016 with 23 points and shooting 10-10 from the floor… Has drawn comparisons to Charles Barkley due to his dimensions, explosiveness and playing style…'
draft_post2000.loc[236,'Has_Page']='yes'

###Noticed a few players before Zion also had issues, manually updating them
#Paul Reed: https://www.nbadraft.net/players/paul-reed/
draft_post2000.loc[233,'PLAYER_NAME']
draft_post2000.loc[233,'Strengths']='6’9 forward who has a 7’2 wingspan and plays with high energy on defense with the ability to play in space … Often turns defense into offense blocking shots both in the paint and on the perimeter … Transition scorer … Can really move his feet defensively and stay in front of smaller, quicker players and has incredible ability to block jumpshots … Wingspan and athleticism allow him to be lethal playing the passing lane and once he comes away with the ball he’s gone in a flash … Vicious at attacking the rim … Capable and willing to finish with his off hand … Soft touch around the rim … Great at catching lobs and dunking in traffic … Can move without the basketball … Strong in pick and roll … Nice spot up jumper with a quick release … Decent shooting range for a bigger forward … Able to shoot off the dribble in mid range … Relentless on the offensive glass … Very versatile and can play more than one position … Averaged over 15 points and 10 rebounds last season shooting over 50 percent from the floor …'
draft_post2000.loc[233,'Weaknesses']='Inconsistent shooting mechanics … Has a very slow release and should look to revamp shooting form … Tends to lean back when shooting … Questionable decision making on offense … Passing and playmaking need work … Not always under control … Prone to committing offensive fouls … Inconsistent playmaking in transition … Low post game needs work … Shoots less than 74 percent from the free throw line … Defensive awareness … Must add body weight to guard 5’s down low … All around feel for the game and fundamentals in need of work …'

#Jalen Harris: has a page but has no analysis on the page
#Sam Merrill: has a page but besides an overall score has no analysis on it

###I just happened to notice "Weaknesses" was spelt incorrectly on Aleksej Pokusevski's page, which affected the script, so correcting that...
#page: https://www.nbadraft.net/players/Aleksej-Pokusevski/
draft_post2000.loc[192,'PLAYER_NAME']
draft_post2000.loc[192,'Weaknesses']='Has a really thin frame … Narrow shoulders, makes it more difficult to bulk up in the future … His hips are also narrow … Lacks elite explosiveness … Not fast enough for a wing and not strong enough for a big … Struggles against physical play due to his frame … Can be too emotional at times, allowing the moment to get the best of him with emotional swings … He doesn’t often take over games as much as he should against inferior competition… Doesn’t really have a true position, which for now is uncertain if it will be a negative in this non-positional era … Still needs some work in Pick and Roll situations as the ball handler … It’s more than obvious that he prefers to either Pop Out or make the short roll after he sets a screen … Has major problems finishing through contact … He has a tendency to force things on offense … His basketball I.Q is exposed at times, since he tries to do things or make passes that either aren’t there … He is a streaky shooter for now … His Post up game needs a lot of work … Can’t really take advantage of any mismatch on offense against a shorter opponent, since he is not strong enough to do it … He must learn to finish plays harder and not just use finesse … His shooting selection can be a little erratic at times … He could be more active on the offensive boards and put pressure on defense … He depends too much on his length on rebounding and has the tendency to forget to box out… Average lateral quickness… Not strong enough to defend on the post … Can’t really bang bodies due to frame, he can be bullied by stronger, more physical opponents … Has problems chasing wing players around screens … He gambles on defense or be a step too slow on defensive rotations …'

###Ideally, any player we use should have both a strength and a weakness for better analysis. Taking a look at how many players have one but not the other to see if it is reasonable to manually update them.

#first players with no strengths but a weakness
for i in draft_post2000.index:
    if pd.isna(draft_post2000.loc[i,'Strengths']) and pd.isna(draft_post2000.loc[i,'Weaknesses'])==False:
        print(i, f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}', f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}')
'''Output from above:
159 Kessler Edwards https://www.nbadraft.net/players/Kessler-Edwards
173 Jericho Sims https://www.nbadraft.net/players/Jericho-Sims
326 Elie Okobo https://www.nbadraft.net/players/Elie-Okobo
380 Anzejs Pasecniks https://www.nbadraft.net/players/Anzejs-Pasecniks
501 Nikola Milutinov https://www.nbadraft.net/players/Nikola-Milutinov
525 Marcus Eriksson https://www.nbadraft.net/players/Marcus-Eriksson
535 Luka Mitrovic https://www.nbadraft.net/players/Luka-Mitrovic
560 Clint Capela https://www.nbadraft.net/players/Clint-Capela
562 Bogdan Bogdanovic https://www.nbadraft.net/players/Bogdan-Bogdanovic
585 Alec Brown https://www.nbadraft.net/players/Alec-Brown
746 Bojan Bogdanovic https://www.nbadraft.net/players/Bojan-Bogdanovic
978 Wilson Chandler https://www.nbadraft.net/players/Wilson-Chandler
1016 Andrea Bargnani https://www.nbadraft.net/players/Andrea-Bargnani
1042 Sergio Rodriguez https://www.nbadraft.net/players/Sergio-Rodriguez        
'''
draft_post2000.loc[159,'Strengths']='A long, wide-shouldered 6’8 combo forward whose physical tools really stand out for a mid-major prospect … Late bloomer who has put on muscle each year of college, and should continue to be able to easily pack more weight into his current 205 lb frame without the loss of athleticism … Good mobility, can comfortably operate on the perimeter or mid-range and shows nice agility and lateral quickness on both ends of the court…Very good perimeter shooting touch, has unorthodox lower body movement in his set up but he gets his legs into his follow through and his release and shot mechanics never change … A knockdown shooter when spotting up in rhythm, but also a factor finding open spots in the defense by moving without the ball, and can hit shots coming off screens from mid-range or 3 … Great pick and pop threat when he’s at the PF spot…Has nice defensive potential, moves his feet & uses his size very well to force tough shots, and has already shown promise being able to defend any frontcourt position as well as some perimeter switches…Above average timing as a shot blocker (1.3 bpg for his career), shows solid instincts and awareness on and off the ball … Can finish strong or with soft touch at the rim, had very few problems converting in the paint against WCC competition … FT shooting improved significantly each season, finished his JR season shooting 87.6% … Unselfish, doesn’t really force the issue and generally gets all his shots within a team structure …'
draft_post2000.loc[173,'Strengths']='Sims is very springy, energetic athlete with great physical traits at 6’10 245, possessing truly explosive leaping ability (ridiculous 44.5” max vert at the NBA Combine, good for 2nd highest ever recorded in the event’s history) and nice speed for a legitimate C prospect … On top of his big-time athleticism, he also possesses an enormous 7’3 wingspan that he uses well on both ends of the court … Runs the floor like a deer, and looks to convert any opportunities he gets in the open court or as a rollman with strong finishes above the rim (impressive 69.6 FG% as a Sr., career 64 FG%) … Possesses a quick second jump, which when coupled with his length allows him to be an effective offensive rebounding/putback threat …. Good upper body strength, can clear the glass and convert in the paint through contact well … The dropstep is definitely his go-to post move, but he will occasionally look comfortable spinning into a jump hook over his right shoulder if it isn’t available … Excellent defensive potential, looks great defending in both the low & high post and in pick & roll switches … Has all the lateral quickness and physical tools needed to defend in today’s NBA as a big…Uses his length and size well as a shotblocker, able to move his feet and reject shots on the high post and perimeter with good timing or patrol the paint in helpside defense…Really improved his defensive fundamentals throughout his 4-year career, which allowed him to make a bigger impact contesting when he was on an island against smaller players during his last 2 seasons … Can impact the game without scoring much or having plays drawn up for him, makes effort plays … Solid hands, doesnt drop or bobble many dump off passes and had some impressive alley oops … Played in a role and scheme at Texas that asked him to do the same things he projects to do at the pro level …'
draft_post2000.loc[326,'Strengths']='Left handed guard with a nice frame that looks that can fill up nicely… Good size and great length for his position (wingspan measured at 6-8 feet) … Good athlete who seems to take care of his body … Explosive guard with an above average first step … Posseses great leaping ability … Versatile player who can play both guard positions … He can be used either as the ball hanlder or as an off ball guard … He plays within the system … His quick first step helps him blow by his opponents and attack miss-matches and closeouts … Has the ability to penetrate from both sides and is really quick going left … When he is concentrated is not afraid to go all the way to the basket and try to score against bigs in the paint … Can create his own shot with ease … His ability to score off the dribble is probably his biggest advantage … Can create seperation from his opponent before shooting with a nice looking step back … His pull-up Jump Shot has improved dramatically through the years … Very good shooter … Has NBA range in his shot, even off the dribble … Has been used in Catch and Shot and Spot Up situations with very good results … His 3-point shooting would be and will be higher when he learns to avoid taking tough shots … Excellent free throw shooter … Has great court vision and can find the open shooter … Has the ability to pass off the bounce and on the move … Watches all the floor, sees the lanes and can find the cutter to the basket … Great in drive and kick situations … Has improved considerably in Pick and Roll situations, looking more willing to find the rolling big … He is a threat in every way possible while playing the Pick and Roll thanks to his shooting ability, explosiveness and court vision … Moves well without the ball and can make good cuts, catching his opponent off guard … Excells in transition as the ball handler … Great lateral quickness when he is motivated and takes a good low stance … Has active hands when he defends the ball … His length and quickness help him create a lot of problems to his opponent … Fills the lanes at an acceptable level … Active weakside defender … He can anticipate plays while he is in the weakside and rotate accordingly … Great at chasing his opponent around the screens … He has the I.Q to be in controll and not lose focus on the defensive rotations … He can make an occasional chase down block … Has lockdown defender potential …  '
draft_post2000.loc[380,'Strengths']='He has an impressive frame at 7-2, with a solid structure and some late developing muscular development … He has a remarkable wingspan, combined with solid mobility … With his size and mobility, he’s the ideal target in the paint area in P&R situations, both for lobs or put backs … He has a nice handle for his size, with the ability to challenge the defender off the dribble and to finish with either hand … He has also shown intriguing potential as a shooter, since he is able to hit shots from behind the three-point line in catch and shot situations … He shows flashes as a defender, both in close outs and as a rim protector, thanks to his footwork and explosiveness … His ability to shoot elevates his stock some as a player teams are looking for … Even though he will turn 22 in December, still seems to have more potential left to develop than the average draft eligible player …'
draft_post2000.loc[501,'Strengths']='Milutinov is a legit seven footer with remarkable wingspan (7-3), to which he combines excellent footwork and great mobility … He runs the court faster than most centers in the ABA league … He has big, soft hands showing ambidextrious ability to use both for hook shots in the middle of the painted area … He’s able to use his size to shoot over opponents, furthermore he has shown solid range as a spot shooter … Despite the limitations with his post game, his skillset has been improving over the last couple years, and with his great court vision and passing skills from inside the paint, he opens the court up for his teammates, becoming a legitimate offensive facilitator … When given openings around the basket, he shows a good variety of finishes in the painted area, including dunks … He’s a solid defensive presence due to his combination of height, wingspan and mobility … He possesses a contribution of intimidation, showing good instincts for the block … He’s considered a solid worker in the gym – training under Dule Vujosevic shall be considered a good business card – and is considered a quality teammate …'
draft_post2000.loc[525,'Strengths']='As a 6-7 SG, he has terrific size for the role, combined with a remarkable wingspan … He’s a sharp shooter with fluid mechanics, smooth and quick release and range … His shooting repertoire is complete, both pull-up jumpers and spot-up … Over the last couple of years he has developed a floater, showing more aggressiveness in attacking the rim … Defensively, his wingspan makes him a solid threat in the passing lanes.'
draft_post2000.loc[535,'Strengths']='Standing 6-foot-8 he has a solid structure and long arms with a frame that should allow him to put on some muscles without struggling … He is a solid athlete despite not being extremely explosive, he’s mobile and smooth running the floor … Offensively he shows solid shooting skills for the role, especially as a spot up shooter – basically in pick and pop situations – beyond three point line … His mobility allows him to be really effective in P&R, where he shows good timing and a sense of position after the screen, for the cut to the rim of the pick and pop … In addition, he’s also really effective in transition, when he can exploit his speed and motor compared to the size … He has good lateral speed and footwork, making him a reliable defender also vs. backcourt players despite the size … He has quick hands and solid instincts in the passing lanes which makes him a good stealer … He has the potential and the technical skills to play in post position even if not yet developed. Great motor, and intensity, really effective in off the ball game …'
draft_post2000.loc[560,'Strengths']='Capela is as good as any athlete ever coming from Europe … What really separates him is his explosiveness around the rim … He isn’t scared of contact and can really finish above the rim with one of two hands … He also can really run the floor on finding easy layups and dunks in transition … Capela is still a very raw talent with a lot of upside, but has also developed solid instinct, particularly on the defensive end … He also makes passes you would not expect from a player like him … His ability to finish around the rim is pretty good, which makes him a very dangerous player on Pick and Roll situation and circling around the basket … Defensively he is very quick, with good lateral movement and when locked in he can really be aggressive showing on pick and roll but also in switching situations … In a good day he can be absolutely dominant in the paint at the European level, and with focus could become an above average NBA defender over time …'
draft_post2000.loc[562,'Strengths']='Standing 6’6” he has a perfect size for the position … With his solid structure and athletic abilities he can definitely adapt his game to the NBA level, with the potential to defend both on guards and small forwards … He possesses natural game comprehension and poise which allows him to play the role of offensive facilitator and game distributor (4 assists of average thus far) … During this season he started playing as a secondary ball handler, but after the injury to Leo Westermann he has been forced to play as a point guard, showing remarkable passing skills and court vision, even if he’s not perfectly suited for the role … He has a slashing style of game, and with his solid structure he’s really effective in drawing contacts when attacking the basket or posting up other backcourt players, exploiting his size and physical strength … This combined with his instincts allow him to also be a reliable rebounder and a consistent defensive presence … On the defensive side he shows great attitude and quick hands, with the ability to put a concrete pressure on his opponent especially in 1vs.1 situations … He’s also effective in the passing lanes, with almost 2 steals per game this season … He has constantly improved his game since his debut at the senior level, becoming an ideal glue guy for a contending team …'
draft_post2000.loc[585,'Strengths']='At 7’1" Alec not only has legitimate NBA center height but also is one of the best shooters in the 2014 NBA Draft … He consistently can hit shot after shot from well beyond the 3 point line …  With 29% of all his shots taken from 3 he was able to make them at a 42% clip and an overall field goal percentage at 47% …  He displays good agility, height, and a solid understanding of his offensive strengths … He can run the floor and is a very dangerous trailer hitting the top of the key jump shot or the occasional dunk … And his strength in the pros should be as a stretch 4 or 5 and used as a pick and pop player on the offensive end.  On the defensive end he was also able to average over 3 blocks per game.  Coming from a mid major Alec was able to give a positive impression to scouts at Adidas Nations while going against some of the best players in college basketball.'
draft_post2000.loc[585,'NBA Comparison']='Matt Bullard'
draft_post2000.loc[746,'Strengths']='Bojan Bogdanovic is a versatile player who can score in lots of different ways. He has a good combination of height and body frame that should allow him to compete with NBA small forwards physically … He can attack the rim effectively with his quick first step, which allows him to beat his immediate defender quite easily … His mid-range game is a continuous threat because he has a good pull up jumper with confidence … He has NBA range on his 3-point shot, and is becoming more adept at putting the ball on the floor and getting to the basket …'
draft_post2000.loc[978,'Strengths']='Chandler has impressive physical tools, with a great combination of strength and athleticism … He can leap out of the building and create some highlight reel dunks … He’s a very smooth athlete with a knack for finishing … Chandler has displayed an improving jump shot and his ball handling skills are slowly developing. As primarily a power forward on DePaul, he has developed some post moves and become a very strong rebounder … His rebounding ability will set him apart from many small forwards a la Shawn Marion … He is also a solid shot blocker who seems to be at his best defensively in the paint …'
draft_post2000.loc[978,'NBA Comparison']='Joey Graham'
draft_post2000.loc[1016,'Strengths']='Good foot work, soft hands even if he’s not particularly strong … A fast player with very quick feet and first step for such a big guy, runs well in the fast break … Shoots it quickly, with solid release and is also a great catch and shoot player … Good one on one player who prefers to face the basket … Bouncy, and will get up multiple times off two feet … Good shot blocker … Played an average of 12 minutes per game this season (2004-05) … Generally impacts game tempo in a positive way when he’s on the floor … Makes plays. The only young 6-10 Italian prospect that can play as pure small forward and not only under the basket as power forward or center… He has a well built body with excellent athletic skills. On the offensive side he prefers going one-on-one and can score in many ways: shooting from outside, finishing a fast-break or on low-post moves. He’s even adept at beating his defender off the dribble due to his great ball-handling skills including dribbling through his legs and behind the back. He runs the court very well with good speed and quickness despite his size. Has soft hands to shoot from three or from the mid range, that’s his best quality. Free throw shooting is another strength. Under the basket he is not so tough, but has nice moves. On the defensive side, he has good fundamentals and likes to block shots and has excellent timing. He’s also a hard worker and coachable.'
draft_post2000.loc[1016,'NBA Comparison']='Dirk Nowitzki'
draft_post2000.loc[1042,'Strengths']='For Sergio Rodriquez, the nickname “Spanish Magician” wasn’t given, it was earned … His imaginitive, charismatic, off the wall playing style combined with innate leadership abilities, has earned him not only respect, but the interest of many NBA scouts … Owns a variety of offense weapons, but is at his best when breaking down defenders with his lethal crossover, or feeding a teammate for an easy basket … Incredible quickness and ball handling ability make taking any defender 1-1 appear effortless … Most impressive is his ability to find the open teammate in traffic … He has the uncanny ability to see passing lanes before they open up, thus enabling him to nearly always make the right decision with the ball. While not incredibly gifted athletically, Sergio always finds a way to get the shot up, or draw the foul when around the basket … Hits open shots, especially in the clutch, and shoots the three ball well enough to keep any defender honest … Absolutely loves to run the fast break, using his extreme quickness and good finishing ability to convert most of the time … Quickness and anticipation make him a capable ball hawk, as well as a solid off the ball defender … His nickname (The Spanish Magician) fits him well, sometimes it appears the moves he makes, and the passes he fires, aren’t real … This kid has a world of talent, and just must continue to develop.'
draft_post2000.loc[1042,'NBA Comparison']='Jason Williams'

#run the tests again:
for i in draft_post2000.index:
    if pd.isna(draft_post2000.loc[i,'Strengths']) and pd.isna(draft_post2000.loc[i,'Weaknesses'])==False:
        print(i, f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}', f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}')

#now players with a weakness but no strength:
for i in draft_post2000.index:
    if pd.isna(draft_post2000.loc[i,'Strengths'])==False and pd.isna(draft_post2000.loc[i,'Weaknesses']):
        print(i, f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}', f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}')
'''Output from above:
220 Jordan Nwora https://www.nbadraft.net/players/Jordan-Nwora
272 Deividas Sirvydis https://www.nbadraft.net/players/Deividas-Sirvydis
276 Eric Paschall https://www.nbadraft.net/players/Eric-Paschall
282 Ignas Brazdeikis https://www.nbadraft.net/players/Ignas-Brazdeikis
300 Trae Young https://www.nbadraft.net/players/Trae-Young
324 Dzanan Musa https://www.nbadraft.net/players/Dzanan-Musa
340 Hamidou Diallo https://www.nbadraft.net/players/Hamidou-Diallo
409 Alec Peters https://www.nbadraft.net/players/Alec-Peters
415 Alpha Kaba https://www.nbadraft.net/players/Alpha-Kaba
705 Izzet Turkyilmaz https://www.nbadraft.net/players/Izzet-Turkyilmaz
758 Malcolm Lee https://www.nbadraft.net/players/Malcolm-Lee
822 Tiny Gallon https://www.nbadraft.net/players/Tiny-Gallon
1310 Kwame Brown https://www.nbadraft.net/players/Kwame-Brown
1312 Pau Gasol https://www.nbadraft.net/players/Pau-Gasol
1315 Shane Battier https://www.nbadraft.net/players/Shane-Battier
1316 Eddie Griffin https://www.nbadraft.net/players/Eddie-Griffin
1318 Rodney White https://www.nbadraft.net/players/Rodney-White
'''
draft_post2000.loc[220,'Weaknesses']='He is an average athlete, which may prove to be problematic when competing against stronger, faster wings … Underwhelming speed and athleticism … Somewhat between positions as a 3/4 without a great ability to defend either position … Tends to hoist up questionable shots … Often forces the action with the shot clock winding down, tossing up a difficult 3-pointer … For the amount of 3-point attempts he takes per game (6.1), he could improve some in that area, shooting 37.4% from beyond the arc as a sophomore … Needs to bulk up his thin frame or else he may get pushed around by more physical players at the next level … Not particularly quick laterally, and lack of a quick first step could limit his offensive effectiveness … 1.3 assists to 2.4 turnovers shows that he needs to get better at protecting the ball. Though being such a focal point of Louisville’s undermanned offense surely played into those numbers …'
draft_post2000.loc[272,'Weaknesses']='Below average athlete by NBA standards … Upper body needs to fill up … Doesn’t really have bounce, which can be a problem at the next level … Quick, but not an explosive player, with an average first step … Short wingspan (measured at about 6-8, same as his height), which limits him on defense … Ball handling is kind of loose, he doesn’t always dribble with his head up, which leads to turnovers … He can’t always create separation from his defender due to his average athleticism and ball handling … Mid-range game isn’t completely there yet … He can be too passive at times, just standing in a corner waiting for the ball to come to him … Not really effective when his shot isn’t falling, at least just yet … He can’t really mix it up on offense … He doesn’t draw fouls due to his playing style because he isn’t attacking the paint, so he doesn’t really go the line a lot, which is bad because he is a very good free throw shooter … He can suffer from tunnel vision at times while driving to the basket … He occasionally tries difficult passes that just aren’t there … Stronger opponents bully him on the defensive boards … Because he mostly stands in the perimeter, he is not a threat on the offensive board … His post up game is minimal, he isn’t really taking advantage of his size and he can’t exploit mismatches … Has an average lateral quickness … Gambles and goes for the steal, putting pressure on team’s defense … Spaces out on Defense, watching the ball and stays in no man’s land, losing his player … His post up defense is problematic, mainly because he is not strong enough to contain bulkier opponents …'
draft_post2000.loc[276,'Weaknesses']='At 6’7’’, he will struggle to guard and rebound against taller players in the post, doesn’t have a real defined position  … Older player (5th year senior who will turn 23 in November) without a ton of upside, having just finished his senior season at Villanova … Struggled in the role of go to scorer in senior year … Hasn’t shown much fluidity/athleticism on offense (possibly a product of the Villanova offense) … Lacks an explosive first step off the dribble and the speed of elite NBA forwards … Not a great spot up shooter … Can be over-aggressive defensively, resulting in unwanted fouls … He must learn to move his feet better instead of relying on his upper body for defense … Got used to handling the ball, but doesn;t seem have the ball skills to handle at the NBA level … Average at best rebounder in college despite his size … Has not shown much in terms of post offense … Has not had to guard down in the post much at Villanova …'
draft_post2000.loc[282,'Weaknesses']='The biggest question will be on the defensive end as he lacks ideal length and in particular foot speed … At 6’7 and not that much bigger of a wingspan (6’8) he won’t possess the quickness and length to stay in front of and check NBA small forwards, nor the length or physicality to match the power forwards… 8’5 standing reach is good for a shooting guard, but small at the 3 position … His toughness will help him initially but over time he will have to figure out how to overcome his defensive deficiencies… Offensively, to become elite he’ll have to improve his vision and playmaking skills for his teammates, otherwise he’ll put himself in a situation where he may be valued only as a spot-up shooter … A good but not great shooter, but could improve as time goes on considering his level of dedication … Shot 77% from the free throw line, which is solid but if shooting is his forte, he’ll need to raise that …'
draft_post2000.loc[300,'Weaknesses']='Young does not have ideal size at 6-2 or length with just a 6-2 wingspan for an NBA guard … These physical limitations can become apparent once he gets all the way to the rim … He lacks the length and vertical explosiveness to finish through NBA length and athleticism and does not project as an efficient scorer at the basket in the NBA … He will have to rely much more on craft to be successful inside at the next level … He has been a bit turnover prone at times this season due to his tendency to make some risky passes at times … However, much of that is due to his unusually high usage rate, and he still has an assist to turnover ratio of better than two to one … Young’s shot selection at times leaves a lot to be desired … As good of a shooter as he is, there are times where he will hoist a very long jumper early in the shot clock without moving the ball and looking for a better opportunity first … As polished as Young is offensively, he is a near liability on defense … He rarely gets into a stance on this end of the floor and shows poor effort guarding the ball … Opposing guards are easily able to get into the lane against him and he is often caught ball-watching when his man does not have the ball … Due to his lack of ideal size and athleticism, he will be a guy for opposing offenses to attack … He is not explosive enough to stay in front of quick guards, and he is also not big enough to stop anyone else on the court … The effort has to improve first and foremost, but even if it does he may never be a reliable defensive player unless he can create a high number of steals … Will need to add strengh to be able to better finish off drives and also defend … Undersized as a scorer at 6’2 with a 6’2 wingspan … Lack of size and physicality will make defense a challenge at the next level … Has a way to go to become a true point guard … Has questionable shot selection and can be too quick with the trigger at times … Sometimes brings the ball up the court and shoots it early in the clock before without anyone else getting a touch or making the defense work … His release is low, starting from his chest … Doesn’t always keep it simple. Tries to do too much at times … Can be turnover prone … Not an elite athlete. Lacks explosiveness. Plays below the rim … May struggle defensively moving forward due to his lack of strength and playing through physicality … Is bothered by length when driving … Gets a number of shots blocked and passes deflected … One of the best at this stage …'
draft_post2000.loc[300,'Notes']='Attended Norman High School in his hometown of Norman, Oklahoma…2015-2016 Oklahoma Gatorade Player of the Year and Oklahoma Class 6A Champion…Consensus five-star recruit…McDonald’s All-American and Jordan Brand Classic Selection … Measured at 6’0, 158 lbs, with a 6’1 wingspan at the 2014 Nike Elite 100 … Measured 6’1.5’’ in shoes, with a 6’2’’ wingspan and 162 lbs at the 2015 Nike Basketball Academy … Averaged 21.3 points per game over the Nike EYBL … On December 19th, 2019 tied the all time NCAA Division I assists record with 22 against Northwestern State …'
draft_post2000.loc[300,'Outlook']='2016 Peach Jam Co-MVP … Young had a strong Nike EYBL season and is one of the top guards in the 2017 high school class.'
draft_post2000.loc[324,'Weaknesses']='Playing with passion isn’t always a good thing, since he occasionally gets caried away by his emotions … He has an incosistent motor, with a lot of ups and downs even in the same game … He is a better athlete than most think, but he is still just an average athlete by NBA standards … He must bulk up considerably … Has a peculiar body structure with a slight hunch back with small chest and waist, so it’s not certain how he can really bulk ip … Lacks elite explosiveness … He doesn’t really have bounce off one foot when driving to the basket … He has a short wingspan (measured at 6-9) … He is mostly a streaky shooter for now.. He needs to improve in Catch and Shot and Spot  Up situations… His off the ball game needs work… He must learn to be effective when he doesn’t have the ball in his hands… Has problems against athletic wings on both ends of the floor… Can’t really finish through contact because his upper body isn’t strong enough… Has problems finishing against length because of his athletic limitations… He really likes going to his left and has problems when he is forced driving to his right… His mid-range game still needs improvement.. Should add some Post moves to his arsenal to explore miss-matches… Must learn to let game come to him when his shot isn’t falling and not force things…. He could improve his on the move passes… Can be turnover prone at times… His short wingspan limits him on defense… His lateral quickness is average… He is not always focused on defense, mainly because he is used to be “The Guy”at smaller level and didn’t want to spend energy on that end of the floor… Struggles on defense against explosive wings… His low stance on defense needs work… Needs to improve his off the ball defense… Gambles and goes for the steal, which adds pressure to his team…'
draft_post2000.loc[340,'Weaknesses']='Still a bit raw offensively … Feel for the game … Relies too much on his athletic ability at this point … He’s not a consistent threat as an offensive option from game-to-game … Has potential as a shooter, but needs to improve his shooting consistency from all areas of the court, including free throws … Needs to tweak his shooting mechanics … Low release on his shot … Not a threat to shoot off the dribble … Overall shot selection needs improvement … Needs to play more controlled and less reckless in the half court … Is sometimes prone to overdriving and getting caught in the lane … Has a tendency to throw up wild floaters and shots in and around the lane … Ball handling has room for improvement (needs to be tightened) … He isn’t a natural passer/playmaker … Will need to work on his playmaking ability for his teammates and overall decision-making … Needs to limit turnovers … Tends to have lapses on defense sometimes … Can get lost when playing off-the-ball defense … Doesn’t react well to opponent movement on-the-ball, and gets lost or lazy off-the-ball … Low steal and block rates despite physical tools … Closeouts are iffy at times … Doesn’t consistently stay in front and when he does he lacks the strength in his 195-pound frame to contain dribble penetration … Can create jump shots but struggles to connect with consistency … Needs to improve in all areas as a shooter; free throws, midrange, and three point. Shot 16.7% from three over the Nike EYBL … Still has room to tighten up his ball handling ability … Can be reckless taking the ball to the basket … Relies upon outjumping defenders at the rim … Can improve his post-game when matched up with smaller guards but does show confidence and willingness to take defenders into the post … Although he can find opportunities to find alley-oops and gaps when cutting backdoor, he can improve moving without the ball especially on the perimeter. Relies on his first step to beat his man after he has the ball … Has room to add upper body strength … Shows some playmaking ability but is strictly a scoring guard at this stage, doesn’t have the combo guard skills yet. Finished with 33 assists to 41 turnovers over the Nike EYBL …'
draft_post2000.loc[340,'Outlook']='Diallo may be the top athlete in high school basketball. His ball skills aren’t quite up to par with his natural ability yet but with some work and time, he can develop into a big time player at the highest levels.'
draft_post2000.loc[340,'Notes']='Measued 6’3.75 without shoes, 6’10.5 wingspan, 179 lbs and a 8’4.5 standing reach in the summer of 2015 … Measured 6’4 without shoes, 6’5 with shoes, 199 lbs, with a 6’11 wingspan and 8’5.5 standing reach at the June 2017 USA U19 tryouts'
draft_post2000.loc[409,'Weaknesses']='Likely to be overwhelmed by NBA athletes. A below average athlete by NBA standards … At 6’8, moves like a center in a small forward’s body … Lack of size at the 4 position (6’8) compounds his athleticism concerns … Very slow laterally and slow to react to plays … He lacks the speed to defend SFs and ideal size and strength to defend in the post … Not especially fast in the open floor. A bit of a plodder. Coming off a season ending injury (stress fracture in his foot) puts draft chances in jeopardy … Level of competition was low playing in the Horizon League … Shot well below expectations as a senior at 36.3% 3p, after hitting 44.0% and 46.6% from 3 as a Junior and Sophomore … Drop in efficiency may be partially due to teams game planning to limit his looks from behind the arch …'
draft_post2000.loc[415,'Weaknesses']='Although he is a good athlete, there are times his feet look a little stiff and slow …  He lacks elite explosiveness … He looks like he isn’t always 100% intense and focused, with too much of a laid back attitude … His reactions on both ends of the floor are a little slow, which could hurt him at the next level … He might be projected as a power forward, but he has been used almost exclusively as a center the last couple of years … His offensive game needs polishing, since he remains a little bit raw and depends too much on his physical attributes … Has some problems finishing against length … He is an incosistent shooter from long range … His footwork in the post is mediorce … His back to the basket game needs a lot of work … He can’t really play physical for now on offense, which makes him too predictable … His face up game is just average … He seems to prefer to play facing the basket rather than with his back to it, which can be a problem because he won’t be able to exploit miss-matches … Improving ball handler, but still not good enough for a power forward … Has good passing instincts, but his passes don’t always connect, because he often telegraphs them … It remains a question mark whether he will be able to follow Stretch-4s on the perimeter despite his great lateral quickness … Not always as commited as he should be on defensive rotations … At times he doesn’t have a good low stance on defense, which makes it easy for opponents to get past him … Not as a good rim protector as you would expect for a player with his physical attributes and skillset … Gambles a lot on defense, going for the steal, which adds presure to his team … Post defense is average and can be exploited by experienced and skillfull Bigs … Can get pushed around at times by bigger opponents … Not the most fluid player, legs are a bit stiff … Seems to prefer to drift out on the perimeter instead of using his size and strength around the basket … He improved his shape over this year, but he absolutely needs to work on his consistency and intensity. .. He’s a decent leaper, but lacks explosiveness and the “second jump” on defensive end … He has to improve his positioning, seems to lack game comprehension, probably due to the lack of experience at a high level … Lacks solid post moves, in order to properly utilize his physical and muscular advantage, he prefers to play facing the basket exploiting only his mobility … Overall impression is of a player currently relying on his physical attributes and instincts, raw potential still to be developed and nurtured in the proper environment … He has to show more competitive fire and aggressiveness to improve and compete at a higher level … Very inconsistent … Needs to get in better physical shape and gain stamina … Lacks a high level of experience, still plays in a junior, under-21 league … Doesn’t always play with enough intensity defensively … Not aggressive enough in closing out and pressuring opponent shots … From time to time too relaxed, especially in defense, Doesn’t protect the rim as much as he could … Can be too casual with his passing … Back to the basket game is underdeveloped, needs to focus a lot of attention to this … At times lacks a fundamental defensive stance, straight legs and up right posture … Rebounds the ball solely based on physical attributes, must learn to properly box out …'
draft_post2000.loc[415,'Notes']='He played the 2014-2015 season for Pau-Lacq-Orthez Espoirs (Juniors) team, then he moved this summer to Mega Leks in Adriatic league, thanks to the solid connections of his agent (Pedja Materic) with Misko Raznatovic, agent and owner of Mega … He had 8 points and 6 rebounds this season in ABA league … Former member of U18 and U20 French national teams …'
#draft_post2000.loc[705,'Weaknesses']=Izzet Turkyilmaz has a barebone analysis with no weaknesses
draft_post2000.loc[758,'Weaknesses']='His position will be a question mark in the NBA… Not a great decision maker, he does not play with the poise nor does he have the mentality of a playmaker … He does not make good decisions in traffic, he has a tendency of leaving his feet to make passes and does not usually deliver the ball on target … Has almost a 1/1 assist to turnover ratio for his career, further showing that he is a liability as a primary ballhandler … As an off-guard, he lacks the frame and mass to be able to battle with the bigger and stronger opponents … His biggest deficiency is the continued inconsistency with his jumpshot, which allows defenders to play off of him and take away his penetration … Even with the improvements, he still shoots a different jumper almost every attempt and his elbow tends to go too far to the outside … The release point also varies and he has a bad habit of hanging too long in the air and shooting on the way down (especially off the dribble) …  '
draft_post2000.loc[822,'Weaknesses']='His main strength is also a weakness, as his size and weight are major limiting factors … He is not a great athlete and does not get off the ground well … Settles for too many outside jumpers even though most of the time he has a distinct advantage by attacking inside … He struggles against taller and longer defenders because even though he can body them out of position, he still has a hard time finishing overtop of them … He undoubtedly carries too much weight and it shows in his footspeed and inability to get up and down the floor.  His lateral quickness is very limited and he has trouble defending quicker faceup forwards, or defending against the pick and roll … His motor is also a bit of a question mark at this point, he didn’t play major minutes this season and at times he seemed winded when playing for longer stretches … Battled with inconsistency all season – he was not able to put together a string of good games at any point during his freshman season … Very turnover prone, he struggles making plays in traffic or when the double team comes, he seems to panic and has a tendency to throw ill-advised passes …'
#draft_post2000.loc[1310,'Weaknesses']=Kwame Brown's analysis did not list weaknesses
draft_post2000.loc[1310,'NBA Comparison']='Kevin Garnett'
#draft_post2000.loc[1312,'Weaknesses']=Pau Gasol's analysis did not list weaknesses
draft_post2000.loc[1312,'NBA Comparison']='Toni Kukoc'
#draft_post2000.loc[1315,'Weaknesses']=Shane Battier's analysis did not have a weaknesses
draft_post2000.loc[1315,'NBA Comparison']='Bobby Jones'
draft_post2000.loc[1316,'Weaknesses']='Ball Handling abilities are not super… Needs to get alot stronger.. Needs to get nastier (on the court).. Character questions have resulted from two separate incidents involving off the court fights with teammates, one in HS one in College.'
draft_post2000.loc[1316,'NBA Comparison']='Tim Duncan'
#draft_post2000.loc[1318,'Weaknesses']=Rodney White's analysis did not have a weakness

#Now run again 
for i in draft_post2000.index:
    if pd.isna(draft_post2000.loc[i,'Strengths'])==False and pd.isna(draft_post2000.loc[i,'Weaknesses']):
        print(i, f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}', f'https://www.nbadraft.net/players/{draft_post2000.loc[i,"First Name"]}-{draft_post2000.loc[i,"Last Name"]}')
'''Output from above (these players did not have weaknesses in their analysis so this is all set):
705 Izzet Turkyilmaz https://www.nbadraft.net/players/Izzet-Turkyilmaz
1310 Kwame Brown https://www.nbadraft.net/players/Kwame-Brown
1312 Pau Gasol https://www.nbadraft.net/players/Pau-Gasol
1315 Shane Battier https://www.nbadraft.net/players/Shane-Battier
1318 Rodney White https://www.nbadraft.net/players/Rodney-White
'''

#check the total number of NAs again
draft_post2000['Overall'].isna().sum() #371, was 372
draft_post2000['NBA Comparison'].isna().sum() #607, was 616
draft_post2000['Strengths'].isna().sum() #378, was 395
draft_post2000['Weaknesses'].isna().sum() #383, was 399
draft_post2000['Outlook'].isna().sum() #1189, was 1192
draft_post2000['Notes'].isna().sum() #711, was 716
draft_post2000['OverTxt'].isna().sum() #1082
draft_post2000['Has_Page'].isna().sum() #0

####output the dataframe
#draft_post2000.to_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv',index=None,header=True)

###input the dataframe
#draft_post2000=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000.csv')

######brief look at data cleaning
import nltk
from nltk.tokenize import word_tokenize
print('\n' + draft_post2000.loc[1,'Strengths'])
temp_data=re.sub(r'[^A-Za-z]+',' ',draft_post2000.loc[1,'Strengths'])
tempWordTok=nltk.word_tokenize(temp_data)
for i in tempWordTok:
    if len(i) <= 3:
        tempWordTok.remove(i)
new_temp_data=' '.join(tempWordTok[i] for i in range(0,len(tempWordTok)))
temp_data=re.sub(r'Miller',' ',new_temp_data)
temp_data=re.sub(r'Brandon',' ',temp_data)
print('\n' + temp_data)

###################now use the sportsreference API to collect the stats that can be used to decide if a player is a bust or not
###################unfortunately the sportsreferenceAPI is no longer supported and appears to be broken, finding another solution

stats_link = 'https://www.basketball-reference.com/players/b/#players'
temptables=pd.read_html(stats_link)

stats_link = 'https://www.basketball-reference.com/players/b/brownja02.html#advanced'
temptables=pd.read_html(stats_link)
temp_DF=temptables[5].copy(deep=True)
temp_DF=temp_DF.set_index('Season',drop=True)
temp_DF.loc['Career','PER']

stats_link = 'https://www.basketball-reference.com/players/b/brownja02.html'
temptables=pd.read_html(stats_link)
temp_DF=temptables[5].copy(deep=True)
temp_DF=temp_DF.set_index('Season',drop=True)
temp_DF.loc['Career','PER']

#####in order to use sportsreference.com to look up stats, need to be able to find a players sportsreference playerID. This is generally formulated by taking the first 5 letters of their last name, plus the first 2 letters of their first name, plus 01 if they are the first player with that combination. So Jason Tatum's sportsreference ID is tatumja01. Jaylen Brown's however, is brownja02, since brownja01 already existed. Can attempt to get this info using the player lookup page.

tempplayerspage=requests.get(stats_link)
playerspagehtml=BeautifulSoup(tempplayerspage.text,'html.parser')
playerspagebody=playerspagehtml.find('tbody')    
playerspagefind=playerspagebody.find_all('a')
playerspagefind
for i in playerspagefind:
    if 'Jaylen Brown' in str(i):
        print(i)
        part_link=re.findall('href="(.*)"',str(i))
        temp_link='https://www.basketball-reference.com' + part_link[0] + '#advanced'
        playerlink=requests.get(temp_link)
        playerlinkhtml=BeautifulSoup(playerlink.text,'html.parser')
        playerlinkbio=playerlinkhtml.find(id='meta')
        playerlinka=playerlinkbio.find_all('a')
        for a in playerlinka:
            if 'NBA Draft' in str(a) and '2016' in str(a):
                print(a)

#####now try to put it all together
player_stats_DF=pd.DataFrame()
#for i in draft_post2000.index:
for i in range(1259,len(draft_post2000.index)):
#for i in range(415,416):
#for i in range(415,421):
    Full_Name=draft_post2000.loc[i,'First Name'] + ' ' + draft_post2000.loc[i,'Last Name']
    first_letter=draft_post2000.loc[i,'Last Name'][0].lower()
    stats_link = f'https://www.basketball-reference.com/players/{first_letter}/#players'
    tempplayerspage=requests.get(stats_link)
    playerspagehtml=BeautifulSoup(tempplayerspage.text,'html.parser')
    playerspagebody=playerspagehtml.find(id='players')    
    playerspagefind=playerspagebody.find_all('a')
    count=0
    for a in playerspagefind:
        if Full_Name in str(a):
            part_link=re.findall('href="(.*)"',str(a))
            for p in range(0,len(part_link)):
                temp_link=f'https://www.basketball-reference.com{part_link[p]}#advanced'
                playerlink=requests.get(temp_link)
                playerlinkhtml=BeautifulSoup(playerlink.text,'html.parser')
                playerlinkbio=playerlinkhtml.find(id='meta')
                playerlinka=playerlinkbio.find_all('a')
                for L in playerlinka:
                    if 'NBA Draft' in str(L) and str(draft_post2000.loc[i,'SEASON']) in str(L):
                        temptables=pd.read_html(temp_link)
                        for T in range(0,len(temptables)):
                            if 'PER' in temptables[T].columns:
                                temp_DF=temptables[T].copy(deep=True)
                                temp_DF=temp_DF.set_index('Season',drop=True)
                                PER=temp_DF.loc['Career','PER']
                                WS48=temp_DF.loc['Career','WS/48']
                                VORP=temp_DF.loc['Career','VORP']
                                TimeForDF=[Full_Name,PER,WS48,VORP]
                                TempStatInfo = pd.DataFrame([TimeForDF])
                                CombStatInfo=[player_stats_DF,TempStatInfo]
                                player_stats_DF=pd.concat(CombStatInfo)
                                count=count+1
                                print('Just finished ' + f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]}' + ', index: ' + str(i) + '. Now sleeping for 15 seconds...')
                                #now wait 15 seconds before doing the next player
                                time.sleep(15)
                                break
    if count==0:
        PER=''
        WS48=''
        VORP=''
        TimeForDF=[Full_Name,PER,WS48,VORP]
        TempStatInfo = pd.DataFrame([TimeForDF])
        CombStatInfo=[player_stats_DF,TempStatInfo]
        player_stats_DF=pd.concat(CombStatInfo)
        print(f'{draft_post2000.loc[i,"First Name"]} {draft_post2000.loc[i,"Last Name"]} does not have a page.' +', index: ' + str(i) + '. On to the next one...')
        time.sleep(15)
        
player_stats_DF=player_stats_DF.reset_index(drop=True)
player_stats_DF.rename(columns={'0':'Full Name','1':'PER','2':'WS_48','3':'VORP'},inplace=True)
#player_stats_DF.to_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/player_stats_DF.csv',index=None,header=True)
#player_stats_DF=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/player_stats_DF.csv')

######now join together the advanced stats with the pre-draft analysis
draft_post2000_comb=draft_post2000.join(player_stats_DF, draft_post2000.index,'left').drop(columns='Full Name')

#export and reimport
#draft_post2000_comb.to_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000_comb.csv',index=None,header=True)
#draft_post2000_comb=pd.read_csv('C:/Users/spitum1/OneDrive - Dell Technologies/Desktop/Misc/Grad/Courses/IST 736/Project/draft_post2000_comb.csv')