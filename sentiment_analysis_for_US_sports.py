import io, json
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import TwitterCookbook as tc
import nltk

nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('stopwords')

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
import string

#stop = stopwords.words('english') these are words that we want to remove from our tweets becuase they have no relevance for positivity / negativity
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
        "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
        'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for','n', 
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'so', 'than', 'too']


states_dictionary = {
'ALABAMA' : "[-88.473227 30.223334 -84.88908 35.008028]",
'ALASKA' : "[-179.148909 51.214183 179.77847 71.365162]",
'ARIZONA' : "[-114.81651 31.332177 -109.045223 37.00426]", 
'ARKANSAS' : "[-94.617919 33.004106 -89.644395 36.4996]",
'CALIFORNIA' : "[-124.409591 32.534156 -114.131211 42.009518]",
'COLORADO' : "[-109.060253 36.992426 -102.041524 41.003444]",
'CONNECTICUT' : "[-73.727775 40.980144 -71.786994 42.050587]",
'DELAWARE' : "[-75.788658 38.451013 -75.048939 39.839007]",
'FLORIDA' : "[-87.634938 24.523096 -80.031362 31.000888]",
'GEORGIA' : "[-85.605165 30.357851 -80.839729 35.000659]",
'HAWAII' : "[-178.334698 18.910361 -154.806773 28.402123]",
'IDAHO' : "[-117.243027 41.988057 -111.043564 49.001146]",
'ILLINOIS' : "[-91.513079 36.970298 -87.494756 42.508481]",
'INDIANA' : "[-88.09776 37.771742 -84.784579 41.760592]",
'IOWA' : "[-96.639704 40.375501 -90.140061 43.501196]",
'KANSAS' : "[-102.051744 36.993016 -94.588413 40.003162]",
'KENTUCKY' : "[-89.571509 36.497129 -81.964971 39.147458]",
'LOUISIANA' : "[-94.043147 28.928609 -88.817017 33.019457]",
'MAINE' : "[-71.083924 42.977764 -66.949895 47.459686]",
'MARYLAND' : "[-79.487651 37.911717 -75.048939 39.723043]",
'MASSACHUSETTS' : "[-73.508142 41.237964 -69.928393 42.886589]",
'MICHIGAN' : "[-90.418136 41.696118 -82.413474 48.2388]",
'MINNESOTA' : "[-97.239209 43.499356 -89.491739 49.384358]",
'MISSISSIPPI' : "[-91.655009 30.173943 -88.097888 34.996052]",
'MISSOURI' : "[-95.774704 35.995683 -89.098843 40.61364]",
'MONTANA' : "[-116.050003	44.358221 -104.039138 49.00139]",
'NEBRASKA' : "[-104.053514 39.999998 -95.30829 43.001708]",
'NEVADA' : "[-120.005746 35.001857 -114.039648 42.002207]",
'NEW_HAMPSHIRE' : "[-72.557247 42.69699 -70.610621 45.305476]",
'NEW_JERSEY' : "[-75.559614 38.928519 -73.893979 41.357423]",
'NEW_MEXICO' : "[-109.050173 31.332301 -103.001964 37.000232]",
'NEW_YORK' : "[-79.762152 40.496103 -71.856214 45.01585]",
'NORTH_CAROLINA' : "[-84.321869 33.842316	-75.460621 36.588117]",
'NORTH_DAKOTA' : "[-104.0489 45.935054 -96.554507 49.00057]",
'OHIO' : "[-84.820159 38.403202 -80.518693 41.977523]",
'OKLAHOMA' : "[-103.002565 33.615833 -94.430662 37.002206]",
'OREGON' : "[-124.566244 41.991794 -116.463504 46.292035]",
'PENNSYLVANIA' : "[-80.519891 39.7198 -74.689516 42.26986]",
'RHODE_ISLAND' : "[-71.862772	41.146339 -71.12057	42.018798]",
'SOUTH_CAROLINA' : "[-83.35391 32.0346 -78.54203 35.215402]",
'SOUTH_DAKOTA' : "[-104.057698 42.479635 -96.436589 45.94545]",
'TENNESSEE' : "[-90.310298 34.982972 -81.6469 36.678118]",
'TEXAS' : "[-106.645646 25.837377 -93.508292 36.500704]",
'UTAH' : "[-114.052962 36.997968 -109.041058 42.001567]",
'VERMONT' : "[-73.43774 42.726853 -71.464555 45.016659]",
'VIRGINIA' : "[-83.675395	36.540738 -75.242266 39.466012]",
'WASHINGTON' : "[-124.763068 45.543541 -116.915989 49.002494]",
'WEST_VIRGINIA': "[-82.644739 37.201483 -77.719519 40.638801]",
'WISCONSIN':"[-92.888114 42.491983 -86.805415 47.080621]",
'WYOMING':"[-111.056888 40.994746 -104.05216 45.005904]"
}
us_state_abbrev = {
    'ALABAMA': 'AL',
    'ALASKA': 'AK',
    'ARIZONA': 'AZ',
    'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA',
    'COLORADO': 'CO',
    'CONNECTICUT': 'CT',
    'DELAWARE': 'DE',
    'FLORIDA': 'FL',
    'GEORGIA': 'GA',
    'HAWAII': 'HI',
    'IDAHO': 'ID',
    'ILLINOIS': 'IL',
    'INDIANA': 'IN',
    'IOWA': 'IA',
    'KANSAS': 'KS',
    'KENTUCKY': 'KY',
    'LOUISIANA': 'LA',
    'MAINE': 'ME',
    'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA',
    'MICHIGAN': 'MI',
    'MINNESOTA': 'MN',
    'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO',
    'MONTANA': 'MT',
    'NEBRASKA': 'NE',
    'NEVADA': 'NV',
    'NEW_HAMPSHIRE': 'NH',
    'NEW_JERSEY': 'NJ',
    'NEW_MEXICO': 'NM',
    'NEW_YORK': 'NY',
    'NORTH_CAROLINA': 'NC',
    'NORTH_DAKOTA': 'ND',
    'OHIO': 'OH',
    'OKLAHOMA': 'OK',
    'OREGON': 'OR',
    'PENNSYLVANIA': 'PA',
    'RHODE_ISLAND': 'RI',
    'SOUTH_CAROLINA': 'SC',
    'SOUTH_DAKOTA': 'SD',
    'TENNESSEE': 'TN',
    'TEXAS': 'TX',
    'UTAH': 'UT',
    'VERMONT': 'VT',
    'VIRGINIA': 'VA',
    'WASHINGTON': 'WA',
    'WEST_VIRGINIA': 'WV',
    'WISCONSIN': 'WI',
    'WYOMING': 'WY'
}

sports= ["Basketball", "Baseball", "Hockey", "Football", "NBA", "MLB", "NHL", "NFL"]

def save_json(filename, data):
    with open('json_files/{0}.json'.format(filename),
              'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        
def load_json(filename):
    with open('json_files/{0}.json'.format(filename), 
              'r', encoding='utf-8') as f:
        return json.load(f)


def gather_tweets(states_dict):
    #Twitter's API connection
    twitter_api = tc.oauth_login() 
    sports= ["Basketball", "Baseball", "Hockey", "Football", "NBA", "MLB", "NHL", "NFL"]
    for state in states_dict: #go through every state
        for sport in sports: #in each state, go through every sport
            #query = 'Basketball -filter:retweets'
            q= sport + '-filter:retweets'

            #the twitter search
            results = tc.make_twitter_request(tc.twitter_search, twitter_api=twitter_api, q=q, lang='en',
                                            max_results=1000, bounding_box=states_dict[state])
            #save results
            save_json(state+'_'+sport, results)
            
            #print(results)
            for i in results:
                print(i['text']) # the actual tweet

def gather_tweets2(states_dict):
    #Twitter's API connection
    twitter_api = tc.oauth_login() 
    sports= ["Basketball", "Baseball", "Hockey", "Football", "NBA", "MLB", "NHL", "NFL"]
    for state in states_dict: #go through every state
        for sport in sports: #in each state, go through every sport
            #query = 'Basketball -filter:retweets'
            q= sport + '-filter:retweets'

            #the twitter search
            results = tc.make_twitter_request(tc.twitter_search, twitter_api=twitter_api, q=q, lang='en',
                                            max_results=1000, bounding_box=states_dict[state])
            #save results
            save_json(state+'_'+sport+"_2", results)
            
            #print(results)
            for i in results:
                print(i['text']) # the actual tweet
        
def checkword(word, inverted):
    dictionary = {
        #bad
        'ass': -0.625, 'sucks':-0.5, 'garbage':-0.625,
        'lame':-0.375, 'stinks':-0.375,
        
        #good
        'lit':0.7, 'dope':0.5, 'fire':0.8,
        'great':0.8
    }
    if word in dictionary:
        if not inverted:
            return dictionary[word]
        else:
            return -1 * dictionary[word]
    else:
        return 0
            

def sentiment_analysis(sentence):
    num = '01'
    count = 0
    word_length = 0
    
    inverted = False
    words = sentence.split(' ')
    for i in words:
        if i in stop: #if our word is part of the stop words, then we remove it. 
            pass
        elif len(i) > 2: 
            val = checkword(i,inverted) #check if the word is part of our dictionary
            if i == 'not':
                inverted = True
            elif val == 0:
                try:   
                    synset = swn.senti_synset(i+'.a.'+num)
                    word = synset.__str__()
                    word_length += 1
                    if not inverted:
                        count += synset.pos_score()
                        count -= synset.neg_score()
                    else:
                        count -= synset.pos_score()
                        count += synset.neg_score()
                        inverted = False
                    #print(i,synset)
                except:
                    try:   
                        synset = swn.senti_synset(i+'.n.'+num)
                        word = synset.__str__()
                        word_length += 1
                        if not inverted:
                            count += synset.pos_score()
                            count -= synset.neg_score()
                        else:
                            count -= synset.pos_score()
                            count += synset.neg_score()
                            inverted = False
                        #print(i,synset)
                    except:
                        try:   
                            synset = swn.senti_synset(i+'.v.'+num)
                            word = synset.__str__()
                            word_length += 1
                            if not inverted:
                                count += synset.pos_score()
                                count -= synset.neg_score()
                            else:
                                count -= synset.pos_score()
                                count += synset.neg_score()
                                inverted = False
                            #print(i,synset)
                        except:
                            try:   
                                synset = swn.senti_synset(i+'.r.'+num)
                                word = synset.__str__()
                                word_length += 1
                                if not inverted:
                                    count += synset.pos_score()
                                    count -= synset.neg_score()
                                else:
                                    count -= synset.pos_score()
                                    count += synset.neg_score()
                                    inverted = False
                                #print(i,synset)
                            except:
                                try:   
                                    synset = swn.senti_synset(i+'.s.'+num)
                                    word = synset.__str__()
                                    word_length += 1
                                    if not inverted:
                                        count += synset.pos_score()
                                        count -= synset.neg_score()
                                    else:
                                        count -= synset.pos_score()
                                        count += synset.neg_score()
                                        inverted = False
                                    #print(i,synset)
                                except:
                                    pass
            else:
                count += val
                word_length += 1
            
    if word_length == 0:
        return 0
    score = count / word_length
    return score

#dictionaries in which we will save our data so we can save them in json files. 
sent_results = {} #ALL the tweets values, sorted by positive and negatives.  
sent_averages_positives = {} #Stores a single positive value for the positive tweets for each state and sport
sent_averages_negatives = {} #stores a single negative value for the negative tweets for each state and sport
def do_analysis_separated():
    for state in states_dictionary: 
        for sport in sports:
            print(state+'_'+sport)
            results = load_json(state+'_'+sport) #this is our raw twitter data from the search
            
            total_positive = 0
            total_negative = 0
            index_positive = 0
            index_negative = 0
            
            for twitter_tweet in results:
                sent_result = sentiment_analysis(twitter_tweet['text']) #run the tweet through our sentiment analysis function
                if sent_result != 0: #if our tweet has sentimental value 
                    if sent_result > 0: #positive, store the score in sent_results, with a specific key
                        sent_results[str(state)+'_'+str(sport)+'_positive_'+str(index_positive)] = sent_result
                        total_positive += sent_result
                        index_positive += 1
                    else: #negative, store the score in sent_results with a specific key 
                        sent_results[str(state)+'_'+str(sport)+'_negative_'+str(index_negative)] = sent_result
                        total_negative += sent_result
                        index_negative += 1
                        
            #since we have two files for each state and sport, this part is a duplicate but for the second set of results.
            print(state+'_'+sport+'_2')
            results = load_json(state+'_'+sport+'_2') 
            for twitter_tweet in results:
                sent_result = sentiment_analysis(twitter_tweet['text'])
                if sent_result != 0:
                    if sent_result > 0:
                        sent_results[str(state)+'_'+str(sport)+'_positive_'+str(index_positive)] = sent_result
                        total_positive += sent_result
                        index_positive += 1
                    else:
                        sent_results[str(state)+'_'+str(sport)+'_negative_'+str(index_negative)] = sent_result
                        total_negative += sent_result
                        index_negative += 1
            
            #once we have analyzed all the tweets, we store the total positive / total tweets. 
            #This is useful for when we want to calculate the average tweet score for the green bar in the graphs.
            sent_averages_positives[str(state)+'_'+str(sport)] = round((total_positive / (index_positive + index_negative))* 1000, 2) 
            sent_averages_negatives[str(state)+'_'+str(sport)] = round((total_negative / (index_positive + index_negative))* 1000, 2) 
            
    #save files
    save_json('sentiment_analysis_data_separated',sent_results)
    save_json('sentiment_analysis_data_averages_separated_positives',sent_averages_positives)
    save_json('sentiment_analysis_data_averages_separated_negatives',sent_averages_negatives)

def graph_state_all_sports_helper(sent_pos, sent_neg, state):
    x_axis_label = []
    positive_data = []
    negative_data = []
    diff_data = []
    for sport in sports:
        key = state+'_'+sport 
        val_pos = sent_pos[str(key)] #positive sentimental score 
        val_neg = sent_neg[str(key)] #negative sentimental score 
        x_axis_label.append(sport)
        positive_data.append(val_pos)
        negative_data.append(val_neg)
        diff_data.append(val_pos + val_neg)
    
    fig = plt.figure(figsize = (12,9))
    ax = plt.subplot(111)
    ax.bar(x_axis_label, positive_data, width=.9, color='b', label='Positive Score')
    ax.bar(x_axis_label, negative_data, width=.9, color='r', label='Negative Score')
    ax.bar(x_axis_label, diff_data, width = .9, color='g', label='Net Score')
    ax.legend()
    plt.title('Sentiment Analysis Scores for ' + state)
    plt.ylim(-200,200)
    plt.savefig('graphs_state_all_sports/'+state+'.png')
    #plt.show()
    
def graph_state_all_sports():
    filename = 'sentiment_analysis_data_averages_separated_positives'
    sent_pos = load_json(filename)
    filename = 'sentiment_analysis_data_averages_separated_negatives'
    sent_neg = load_json(filename)
    for state in states_dictionary:
        graph_state_all_sports_helper(sent_pos,sent_neg, state)

def graph_sports_all_states_helper(sent_pos, sent_neg, sport):
    x_axis_label = []
    positive_data = []
    negative_data = []
    diff_data = []
    
    list_of_states = states_dictionary.keys()
    for state in list_of_states: #for each state
        key = state+'_'+sport 
        val_pos = sent_pos[str(key)] #positive score
        val_neg = sent_neg[str(key)] #negative score
        x_axis_label.append(us_state_abbrev[state]) # the label is the abbreviation of the state
        positive_data.append(val_pos) 
        negative_data.append(val_neg)
        diff_data.append(val_pos + val_neg)
        
    fig = plt.figure(figsize = (24,16))
    ax = plt.subplot(111)
    ax.bar(x_axis_label, positive_data, width=.9, color='b', label='Positive Score')
    ax.bar(x_axis_label, negative_data, width=.9, color='r', label='Negative Score')
    ax.bar(x_axis_label, diff_data, width = .9, color='g', label='Net Score')
    ax.legend()
    plt.title('Sentiment Analysis Scores for ' + sport)
    plt.ylim(-200,200)
    plt.savefig('graphs_sports_all_states/'+sport+'.png')
    
    
def graph_sports_all_states():
    filename = 'sentiment_analysis_data_averages_separated_positives'
    sent_pos = load_json(filename)
    filename = 'sentiment_analysis_data_averages_separated_negatives'
    sent_neg = load_json(filename)
    for sport in sports:
        graph_sports_all_states_helper(sent_pos,sent_neg,sport)

def graph_states_all_sports_passion_helper(sent_scores, state):
       
    x_axis_label = []
    positive_data = []
    negative_data = []
    diff_data = []
    for sport in sports:
        sent_positives = 0
        sent_positive_count = 0
        sent_negatives = 0
        sent_negative_count = 0
        #get positives
        key = state + "_"+sport+"_positive_"+str(sent_positive_count)
        while key in sent_scores:
            sent_positives += sent_scores[key]
            sent_positive_count+=1
            key = state + "_"+sport+"_positive_"+str(sent_positive_count)
            
        #get negatives
        key = state + "_"+sport+"_negative_"+str(sent_negative_count)
        while key in sent_scores:
            sent_negatives += sent_scores[key]
            sent_negative_count+=1
            key = state + "_"+sport+"_negative_"+str(sent_negative_count)

        x_axis_label.append(sport)
        val_pos = round((sent_positives/sent_positive_count) * 1000, 2) 
        val_neg = round((sent_negatives/sent_negative_count) * 1000, 2)
        positive_data.append(val_pos)
        negative_data.append(val_neg)
        diff_data.append(val_pos + val_neg)
    
    fig = plt.figure(figsize = (12,9))
    ax = plt.subplot(111)
    ax.bar(x_axis_label, positive_data, width=.9, color='b', label='Positive Score')
    ax.bar(x_axis_label, negative_data, width=.9, color='r', label='Negative Score')
    ax.bar(x_axis_label, diff_data, width = .9, color='g', label='Net Score')
    ax.legend()
    plt.title('Sentiment Analysis Scores for ' + state + " Passion Comparison")
    plt.ylim(-200,200)
    plt.savefig('graphs_states_passion/'+state+'.png')

def graph_states_all_sports_passion():
    filename = 'sentiment_analysis_data_separated'
    sent_file = load_json(filename)
    for state in states_dictionary:
        graph_states_all_sports_passion_helper(sent_file, state)
        
def main(): 
    #to gather tweets from all 50 states, you must have a folder called json_files in the same directory as this file. 
    #All tweets will be stored in there, labeled by their {state}_{sport}.
    
    #gather_tweets(states_dictionary)
    #gather_tweets2(states_dictionary)
    
    #Gathered the tweets twice, once on an earlier date and once again a few days later. We have two different functions for them becuase the we saved the data in different files for each state + sport
    ###############################################################
    #to run the tweets through our sentimental analysis, call:
    
    #do_analysis()
    
    #this will put all the scores into json_files, in files called sentiment_analysis_{...} For the scores, do_analysis() combines positive and negatives by adding or subtracting
    #in our graphs, we display both the negative and positive values calculated, so we saved the data into two parts, the negative scores and the positive scores. This can be done
    #by calling:
    
    #do_analysis_separated()
    
    #tweets saved in 'json_files/sentiment_analysis_data_separated'
    #positive scores saved in 'json_files/sentiment_analysis_data_averages_separated_positives'
    #negative scores saved in 'json_files/sentiment_analysis_data_averages_separated_negatives'
    
    ###############################################################
    #to produce the graphs that we created, you can call these two functions. You must create folders called graphs_state_all_sports and graphs_sports_all_states in the same directory as this file.
    
    #produces a graph for each sport, where each graph has all 50 states on it
    
    #graph_sports_all_states()
    
    #produces a graph for each state, where each graph has all the sports on it
    
    #graph_state_all_sports()
    
    ##############################################################
    #passion: (graph 1 in report) You must have a folder called graphs_states_passion in the same directory as this folder. 
    #graph_states_all_sports_passion()
    
    pass

main()