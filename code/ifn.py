import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import os
import pickle
from feature_extraction import *
import warnings
warnings.filterwarnings('ignore')

features = [ 'SQ25', 'SQ75','F0_median', 'sdmedian', 'IDR', 'SVMeanRange','SVMaxCurv']


def setup_df(PATH):
    all_files = os.listdir(PATH)
    df = pd.DataFrame(all_files, columns=['file'])
    df['speaker'] = df['file'].apply(lambda x: int(x.split('.')[0].split('-')[-1]))
    # 0 for neutral, 1 for emotional
    df['speech_type'] = df['file'].apply(lambda x: 0 if int(x.split('.')[0].split('-')[2])==1 else 1)
    df['F0_contour'] = df['file'].apply(lambda x: get_F0_contour(PATH+x))
    return df

def get_audio_features(df, norm=False):
    if norm==False:
        df['SQ25'] = df['F0_contour'].apply(get_SQ25)
        df['SQ75'] = df['F0_contour'].apply(get_SQ75)
        df['IDR'] = df['F0_contour'].apply(get_IDR)
        df['voiced_segments'] = df['F0_contour'].apply(get_voiced_segments)
        df['F0_median'] = df['F0_contour'].apply(get_F0_median)
        df['sdmedian'] = df['F0_contour'].apply(get_sdmedian)
    else:
        df['SQ25'] = df['F0_contour_norm'].apply(get_SQ25)
        df['SQ75'] = df['F0_contour_norm'].apply(get_SQ75)
        df['IDR'] = df['F0_contour_norm'].apply(get_IDR)
        df['voiced_segments'] = df['F0_contour_norm'].apply(get_voiced_segments)
        df['F0_median'] = df['F0_contour_norm'].apply(get_F0_median)
        df['sdmedian'] = df['F0_contour_norm'].apply(get_sdmedian)
    
    df['SVMeanRange'] = df['voiced_segments'].apply(get_voiced_segment_range)
    df['SVMaxCurv'] = df['voiced_segments'].apply(get_max_voiced_curvature)
    
    return df

def get_trained_GMMs(df):
    trained_GMMs = {}

    for feature in features:
        gmm = GaussianMixture(n_components=2)
        gmm.fit(df[feature].values.reshape(-1,1))
        trained_GMMs[feature] = gmm
    return trained_GMMs

def infer_GMM(df, trained_GMMs):
    infered = np.zeros(shape=(1,len(features)))
    for i in range(len(features)):
        res = trained_GMMs[features[i]].score_samples([[df[features[i]]]])
        infered[:,i] = res
    return infered[0]


def get_avg_F0_ref(ref_df):
    ref_df['F0_contour_sum'] = ref_df['F0_contour'].apply(sum)
    ref_df['F0_contour_length'] = ref_df['F0_contour'].apply(len)
    ref_df['F0_contour_mean'] = ref_df['F0_contour_sum']/ref_df['F0_contour_length']
    return np.sum(ref_df['F0_contour_mean']) / 24

def stratified_sample_df(df):
    per_speaker_neutral_number = df[df['speech_type']==0]['speaker'].value_counts().values[0]
    sampled_df = []
    sampled_df.append(df[df['speech_type']==0])
    speakers = df['speaker'].unique()
    for speaker in speakers:
        speaker_df = df[df['speaker'] == speaker]
        speaker_emotional_df= speaker_df[speaker_df['speech_type']==1]
        sampled_df.append(speaker_emotional_df.sample(n=per_speaker_neutral_number))
    return pd.concat(sampled_df).reset_index(drop=True)


def get_changed_labels(neutral_list, emotional_list, row):
    if row['file'] in neutral_list:
        return 0
    else:
        return 1

def get_S_s_F0(F0_ref, df, UPPER_CAP=100, LOWER_CAP=0.00001):
    df_neu = df[df['changed_speech_type']==0]
    speakers = df['speaker'].unique()
    grouped_df_neu = {}
    UPPER_CAP = 100
    LOWER_CAP = 0.00001
    for speaker in speakers:
        speaker_df_neu = df_neu[df_neu['speaker']==speaker]
        speaker_mean = (speaker_df_neu['F0_contour_sum']/speaker_df_neu['F0_contour_length']).mean()
        if F0_ref/speaker_mean > UPPER_CAP:
            grouped_df_neu[speaker] = UPPER_CAP
        elif F0_ref/speaker_mean < LOWER_CAP:
            grouped_df_neu[speaker] = LOWER_CAP
        else:
            grouped_df_neu[speaker] = F0_ref/speaker_mean

    return grouped_df_neu
    
    
def get_normalised_df(df, avg_F0_ref, get_S_s_F0, UPPER_CAP=100, LOWER_CAP=0.00001):
    df_norm = get_S_s_F0(avg_F0_ref, df, UPPER_CAP=UPPER_CAP, LOWER_CAP=LOWER_CAP)
    df['F0_contour_norm'] = df.apply(lambda x: x['F0_contour']/df_norm[x['speaker']], axis=1)
    return df_norm, df
        

def get_normalised_df_infer(df, df_norm):
    df['F0_contour_norm'] = df.apply(lambda x: x['F0_contour']/df_norm[x['speaker']], axis=1)
    return df_norm, df

def get_pred_labels(df, CLASSIFICATION_THRESHOLD=0.5):
    
    grouped_sampled_df = df.groupby('speaker')
    neutral = []
    emotional = []

    for name, group in grouped_sampled_df:
        neu_result = group[group['predicted_likelihood'] >= CLASSIFICATION_THRESHOLD]
        emo_result = group[group['predicted_likelihood'] < CLASSIFICATION_THRESHOLD]

        total = group.shape[0]
        to_add = int(np.ceil(0.2 * total)) - neu_result.shape[0]
        converted_neu_add = neu_result
        if to_add > 0:
            emo_result_sort = emo_result.sort_values('predicted_likelihood', ascending=False)
            converted_neu_add = emo_result_sort.head(to_add)
            emo_result.drop(converted_neu_add.index, inplace=True)

        neutral.extend(converted_neu_add['file'].tolist())
        emotional.extend(emo_result['file'].tolist()) 

    df['changed_speech_type'] = df.apply(lambda x:get_changed_labels(neutral,emotional,x), axis=1) 
    
    return df
    
def get_stopping_criteria(df, count, CLASSIFICATION_THRESHOLD=0.5):
    df = get_pred_labels(df,CLASSIFICATION_THRESHOLD=0.5)   
    if count==0:
        return df, 1000000
    else:
        changed_dict = (df['prev_changed_speech_type'] != df['changed_speech_type']).value_counts().to_dict()

        if True not in changed_dict.keys():
            changed_dict[True] = 0
            epsilon = changed_dict[True]/changed_dict[False]
        elif False not in changed_dict.keys():
            epsilon = 1000000
        else:
            epsilon = changed_dict[True]/changed_dict[False]
    
    return df, epsilon
