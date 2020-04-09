from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import pandas as pd
from ifn import *
import warnings
import argparse
warnings.filterwarnings('ignore')


TRAIN_PATH = "../data/processed/train2/"
REF_PATH = "../data/processed/reference/"
WRITE_PATH = "../data/models/"
DF_PATH = "../data/dataframes/"
CLASSIFICATION_THRESHOLD = 0.45
UPPER_CAP = 100
LOWER_CAP = 0.00001
EXPERIMENT_TYPE = 'ldc_0.45'

ref_df = None
temp_df = None

if os.path.exists(DF_PATH) == False:
	os.mkdir(DF_PATH)
	print('Created directory - ', DF_PATH)

if os.path.exists(DF_PATH+'ref_df.pkl') == False:
	ref_df = setup_df(REF_PATH)
	ref_df = get_audio_features(ref_df)
	ref_df.to_pickle(DF_PATH+'ref_df.pkl')
	print('Created file - ', DF_PATH+'ref_df.pkl')
else:
	ref_df = pd.read_pickle(DF_PATH+'ref_df.pkl')
if os.path.exists(DF_PATH+'temp_df.pkl') == False:
	temp_df = setup_df(TRAIN_PATH)
	temp_df.to_pickle(DF_PATH+'temp_df.pkl')
	print('Created file - ', DF_PATH+'temp_df.pkl')
else:
	temp_df = pd.read_pickle(DF_PATH+'temp_df.pkl')


trained_GMMs = get_trained_GMMs(ref_df)
avg_F0_ref = get_avg_F0_ref(ref_df)



def dump_train(to_dump, WRITE_PATH, EXPERIMENT_TYPE):
	with open(WRITE_PATH+EXPERIMENT_TYPE+'.pickle', 'wb') as handle:
		pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_global():

	# for every epoch
	epoch = 0

	ITERATION_CLF_REPORT = []
	ITERATION_FILE_CHANGE = []
	LDC_CLFS = []


	for iterations in range(400):

		sampled_df = stratified_sample_df(temp_df)
		
		train_df, test_df = train_test_split(sampled_df, test_size = 0.33, stratify = sampled_df[['speaker','speech_type']])
		
		train_df = get_audio_features(train_df)
		test_df = get_audio_features(test_df)
		
		train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
		train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)

		count = 0

		sampled_df_norm = None
		ldc_clf = None
		epsilon = 1000000
		max_iters = 1000
		
		ldc_clf = LinearDiscriminantAnalysis(solver='lsqr')
		
		print('*************************************************') 
		print('ITERATION NUMBER - ',iterations)
		print('*************************************************') 

				
		for stage in ['train', 'test']:

			print('----------------------------------------')   
			print('Stage - ', stage)


			if stage == 'train':
				
				ITERATION_FILE_CHANGE_ELEM = []
				
				while epsilon > 0.05 and count < max_iters:
			
					print('=========================================')   
					

					if count ==0 :
						train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
						train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)
						train_df = get_audio_features(train_df, norm=False)
					else:
						train_df['F0_contour_sum'] = train_df['F0_contour_norm'].apply(sum)
						train_df['F0_contour_length'] = train_df['F0_contour_norm'].apply(len)
						# Change above to F0_contour_norm when norm=True
						train_df = get_audio_features(train_df, norm=True)
							
					train_df['inferred'] = train_df.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)
					
					ldc_clf.fit(np.array(train_df['inferred'].tolist()),train_df['speech_type'].values)

					train_df['predicted_likelihood'] = ldc_clf.predict_proba(np.array(train_df['inferred'].tolist()))[:,0]

					train_df, epsilon = get_stopping_criteria(train_df, count, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)
					
					print(count)
					print(epsilon)
					ITERATION_FILE_CHANGE_ELEM.append(epsilon)
					LDC_CLFS.append(ldc_clf)
					
					
					train_df['prev_changed_speech_type'] = train_df['changed_speech_type']
									
					sampled_df_norm, train_df = get_normalised_df(train_df, avg_F0_ref=avg_F0_ref, get_S_s_F0=get_S_s_F0_global)                
					count+=1
					
				ITERATION_FILE_CHANGE.append(ITERATION_FILE_CHANGE_ELEM)

			else:
					sampled_df_test = test_df
					if count!=0:
						_, sampled_df_test = get_normalised_df_infer(test_df, sampled_df_norm)
					if count == 0 :
						sampled_df_test = get_audio_features(sampled_df_test, norm=False)
					else:
						sampled_df_test = get_audio_features(sampled_df_test, norm=True)
					sampled_df_test['inferred'] = sampled_df_test.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

					sampled_df_test['predicted_likelihood'] = ldc_clf.predict_proba(np.array(sampled_df_test['inferred'].tolist()))[:,0]

					sampled_df_test = get_pred_labels(sampled_df_test, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

					clf_report = classification_report(sampled_df_test['speech_type'],sampled_df_test['changed_speech_type'], output_dict=True)
					ITERATION_CLF_REPORT.append(clf_report)
					print(clf_report)
					
					

	to_dump = {'clf_report':ITERATION_CLF_REPORT, 'file_change':ITERATION_FILE_CHANGE, 'ldc_clfs':LDC_CLFS, 'gmms': trained_GMMs}

	return to_dump


def train_ifn():

	# for every epoch
	epoch = 0

	ITERATION_CLF_REPORT = []
	ITERATION_FILE_CHANGE = []
	LDC_CLFS = []


	for iterations in range(400):

		sampled_df = stratified_sample_df(temp_df)
		
		train_df, test_df = train_test_split(sampled_df, test_size = 0.33, stratify = sampled_df[['speaker','speech_type']])
		
		train_df = get_audio_features(train_df)
		test_df = get_audio_features(test_df)
		
		train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
		train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)

		count = 0

		sampled_df_norm = None
		ldc_clf = None
		epsilon = 1000000
		max_iters = 1000
		
		ldc_clf = LinearDiscriminantAnalysis(solver='lsqr')
		
		print('*************************************************') 
		print('ITERATION NUMBER - ',iterations)
		print('*************************************************') 

				
		for stage in ['train', 'test']:

			print('----------------------------------------')   
			print('Stage - ', stage)


			if stage == 'train':
				
				ITERATION_FILE_CHANGE_ELEM = []
				
				while epsilon > 0.05 and count < max_iters:
			
					print('=========================================')   
					

					if count ==0 :
						train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
						train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)
						train_df = get_audio_features(train_df, norm=False)
					else:
						train_df['F0_contour_sum'] = train_df['F0_contour_norm'].apply(sum)
						train_df['F0_contour_length'] = train_df['F0_contour_norm'].apply(len)
						# Change above to F0_contour_norm when norm=True
						train_df = get_audio_features(train_df, norm=True)
							
					train_df['inferred'] = train_df.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)
					
					ldc_clf.fit(np.array(train_df['inferred'].tolist()),train_df['speech_type'].values)

					train_df['predicted_likelihood'] = ldc_clf.predict_proba(np.array(train_df['inferred'].tolist()))[:,0]

					train_df, epsilon = get_stopping_criteria(train_df, count, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)
					
					print(count)
					print(epsilon)
					ITERATION_FILE_CHANGE_ELEM.append(epsilon)
					LDC_CLFS.append(ldc_clf)
					
					
					train_df['prev_changed_speech_type'] = train_df['changed_speech_type']
									
					sampled_df_norm, train_df = get_normalised_df(train_df, avg_F0_ref=avg_F0_ref, get_S_s_F0=get_S_s_F0)                
					count+=1
					
				ITERATION_FILE_CHANGE.append(ITERATION_FILE_CHANGE_ELEM)

			else:
					sampled_df_test = test_df
					if count!=0:
						_, sampled_df_test = get_normalised_df_infer(test_df, sampled_df_norm)
					if count == 0 :
						sampled_df_test = get_audio_features(sampled_df_test, norm=False)
					else:
						sampled_df_test = get_audio_features(sampled_df_test, norm=True)
					sampled_df_test['inferred'] = sampled_df_test.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

					sampled_df_test['predicted_likelihood'] = ldc_clf.predict_proba(np.array(sampled_df_test['inferred'].tolist()))[:,0]

					sampled_df_test = get_pred_labels(sampled_df_test, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

					clf_report = classification_report(sampled_df_test['speech_type'],sampled_df_test['changed_speech_type'], output_dict=True)
					ITERATION_CLF_REPORT.append(clf_report)
					print(clf_report)
					
					

	to_dump = {'clf_report':ITERATION_CLF_REPORT, 'file_change':ITERATION_FILE_CHANGE, 'ldc_clfs':LDC_CLFS, 'gmms': trained_GMMs}

	return to_dump



def train_optimal():

		# for every epoch
	epoch = 0

	ITERATION_CLF_REPORT = []
	ITERATION_FILE_CHANGE = []
	LDC_CLFS = []


	for iterations in range(400):

		

		sampled_df = stratified_sample_df(temp_df)

		train_df, test_df = train_test_split(sampled_df, test_size = 0.33, stratify = sampled_df[['speaker','speech_type']])

		train_df = get_audio_features(train_df)
		test_df = get_audio_features(test_df)

		train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
		train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)

		# for 400 iterations
		ldc_clf = None
		epsilon = 1000000
		max_iters = 1000

		ldc_clf = LinearDiscriminantAnalysis(solver='lsqr')

		print('*************************************************') 
		print('ITERATION NUMBER - ',iterations)
		print('*************************************************') 


		for stage in ['train', 'test']:

			print('----------------------------------------')   
			print('Stage - ', stage)


			if stage == 'train':

				ITERATION_FILE_CHANGE_ELEM = []
				sampled_df_norm, train_df = get_normalised_df(train_df, avg_F0_ref=avg_F0_ref, get_S_s_F0=get_S_s_F0_optimal)

				train_df['F0_contour_sum'] = train_df['F0_contour_norm'].apply(sum)
				train_df['F0_contour_length'] = train_df['F0_contour_norm'].apply(len)

				train_df = get_audio_features(train_df, norm=True)

				train_df['inferred'] = train_df.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

				ldc_clf.fit(np.array(train_df['inferred'].tolist()),train_df['speech_type'].values)

				train_df['predicted_likelihood'] = ldc_clf.predict_proba(np.array(train_df['inferred'].tolist()))[:,0]

				train_df, epsilon = get_stopping_criteria(train_df, 0, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

				print(epsilon)
				ITERATION_FILE_CHANGE_ELEM.append(epsilon)
				LDC_CLFS.append(ldc_clf)

				train_df['prev_changed_speech_type'] = train_df['changed_speech_type']

				ITERATION_FILE_CHANGE.append(ITERATION_FILE_CHANGE_ELEM)

			else:

				_, sampled_df_test = get_normalised_df_infer(test_df, sampled_df_norm)

				sampled_df_test['inferred'] = sampled_df_test.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

				sampled_df_test['predicted_likelihood'] = ldc_clf.predict_proba(np.array(sampled_df_test['inferred'].tolist()))[:,0]

				sampled_df_test = get_pred_labels(sampled_df_test, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

				clf_report = classification_report(sampled_df_test['speech_type'],sampled_df_test['changed_speech_type'], output_dict=True)
				ITERATION_CLF_REPORT.append(clf_report)
				print(clf_report)
					
	to_dump = {'clf_report':ITERATION_CLF_REPORT, 'file_change':ITERATION_FILE_CHANGE, 'ldc_clfs':LDC_CLFS, 'gmms': trained_GMMs}

	return to_dump

def train_unnorm():

		# for every epoch
	epoch = 0

	ITERATION_CLF_REPORT = []
	ITERATION_FILE_CHANGE = []
	LDC_CLFS = []


	for iterations in range(400):

		sampled_df = stratified_sample_df(temp_df)

		train_df, test_df = train_test_split(sampled_df, test_size = 0.35, stratify = sampled_df[['speaker','speech_type']])

		train_df = get_audio_features(train_df)
		test_df = get_audio_features(test_df)

		train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
		train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)


		# for 400 iterations
		ldc_clf = None
		epsilon = 1000000
		max_iters = 1000

		ldc_clf = LinearDiscriminantAnalysis(solver='lsqr')

		print('*************************************************') 
		print('ITERATION NUMBER - ',iterations)
		print('*************************************************') 


		for stage in ['train', 'test']:

			print('----------------------------------------')   
			print('Stage - ', stage)


			if stage == 'train':

				ITERATION_FILE_CHANGE_ELEM = []

				train_df['inferred'] = train_df.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

				ldc_clf.fit(np.array(train_df['inferred'].tolist()),train_df['speech_type'].values)

				train_df['predicted_likelihood'] = ldc_clf.predict_proba(np.array(train_df['inferred'].tolist()))[:,0]

				train_df, epsilon = get_stopping_criteria(train_df, 0, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

				print(epsilon)
				ITERATION_FILE_CHANGE_ELEM.append(epsilon)
				LDC_CLFS.append(ldc_clf)

				train_df['prev_changed_speech_type'] = train_df['changed_speech_type']

				ITERATION_FILE_CHANGE.append(ITERATION_FILE_CHANGE_ELEM)

			else:

				sampled_df_test = test_df

				sampled_df_test['inferred'] = sampled_df_test.apply(lambda x: infer_GMM(x, trained_GMMs),axis=1)

				sampled_df_test['predicted_likelihood'] = ldc_clf.predict_proba(np.array(sampled_df_test['inferred'].tolist()))[:,0]

				sampled_df_test = get_pred_labels(sampled_df_test, CLASSIFICATION_THRESHOLD=CLASSIFICATION_THRESHOLD)

				clf_report = classification_report(sampled_df_test['speech_type'],sampled_df_test['changed_speech_type'], output_dict=True)
				ITERATION_CLF_REPORT.append(clf_report)
				print(clf_report)
					
	to_dump = {'clf_report':ITERATION_CLF_REPORT, 'file_change':ITERATION_FILE_CHANGE, 'ldc_clfs':LDC_CLFS, 'gmms': trained_GMMs}

	return to_dump

def train_raw():

	print('Training Raw Ablation')

		# for every epoch
	epoch = 0

	ITERATION_CLF_REPORT = []
	ITERATION_FILE_CHANGE = []
	LDC_CLFS = []


	for iterations in range(400):

		sampled_df = stratified_sample_df(temp_df)

		train_df, test_df = train_test_split(sampled_df, test_size = 0.33, stratify = sampled_df[['speaker','speech_type']])

		train_df = get_audio_features(train_df)
		test_df = get_audio_features(test_df)

		train_df['F0_contour_sum'] = train_df['F0_contour'].apply(sum)
		train_df['F0_contour_length'] = train_df['F0_contour'].apply(len)


		# for 400 iterations
		ldc_clf = None

		ldc_clf = LinearDiscriminantAnalysis(solver='lsqr')

		print('*************************************************') 
		print('ITERATION NUMBER - ',iterations)
		print('*************************************************') 


		for stage in ['train', 'test']:

			print('----------------------------------------')   
			print('Stage - ', stage)


			if stage == 'train':

				ITERATION_FILE_CHANGE_ELEM = []

				train_df['inferred'] = train_df.apply(lambda x: infer(x),axis=1)

				ldc_clf.fit(np.array(train_df['inferred'].tolist()),train_df['speech_type'].values)

				LDC_CLFS.append(ldc_clf)

			else:

				sampled_df_test = test_df

				sampled_df_test['inferred'] = sampled_df_test.apply(lambda x: infer(x),axis=1)

				sampled_df_test['predicted_likelihood'] = ldc_clf.predict(np.array(sampled_df_test['inferred'].tolist()))

				clf_report = classification_report(sampled_df_test['speech_type'],sampled_df_test['predicted_likelihood'], output_dict=True)
				ITERATION_CLF_REPORT.append(clf_report)
				print(clf_report)
					
	to_dump = {'clf_report':ITERATION_CLF_REPORT, 'file_change':ITERATION_FILE_CHANGE, 'ldc_clfs':LDC_CLFS, 'gmms': trained_GMMs}

	return to_dump


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train Iterative Feature Normalisation and its variants")
	parser.add_argument("--experiment", help="Type of variant to run. A", choices=['norm','without','global','optimal','ablation'] ,type=str, required=True, default='norm')
	parser.add_argument("--train_path", help="Path to the train set", type=str, required=False, default="../data/processed/train2/")
	parser.add_argument("--ref_path", help="Path to the reference corpus", type=str, required=False, default="../data/processed/reference/")
	parser.add_argument("--write_path", help="Path to save the models", type=str, required=False, default="../data/models/")
	parser.add_argument("--dataframe_path", help="Path to the dataframes if available", type=str, required=False, default="../data/dataframes/")
	parser.add_argument("--threshold", help="Classification Threshold", type=float, required=False, default=0.5)
	parser.add_argument("--upper_cap", help="Upper cap on the scaling factor values", type=int, required=False, default=100)
	parser.add_argument("--lower_cap", help="Lower cap on the scaling factor values", type=float, required=False, default=0.00001)

	args = parser.parse_args()
	if args.train_path != None:
		TRAIN_PATH =  args.train_path
	if args.ref_path != None:
		REF_PATH = args.ref_path
	if args.write_path != None:
		WRITE_PATH = args.write_path 
	if args.dataframe_path != None:
		DF_PATH = args.dataframe_path
	if args.threshold != None:
		CLASSIFICATION_THRESHOLD = args.threshold
	if args.upper_cap != None:
		args.upper_cap = 100
	if args.lower_cap != None:
		LOWER_CAP = args.lower_cap

	EXPERIMENT = agrs.experiment


	EXPERIMENT_TYPE = 'ldc_0.45'

	to_dump = None
	if EXPERIMENT == 'norm':
		to_dump = train_ifn()
	elif EXPERIMENT == 'without':
		to_dump = train_unnorm()
	elif EXPERIMENT == 'global':
		to_dump = train_global()
	elif EXPERIMENT == 'optimal':
		to_dump = train_optimal()
	elif EXPERIMENT == 'ablation':
		to_dump = train_raw()
	dump_train(to_dump, WRITE_PATH, EXPERIMENT_TYPE)
	print('------------------COMPLETE------------------')








