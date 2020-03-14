# SER_LSTM_librosa
 
Speech Emotion Recognition System using LSTM(Long short-term memory, RNN)

Dependencies: librosa, numpy, sklearn, keras (Tensorflow), matplotlib

Corpos: RAVDESS (English, 1500 audios from 24 people)

Train (Features): librosa features sets*

flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent, maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd, pitch_tuning_offset, meanrms, maxrms, stdrms, mfccs, mfccsstd, mfccmax, chroma, mel, contrast
Parameter seeting: alpha = 1.9, max_iter = 700

Result: Accuracy: 69.1% (1 Epochs) --> 93.4% (20 Epochs)

Parameter seeting: n_epochs = 20, batch_size = , activation='relu' (Rectified Linear Unit(ReLU))

*MEMO: Files (corpus) should be categorized and stored in different folders by labels (emotions)
